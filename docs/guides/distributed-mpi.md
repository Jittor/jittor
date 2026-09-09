# 用 MPI 做分布式训练

Jittor 可以用 MPI 启动数据并行的多个 worker。集合通信在 NVIDIA 设备上可以走 NCCL，
训练算法本身仍然写在 Python 里。

## 安装与探测 Open MPI

Debian / Ubuntu 上安装启动器与开发头文件：

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

Jittor 从 `PATH` 里找 `mpicc`。只有当编译器包装器装在非标准位置时才需要显式指定：

```bash
export mpicc_path=/opt/openmpi/bin/mpicc
```

用 `mpirun` 启动同一个训练程序：

```bash
# 单进程
python train.py

# 四进程
mpirun -np 4 python train.py

# 指定两张 NVIDIA 卡
CUDA_VISIBLE_DEVICES=2,3 mpirun -np 2 python train.py
```

数据集层会在 worker 之间划分数据，受支持的优化器会同步梯度，同步 BN 层会交换统计量。
**把数据集配置的批次大小当作全局批次大小**，并自行确认自定义 sampler 的行为。

## 副作用只放在 rank 0

每个 worker 都会执行整个 Python 程序。把文件写入与日志限制在 rank 0：

```python
optimizer.step(loss)

if jt.rank == 0:
    writer.add_scalar("train/loss", float(loss.item()))
```

**不要把 Jittor 算子放进只有某个 rank 才进入的分支。** 所有 worker 必须构造兼容的图、
以相同顺序调用集合通信，否则会死锁。

## 聚合验证指标

作为全局指标使用之前先归约：

```python
correct = local_correct(output, labels)
correct.sync()
if jt.in_mpi:
    correct = correct.mpi_all_reduce("add")

if jt.rank == 0:
    print("global correct:", int(correct.item()))
```

要让某个操作只在一个进程里跑而又不想手写分支，用 `jt.single_process_scope()`：

```python
@jt.single_process_scope()
def write_summary(metrics):
    save_summary(metrics)
```

## 公开的 MPI 状态与集合通信

| 名称 | 含义 |
| --- | --- |
| `jt.in_mpi` | 本进程是否处在 MPI world 中 |
| `jt.world_size` | worker 数量；非 MPI 时为 `1` |
| `jt.rank` | 当前 worker 编号；非 MPI 时为 `0` |
| `Module.mpi_param_broadcast(root=0)` | 广播模块参数 |
| `Var.mpi_reduce(op="add", root=0)` | 归约到某一个 worker |
| `Var.mpi_broadcast(root=0)` | 从某一个 worker 广播 |
| `Var.mpi_all_reduce(op="add")` | 归约并把结果发回每个 worker |

每个 worker 贡献相同时用 `"mean"`；否则分别归约**求和值**与**样本数**。

## 同步统计量

自定义的同步归一化可以跨 worker 归约矩：

```python
x_mean = jt.mean(x, dims=[0, 2, 3], keepdims=1)
x2_mean = jt.mean(x * x, dims=[0, 2, 3], keepdims=1)
if jt.in_mpi:
    x_mean = x_mean.mpi_all_reduce("mean")
    x2_mean = x2_mean.mpi_all_reduce("mean")
x_var = x2_mean - x_mean * x_mean
```

能用 Jittor 维护的同步归一化层就优先用它；上面的例子演示的是**集合通信的调用顺序**，
不是一个完整的替代实现。
