# 原生单机多卡训练

Jittor 原生数据并行使用 `jtrun` 启动进程、`jittor.distributed` 管理 process group，
并由 `jittor.nn.parallel.DistributedDataParallel` 同步模型参数与梯度。NVIDIA 首期后端为
NCCL；MPI 可继续用于已有 legacy 程序。本指南不依赖 `jittor.compat.torch`。

首期范围是单机多 GPU 数据并行，不包括多机 rendezvous、原生 FSDP、TP、ZeRO 或分片
checkpoint。Jittor 的反向与更新入口是 `optimizer.backward(loss)` 和 `optimizer.step()`；
Torch 风格的 `loss.backward()` 不属于原生 DDP 契约。

## 启动

安装 Jittor 后，显式列出可见设备并启动两个 rank：

```bash
CUDA_VISIBLE_DEVICES=0,1 jtrun --nproc-per-node=2 examples/distributed/native_ddp.py
```

当前已验证的单机条件要求显式设置 `NCCL_P2P_DISABLE=1`。`jtrun` 对 NCCL 子进程也会强制
写入这个值，保证所有 rank 选择相同的共享内存传输路径；在具备可靠 P2P 的主机上重新验证
后，才能移除这项限制。

`python -m jittor.distributed.launch` 是同一 launcher 的模块入口。每个进程获得
`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`；
NCCL 子进程只看见分配给自己的 GPU。launcher 在任一 rank 失败时终止其余 rank，并返回非零
状态。

## 训练循环

下面的[完整示例](../../examples/distributed/native_ddp.py)使用原生 `Dataset` 和一层线性模型。
`Dataset.batch_size` 表示所有 rank 合计的全局 batch size；各 rank 按相同的 seed 与 epoch
生成共同 shuffle 顺序，再把每个完整 global batch 按 rank 切开。为保证每个 rank 的完整
local batch 一样大，global batch size 应能被 world size 整除；尾部样本在 `drop_last=False`
时按 ceil 规则分片，不够的一侧会重复尾部索引。每轮调用 `set_epoch(epoch)` 推进 shuffle，
并按 seed、epoch、rank 和 worker id 派生可复现的 worker seed。

```python
import jittor as jt
from jittor import distributed as dist
from jittor.nn.parallel import DistributedDataParallel

dist.init_process_group(backend="nccl")
model = DistributedDataParallel(MyModel())
optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    loss = model(batch).mean()
    optimizer.backward(loss)
    optimizer.step()
```

DDP 构造时从 group rank 0 广播参数和启用广播的 buffers；每次同步 backward 前，reducer
按稳定参数顺序对累积梯度做 mean all-reduce。NCCL 后端会把同一轮参数梯度放进一个
collective bucket，减少逐参数的 stream join 和 host launch；`with model.no_sync():` 会暂存
本地梯度，退出后需要一次同步的 `optimizer.backward(loss)` 才能调用 `optimizer.step()`。
该实现每个进程只使用一张设备，不会替用户搬运模型；先把模型放到当前 rank 的设备上。

文件写入、日志和普通 checkpoint 只由 rank 0 执行：

```python
if dist.get_rank() == 0:
    model.module.save("checkpoint.pkl")
```

需要断点后保持 Adam/SGD 动量与步数时，还要保存并恢复 optimizer 状态；状态文件同样只由
rank 0 写入，恢复后在 barrier 之后由各 rank 读取：

```python
if dist.get_rank() == 0:
    model.module.save("model.pkl")
    jt.save(optimizer.state_dict(), "optimizer.pkl")
dist.barrier()
model.module.load("model.pkl")
optimizer.load_state_dict(jt.load("optimizer.pkl"))
```

`destroy_process_group()` 会关闭已注册的 Store 并清除 Python process-group 状态。当前
NCCL/HCCL wrapper 没有可调用的 communicator teardown，因此底层 communicator 在 worker
进程退出时释放。`async_op=True` 目前返回的是同步操作已完成的 `Work`，不代表通信与计算重叠；
timeout/cancel 也未实现。MPI 的 `jt.in_mpi`、`jt.rank`、`jt.world_size` 与 `mpi_*` 别名继续
服务 legacy 程序，不是新原生训练脚本的接口。
