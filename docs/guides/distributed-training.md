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

`jtrun` 会把调用方的 `NCCL_P2P_DISABLE` 原样传给每个 NCCL rank，不会无条件覆盖传输策略。
在支持 P2P 的主机上可以不设置该变量，或显式设置为 `0` 允许 NCCL 选择 P2P；需要共享内存
回退时再显式设置 `NCCL_P2P_DISABLE=1`。允许 P2P 后，NCCL 仍会按拓扑选择传输方式，
跨 NUMA 链路可能使用共享内存。设置 `NCCL_DEBUG=INFO`，通过实际 `Channel ... via`
日志确认 `P2P/CUMEM` 或 `SHM`；在已验证 peer access 的主机上，可显式使用
`NCCL_P2P_LEVEL=SYS` 测试跨 NUMA 的 P2P。启动器会统一继承这个设置。

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

DDP 构造时验证跨 rank 的参数/buffer 结构，并从 group rank 0 广播参数和启用广播的
buffers。构造后应保持参数对象、shape、dtype、可训练状态和设备固定；每次 backward
在本地检查结构及优化器参数覆盖，优化器应包含该 wrapper 的全部可训练参数且不重复。
`parameter.update()` 和同结构的 `load_state_dict()` 可以保留已有参数对象并更新其值。
本地契约错误会抛出异常，`jtrun` 随后终止其他 rank；自行管理进程时也需要负责停止
可能正等待 collective 的同伴。

同步 backward 按稳定模型顺序对累积梯度求平均。NCCL 使用同 dtype、同设备的连续
梯度块，每块通常不超过 25 MiB 且最多含 64 个参数；单个超出容量的参数单独通信。
分组内只提交原始 sum，退出 NCCL group 并建立流依赖后才平均和拆回优化器梯度。
这减少了逐参数通信及构图成本，当前没有计算通信重叠。`with model.no_sync():` 暂存
本地梯度，退出后需要一次同步的 `optimizer.backward(loss)` 才能调用 `optimizer.step()`。
该实现每个进程只使用一张设备，不会替用户搬运模型；先把模型放到当前 rank 的设备上。

使用 AdamW 的 CUDA 训练可显式选择已有的融合更新，以减少 Python 构图和 kernel
launch 成本；对照其他框架时，两侧应使用相同的融合设置并独立验证训练与恢复结果：

```python
optimizer = jt.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                           betas=(0.9, 0.999), eps=1e-8, fused=True)
```

原生 DDP 的 buffers 在构造时广播，当前不会像 PyTorch DDP 的默认配置那样在每次
forward 前广播；需要全局 batch 统计时使用原生 `SyncBatchNorm`。

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
