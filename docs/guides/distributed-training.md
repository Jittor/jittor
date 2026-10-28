# 原生单机多卡训练

Jittor 原生数据并行使用 `jtrun` 启动进程、`jittor.distributed` 管理 process group，
并由 `jittor.nn.parallel.DistributedDataParallel` 同步模型参数与梯度。NVIDIA 首期后端为
NCCL；MPI 可继续用于已有 legacy 程序。本指南不依赖 `jittor.compat.torch`。

本指南的训练示例是单机多 GPU 数据并行。原生 FSDP、TP、ZeRO 和分片 checkpoint 的
验证范围分别见[维护者结果](../results/index.md)，不能由启动成功推断训练分片已验收。
Jittor 的反向与更新入口是 `optimizer.backward(loss)` 和 `optimizer.step()`；
Torch 风格的 `loss.backward()` 不属于原生 DDP 契约。

## 启动

安装 Jittor 后，显式列出可见设备并启动两个 rank：

```bash
CUDA_VISIBLE_DEVICES=0,1 jtrun --nproc-per-node=2 examples/distributed/native_ddp.py
```

在已验证的无 P2P 单机条件下设置 `NCCL_P2P_DISABLE=1`。`jtrun` 会先探测本地设备间的
peer access；每个节点只暴露一张卡时，探测不能判定跨节点或其他设备的 P2P 能力，需按
实际硬件配置传输。本指南不据此声明网络或通信性能。

`python -m jittor.distributed.launch` 是同一 launcher 的模块入口。每个进程获得
`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`；
NCCL 子进程只看见分配给自己的 GPU。launcher 在任一 rank 失败时终止其余 rank，并返回非零
状态。

### 固定节点控制面（实验性）

Linux/NCCL 入口接受 `--nnodes`、`--node-rank` 和 `--rdzv-id`。每个节点运行一个
`jtrun`，使用同一个可从所有节点访问的 master 地址、非零端口、job id 和本地进程数：

```bash
# 节点 0；MASTER_ADDR 指向该节点可被其他节点访问的地址。
jtrun --nnodes=2 --node-rank=0 --rdzv-id=training-job \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --nproc-per-node=2 --device-ids="<local-device-list>" train.py
# 节点 1，地址、端口、job id 和本地进程数保持一致。
jtrun --nnodes=2 --node-rank=1 --rdzv-id=training-job \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --nproc-per-node=2 --device-ids="<local-device-list>" train.py
```

全局 rank 为 `node_rank * nproc_per_node + local_process_index`；`WORLD_SIZE` 是总进程数，
`LOCAL_WORLD_SIZE` 是本节点进程数，`GROUP_RANK/GROUP_WORLD_SIZE` 标识节点。
单卡掩码下 `LOCAL_RANK` 仍为设备索引 0。

节点 0 的 launcher 持有原生 TCPStore 服务；worker 只连接该服务，避免 GPU 集合通信持有
Python GIL 时阻塞 rendezvous。WORLD 与 subgroup 的 NCCL unique id 经 Store 交换，
因此不要求节点共享 root-info 文件。所有 WORLD rank 必须以相同顺序调用 `new_group`；
不一致的 rank 顺序会在进入 NCCL 初始化前拒绝。节点 agent 经 Store 传播退出与心跳；
NCCL 心跳文件仅监视本节点 worker。agent 被终止时，其本地 worker 通过 Linux
parent-death signal 收尾。

这是固定节点控制面，不支持弹性重启、节点增减或多节点 HCCL。控制 Store 应位于可信网络；
job id 只作命名空间隔离。`--timeout` 约束节点会合与心跳等待，首次 JIT 应提前串行预热，
每个 rank 使用独立缓存。日志目录是各节点本地路径；checkpoint 的既有共享文件路径要求
仍由 checkpoint 协议决定，控制面不会自动传输 checkpoint 文件。

单机上的多个 agent 只能验证控制协议；实际跨主机连通、网络故障、训练及 fresh-process
恢复仍需真实多主机验收。不要将 loopback 或两个本地 rank 的结果视为多机通过。

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
