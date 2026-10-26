# 原生 ZeRO 第一阶段优化器

`jittor.distributed.ZeroRedundancyOptimizer` 在**训练期间分片优化器状态**。每个 rank 只为
自己拥有的参数实例化一个既有的原生 Adam、AdamW 或 SGD。不属于自己的动量矩从不构建。
模型参数与梯度保持复制——这是 ZeRO 第一阶段。

```python
import jittor as jt
from jittor import distributed as dist
from jittor.distributed import ZeroRedundancyOptimizer, checkpoint
from jittor.nn.parallel import DistributedDataParallel

dist.init_process_group("nccl", init_method="env://")
model = DistributedDataParallel(jt.nn.Linear(3, 2))
optimizer = ZeroRedundancyOptimizer(
    model.parameters(), jt.optim.Adam, lr=0.01, betas=(0.85, 0.95))

loss = ((model(jt.ones((4, 3))) - jt.zeros((4, 2))) ** 2).mean()
optimizer.step(loss)
checkpoint.save_training_state(model, optimizer, "checkpoint")

# Construct fresh owners with the same ordered groups and DDP trainability.
fresh = DistributedDataParallel(jt.nn.Linear(3, 2))
resumed = ZeroRedundancyOptimizer(
    fresh.parameters(), jt.optim.Adam, lr=0.01, betas=(0.85, 0.95))
checkpoint.load_training_state(fresh, resumed, "checkpoint")
```

每个 rank 按相同顺序执行这些操作，checkpoint 目录必须由所有 rank 共享。公开的原生启动器
提供真实的 CUDA/NCCL worker。CPU 与 CUDA 单进程使用不需要分布式 Store；当前的多 rank
实现明确要求 CUDA/NCCL WORLD 与原生 Store。子组、HCCL/MPI 多 rank 训练以及 world-size
迁移都在本实现已验证的契约之外。

## 归属与通信

参数按元素数降序整块分配给当前负载最轻的 rank，分组/参数顺序相同时结果稳定。小模型可能
让某个 rank 分不到参数；该 rank 会建立空的算法状态列表，仍然参与梯度同步与所有者广播。
整块分配不保证单个大参数在逐 rank 状态上均衡。

包装器的完整 `param_groups` 属于原生基类 Optimizer，由它负责反向、梯度累加、裁剪以及
既有的 DDP reducer。`local_optimizer` 暴露既有原生算法所拥有的状态供检查。更新时调用它
的 `step()` 且不传 loss：因此不再做一次反向或梯度归约。分组覆盖与当前默认值会被转发，
包括移除某项覆盖。新增参数分组、改变参数绑定/形状/dtype、以及优化器分组里重复的参数
别名都会被拒绝；对 tied weights 请传入 module 去重后的参数。rank 间一致性要求超参数是
可 JSON 序列化的主机侧值。

在做非对称的所有者更新之前，所有 rank 都以 `weak_sync=False` 完成精确的集合梯度根。
随后参数按参数顺序从各自的所有者广播。Store 一致性检查校验 rank 配置与可训练性；peer
在托管 server 的 rank 进入集合通信之前，以原生 `arrive` 契约完成它们最后一次 RPC。
rank 零清理临时键。这些显式边界以正确性为先；这里不声称吞吐。

DDP 既有的 `no_sync` 契约得到保留：中间的 micro batch 可以在本地累加，而更新之前必须有
一次同步的反向。原生 DDP 在构造时固定可训练性。用保存下来的可训练性构造续跑用的 DDP
模型；独立的 ZeRO 支持延迟解冻，同时保留既有算法逐参数的 first-touch 规则。

## checkpoint 与精度

原生训练 checkpoint 通过各自既有的所有者保存完整模型状态、有序 FQN 绑定、可训练性与
RNG，并把优化器状态保存为 `RankLocal` 拥有的张量。分片元数据记录算法、rank、world size、
所有权以及参数形状/dtype。加载会在修改模型参数之前校验这些元数据与本地状态。保存时
缺失的分组覆盖也会移除新所有者上任何陈旧的覆盖。训练 checkpoint 仍然明确不支持挂起的
梯度累加。

最初验证的范围是 CPU 单进程与单主机两 rank CUDA/NCCL 上的原生 FP32 Adam/AdamW/SGD，
外加原生 BF16 参数/动量 SGD 并保留原始 BF16 checkpoint。这次 BF16 SGD 检查不构成对
FP32 master 参数的 AMP，或对 BF16 Adam/AdamW 支持的证明。活动的计算在 NumPy I/O 之前
于 CUDA 上被观测；未被触碰的冻结零状态则在显式实体化其声明的存储放置之后单独检查。

这个 API 不提供参数/梯度分片、原生 FSDP、TP、ZeRO 第二/第三阶段、弹性重新分片，或多
主机 rendezvous 验收。更广的分布式能力需求仍只是部分实现。见
[验证报告](../results/2026-10-15-native-zero-optimizer.md)。
