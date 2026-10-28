# 原生分片 checkpoint 文件

状态：实验性的原生 I/O 与按已完成的步续跑训练。复查日期：2026-10-15。基线：
`8e388d108eb83c1c438936709f29855f47ba2a69`。Owner：序列化与分布式维护者。当原生
Module、Optimizer、RNG 或进程组协议变化时重新验证。

`jittor.distributed.checkpoint` 使用原生 Jittor 所有者，从不导入 Torch 或它的兼容前端。
它切分的是 checkpoint 的载荷文件；它不分片训练参数、梯度或优化器算术。原生 FSDP、
张量并行、ZeRO 与多节点验收是另外的能力。

## 训练续跑

```python
from jittor import distributed as dist
from jittor.distributed import checkpoint

# The launcher owns the real communicator. Register a Store for multi-rank I/O.
dist.init_process_group("nccl", init_method="env://")
# Native model/optimizer train normally; every rank calls the checkpoint API.
optimizer.step(loss)
checkpoint.save_training_state(model, optimizer, checkpoint_directory,
                               extra={"completed_steps": 3})
extra = checkpoint.load_training_state(model, optimizer, checkpoint_directory)
```

该辅助函数保存模型 FQN 与可训练性标志、优化器类、有序的参数绑定、默认值、动量矩缓冲与
计数器，以及 rank 本地的 Python、NumPy、原生 CPU 与可见 CUDA 生成器状态。保存不会改动
模型原有的标志。加载通过原生 Module 所有者应用保存下来的标志，并通过优化器既有的公开
所有者安装状态。原生优化器的数学不被替换。特别地，原生 Adam 的 epsilon 位置与
`torch.optim.Adam` 不同；数值验证必须使用原生约定。

在完成一次优化器 step 或 `zero_grad()` 之后保存。挂起的累加会被显式拒绝，因为原生公开的
优化器状态里不包含它的反向锁存。恢复要求模型 FQN、形状与 dtype，优化器类与有序的
FQN/分组绑定，执行后端与可见的 CUDA RNG 设备数，以及全局 rank 集合都匹配。
模型/优化器元数据与全部载荷文件都在改动目标所有者之前完成校验。非法的生成器状态会回滚
当前 RNG 状态。自定义优化器加载钩子与对目标所有者的并发修改不是事务性扩展点。

支持 CPU 与 CUDA 的原生张量载荷。BF16 使用显式的 UInt16 存储视图，并恢复原始的 BF16
dtype 与原始位；单靠 `numpy()` 会把 BF16 按数值导出为 FP32，从而丢失原始存储标识。
未验证的原生后端会被显式拒绝。

## 通用状态树与本地张量区间

`save(state, directory)` 接受字符串键的字典、列表、元组、JSON 原始值、NumPy
标量/数组、原生张量以及下面两个描述符。不会执行任何 pickle 全局对象。原生张量叶子在
当前原生设备上以 detached 方式加载；训练辅助函数另行恢复记录下来的可训练性。

```python
from jittor.distributed.checkpoint import RankLocal, TensorShard, save, load

# offset is a flat ELEMENT interval in the given global shape.
save({"weight": TensorShard(local_weight, global_shape, element_offset),
      "extra": RankLocal(per_rank_state)}, directory)
local = load(directory, tensor_shards={
    "/weight": TensorShard(target_tensor, global_shape, new_offset)
})["weight"].tensor
```

目标控制返回的本地形状与原生放置。它原有的内容不会被修改。dtype 与原生/NumPy 所有者必须
与保存的叶子一致。张量路径使用 JSON pointer 转义（`~0`、`~1`）。目标区间可以改变，但
rank 本地状态不能迁移到不存在的全局 rank。训练 RNG 的重新分片与 world-size 变化会被拒绝。

普通张量输入采用复制语义：每个 rank 在主机上快照完整输入，校验全字节摘要一致，然后写入
一段均衡的扁平区间。这条路径减少文件重复，但不承诺主机内存有界。`TensorShard` 则只消费
已有的本地输入，校验全局覆盖，并且能在不分配完整全局输出的情况下恢复请求的本地区间。
两个描述符都不提供训练分片。

## 完成与失败

所有 rank 必须以相同顺序、相同路径与相同覆盖选项调用 `save`，使用共享文件系统与原生
进程组 Store。Store 交换状态与元数据；每个 rank 写自己的载荷文件。载荷与逐 rank 元数据会
被检查：缺失的 rank、不一致的副本、形状/dtype、重叠或缺失的区间，以及 SHA256 损坏。
只有全部通过，rank 零才会以完整的一代原子替换 `manifest.json`。

失败会通知参与的 peer 并抛错，而不是产出一个完整标记。`overwrite=True` 在新的写入失败时
保留上一代已提交的版本。旧的和不完整的代留待操作者显式清理。超时不会取消 I/O，也不构成
对 worker 死亡的容错。不支持并发保存到同一目录。读者校验与恢复期间，各代文件必须保持
不可变；文件系统的崩溃保证仍取决于实际部署的共享文件系统。
