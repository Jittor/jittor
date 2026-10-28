# 原生 FSDP 的存储、反向与 checkpoint 归属

状态：有界的 CUDA/NCCL 正确性实现。FP32 训练与固定布局的换进程续跑已有维护者证据。
BF16 计算也满足记录的容差与精确续跑，但它与二进制 Torch 之间的微小梯度差异仍是一项
明确的数值限制。实测来源、阈值与条件见[验证记录](../results/2026-10-15-native-fsdp.md)。
Owner：Jittor 核心维护者；当存储、集合通信、优化器梯度、精度策略或 checkpoint 编码
发生变化时复查。

## 原生入口

`jittor.distributed.fsdp.fully_shard` 把一个原生 module 的参数替换为 rank 本地的 FP32
存储。`FullyShardedDataParallel` 是围绕该操作的原生 `nn.Module` 包装；`jittor.nn.parallel`
惰性暴露同一个类。两个入口都不导入 Torch 或兼容包。原生优化器要在分片**之后**构建，
并且每个 rank 都从同一份完整的参数值起步。不同的初值会在参数集合通信之前被拒绝。

CUDA 启动器必须提供两个或更多真实 rank，并初始化原生 NCCL WORLD 进程组。通常的原生
训练调用保持不变：

```python
import jittor as jt
from jittor import distributed as dist
from jittor.distributed.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy

# Run under jittor.distributed.launch with an isolated cache per rank.
dist.init_process_group("nccl", init_method="env://")
jt.set_global_seed(42)
module = jt.nn.Sequential(jt.nn.Linear(5, 7), jt.nn.Tanh(), jt.nn.Linear(7, 3))
module = FullyShardedDataParallel(module)
optimizer = jt.optim.AdamW(module.parameters(), lr=0.001)
inputs = jt.ones((3, 5))
target = jt.zeros((3, 3))
optimizer.step(((module(inputs) - target) ** 2).mean())
```

对于有界的 BF16 策略，给包装器传
`mixed_precision=MixedPrecisionPolicy(param_dtype="bfloat16", reduce_dtype="float32",
output_dtype="float32")`。参数与动量矩保持为 rank 本地的 FP32 分片。临时的 all-gather
输入、gather 出来的参数以及浮点输入使用 BF16；reduce-scatter 与优化器梯度使用 FP32。
这是一个 module 级策略，不启用全局 autocast，也不改变原生优化器的更新公式。

## 只有一份分片实现

`distributed/fsdp/storage.py` 负责参数切分、padding、临时 all-gather 参数、module 属性
替换以及完整引用的释放。`gradients.py` 负责把完整梯度归约进对应的本地分片。兼容层的
FSDP 实现通过它自己的元数据与集合通信适配器委托给这些函数。它的 DTensor、前端作用域、
mesh 与 Torch 优化器接口仍留在 `compat/fsdp2`；原生执行不会反过来进入那个前端。

对每个参数，原生实现把扁平化后的元素区间均匀切分，必要时给最后一段补 padding。整个
训练过程中，优化器拥有这些本地参数对象。前向 gather 出临时的完整参数；反向对这批图对象
本身求导，reduce-scatter 梯度，同步实际的集合通信结果，然后释放完整参数的引用。原生
SGD/Adam/AdamW 通过它们既有的更新算法更新本地 FP32 参数。独立轨迹的验收目前覆盖 SGD
与 AdamW，而不是每一个已接受的优化器。

优化器接缝校验 rank、world size、shape、dtype、device、可训练性、参数标识与分组控制。
FSDP 参数绕过旧的梯度 all-reduce 与周期性全参数广播。既有的 DDP 与张量并行接缝保持
独立。分片之前构建的陈旧优化器、部分参数所有权、相互冲突的包装器，以及没有新一次前向
就发起的额外反向，都会显式失败。

## checkpoint 叶子与续跑

既有的 `jittor.distributed.checkpoint.save_training_state` 与 `load_training_state` API
把每个真正分片的参数、梯度与优化器动量矩编码成一个 `TensorShard`，带上它的完整形状和
有效的扁平区间。padding 不落盘，恢复时重建为零。缓冲区、优化器标量与 FSDP 布局元数据
仍然复制保存并跨 rank 校验；完整模型不会藏进 `RankLocal`。既有的 rank 本地 RNG 处理
仍由原生 checkpoint 实现负责。

恢复会在修改模型之前校验原始 FSDP 形状、master dtype、world size、分片长度与精度策略。
验收序列使用三次不同的启动器调用：独立 oracle 对拍；三步加上保存与两步实时步；以及
换进程恢复后接着跑那最后两步。最后一段要求输出、输入/参数梯度、本地参数、优化器状态、
真实 dropout 输出，以及每个 rank 的 Python、NumPy、原生 CPU 与 CUDA RNG 轨迹完全相等。

## 有意的限制

多 rank 验收目前覆盖固定 world size、逐参数切分的 CUDA/NCCL WORLD，FP32 master 以及
可选的 CUDA BF16 计算。原生 CPU/world-one 测试演练算术与校验契约，不构成对分布式 CPU
支持的证明。嵌套 FSDP、互为别名的参数、参数容器、与 DDP/TP/ZeRO 的组合、额外的优化器
输入参数、保留图的反向以及 world-size 迁移被拒绝或不支持。其它加速器后端、多主机、
重叠通信与每一种优化器变体在这里都没有验收声明。本地存储计数只证明切分，不证明峰值
显存节省或性能。
