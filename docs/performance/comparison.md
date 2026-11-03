# 与其他框架比性能

[本栏目目录](index.md) · [English](comparison.en.md)

Jittor 异步执行，并在首次使用时编译算子。所以一次有意义的基准测试必须**把编译和
稳态执行分开**，并在测量区间前后同步设备。

按比较目标选择工具：[基准测试](benchmarking.md)覆盖本地算子和跨提交 ASV 结果；[Torch 对照套件](https://github.com/Jittor/jittor/blob/2.0-refactor/benchmarks/torch_compat/README.md)在两个解释器中运行同一份 `import torch` 负载。库专属负载由对应生态仓库维护。

## 先准备可比较的输入

对齐权重、输入、dtype、精度设置、训练或推理模式和设备。计时之外比较输出；训练还要比较所需梯度。一次计时成功不等于数值对齐。

## 测量本地推理负载

下面示例预热后逐次同步推理，计入每轮同步成本；参考框架也应使用相同策略。执行后端需要明确选择并记录。



```python
import time

import jittor as jt
from jittor.models import resnet50

jt.flags.use_cuda = int(jt.has_cuda)
warmup = 10
repetitions = 100
batch_size = 8

data = jt.random((batch_size, 3, 224, 224))
model = resnet50()
model.eval()

jt.sync_all(True)
for _ in range(warmup):
    model(data).sync()
jt.sync_all(True)

start = time.perf_counter()
for _ in range(repetitions):
    model(data).sync()
jt.sync_all(True)
elapsed = time.perf_counter() - start

print("Jittor images/s:", repetitions * batch_size / elapsed)
```

每个框架都要用**相同的**预热次数、输入形状、dtype、eval 模式、加速器、库版本和同步
策略。**不要拿 Jittor 的首次编译去比另一个框架的热缓存。**

## 先验证数值一致

算得更快但算的不是同一个模型，没有意义。计时之前：

1. 传入完全相同的参数与输入值；
2. 在计时区间**之外**比较输出；
3. 按 dtype 和后端选择合适的容差；
4. 确认两个进程都用的是预期的加速器；
5. 盯住显存，别让换出或分配压力只拖慢其中一边。

PyTorch 侧在测量循环前后调用 `torch.cuda.synchronize()`，Jittor 侧在相同位置调用
`jt.sync_all(True)`。报告结果时给出硬件、软件版本、确切命令、样本数和离散度，而不是
一个没有说明的吞吐数字。

CPU 上的对比**必须绑核**（`OMP_PROC_BIND=close`、`OMP_PLACES=cores`）：不绑核的同一个
步能测出相差 25% 的两个簇，足以把结论倒过来。实测数字见
[基准测试的"CPU 线程绑核"](benchmarking.md)。

## 为什么 GPU 会闲着

如果 kernel 更快但整步更慢，多半是 CPU 建图与设备计算没有重叠。见
[流水式惰性执行](../notes/runtime/pipelined-execution.md)，那里说明 `auto_flush_ops` 的作用及其显存代价。
