# 核心概念

[文档首页](../index.md) · [English](index.en.md)

这些页面解释写模型和调优时需要理解的行为。先看用法与约束，再按需阅读实现细节。

## 设备与数值

混用设备前先读设备与放置。使用低精度时，区分输入 dtype 与累加精度；特殊值和跨设备结果的规则集中在数值契约。

- [设备与放置](device-placement.md)
- [混合精度训练](../guides/mixed-precision.md)
- [float32 累加精度](numerics/float32-precision-policy.md)
- [复数 dtype](numerics/complex-dtype.md)
- [数值契约](numerics/numerics-contract.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

device-placement
numerics/float32-precision-policy
numerics/complex-dtype
numerics/numerics-contract
```
-->

## 运行时机制

了解计算何时执行、如何只读查询状态，以及 CUDA 库计划如何缓存。测量时间和显存请走[性能工作流](../performance/index.md)。

- [流水式惰性执行](runtime/pipelined-execution.md)
- [运行时内省](runtime/runtime-introspection.md)
- [CUDA 库的 plan 缓存](runtime/cuda-library-plan-caches.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

runtime/pipelined-execution
runtime/runtime-introspection
runtime/cuda-library-plan-caches
```
-->