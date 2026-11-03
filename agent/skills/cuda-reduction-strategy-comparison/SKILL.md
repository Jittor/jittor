---
name: cuda-reduction-strategy-comparison
description: 评估共享内存、warp 或多级归约时，用同一已编译 kernel 的数值与性能证据选择策略。
---

# 比较 CUDA 归约策略

## 操作

- 从当前 `src/` 实现读取 `use_shared_reduce` 等选项的有效值；不要照抄旧实验表。
  `0` 可禁用共享归约，`2` 也是有效路径，不应把选择简化成只比较 level 3/4。
- 使用覆盖归约轴、连续/跨步布局、尾部不足、dtype 和实际模型尺寸的形状集。
  一个“代表形状”不能支持全局默认策略。
- 固定输入，用独立高精度参考检查每个策略的输出与所需梯度；显式拒绝 NaN/Inf 和超预算误差。
- 每种策略的 JIT key 必须区分，选项作用域应覆盖构图、编译、warmup、**数值检查和计时**。
  只在 profiler 内切 flag，可能验证旧 kernel 却测量新 kernel。
- 检查实际生成源码和产物确认命中策略，再做交替 A/B；数值错误不能折算成性能提升。
- 记录寄存器、共享内存、同步和 kernel 次数；端到端性能包含归约前后转换，不能只看单个 kernel。

从现有 [shared-reduce 回归](../../../tests/backends/cuda/test_shared_reduce.py)
开始扩展策略与数值覆盖；通用回归放对应 `tests/ops/` 或 `tests/codegen/`；模型专属实验放 lab 或独立生态仓库。
通用计时入口见[性能分析](../jittor-performance-profiling/SKILL.md)。
