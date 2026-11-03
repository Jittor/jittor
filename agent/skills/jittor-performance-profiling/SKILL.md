---
name: jittor-performance-profiling
description: 选择核心算子、梯度管理与设备前端的性能诊断入口，避免把首次编译、延迟执行或错误前端当成性能结论。第三方库与模型专属验收到 JittorEcosystem 对应仓库执行。
---

# 核心与 compat 性能诊断

可执行工具与环境命令维护在 [benchmarks/profiling](../../../benchmarks/profiling/README.md)，
不在 agent 下复制另一套脚本或结果。真实模型的通用对照见
[benchmarks/torch_compat](../../../benchmarks/torch_compat/README.md)；下游库的版本、模型
支持和官方用例由 [JittorEcosystem](https://github.com/JittorEcosystem) 各仓库维护。

1. 先用最小数值复现确认瓶颈涉及的操作；数值或梯度错误优先修复，不计为性能收益。
2. 按工具实际前端和设备选择入口。原生 Jittor 与独立 compat 不可用导入别名互相替代。
3. 在真实目标设备验证驻留与回退；导入成功、CPU fallback 和显存占用不能替代设备证据。
4. 记录提交和包来源，隔离 JIT 缓存，串行预热。冷编译、构图、逐调用延迟和排队吞吐分别报告。
5. 保存全部输出与梯度直到同步，确认计时覆盖实际计算；再用相同工作负载比较独立参考。
6. 结果留在仓库外，issue/PR 写清环境、样本、波动和限制。旧实测数据不能作为当前提交验收。

深入分析按需使用 [单核冷编译](../jittor-single-kernel-cold-profile/SKILL.md)、
[逐元素带宽](../cuda-elementwise-bandwidth-roofline/SKILL.md)、
[CUDA 归约](../cuda-reduction-strategy-comparison/SKILL.md)。

## 构图与规划成本

Python 构图、核心图记账/规划和设备执行分别计时。构图阶段避免 `.numpy()`、`.sync()` 等
强制执行；保留输出与梯度直到显式同步，防止对象提前释放把工作裁掉。
纯构图测量还要在作用域内设置 `auto_flush_ops=0`，避免达到阈值后自动发射；
端到端测量保留实际配置，两种口径分别记录。
使用固定规模的链、分叉、共享子图与实际工作负载观察增长趋势，不用单点延迟推断复杂度。

记账改动先用存活数、遍历次数或分配次数的增量验证机制，再测性能；计数器本身也要核对作用域。
基线与候选在独立 worktree、相同配置与相同冷热状态下交替测量，报告样本分布和规模。
不要为了测旧实现覆盖当前工作树，也不要把减少检查而引入的正确性缺陷算作加速。
