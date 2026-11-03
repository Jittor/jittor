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
