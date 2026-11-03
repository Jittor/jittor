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

## ACL 数值与混合前缀诊断

通用探针统一位于 `benchmarks/profiling/`，不恢复逐库技能或支持表。先加载 CANN、
选择已分配设备，分别配置原生 Torch CPU、原生 Torch NPU 和 Jittor 解释器与缓存；
Jittor CPU 明确使用 `JT_BACKEND=cpu`，涉及 compat 的进程激活 `JITTOR_TORCH_SHIM=1`。
所有输出留在 `JITTOR_LAB_ROOT` 下且在 checkout 外。模型相关入口与参数见
[工具目录](../../../benchmarks/profiling/README.md)，存在入口不代表对应模型已验收。

| 通用入口 | 验证内容 |
| --- | --- |
| `probe_acl_arg_reduce.py` | 极值、索引及索引对应原值；public / raw SDK 分开，包含轴、dtype 和行边界 |
| `probe_acl_vector_norm.py` | norm/normalize 的有限误差、NaN/Inf mask、溢出/下溢和 epsilon |
| `probe_tensor_ne_identity.py` | operator/method 的同对象实数自比较、输出类型/设备；function/alias 可单独诊断 |
| `probe_acl_linear.py` | 公共 Linear、非连续输入、抵消/小值与可选特殊值；不安装新 Linear 分派 |

```bash
probe_tool="$PWD/benchmarks/profiling/probe_tensor_ne_identity.py"
probe_state="$JITTOR_LAB_ROOT/_state/profiling/ne"
<real-torch-cpu-python> "$probe_tool" --backend torch --device cpu \
  --source-root "$PWD" --output "$probe_state/cpu.json"
JITTOR_TORCH_SHIM=1 JT_BACKEND=cpu <jittor-python> "$probe_tool" \
  --backend jittor --device cpu --reference "$probe_state/cpu.json" \
  --source-root "$PWD" --require-exact --output "$probe_state/jittor-cpu.json"
JITTOR_TORCH_SHIM=1 <jittor-python> "$probe_tool" \
  --backend jittor --device npu --reference "$probe_state/cpu.json" \
  --source-root "$PWD" --require-exact --output "$probe_state/jittor-npu.json"
```

检查报告的 `passed` / `reference_matched`、全部数组、驻留和 Jittor fallback 计数；
普通诊断退出 0 不等于数学通过。`--require-exact` / `--require-match` 将不匹配变为非零退出。
原生设备 SDK 本身可能不满足独立 CPU 的特殊值契约，相同 SDK 错误不能当作正确参考。
NaN、正负零和整数边界分别验证，不从普通有限值推断全域支持。特别是 raw scalar
比较的弱标量转换不同于同 dtype Tensor 比较；半精度量化和大整数需要独立参考。

自动重放可每次用普通执行器重新计算封闭显式 CPU 前缀和 H2D，再启动设备后缀录图。
CPU 计算、等待、传输和结果复制均计入整模时间，不缓存 mask 或模型输出。
opaque code、随机数、待执行 D2H 和未证明的依赖仍拒绝录图；`_is_constant` 或 CPU
placement 本身不是纯度证据。前缀每次重算后复核边界地址，迁移即释放旧图再走执行器；
原全设备图不增加前缀等待。

报告中的 `device_graph_host_prefix` 证明是否实际使用此路径；
`host_prefix_retained_bytes` 是额外前缀准入预算上界，可重复计数别名，独立于原有
`estimated_retained_bytes`，不能作为实测 CPU 占用或峰值。读取统计时不执行张量。
CPU 私有输入 placement、kept fused 内部条件的重算、native pin 图记账、排队、源修改、
实际存储迁移、低层 launch 和预算/释放都要验证。CUDA/其他 SoC 的旧全设备图测试
不能替代其动态 CPU 前缀验证。

同一个实数 Tensor 对象的 `x != x` 可复用 canonical `isnan`，结果应保留 frontend
类型和设备；数字标量在输入设备上构造，避免混合前端作用域产生多余往返。
不同 holder、complex、独立函数入口与一般 scalar 签名不由这一身份式获得验收。

异构 CPU 上记录 affinity、线程数与整个样本序列。可配置 `taskset -c "$BENCH_CPUSET"`
作相同物理核预算的对照，但也保留较快的原生默认配置，不能把 affinity 改变算作代码收益。
若早期时延持续上升，增加完整前向预热后重测，不挑较快前几个样本。比较器保存原始
顺序的全部输出；效果一致性和有真值的数据集质量分别报告。

当前处置与新验证写入 [PR #18043](https://github.com/Jittor/jittor/pull/18043)。
此前的[问题记录](https://github.com/Jittor/jittor/blob/3cf654e15aea6de02e5c9b0b4064e286ea19b24c/agent/manuals/known-issues.md)和
[硬件验收记录](https://github.com/Jittor/jittor/blob/3cf654e15aea6de02e5c9b0b4064e286ea19b24c/agent/manuals/deferred-hardware.md)
保留在固定提交，不能当作当前 head 已通过；清理目录也不关闭已有缺陷。
测量实现 `013bc35bdeb3d460c269c85dc9781e841731764e` 的
[历史视觉记录](https://github.com/Jittor/jittor/blob/eecff5ce5aa0d02aba3a246fcfb5632596732b44/docs/results/2026-11-02-vision-ascend310p-replay.md)
仅覆盖报告内配置，不外推新整合、NPU BF16、训练或其他设备。
