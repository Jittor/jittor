---
name: cuda-elementwise-bandwidth-roofline
description: 比较 CUDA 逐元素 kernel 的实际流量、copy 带宽和 profiler 时间，区分带宽、缓存、调度及计算瓶颈。
---

# 诊断逐元素算子的带宽瓶颈

## 计量口径

1. 用相同设备上的大数组 copy 测可达带宽；数据应超过相关缓存容量，避免把 L2 带宽当 HBM 带宽。
2. 按实际外部读写字节计算流量，明确融合、中间量和读复用；不能用理论峰值替代实测上界。
3. 先验数值再计时，区分 Jittor profiler 的 rerun 次数、每次 kernel 时间与真实端到端 timeline。
4. 按算子语义、shape/dtype 和执行次数对齐桶，不直接比较两套框架的 kernel 名称。
5. profiler 空匹配或缺少目标事件时停止形成结论；诊断脚本返回 0 本身不代表验收通过。

## 工具

| 工具 | 用途 |
| --- | --- |
| [roofline_copy.py](scripts/roofline_copy.py) | 真实 CUDA copy 带宽参考 |
| [profile_step.py](scripts/profile_step.py) | Jittor profiler/nsys 工作负载采样 |
| [profile_step_torch.py](scripts/profile_step_torch.py) | 独立真实 PyTorch 采样 |
| [classify.py](scripts/classify.py) | 分桶已有 profiler/nsys/Torch 报告 |

先查看各工具 `--help`。两个 step 工具使用主仓维护的 `_ecosystem_speed` fixture；
它们是性能诊断入口，不证明某个下游库已受支持。第三方专属基准在 JittorEcosystem 维护。
报告和 trace 写入 lab，记录实际导入路径、设备、工作量、预热、rerun 及同步边界。
