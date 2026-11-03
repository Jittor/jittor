---
name: gate-tier-budget
description: 调整 core、smoke 或 full 测试选择与并发配置时，用同口径实测和算术预算维护反馈速度。
---

# 测量与维护门禁预算

## 选择与测量

唯一选择表在 `tests/_helpers/tiers.py`：core 是最小反馈白名单，smoke 默认纳入并排除
慢文件，full 保留完整范围。新增 core 项需要实测秒数和所守的基本行为，不能靠行数判断成本。

用实际门禁命令加 `--durations=0`，按文件聚合 setup/call/teardown；记录机器负载、CPU
配额、线程数、模式、编译并发和缓存温度。native 与 Torch 模式分别测，不混在一次 pytest。
对比期间固定源码和环境，在各自缓存里完成预热；中途同步或切工具链后重新留受影响基线。

## 预算与并行

- `tools/gate_budget_report.py` 报告已测的 CPU 预算，CUDA 需另行同口径实测；`tests/structure/test_gate_tiers.py` 验证选择表。
- 算术下界包括 `max(文件耗时总和 / workers, 最长文件耗时)`，再加启动成本；
  顺序执行的两种进程模式需相加。不要把运行时墙钟上限作为稳定断言。
- `--dist loadfile` 按文件分发；只有确实独立的生成用例才考虑按用例分发。
- workers 与 OpenMP/BLAS 线程共同消耗配额，应一起规划。首次 JIT 仍需串行准备。
- 并行化必须保留完整的 nodeid/结论集合，不能把 worker 崩溃造成的少跑视为提速。
  按[门禁执行核验](../verifying-a-gate-actually-ran/SKILL.md)比较。

超预算时先找实际瓶颈。清单路径重命名、过期推迟项与只剩空选择集应由结构测试拒绝。
