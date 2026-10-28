# 整合回归与 NCCL 缺陷收口

- 状态：KI-DIST-002、KI-DIST-003 的退出条件已通过；CPU/CUDA 定向整合验证。
- 日期：2026-10-15。
- 整合基线：`f02e7aec0b43daf1b275a97eab340de7c4e917d3`，同步
  `origin/2.0-refactor` 后保留本地修改，本文与最终整合改动一起提交。
- 维护者：Jittor 核心、Torch 兼容与分布式维护者。
- 复查条件：collective 调度、张量默认放置、FSDP 分片/优化器、NCCL 版本或门禁进程模式变化。

## 真实 NCCL 2.18.3 验证

两个 RTX 4090 rank 使用 Python 3.11.16、nvcc 12.2.140 和驱动
580.178.04。官方 `nvidia_nccl_cu12-2.18.3` wheel 的 SHA256 为
`89cf220910c43e53df992f54f7aa2a35c6d606b077c9ae95b1f17c544d3ad3f3`。
每个 rank 串行预热时，实际 `ncclGetVersion` 返回 `21803`，并检查
进程映射只加载该版本的真实库；不以 Python 包元数据代替动态库证据。

| 定向范围 | 每 rank 结果 |
| --- | --- |
| 外置纯原生快速 collective 序列 | 75 次精确检查，未加载 Torch shim，fallback 增量 0 |
| PR #18031 原生 smoke 的计算检查 | 20 次检查，两个 rank 退出码均 0，包括 bf16、DDP 与训练步 |
| 新维护测试 `tests/backends/comm/nccl/test_native_collectives.py` | 1 passed，0 skipped，75 次精确检查，fallback 增量 0 |
| reordered DeviceMesh + Torch Adam 子类 | 1 passed，0 skipped，三步 NumPy 参考与旧视图生命周期通过，fallback 增量 0 |

快速序列将 reduce、bf16 reduce、broadcast、gather、scatter 轮流置于
第二个位置，不在 collective 之间添加设备级同步来掩盖旧现象。
原生 smoke 仅调整旧脚本的 local-rank 断言：launcher 的逻辑设备编号和
`LOCAL_RANK` 不同，断言改用环境中的 `LOCAL_RANK`，其计算检查保持原样。

KI-DIST-002 在当前树未再复现，退出条件已验证，原始原因没有隔离，不能
归因于某一个修复提交。KI-DIST-003 结合既有张量放置 owner 修正
`0745b1bab5` 和测试显式 CUDA 输入，已验证真实 device 上的完整三步计算。
两个开放条目从问题总账移除；这里保留验收范围与复查条件。

`nox -s nccl` 现在串行预热各 rank，通过 canonical `probe_library` 实际
加载 NCCL，然后分别启动 native 和 Torch 模式，使用各自日志目录。
新增 host 契约验证进程模式、独立 rank cache、预热先于启动和节点选择参数。
本次实际双卡执行的是上表两项维护测试，没有声称整份 FSDP2 文件或
完整 nox 会话均已验收。单节点证据也不代表多节点、HCCL 或 ACL 验收。

在分配两张设备并按环境手册隔离编译缓存后，可复验定向范围：

```bash
python -m nox -s nccl -- tests/backends/comm/nccl/test_native_collectives.py
python -m nox -s nccl -- compat/tests/fsdp2/test_fsdp2_nccl.py::TestFSDP2Nccl::test_mesh_reordered_shards_custom_adam_and_lifetime
```

## 同批整合范围

整合了 [SDPA 掩码](2026-10-14-cuda-sdpa-mask-coverage.md)、
[图重放预算](2026-10-14-graph-replay-retain-budget.md)、
[CUDA 13 工具链](2026-10-15-cuda13-toolchain-packaging.md)、
[下游训练 scorecard](2026-10-14-downstream-oracle-scorecards.md)、
[TorchMetrics/TensorDict](2026-10-15-metric-state-and-tensordict-oracles.md)、
[bf16 FULL 续训](2026-10-15-bf16-fsdp2-checkpoint-continuation.md) 和
[checkpoint 选项](2026-10-15-checkpoint-state-options.md)。各报告中的
设备、组件级范围、舍入差异与未验收项继续适用。

Checkpoint 选项整合前，主树 CPU core 为 322 passed、42 skipped、
1 个既有 xfailed；CUDA 原生 SDPA/CUB/graph-retention 定向回归为
37 passed、0 skipped；compat 与工具 CPU 回归为 161 passed、0 skipped。
当时完整结构门禁为 1441 passed、6 skipped、0 failed。
新增 checkpoint 选项后的最终复验：

| 主树最终范围 | 结果 |
| --- | --- |
| CPU core，两种进程模式 | 322 passed、42 skipped、1 个既有 xfailed |
| CPU checkpoint，包括真实 Accelerate 保存/续训 | 30 passed、0 skipped |
| CUDA checkpoint 与 Module 自定义状态 | 37 passed、0 skipped |
| NCCL 2.18.3 双卡 bf16 FULL checkpoint | 每 rank 1 passed、0 skipped，模型/优化器恢复、两步续训与四类 RNG 精确相等，fallback 0 |
| 完整 `tests/structure` | 1441 passed、6 skipped、0 failed，167.27 s |
| 布局与空白 | 通过 |

初次 checkpoint 复验发现共享环境的 `safetensors` 安装不完整，真实
Accelerate 在导入阶段失败。改用外置、已安装完整依赖的目录后，上述
CPU/CUDA 复验通过，未把导入失败改为 skip。初次 core 启动继承 CUDA
环境变量，与 CPU 构建不一致；明确设置 `JT_USE_CUDA=0` 后两个模式通过。
一个额外 CUDA nodeid 计数检查不识别这些测试的命名形式，37 个 case
虽通过但该次会话退出 1；最终定向运行保留真实 CUDA 能力与执行检查，
按测试自身的物理位置、dispatch 和 fallback 断言验证，没有用该计数
失败的会话充当通过证据。上述初次日志均保留。

各子任务 worktree 中报告入口或 MANIFEST 尚未登记造成的结构失败，保留其
原基线记录；整合树统一登记结果索引并重新生成 MANIFEST，以最终门禁为准。
首次 JIT 串行预热，CPU、CUDA 和每个分布式 rank 使用隔离缓存，未与
benchmark 共享编译缓存。

原始日志、官方 wheel 元数据、外置脚本和 rank 输出未版本化，位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/`。整合日志包括
`logs/sdpa-graph-cuda13-native-cuda-recheck.log`、
`logs/integrated-compat-tools-cpu.log`、
`logs/integrated-legacy-nccl218-maintained-native.log` 和
`logs/integrated-legacy-nccl218-reordered-mesh.log`；逐 rank 日志分别在
`legacy-nccl/maintained-native-logs/` 与
`legacy-nccl/maintained-reordered-mesh-logs/`。最终整合日志为
`logs/final-integrated-core-cpu-recheck.log`、
`logs/final-checkpoint-options-cpu-deps.log`、
`logs/final-checkpoint-module-cuda-physical-recheck.log`、
`logs/final-integrated-structure.log` 和
`logs/final-legacy-nccl218-full-checkpoint.log`。
