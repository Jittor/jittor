# 验证结论

这一节保存**可复现的维护者验证结论**：一个具体的现象、它的根因、修复，以及修复
之后在真实设备上量到的数字。写的是"我们查过什么、结果怎么样"，不是项目历史，也
不是教程——想让某个模型或后端跑起来、想对照某个已知问题的结论时看这里。

每条结论开头都注明状态、日期、基线提交、验证范围、维护者和复查条件。过期的结论会
被删除或归档，不在多个文件里重复维护；压缩过的报告只保留结论、最终数字、复现命令与
未结事项，逐步的记录留在 Git 历史里。

```{toctree}
:maxdepth: 1

2026-08-28-ascend-910b-validation
2026-08-30-qwen3-ascend-training
2026-08-31-vllm-ascend-jittor-bootstrap
2026-09-02-verl-ascend-core-algorithms
2026-09-04-import-jittor-cost-attribution
2026-09-12-minimax-h3-torch-compat
2026-09-12-cuda-metaop-launch-index-scalar
2026-09-12-batched-linear-graph-nodes
2026-09-13-host-path-per-node-cost
2026-09-14-autocast-conv-mixed-dtype
2026-09-14-exit-heap-corruption
2026-09-14-jittor-vs-pytorch
2026-09-14-vllm-omni-h3-enablement
2026-09-19-torch-compat-runbook-verification
2026-09-21-flowey-merge-verification
2026-09-22-complex-svd-grad-and-complex128
2026-09-22-sgd-dampening-and-native-vmap-gaps
2026-09-24-torch-compat-real-models
2026-09-25-profiling-tools
2026-10-05-high-rank-and-downstream-core-gaps
2026-10-09-downstream-p1-allocator-protocol-transpose
2026-10-14-benchmark-coverage
2026-10-14-downstream-coverage-audit
2026-10-14-optimizer-checkpoint-continuation
2026-10-14-checkpoint-rng-fsdp2-continuation
2026-10-14-amp-bfloat16-gradscaler
2026-10-15-distribution-constraint-publication
2026-10-15-training-registry-denominators
2026-10-14-cuda-sdpa-mask-coverage
2026-10-14-graph-replay-retain-budget
2026-10-14-downstream-oracle-scorecards
2026-10-15-cuda13-toolchain-packaging
2026-10-15-bf16-fsdp2-checkpoint-continuation
2026-10-15-metric-state-and-tensordict-oracles
2026-10-15-checkpoint-state-options
2026-10-15-integration-nccl-coverage
2026-10-15-upstream-pr-disposition
2026-10-15-verl-registered-component-oracles
```

## 按主题索引

| 结论 | 状态 | 日期 |
| --- | --- | --- |
| [Ascend 910B3：冷缓存启动、ACL 执行与 NPU 门禁](2026-08-28-ascend-910b-validation.md) | NPU 门禁 397 passed / 9 skipped；skip 对应的能力边界仍开放 | 2026-08-28，2026-09-01 复查 |
| [Qwen3-0.6B 在 Ascend 910B3 上训练](2026-08-30-qwen3-ascend-training.md) | FP32 训练与 BF16 单步对齐接受；BF16 精确路径慢约 6%、训练轨迹未验收 | 2026-08-30，2026-09-02 |
| [vLLM 在 Ascend 910B3 上经 Jittor 运行](2026-08-31-vllm-ascend-jittor-bootstrap.md) | 单请求 Qwen3-0.6B 正确且快 6.8%；外置 NPU 插件未入仓，多请求未测 | 2026-08-31，2026-09-02 |
| [verl 核心算法在 Ascend 910B3 上](2026-09-02-verl-ascend-core-algorithms.md) | 六类 loss 与梯度逐位一致；五条慢 1.25–1.75 倍，NPU 端到端 PPO 未跑 | 2026-09-02 |
| [热缓存 `import jittor` 的耗时归因](2026-09-04-import-jittor-cost-attribution.md) | 归因接受，热缓存已低于 1 s；冷缓存仍在 import 时编译核心 | 2026-09-04 |
| [MiniMax-H3 在 Torch 兼容层下跑通](2026-09-12-minimax-h3-torch-compat.md) | 端到端跑通，达到 parity 速度 | 2026-09-12，2026-09-14 复验 |
| [autocast 请求了混合 dtype 的卷积](2026-09-14-autocast-conv-mixed-dtype.md) | 已修复，真实 CUDA 设备验证 | 2026-09-14 |
| [退出期 "corrupted double-linked list"](2026-09-14-exit-heap-corruption.md) | 已修复，多次真实运行验证 | 2026-09-14 |
| [jittor + torch-compat + vLLM-Omni 接入 H3 暴露的 shim 缺陷](2026-09-14-vllm-omni-h3-enablement.md) | 单卡与 TP2 请求端到端正确（710.7 s → 39.6 s），部署侧视频噪声已修；参考 VAE 解码仍比 PyTorch 慢 1.33x | 2026-09-14 至 2026-09-22 |
| [CUDA 元算子：发射配置、strided 下标与标量常量融合](2026-09-12-cuda-metaop-launch-index-scalar.md) | 部分完成，H20 上实测 | 2026-09-12 |
| [主机受限步长：把每算子的 Python 开销从图构建里拿掉](2026-09-13-host-path-per-node-cost.md) | Python 路径已完成 | 2026-09-13 |
| [batched Linear 的两个展平节点](2026-09-12-batched-linear-graph-nodes.md) | 已落地，H20 上实测；对拍结论未附 | 2026-09-12，2026-09-23 归档 |
| [Jittor vs 真 PyTorch 2.9.1：现在差在哪](2026-09-14-jittor-vs-pytorch.md) | 快照：对 eager 赢 6 平 5 输 1，对 `torch.compile` 赢 2 平 2 输 5 | 2026-09-14 |
| [Torch 兼容层在真实模型上对 PyTorch：差距表、显存与性能修复](2026-09-24-torch-compat-real-models.md) | 11 项全部跑通，几何平均 1.26x，进程显存峰值为 PyTorch 的 0.87–1.45 倍；余下差距在主机侧 | 2026-09-24 |
| [性能/显存分析工具：审计与重写](2026-09-25-profiling-tools.md) | 已实现，RTX 4090 上验证；未合入 | 2026-09-25 |
| [张量维度上限 10 → 32 与下游评测的核心缺口](2026-10-05-high-rank-and-downstream-core-gaps.md) | 已修复，CPU 与 RTX 4090 验证 | 2026-10-05 |
| [下游 P1：分配器并存上限、数值协议与 transpose 的运行时置换](2026-10-09-downstream-p1-allocator-protocol-transpose.md) | 已修复，CPU 验证，transpose/协议另在 RTX 4090 验证 | 2026-10-09 |
| [Benchmark 覆盖与比较协议补齐](2026-10-14-benchmark-coverage.md) | 卷积 CPU/CUDA、Accelerate CPU、模型 CPU harness 已验；其余设备与场景缺口保留 | 2026-10-14 |
| [下游注册表、门禁与覆盖审计](2026-10-14-downstream-coverage-audit.md) | 已审计 13 个 runbook 与五个上游注册表；未授予新的设备验证级别 | 2026-10-14 |
| [优化器 FQN checkpoint 与确定性续训](2026-10-14-optimizer-checkpoint-continuation.md) | 单进程确定性数值续训已验；RNG/双卡验收见后续报告，ACL 待验 | 2026-10-14 |
| [完整 RNG 与 FSDP2 FULL checkpoint 续训](2026-10-14-checkpoint-rng-fsdp2-continuation.md) | CPU/CUDA 随机流与双卡 FULL 续训已验；分片 I/O 与 ACL 待验 | 2026-10-14 |
| [bf16 autocast 与 GradScaler 边界](2026-10-14-amp-bfloat16-gradscaler.md) | CPU/CUDA 实际 dtype 与边界对拍通过；fidelity 保持 APPROXIMATE | 2026-10-14 |
| [分布约束命名空间发布](2026-10-15-distribution-constraint-publication.md) | 原生属性与工厂发布已修；CPU/CUDA 边界检查对拍通过 | 2026-10-15 |
| [训练注册表分母补齐](2026-10-15-training-registry-denominators.md) | 五库源码分母已重导；补 ms-swift/verl 训练轴，未授予数值覆盖 | 2026-10-15 |
| [CUDA SDPA 掩码覆盖](2026-10-14-cuda-sdpa-mask-coverage.md) | 180 个真实 CUDA 场景对拍；融合路径扩展，边界保持显式 | 2026-10-14 |
| [图重放保留预算](2026-10-14-graph-replay-retain-budget.md) | 动态预算约束已缓存的 replay；默认预算保留，CPU/CUDA 实测 | 2026-10-14 |
| [下游独立 oracle 与训练 scorecard](2026-10-14-downstream-oracle-scorecards.md) | 四库 CPU/CUDA 代表场景已验；LoRA/component 范围限定，未授予 L4/L5 | 2026-10-14 |
| [CUDA 13 工具链与 uv 安装](2026-10-15-cuda13-toolchain-packaging.md) | Linux/SM89 工具链、CUB、矩阵乘与卷积实测；Cu122 回归通过 | 2026-10-15 |
| [双卡 bf16 FSDP2 续训](2026-10-15-bf16-fsdp2-checkpoint-continuation.md) | bf16 真计算与 FULL/RNG 精确恢复已验；跨框架舍入差异保留 | 2026-10-15 |
| [TorchMetrics state 与 TensorDict 对拍](2026-10-15-metric-state-and-tensordict-oracles.md) | 两库 CPU/CUDA 全梯度通过；TensorDict 三步训练通过，限定组件范围 | 2026-10-15 |
| [Checkpoint 选项与子模块契约](2026-10-15-checkpoint-state-options.md) | CPU/CUDA 单进程选项和实际 Torch checkpoint 互通已验；多 rank 选项与 ACL 待验 | 2026-10-15 |
| [整合回归与 NCCL 缺陷收口](2026-10-15-integration-nccl-coverage.md) | NCCL 2.18.3 双卡退出条件通过，KI-DIST-002/003 收口；限定定向范围 | 2026-10-15 |
| [上游 PR 处置与 NumPy ABI](2026-10-15-upstream-pr-disposition.md) | 十个 PR 已核对；NumPy 1/2 真正复制与 shape 回归通过，Windows 待真机 | 2026-10-15 |
| [VERL 注册算法组件对拍](2026-10-15-verl-registered-component-oracles.md) | GAE/vanilla PPO 与 k3 组件 CPU/CUDA 全梯度及三步轨迹通过；不计 rollout/Ray/Hydra | 2026-10-15 |
