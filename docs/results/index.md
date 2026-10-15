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
2026-10-15-sdpa-opmath
2026-10-15-like-factory-leaves
2026-10-15-diffusers-component-scorecards
2026-10-15-mpi-bfloat16-process-groups
2026-10-15-cpu-bfloat16-accumulators
2026-10-15-h3-silu-contracts
2026-10-15-accelerate-protocol-scorecards
2026-10-15-deepspeed-paired-protocol
2026-10-14-graph-replay-retain-budget
2026-10-14-downstream-oracle-scorecards
2026-10-15-cuda13-toolchain-packaging
2026-10-15-bf16-fsdp2-checkpoint-continuation
2026-10-15-metric-state-and-tensordict-oracles
2026-10-15-metrics-tensordict-scorecards
2026-10-15-vllm-component-scorecards
2026-10-15-checkpoint-state-options
2026-10-15-integration-nccl-coverage
2026-10-15-upstream-pr-disposition
2026-10-15-verl-registered-component-oracles
2026-10-15-transformers-model-family-matrix
2026-10-15-native-sharded-checkpoint
2026-10-15-longformer-32768-context-oracles
2026-10-15-deepspeed-engine-prerequisites
2026-10-15-deepspeed-zero-engine
2026-10-15-native-zero-optimizer
2026-10-15-native-tensor-parallel
2026-10-15-native-fsdp
2026-10-15-bfloat16-linear-bias-accumulation
2026-10-15-native-tensor-parallel-adamw-bf16
2026-10-15-fsdp-compat-lifecycle
2026-10-15-openmmlab-training-oracles
2026-10-15-tensor-detach-and-factory-leaf
2026-10-15-cuda-extension-philox
2026-10-15-flash-attention-training-oracles
2026-10-15-deepspeed-bfloat16-multirank
2026-10-15-deepspeed-maintained-gate
2026-10-15-tanh-fsdp-bfloat16
2026-10-15-trellis-model-validation
2026-10-15-pytree-transformers-513
2026-10-15-rng-native-generator-contract
2026-10-15-rmsnorm-triton-alias
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
| [训练注册表分母补齐](2026-10-15-training-registry-denominators.md) | 补 ms-swift/verl 训练轴与 vLLM/Omni 分母、SCM wheel 源码身份校验；枚举不授予数值覆盖 | 2026-10-15 |
| [CUDA SDPA 掩码覆盖](2026-10-14-cuda-sdpa-mask-coverage.md) | 180 个真实 CUDA 场景对拍；融合路径扩展，边界保持显式 | 2026-10-14 |
| [图重放保留预算](2026-10-14-graph-replay-retain-budget.md) | 动态预算约束已缓存的 replay；默认预算保留，CPU/CUDA 实测 | 2026-10-14 |
| [下游独立 oracle 与训练 scorecard](2026-10-14-downstream-oracle-scorecards.md) | 四库 CPU/CUDA 代表场景已验；LoRA/component 范围限定，未授予 L4/L5 | 2026-10-14 |
| [CUDA 13 工具链与 uv 安装](2026-10-15-cuda13-toolchain-packaging.md) | Linux/SM89 工具链、CUB、矩阵乘与卷积实测；Cu122 回归通过 | 2026-10-15 |
| [双卡 bf16 FSDP2 续训](2026-10-15-bf16-fsdp2-checkpoint-continuation.md) | bf16 真计算与 FULL/RNG 精确恢复已验；跨框架舍入差异保留 | 2026-10-15 |
| [TorchMetrics state 与 TensorDict 对拍](2026-10-15-metric-state-and-tensordict-oracles.md) | 两库 CPU/CUDA 全梯度通过；TensorDict 三步训练通过，限定组件范围 | 2026-10-15 |
| [vLLM 组件 CPU/CUDA 评分卡](2026-10-15-vllm-component-scorecards.md) | 三个参数无关组件 L2，各 3/40 CustomOp；模型 0/376，整引擎阻塞另列 | 2026-10-15 |
| [TorchMetrics/TensorDict 统一 scorecard](2026-10-15-metrics-tensordict-scorecards.md) | 新 CPU/CUDA 原始证据接受 4 条；有界顶层 API 各 2/105 L2 与 2/60 L3 | 2026-10-15 |
| [Checkpoint 选项与子模块契约](2026-10-15-checkpoint-state-options.md) | CPU/CUDA 单进程选项和实际 Torch checkpoint 互通已验；多 rank 选项与 ACL 待验 | 2026-10-15 |
| [整合回归与 NCCL 缺陷收口](2026-10-15-integration-nccl-coverage.md) | NCCL 2.18.3 双卡退出条件通过，KI-DIST-002/003 收口；限定定向范围 | 2026-10-15 |
| [上游 PR 处置与 NumPy ABI](2026-10-15-upstream-pr-disposition.md) | 十个 PR 已核对；NumPy 1/2 真正复制与 shape 回归通过，Windows 待真机 | 2026-10-15 |
| [VERL 注册算法组件对拍](2026-10-15-verl-registered-component-oracles.md) | GAE/vanilla PPO 与 k3 组件 CPU/CUDA 全梯度及三步轨迹通过；不计 rollout/Ray/Hydra | 2026-10-15 |
| [Transformers 模型族矩阵](2026-10-15-transformers-model-family-matrix.md) | 九项 CPU/CUDA L3 场景；8/374 模型族、2/46 任务，含 CLIP/Mixtral/2048 token RoPE | 2026-10-15 |
| [DeepSpeed Engine 前置所有权与梯度契约](2026-10-15-deepspeed-engine-prerequisites.md) | 原生模块注册与反向回调验证；真实 Engine 后续验收见专用报告 | 2026-10-15 |
| [DeepSpeed 单 rank FP32 训练与恢复](2026-10-15-deepspeed-zero-engine.md) | CPU、CUDA 各 12 配置通过；双 rank bf16 尚未验收 | 2026-10-15 |
| [原生 ZeRO1 状态分片](2026-10-15-native-zero-optimizer.md) | CPU 与双卡 NCCL 训练恢复已验；限定 Stage 1，不含 FSDP/TP 或多机 | 2026-10-15 |
| [OpenMMLab 真实训练对拍](2026-10-15-openmmlab-training-oracles.md) | MMEngine/MMCV 的 CPU、CUDA 四条 L3 证据；限定真实 train_step 场景 | 2026-10-15 |
| [Tensor detach 与工厂梯度边界](2026-10-15-tensor-detach-and-factory-leaf.md) | CPU/CUDA 原地 detach、默认梯度发布和 bf16 新叶子契约通过；双卡 Engine 仍待验 | 2026-10-15 |
| [CUDA 扩展 Philox 状态](2026-10-15-cuda-extension-philox.md) | 实际 CUDA RNG 重放与融合 FlashAttention 训练回归；独立 Torch 对拍仍待验 | 2026-10-15 |
| [FlashAttention 公开训练 API 对拍](2026-10-15-flash-attention-training-oracles.md) | 六入口历史 48 轨迹及正式 12 CUDA 场景通过；统一 L3 记录已验，CPU 阶梯缺失使广度仍 0/7 | 2026-10-15 |
| [DeepSpeed 双卡 BF16 训练与恢复](2026-10-15-deepspeed-bfloat16-multirank.md) | ZeRO1/2 的 AdamW 三步训练、完整梯度与严格续训已验；限定小模型与实际 CUDA 双卡 | 2026-10-15 |
| [DeepSpeed 维护门禁与 CPU 状态舍入](2026-10-15-deepspeed-maintained-gate.md) | 双卡 BF16 四格及 CPU/CUDA 各 12 项训练续训通过；修复 CPU Adam 状态取值顺序差异 | 2026-10-15 |
| [tanh 反向与原生 FSDP BF16 对拍](2026-10-15-tanh-fsdp-bfloat16.md) | 修复低精度保存输出求导；固定双卡 BF16 的 80 个参数梯度逐位对齐，续训 368 数组逐位一致 | 2026-10-15 |
| [TRELLIS.2 真实模型训练对拍](2026-10-15-trellis-model-validation.md) | 三个真实模型的 CPU/CUDA 三步训练、全部梯度和独立扰动通过；统一 models 广度各 3/9 | 2026-10-15 |
| [自定义 pytree 与 Transformers 5.13.0](2026-10-15-pytree-transformers-513.md) | 自定义节点与键路径修复；固定 GPT2 的 CPU/CUDA 三步、全部参数和输入梯度通过 | 2026-10-15 |
| [原生 RNG 与完整位宽 seed](2026-10-15-rng-native-generator-contract.md) | CPU/CUDA 原生流、64 位 seed、显式生成器及双设备 checkpoint 恢复通过；不承诺跨框架样本逐位一致 | 2026-10-15 |
| [RMSNorm 可选契约与 Triton 原地别名](2026-10-15-rmsnorm-triton-alias.md) | 保护模块原始归一化契约，修复真实 CUDA 原地写入被覆盖，并接入 vLLM 组件门禁审计 | 2026-10-15 |
| [原生 TP 线性层训练与恢复](2026-10-15-native-tensor-parallel.md) | CPU 与双卡 NCCL 的 FP32 等分线性层、完整梯度和固定 world 续训已验；bf16 与多机待验 | 2026-10-15 |
| [原生 FSDP 分片与新进程恢复](2026-10-15-native-fsdp.md) | 原生双卡训练与严格恢复已验；bf16 小梯度误差和旧兼容失败单列 | 2026-10-15 |
| [CUDA BF16 Linear 偏置舍入](2026-10-15-bfloat16-linear-bias-accumulation.md) | 真实 BF16 GEMM 保留 FP32 累加到偏置之后；CPU 既有路径与二阶梯度回归已验 | 2026-10-15 |
| [原生 TP AdamW 与 BF16 续训](2026-10-15-native-tensor-parallel-adamw-bf16.md) | 双卡 BF16 完整梯度、状态分片和新对象恢复通过固定误差预算；不含多机 | 2026-10-15 |
| [FSDP 兼容层优化器生命周期](2026-10-15-fsdp-compat-lifecycle.md) | 优化器参数替换与原生模块 checkpoint 分派修复，CPU/CUDA 与双 rank 定向复验 | 2026-10-15 |
