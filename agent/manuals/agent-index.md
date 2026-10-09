# Jittor Agent 文档索引

`agent/` 只保存操作手册（`manuals/`）与可复用 skill（`skills/`），这里是两者的唯一入口。
仓库维护命令在 `tools/`；设计与机制说明在 `docs/development/`、`docs/notes/` 和
`docs/compatibility/`，验证结论在 `docs/results/`；任务状态、交接和验证证据记在
GitHub issue 与 PR 中。原始日志、缓存和大体积产物放在 `$JITTOR_LAB_ROOT`（未设置时用
仓库同级的 `jittor-lab/`），见[工作区边界](collaboration.md#工作区边界)。

## 开始工作

1. 按 [`AGENTS.md`](../../AGENTS.md) 同步目标远端分支，再读[协作手册](collaboration.md)。
2. 经 [jittor-dev-context](../skills/jittor-dev-context/SKILL.md) 读[项目上下文](project-context.md)。
3. 按任务读[环境规则](environment.md)、[已知问题总账](known-issues.md)、
   [硬件延迟清单](deferred-hardware.md)与对应的 `docs/` 文档。
4. 对拍、验证或基准先找下面已有的 skill，不重新造轮子。

## Manuals

- [collaboration.md](collaboration.md)：协作流程、文档纪律、验证原则、JIT 并发与工作区边界。
- [project-context.md](project-context.md)：项目是什么、当前状态、信息在哪、当前重点。
- [environment.md](environment.md)：隔离运行模板、Python 与工具、各后端门禁的前置条件。
- [known-issues.md](known-issues.md)：未关闭缺陷与已知限制的总账（条目格式与编号规则在其头部）。
- [deferred-hardware.md](deferred-hardware.md)：代码已写、等硬件验收的事项与上机命令。
- [hccl-on-device-verification.md](hccl-on-device-verification.md)：HCCL 集合通信去同步的上机验证步骤。

## Skills

### 流程、协作与门禁

- [jittor-dev-context](../skills/jittor-dev-context/SKILL.md)：开工时的上下文路由入口。
- [github-collaboration-commit](../skills/github-collaboration-commit/SKILL.md)：同步、提交、快进推送与 PR 流程。
- [git-worktree-shared-state](../skills/git-worktree-shared-state/SKILL.md)：多个 worktree 之间共用与不共用的 git 状态。
- [jittor-worktree-verification](../skills/jittor-worktree-verification/SKILL.md)：确认跑的是本 worktree 的代码。
- [verifying-a-gate-actually-ran](../skills/verifying-a-gate-actually-ran/SKILL.md)：相信一次红或绿之前先证明门禁真的跑了。
- [gate-tier-budget](../skills/gate-tier-budget/SKILL.md)：门禁分层与按算术校验的耗时预算。
- [codebase-wide-fix-as-rule](../skills/codebase-wide-fix-as-rule/SKILL.md)：把全树修复写成结构规则并用反例证明它有牙。
- [pure-code-motion-refactor](../skills/pure-code-motion-refactor/SKILL.md)：证明拆分或搬移代码行为逐字不变。

### 核心 C++、构建与绑定

- [jittor-core-cpp-edit-loop](../skills/jittor-core-cpp-edit-loop/SKILL.md)：改 `src/` 的编辑、重编、验证循环。
- [jittor-build-change-verification](../skills/jittor-build-change-verification/SKILL.md)：构建系统改动的冷/热缓存、并发与切 flag 验证。
- [jittor-build-time-capabilities](../skills/jittor-build-time-capabilities/SKILL.md)：同一源码为何编出能力不同的构建。
- [cuda12-pip-stack-acceptance](../skills/cuda12-pip-stack-acceptance/SKILL.md)：系统 CUDA 作干扰项时验收 `jittor[cuda12]` 的干净环境安装。
- [jittor-pyjt-bindings](../skills/jittor-pyjt-bindings/SKILL.md)：pyjt 绑定生成器与转换层的验证。
- [jittor-single-kernel-cold-profile](../skills/jittor-single-kernel-cold-profile/SKILL.md)：单个生成 kernel 的冷编译分阶段计时。
- [jit-compile-failure-attribution](../skills/jit-compile-failure-attribution/SKILL.md)：JIT 编译失败是否归因到闯祸的算子。
- [core-invariant-property-tests](../skills/core-invariant-property-tests/SKILL.md)：给图、liveness 与执行器写属性测试。
- [graph-inference-to-explicit-record](../skills/graph-inference-to-explicit-record/SKILL.md)：把图反推改成在产生处显式记录。
- [process-global-state-and-optin](../skills/process-global-state-and-optin/SKILL.md)：找出全进程副作用并改成可撤销的 opt-in。
- [hang-that-holds-the-gil](../skills/hang-that-holds-the-gil/SKILL.md)：定位持有 GIL 的挂死并改成有界等待。

### 正确性对拍与语义

- [jittor-op-parity-oracle](../skills/jittor-op-parity-oracle/SKILL.md)：Python 层算子的数值对拍基准与容差。
- [jittor-grad-silence-probe](../skills/jittor-grad-silence-probe/SKILL.md)：判断梯度是否被静默吞掉、查询反向图形状。
- [optimizer-semantics-diffing](../skills/optimizer-semantics-diffing/SKILL.md)：优化器语义（累积、裁剪、偏差修正）的对拍。
- [native-api-sweep](../skills/native-api-sweep/SKILL.md)：原生公开接口面的存续、数值与显存扫描。
- [jittor-allocator-flag-matrix](../skills/jittor-allocator-flag-matrix/SKILL.md)：分配器缺陷的 flag 矩阵与竞争复现。
- [multi-device-verification](../skills/multi-device-verification/SKILL.md)：多卡上证明张量真的在声称的设备上。
- [jittor-distributed-verification](../skills/jittor-distributed-verification/SKILL.md)：单机多进程验证 MPI/NCCL 与 rendezvous。
- [torch-checkpoint-stride-oracle](../skills/torch-checkpoint-stride-oracle/SKILL.md)：PyTorch checkpoint 的 storage/stride 读取对拍。

### 性能

- [jittor-core-planning-cost](../skills/jittor-core-planning-cost/SKILL.md)：核心记账改动的规划开销定价。
- [jittor-graph-construction-cost](../skills/jittor-graph-construction-cost/SKILL.md)：Python 前端建图的主机开销。
- [codegen-optimization-in-effect](../skills/codegen-optimization-in-effect/SKILL.md)：证明指令级 codegen 优化真的生成且有收益。
- [cuda-elementwise-bandwidth-roofline](../skills/cuda-elementwise-bandwidth-roofline/SKILL.md)：逐元素 kernel 的带宽屋顶线与同口径对比。
- [cuda-reduction-strategy-comparison](../skills/cuda-reduction-strategy-comparison/SKILL.md)：CUDA 归约三种策略的选择与测量。
- [cuda-backend-choice-proof](../skills/cuda-backend-choice-proof/SKILL.md)：证明 CUDA 库算子选到了预期算法、精度或缓存键。
- [jittor-transformers-perf](../skills/jittor-transformers-perf/SKILL.md)：Transformers 在 shim 与 PyTorch 上的 CUDA/昇腾性能分析。

### 无目标硬件时的后端验证

- [acl-host-syntax-check](../skills/acl-host-syntax-check/SKILL.md)：无 CANN 主机上对 ACL C++ 做真实 TU 语法检查。
- [static-launch-equivalence](../skills/static-launch-equivalence/SKILL.md)：去样板前后设备调用的静态等价性证明。
- [cuda-negative-path-verification](../skills/cuda-negative-path-verification/SKILL.md)：真正跑 CUDA 负向用例并判断绿色是否算数。

### Torch 兼容层

- [torch-shim-noop-audit](../skills/torch-shim-noop-audit/SKILL.md)：判断一个 torch API 是真实现还是签名齐全的空操作。
- [torch-api-cohort-promotion](../skills/torch-api-cohort-promotion/SKILL.md)：把一族 torch API 写成模块级一等对象并登记保真度。
- [jittor-torch-diff](../skills/jittor-torch-diff/SKILL.md)：shim 与真 PyTorch 的双解释器前反向对拍与梯度调试。

### 下游生态库

- [downstream-library-adaptation](../skills/downstream-library-adaptation/SKILL.md)：接入下游库的分流判据、device 阶梯与 adapter 准入。
- [torch-compat-repo-runbook](../skills/torch-compat-repo-runbook/SKILL.md)：下游库 runbook 模板与四轴验收协议。
- [transformers-torch-compat](../skills/transformers-torch-compat/SKILL.md)：HuggingFace Transformers。
- [diffusers-torch-compat](../skills/diffusers-torch-compat/SKILL.md)：diffusers。
- [peft-torch-compat](../skills/peft-torch-compat/SKILL.md)：PEFT（LoRA）。
- [ms-swift-torch-compat](../skills/ms-swift-torch-compat/SKILL.md)：ms-swift 自带 LoRA tuner。
- [mmcv-torch-compat](../skills/mmcv-torch-compat/SKILL.md)：mmcv（mmcv-lite 的 `mmcv.cnn`）。
- [mmengine-torch-compat](../skills/mmengine-torch-compat/SKILL.md)：MMEngine 的模型基类。
- [tensordict-torch-compat](../skills/tensordict-torch-compat/SKILL.md)：TensorDict。
- [torchmetrics-torch-compat](../skills/torchmetrics-torch-compat/SKILL.md)：TorchMetrics。
- [flash-attention-torch-compat](../skills/flash-attention-torch-compat/SKILL.md)：官方 flash-attention 与 SDPA 接线。
- [vllm-torch-compat](../skills/vllm-torch-compat/SKILL.md)：vLLM。
- [vllm-omni-torch-compat](../skills/vllm-omni-torch-compat/SKILL.md)：vLLM-Omni 的 MiniMax-H3 推理。
- [verl-torch-compat](../skills/verl-torch-compat/SKILL.md)：verl（PPO、FSDP2、权重传输）。
- [trellis-torch-compat](../skills/trellis-torch-compat/SKILL.md)：TRELLIS.2 3D 生成 pipeline。
- [torchquantum-readme-validation](../skills/torchquantum-readme-validation/SKILL.md)：TorchQuantum README 用法验证。
