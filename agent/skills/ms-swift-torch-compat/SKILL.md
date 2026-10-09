---
name: ms-swift-torch-compat
description: 设计、实现和验证 ms-swift 在 Jittor torch shim 上的兼容性。覆盖实际使用的模型、tuner、训练/推理入口、状态恢复、数据与序列化、分布式和目标设备；不把单个 LoRA case 当作完整适配。
---

# ms-swift Jittor Torch 兼容

## 目标

本 skill 的对象是 **ms-swift 通过公开 torch API 使用 Jittor** 的完整兼容面，而不是某一个模型或某一个 LoRA 示例。首个 `ms_swift_lora_llama` case 只是快速定位和回归入口；它通过后仍必须按适配面矩阵扩展验证，未执行的面必须明确记录为 `not-run`，不能从首个 case 推广结论。

适配范围由任务声明的 ms-swift 版本、安装的可选依赖和用户要求共同确定。开始时先盘点真实版本暴露的公开入口，再建立范围矩阵；不要凭印象声称“支持整个 ms-swift”。

## 统一等级与本轮完成边界

本任务只使用[验收合同](references/verification.md)的飞书 L0–L5：L0 构造、L1 前向、L2 梯度与更新、L3 完整恢复、L4 公开入口、L5 稳态性能。通用 `downstream-library-adaptation` 的同名等级不可换算。缺陷 C 类、首个失败 L 层和 core/compat/adapter 归属分别记录。

先冻结本轮必须交付的代表性“功能面 × 模型/流程 × 设备 × 拓扑”清单。选实际共享机制的代表用例，不逐个穷举全部注册模型；未执行者标 `not-run`。每格分开记录“公开入口实际运行/产出”和“层级验收状态”：CLI 跑完但 dtype、激活重算或恢复语义失败，只能记入口运行，不得授予 L4。诊断性性能可定位瓶颈，不能充当 L5。

## 适配面矩阵

至少逐项盘点并给出状态、首个断点、运行键和证据位置：

| 面 | 必须覆盖的内容 |
| --- | --- |
| 入口与配置 | `swift` CLI/launcher、配置解析、runtime 初始化、设备/精度/随机性配置 |
| 模型 | 任务要求的模型族；至少区分 causal LM、seq2seq/encoder、视觉或多模态入口（实际暴露者） |
| tuner | ms-swift 自有 LoRA 及版本中实际公开的其他 tuner；PEFT LoRA 只能作为互操作路径单独标注 |
| 训练 | SFT/监督训练；任务或安装依赖要求时增加 preference/DPO、RL/奖励训练等公开流程 |
| 状态 | optimizer、scheduler、梯度累积、混合精度、保存、全新进程恢复和继续训练 |
| 推理 | adapter 加载、generate、greedy/采样配置、cache、批处理和公开推理入口 |
| 数据链路 | datasets、tokenizers、collator、padding/label、离线数据读取及有效样本划分 |
| 序列化 | safetensors、模型/adapter export、checkpoint 元数据与设备迁移 |
| 分布式 | 单卡、单机多卡和在资源具备时的多机；实际 backend、rank、同步与恢复 |
| 生态调用 | transformers、peft、accelerate 等被 ms-swift 实际路径调用的 API；只证明调用路径，不能冒充这些库的独立全量支持 |
| 设备 | CPU 参考、任务目标加速器（如 Ascend ACL/NPU）；CUDA/其他后端仅在任务声明时加入 |
| 性能 | 真实目标尺寸的训练/推理稳态性能、显存和零 fallback；tiny case 只用于正确性定位 |

“不适用”只能在版本/配置确实没有该入口并留有盘点证据时使用；资源不足、依赖缺失、尚未执行都写 `resource-blocked` 或 `not-run`，不得改写成通过。

## 工作流程

1. 遵循仓库协作规则：检查 dirty worktree 和运行中测试，保留本地改动，整合并记录目标远端 SHA；不丢弃改动、不自动 stash。所有运行继续使用独立 state、解释器、`JITTOR_HOME` 和临时目录。
2. 读取[环境合同](references/environment.md)、[验收合同](references/verification.md)和 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)。先记录 ms-swift、transformers、peft、accelerate、datasets、tokenizers、safetensors 版本和实际模块路径。
3. 从 ms-swift 的公开 CLI、launcher、model/tuner registry 和已安装依赖生成适配面矩阵；不把内部手写循环当作公开入口，也不把不存在的可选功能加入必验范围。
4. 每个面先做裸跑最小复现，再按三行判据分流：能力缺失修 Jittor core/ACL/HCCL，torch API/状态语义差异修 `jittor.compat.torch`，只有可迁出且确属 ms-swift 私有行为才允许 adapter。
5. 先用 CPU 外部 PyTorch oracle 收敛公共语义，再在真实目标设备用原生 PyTorch/torch_npu oracle 对拍。设备运行必须确认实际 provider、tensor residency、backend 和完整工作阶段 `fallback_policy=error`；import 成功或 CPU fallback 不算通过。
6. 按“功能面 × 模型/流程 × 设备 × 执行拓扑”推进 L0-L5。相同运行键且证据完整时复用，源码、依赖、设备、协议或输入改变则生成新键；保留旧失败，不事后改判。
7. 维护性回归进入生态门禁。新增公共 shim 行为必须有定向结构/API/数值测试；公开入口、分布式、恢复和性能证据必须分别记录，不能用通信微检或单步循环替代。

## 短预检、根因与续接

真机长作业前用短时 worker 预检解释器与模块身份、依赖 pin、固定模型/数据摘要、CLI 参数、钩子签名、审计变量、短 `TMPDIR`、私有 `CCACHE_DIR`、结果读取器和退出码。先证明两侧逐步样本身份、输入、初始权重与参数映射一致，再比较 loss/logits/梯度；同 seed 不保证相同 batch。harness/调度失败单列无效运行，不升级兼容结论。

按源码、ABI、编译器、CANN 和设备指纹分离冷编译与模型验证；只有指纹一致才复用预编译产物，并发作业各自独占可写缓存。冷编译有进展时看编译产物与活跃进程，不只看 pytest 百分比；冷编译不计入 L5。前置失败就收集根因并取消无效依赖作业，修复后用新运行键。每个根因最多五轮不同假设的修复，仍失败则留断点并推进独立面。

监督脚本以作业终态、新错误、完整证据和需要决策为 Codex 唤醒事件；等待作业时用轻量状态检查，不按分钟重新读全套 Skill/Git/日志。Git fetch 只在新任务、集成或推送边界执行，网络失败退避并保留最近可信 SHA。矩阵仅在验收结论、根因或范围改变时更新，逐次日志保留在 state。

若当前只剩等待 Slurm 作业，在回复末尾独占一行写 WAIT_JOBS=123,456（真实作业号）；监督脚本仅用 shell 等状态变化或最多 30 分钟再唤起 Codex。

## 代表性测试策略

不要穷举每个模型，而要覆盖每种共享机制，并把未覆盖的组合列出来：

- **模型族**：从实际任务中选 causal LM、seq2seq/encoder、视觉/多模态代表；同一共享 transformer 模块的模型可复用算子证据，但仍需验证各自输入/输出结构。
- **tuner**：首个 Swift LoRA smoke 后，逐一加入实际要支持的 tuner/target module、frozen/trainable 参数策略、adapter 保存与加载；PEFT 与 Swift tuner 的结果分开。
- **训练阶段**：L0 构造，L1 forward，L2 全部适用梯度和真实 optimizer 更新，L3 完整 checkpoint 新进程恢复，L4 真实公开 CLI，L5 锁定真实尺寸稳态测量。
- **输入与状态**：整数 token、浮点输入、label/mask、梯度累积、scheduler、AMP、随机状态、dataloader/sampler 游标和多 rank 数据划分按实际使用情况逐项锁定。
- **入口与依赖**：公开 `swift` 命令必须与原生入口分别运行；依赖库只验证 ms-swift 实际使用的 API。若用户另要求某个依赖库整体适配，改用该库自己的 runbook。

## 分布式与设备规则

四种执行拓扑是 Ascend/NPU profile 的验收维度，不是全部适配面：单 NPU 训练、单机多 NPU 训练、真实多机多 NPU 训练和单 NPU 推理分别记录；CPU profile 则至少记录单进程训练/推理及公开入口。单机微检不能代表公开分布式训练；单机不能模拟多机。每个 rank 记录 hostname、global/local rank、world size、目标设备 ID、process group/backend、同步前后梯度/状态、退出结果和恢复结果。没有目标硬件就记未验证，不用 CPU fallback 冒充。

## 交付与边界

仓库只保留 skill、源码、测试、可复用工具和简洁结果；原始日志、checkpoint、缓存和运行环境放在任务 state。报告必须写精确 baseline、dirty diff、依赖/设备、命令、运行键、每个适配面的状态和边界。完成 `git diff --check`、布局检查和受影响结构门禁。

本 skill 不自动扩大到 vLLM、DeepSpeed、Megatron 或其他下游产品；这些路径只有在用户明确要求且按对应 adapter/runbook 分流后才加入矩阵。不得为了让某个下游流程变绿而在 ms-swift 中复制 Jittor 或 torch 的第二套实现。
