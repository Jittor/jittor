# 2026-09-22：ms-swift LoRA 的 Ascend torch shim 验证

**状态（2026-10-09 19:19 CST 复核）：完整逐格矩阵见下文。tiny 单卡 LLaMA LoRA 公开 SFT/恢复 L0–L4 通过，L5 未运行；两卡 tiny IA3 fused AdamW、ORPO 与目标 2.0-refactor DPO 锁定配置的公开训练/完整恢复及逐 rank 设备审计均通过 L0–L4，L5 未运行。另有 tiny LLaMAPro 单 NPU公开 SFT 通过 L0–L2、L4 和逐 rank 设备/数值审计；其原生 torch_npu checkpoint 新进程恢复在修补索引元数据后仍因恢复 optimizer 参数组与当前零可训练参数不匹配而失败，故 L3 未通过且未启动候选恢复。tiny LLaMA Swift Adapter 双 NPU公开 SFT 与完整 checkpoint 新进程恢复通过 L0–L4；连续四步与三步保存后续训对拍 11,637 项、0 失败，逐 rank 驻留及 strict fallback=0。此前原生恢复断点及比较器协议失败仍保留为历史记录。其单 NPU公开推理另完成 2461 项严格比较，L0/L1/L4 通过，L5 未运行。DPO int64 `labels.roll()` 的 ACL dtype 门禁由提交 `ac515b406` 修复；公开 DPO 新运行键完成四步连续训练、checkpoint-3 全新进程恢复及 170 项严格对拍，0 失败，最大绝对差 `1.1920928955078125e-07`。旧候选与比较器失败均保留为历史证据，详见 DPO 公开入口修复验收。结构门禁 job 1911 在隔离安装 `pytest-xdist==3.6.1` 后于 NPU worker 1389 passed、6 skipped、0 other skips，布局检查通过。Qwen3 tiny 公开 SFT 的 L2 失败，tiny 推理 L0/L1/L4 通过，约 1.1B 推理严格数值比较失败。所有通过只适用于各自锁定配置。真实多机仍受资源阻塞。**

各历史运行均绑定其 manifest 中的源基线与 dirty source，不因后续同步而改判。IA3 r3 工作负载的集成代码基线为 `a8dbba3984931a699aa814c31bcec7edf09fcb73`，upstream `2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；此次追补的是同一原始运行产物的 comparator 协议 v2/v3，不重跑模型。此前报告记录的较早 SHA 仍是对应历史运行的真实基线。维护者：Torch compatibility / ACL backend maintainers。核心初始化、梯度状态语义、依赖版本、后端/驱动或协议变化时重新验证。本结果仍在持续验收；不创建 PR 或合并 PR。

## 验收范围与当前状态

本报告采用[统一验收合同](../../agent/skills/ms-swift-torch-compat/references/verification.md)的飞书唯一刻度：L0 导入/构造，L1 前向/推理，L2 训练/梯度，L3 完整 checkpoint 保存、全新进程恢复并续训，L4 真实公开 ms-swift CLI/launcher 端到端，L5 正确性通过后的稳态性能。后端身份、NPU 驻留、分布式 HCCL、严格零回退、有限值和精确键/形状是每层门禁，不独立占一个等级。

| 轨道 | 当前证据 | 未完成项或阻塞 |
| --- | --- | --- |
| 单 NPU 训练 | 锁定 tiny case 的公开三步对拍、完整 checkpoint fresh-process 精确续训通过，覆盖 L0–L4 适用门禁 | L5 真实尺寸稳态性能未完成；不推广其他模型或浮点输入 |
| 单机多 NPU 训练 | IA3/fused AdamW、ORPO 与目标 2.0-refactor DPO 锁定配置的公开双 rank 训练、严格 ACL、逐 rank 设备驻留及完整恢复对拍通过 L0–L4 | 各配置 L5 真实尺寸稳态性能未运行；其它模型/tuner 未覆盖；真实多机仍资源阻塞 |
| 真实多机多 NPU 训练 | `resource-blocked`：2026-10-08 21:24 CST 的 Slurm `sinfo -N -p npu` 仅列出实际 NPU 节点 `cscg-hw01`（`gpu:8`）；没有第二个 `cscg-hw00` 调度节点 | 需要至少两个真实主机且每主机至少两张 NPU；不能用单机多进程或主机别名代替。解除条件是调度器提供第二台实际 NPU 主机并能分配到两台各至少两卡 |
| 单 NPU 推理 | tiny Qwen3 与 tiny Swift Adapter 公开 Swift 推理 L0/L1/L4 通过；约 1.1B 公开入口执行完成但 logits/KV 严格比较失败 | L5 稳态性能未运行；训练 L2/L3 不适用 |

### 按锁定工作负载逐格记录的 L0–L5 矩阵（2026-10-09）

格子状态只使用验收合同规定的 `passed`、`failed`、`not-run`、`resource-blocked`、`not-applicable`。只有同一行锁定配置及其原始运行证据支持的格子才标为通过；其它模型、拓扑或依赖不从该行外推。

| 轨道 × 锁定工作负载 | L0 | L1 | L2 | L3 | L4 | L5 | 证据与首个未闭合点 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 单 NPU 训练：tiny LLaMA、Swift q/v LoRA、FP32、固定整数 token，公开 SFT | passed | passed | passed | passed | passed | not-run | 第 292、349 节的三步轨迹与全新进程恢复；真实尺寸稳态性能及浮点输入梯度未验证。 |
| 单 NPU 训练：Qwen3 tiny、Swift LoRA、FP32、bool-mask SDPA、公开 SFT | passed | passed | failed | not-run | failed | not-run | 第 404 节：第二次 AdamW 更新超出预置容差；L3 未进入，L4 端到端严格验收失败。 |
| 单 NPU 推理：Qwen3 tiny、原生 checkpoint adapter、greedy 4 token、公开 infer | passed | passed | not-applicable | not-applicable | passed | not-run | 第 404 节：加载状态、logits 与 token ID 精确相等；tiny 计时不满足 L5。 |
| 单 NPU 推理：tiny LLaMA、Swift Adapter length=16/GELU、greedy 4 token、公开 `swift.cli.main infer` | passed | passed | not-applicable | not-applicable | passed | not-run | 原生 job 1940（run r4）→ strict ACL job 1975（run r6）→ worker comparator job 1976；2461 项、0 失败，输入/token 精确一致，参数最大差 `4.47035e-8`、logits `1.19209e-7`。初始 Adapter 线性层在 CPU，Swift Adapter 首次前向按输入设备延迟迁移；之后 16 次前向的 29 个参数均为 `npu:0`。候选 strict `backend_fallback=error`，完整阶段 fallback 起止均为 0。L5 未运行。 |
| 单 NPU 推理：约 1.1B Qwen3、公开 infer | passed | failed | not-applicable | not-applicable | failed | not-run | 第 349 节：公开响应/tokens 精确，但 logits/KV 数值有 34/1618 项超出固定阈值；L5 不得启动。 |
| 单机 2 NPU 训练：IA3、旧 fused AdamW 设备/单步审计、公开 SFT | passed | failed | failed | failed | failed | not-run | 第 420 节：forward 数值未保存，故 L1/L2 严格比较失败；候选 `optimizer.pt` 非有效 ZIP，L3 失败。局部参数、梯度、更新与设备观察保留为部分证据，不提升等级。 |
| 单机 2 NPU 训练：IA3、FP32、`adamw_torch_fused`、公开 `swift.cli.sft` 四步及 checkpoint-3 fresh-process 恢复 | passed | passed | passed | passed | passed | not-run | 新运行键 `ia3-world2-fullstate-fused-l3-r3-20261008`；Slurm 1713/1714 执行候选，1724 在 worker 上完成 protocol v3 比较。checkpoint 224 项与轨迹/设备 298 项零失败；optimizer 浮点状态按 `1e-7` 比较、adapter/参数轨迹及 forward/gradient/update 按预锁容差比较；每 rank NPU/HCCL、候选 strict error、fallback=0。L5 未运行。首次比较器 `1715` 与 protocol v2 的失败原样保留，原因及修正见下文。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 `swift.cli.rlhf --rlhf_type dpo` 三步与 checkpoint-3 fresh-process 恢复（历史 Jittor 1.3.11） | not-run | not-run | not-run | not-run | not-run | not-run | 运行键 `dpo-world2-20261003` 的候选日志实际为 Jittor 1.3.11，目标为 Jittor 2.0-refactor，故不计入本矩阵。旧 case 有公开双卡训练/恢复与 adapter/optimizer 数值比较，但没有全梯度、每 rank batch/forward 输出和计算张量物理设备审计；不得据此授予目标 L0–L4。需在当前基线下 oracle-first 完整审计。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 `swift.cli.rlhf --rlhf_type orpo` 四步及 checkpoint-3 fresh-process 恢复（历史 Jittor 1.3.11） | not-run | not-run | not-run | not-run | not-run | not-run | 运行键 `orpo-world2-rngfix-20261007` 的候选日志实际为 Jittor 1.3.11，不是本任务目标 Jittor 2.0-refactor；故不计入本矩阵。该历史运行的 native/candidate 数值轨迹与恢复 comparator 均为零失败，但 observer 未保存 forward 值或每 rank batch/参数/gradient/optimizer 物理 device。需在当前 2.0 基线下 oracle-first 重跑完整审计。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32、公开 DPO 四步与 checkpoint-3 fresh-process 恢复（目标 2.0-refactor） | passed | passed | passed | passed | passed | not-run | 新运行键 `dpo-world2-2x-full-audit-20261009-int64roll-fix01`：复用先行 native oracle job 1783；strict ACL candidate job 1887 连续训练四步，job 1888 在新进程从 checkpoint-3 恢复并续至 step 4；比较 job 1905 完成 170 项比较、0 失败，最大绝对差 `1.1920928955078125e-07`。两 rank 均为 HCCL/world-size=2；candidate `backend_fallback=error`，连续与恢复阶段 fallback 均为 0。每 rank 参数、实际 batch/计算输入、前向输出、梯度和 optimizer state 均驻留对应 NPU；原生 torch_npu 通用 fallback counter unavailable/unknown。L0–L4 仅适用于此锁定配置，L5 未运行。旧候选/比较器失败原样保留，根因与完整证据见下文。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift LoRA、FP32 fused AdamW、公开 ORPO 四步与 checkpoint-3 fresh-process 恢复（目标 2.0-refactor） | passed | passed | passed | passed | passed | not-run | 新键 `orpo-world2-2x-full-audit-20261008-jobguardfix01`；native oracle job 1800 → strict ACL candidate job 1801 → fresh-process resume job 1802 → comparator job 1803。公开 `swift.cli.rlhf --rlhf_type orpo`，world-size 2、HCCL、四步固定轨迹和 checkpoint-3 续训；146 项比较、0 失败。两侧 rank0/rank1 的参数、实际 batch/计算输入、前向输出、梯度及 84 个 optimizer-state 张量均在对应 NPU；candidate runtime=Jittor shim/provider=ACL、`backend_fallback=error`、起始/结束 fallback=0。native runtime/provider=torch_npu，通用 fallback counter `unavailable/unknown`。L5 真实尺寸稳态性能未运行；结论只适用于该锁定配置。未版本化原始证据位于 `$TASK_STATE/runs/orpo-world2-2x-full-audit-20261008-jobguardfix01/`。 |
| 单 NPU 训练：tiny LLaMA、Swift LLaMAPro、FP32 fused AdamW、公开 `swift.cli.sft` 三步 | passed | passed | passed | failed | passed | not-run | 运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-r1`。native oracle job 1917 先运行；strict ACL candidate job 1922 与 worker comparator job 1926 对 30 项比较、0 失败，最大绝对差：参数 `5.97909e-7`、optimizer `2.7940e-9`、logits `8.9407e-8`、梯度 `1.4901e-8`、更新 `5.9605e-7`，均在预锁容差内。公开三步训练的 L0–L2/L4 通过；candidate `backend_fallback=error`、HCCL world-size=1、fallback=0，参数/batch/forward/gradient/optimizer 均在 NPU。L3 native 新进程恢复 job 1928 首次因 SwiftModel checkpoint 缺根目录 shard index 失败；新运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-resume-index-r1` 的 job 2141 仅为原 checkpoint 复制并添加索引元数据后重试 native oracle，随后通过模型加载并在 optimizer load 处失败：SwiftModel 统计 0 个可训练参数，保存的 optimizer 参数组含训练参数，报 `loaded state dict contains a parameter group that doesn't match the size of optimizer's group`。安装版 Swift 4.5.2 `TunerMixin.prepare_model` 传入 `is_trainable=True`，但 `SwiftModel.from_pretrained` 的公开签名默认 `inference_mode=True` 且未消费 `is_trainable`；这与零可训练参数现象一致，是当前证据指向的上游恢复语义根因。两次 native 失败均未运行 candidate resume；须先有 native 公共恢复通过，不能用 adapter-only 加载替代完整状态。此配置 L3 failed、L5 not-run。原始证据：`$TASK_STATE/runs/llamapro-world1-full-audit-20261009-jittor2-fused-r1/`、`...-resume-r2/`、`...-resume-index-r1/`。 |
| 单机 2 NPU 训练：tiny LLaMA、Swift Adapter（length=16/GELU）、FP32 fused AdamW、公开 `swift.cli.sft` 三步及完整恢复到 step 4 | passed | passed | passed | passed | passed | not-run | 初始三步公开 SFT 运行键 `adapter-world2-full-audit-20261009-jittor2`：native oracle 1930 → strict ACL candidate 1931 → worker comparator 1935，2092 项、0 失败。L3 复核运行键 `adapter-world2-resume-audit-20261009-jittor2-r2/r4` 与 `adapter-world2-uninterrupted-audit-20261009-jittor2-r2`：native resume 2082、strict ACL resume 2091、native uninterrupted 2135、strict ACL uninterrupted 2136；worker compare-r6 job 2139 检查 11,637 项、0 失败。含两侧逐 rank 新进程 checkpoint 恢复/续训与不中断轨迹、batch 精确一致、checkpoint adapter/optimizer/scheduler/trainer/RNG，参数、forward、gradient、update 与 optimizer 值在预锁类别容差内；最大绝对差 logits `1.1920928955078125e-7`、梯度 `5.960464477539063e-8`、参数/更新 `4.470348358154297e-8`、optimizer `1.1175870895385742e-8`。两 rank HCCL 与实际 NPU 驻留逐项通过；候选 ACL `backend_fallback=error` 且起止 fallback=0。L5 未运行。先前 native checkpoint 分支和 comparator r4/r5 协议/元数据失败仍保留，不用于本 PASS；详见后续 L3 复核记录。 |
| 单机多 NPU 训练：Swift LoRA 候选公开分布式路径 | not-run | not-run | failed | not-run | failed | not-run | 第 349 节：已有真实 HCCL 微检，但候选公开 Trainer 路径先后在 DDP hook 导入、token-count gather 处失败；未产生可验收的分布式训练轨迹。原生 2/4/8 NPU 仅为 oracle。 |
| 真实多机训练：每台至少 2 NPU、至少 2 台主机 | resource-blocked | resource-blocked | resource-blocked | resource-blocked | resource-blocked | resource-blocked | 第 316、349 节：现有真实分配只有单主机；解除需实际多机 Slurm 资源。单机模拟不计。 |
| 单 NPU 推理：其它模型族、其它 tuner 或推理配置 | not-run | not-run | not-applicable | not-applicable | not-run | not-run | 需按实际 registry/可选依赖另立锁定运行键；不从 Qwen3 tiny 推广。 |

IA3 r3 已完成：native oracle 连续训练与 fresh-process resume 先运行；候选随后通过真实双 NPU `swift.cli.sft` 连续四步与 checkpoint-3 新进程续训。Slurm 1715 原始比较失败是因 comparator 把内部 optimizer 张量设为精确零容差；cmpv2 仅修正该项后 checkpoint 224 项零失败，但轨迹 comparator 又把跨 runtime 恢复初始可训练参数错误设为零容差。cmpv3 将这项恢复初始参数比较改为预置轨迹阈值 `1e-6`，其余键/shape/dtype/有限值、冻结参数精确、optimizer、scheduler、trainer、RNG、逐 rank HCCL/device/fallback 检查保持不变；Slurm 1724 完成，checkpoint 224 项及完整轨迹/设备 298 项均零失败。候选两 rank 使用 ACL、`backend_fallback=error`、结束回退计数均为 0；batch、forward output、参数、梯度和 optimizer state 的 rank 本地物理设备记录均为对应 NPU。原始运行及 comparator 协议/结果未版本化，位于 `$TASK_STATE/runs/ia3-world2-fullstate-fused-l3-r3-20261008/` 与 `...-cmpv2-20261008/`、`...-cmpv3-20261008/`。这是这一锁定配置的 L0–L4 证据，不覆盖 L5 或其它模型/tuner。

### 按适配功能面列出的覆盖边界

| 功能面 | 全范围状态 | 已验证的锁定配置与缺口 |
| --- | --- | --- |
| 入口与配置 | not-run | tiny 单卡 `swift` SFT 与 Qwen3 tiny `swift infer` 已在上表逐配置判定；完整 CLI/launcher 与配置组合矩阵未跑完。 |
| 模型族 | not-run | causal LM 的 tiny LLaMA、Qwen3 有配置级证据；Qwen3 约 1.1B 推理严格数值失败。当前 ms-swift 4.5.2 worker registry 盘点有 222 个 model type，其中 118 个标记多模态；无 T5 type，但含 BERT/ModernBERT encoder/reranker 与视觉/多模态类型。该盘点不代表模型运行：除已列 case 外，encoder 与视觉/多模态 L0–L5 仍未运行。 |
| tuner | not-run | Swift LoRA tiny 单卡 SFT、IA3 两卡 FP32 fused AdamW、ORPO 锁定配置均有 L0–L4 证据；LLaMAPro 单 NPU与 Adapter 双 NPU训练各有配置级 L0–L2/L4 证据，但 native L3 均未通过；其它 Swift tuner 与独立 PEFT 互操作未运行。 |
| 训练流程 | not-run | 上表锁定的单卡 SFT 有通过/失败项；IA3 fused AdamW 与 ORPO 双卡公开流程通过 L0–L4；Adapter 双卡公开训练 L0–L2/L4 通过但 L3 native oracle 失败；DPO 在公开训练首步失败，其它 RL/奖励训练组合尚未完成 oracle-first 对拍。 |
| optimizer、scheduler 与 AMP | not-run | tiny FP32 AdamW、ORPO 与 IA3 fused AdamW 的 optimizer checkpoint/恢复状态及逐 rank 物理驻留按锁定运行验收；AMP及其它 optimizer 组合未运行。 |
| checkpoint 与恢复 | not-run | tiny 单卡 LoRA、IA3 fused AdamW、ORPO、Swift Adapter 双卡有严格全新进程恢复证据；LLaMAPro native 新进程恢复先后遇到 SwiftModel shard-index 路径及零可训练参数/optimizer 组不匹配，Adapter 的首次 native oracle 曾遇 shard-index 问题、后续通过索引元数据诊断并已完成严格 L3；其它模型、tuner 与状态组合未运行。 |
| 推理与生成 | not-run | Qwen3 tiny adapter 加载、logits 与 greedy token ID 通过；约 1.1B logits/KV 严格比较失败；采样、cache 变体和批处理矩阵未运行。 |
| datasets、tokenizers、collator | not-run | 固定本地 JSONL 与锁定 tokenizer/collator 路径随已列训练 case 被执行；通用数据源、流式数据、划分策略及更广 collator 组合未运行。 |
| safetensors、导出与设备迁移 | not-run | 已列 tiny 与 IA3 锁定配置 checkpoint/adapter 路径通过；其它 export 格式和跨设备迁移组合未运行。 |
| Transformers、PEFT、Accelerate 等依赖调用 | not-run | 只验证 ms-swift 在上述公开 case 实际经过的调用路径；不代表任一依赖库独立全量支持，未覆盖 API 未运行。 |
| 分布式 | not-run | IA3 与 ORPO 锁定配置候选公开训练/恢复在 HCCL 双卡严格对拍和逐 rank 驻留审计通过；其它公开流程未覆盖，真实多机资源阻塞。原生多卡 oracle 不等于候选兼容。 |
| 稳态性能 | not-run | 所有轨道尚未同时满足相应正确性与 L4 前置，现存 tiny/编译计时不作 L5。 |

功能面表中的多个状态表示不同锁定配置的分项状态，不是“部分通过”的合成等级；具体模型、流程、设备和拓扑以紧邻的工作负载矩阵及后续章节为准。

四轨分别验收，不能用单卡 case 或通信微检替代分布式公开 launcher。CPU 模型、对拍、训练与性能不在本任务范围。没有新增 adapter，也没有修改 ms-swift 掩盖缺口。已有记录证明的是下列确实执行的部分，不再沿用“单卡 L0–L5 完成”的总体声明。

## 已有运行环境

验证 ms-swift 自身 `swift.tuners.LoRAConfig` 经 Jittor torch shim / ACL 执行，与独立原生 PyTorch/torch_npu NPU oracle 比较；已有数值和计时运行均使用单 Ascend 910B3。

- Python 3.11.15；oracle/shim 使用独立 package site。
- ms-swift 4.5.2、PEFT 0.17.1、Transformers 4.57.6，两侧一致。
- 原生 torch 2.10.0+cpu wheel 配 torch_npu 2.10.0，实际在 NPU 执行；wheel 标签不作为设备证据。
- CANN 9.0.0、driver 25.5.1；官方镜像 digest：`sha256:8742779115ff73944113f5826836c7aa538757c7401e1a636696d184dba8bc4e`。
- 同完整初始 parameters/buffers、同离线固定输入；已有运行键包含源码、版本、设备、协议、权重、输入、环境及包装器哈希。并行进程使用独立 JITTOR_HOME/可写缓存；串行可独占复用已完成的编译缓存，来源记入 manifest，训练及恢复仍使用新进程。相同键复用保存的结果。

原始文件均在 `$TASK_STATE`，不提交日志、NPZ、模型、容器文件或缓存。精确环境见 `session/host-environment-facts.json`；各运行 `manifest.json`、`source.patch`、`source-untracked/` 界定源码范围，不能仅凭共同 HEAD 把不同运行当作同一源码结果。新增四轨运行按环境合同补齐拓扑、每 rank 固定数据划分、全局 batch、优化器和 checkpoint 等运行键输入；不改写旧 manifest。

## 按统一刻度归属已有证据

| 证据归属 | 结果与实际范围 |
| --- | --- |
| L0/L1/L2 定位用例：`ms_swift_lora_llama` | Swift 构造、完整 NPU 前向/反向及独立 oracle 对拍通过。输出误差 0；8 个必需 LoRA 梯度最大归一化误差 `3.7604574904210894e-7`。21 个主干参数保持冻结；integer input_ids 输入梯度不适用。40 项 NPU inventory、线程数 2 和版本/输入/权重条件通过。 |
| L2 多步训练证据：`ms_swift_lora_llama_adamw3` | 固定输入、3 次真实 AdamW 更新通过；83 项初始参数、逐步 loss/梯度/参数/update-delta 轨迹齐全。梯度和 delta 独立按对应 oracle 幅度归一化，避免大初始参数掩盖错误更新。逐步 backward/update 驻留及冻结集合通过；允许首步 LoRA A 数学零梯度，要求整体确实更新。实际 harness 容差：loss/参数 0.002，梯度/delta 0.01。该证据不包含完整 checkpoint，不能归为 L3。 |
| L5 所需的历史测量材料：约 1.1B FP32 Swift LoRA | 3 次预热与 10 次完整训练步、177 项产物及真实 NPU 驻留比较通过。未完成 L3/L4 前置，尚不能通过 L5；实测数字保留下表。 |
| 所有已执行 case 的候选后端/回退证据 | 实际 ACL 身份、单 NPU 驻留、`fallback_policy=error` 和核心计数 0。它们是已执行阶段的硬门禁证据，不构成独立 L5，也不能推出未运行轨道通过。 |

原生端没有已验证的通用 fallback counter：较新的原始报告使用 `None` 表示未提供，绝不是零计数通过。早期单步 oracle 报告中的 0 是旧报告占位，不能视为原生实测，原始产物保留不改。原生 NPU tensor residency 与已知 CPU-fallback 告警检查是独立证据，不替代通用计数。候选的零计数来自真实核心接口和严格策略。

候选尚无证据证明每个分布式 rank 的 hostname、global/local rank、world size、NPU ID、HCCL、梯度、优化器与 checkpoint 状态符合新协议，也尚无完整训练状态的全新进程恢复与续训结果；原生参考的新增证据见下文。上述旧证据只按其已覆盖内容复用，不能补推这些缺项。

## 新协议公开入口与实际断点（01:20 更新）

以下为同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 上的新增证据。
原生通过只建立 oracle，不意味着候选相应等级通过。两侧锁定 tiny Llama（hidden 64、
2 层、2 heads、vocab 128）完整初始模型和 Swift q/v LoRA；FP32、HF32 关闭，
全局 batch 8、固定顺序、AdamW 三步。CPU 模型执行不在范围。

| 项目 | 实际结果与运行键 |
| --- | --- |
| 原生公开训练与完整恢复 | control `5863b9829ebcd8f8864f4af32e47c185384b48213a6dc132400d68b3c2685182`，fresh restore `d5f276cce0420015129475dd8bdaf5f988f9741f999dca3a01c2b4dc9be3ba9e`；比较 `2c07a6603be0ea89806fb7725d9ed4c0a29c53ddfc0984cf1763c448154f45dc` 全项严格一致。 |
| 原生公开推理 | `f99cfd9668f4ab1e36c76e51fb636976218ff9d161656f33fd4f7b0a33b01234`：实际模型及 adapter 加载，4 个请求/16 个 greedy tokens，记录 logits 与完整 KV cache；候选对拍未执行。 |
| 原生 RNG 恢复 | 汇总 `bda8ca45308e6f72f4629b113fc115fac69e0551d855d5b44acee1d811ff3047`：rand/randn/非零 dropout 的进程内、保存、独立进程恢复共 9 个阶段通过；不要求两运行时 RNG 字节格式相同。 |
| 候选公开训练首断点 | `7efb6e6d4fc373337faa4384cbf11aeb854ca97f37cd01eaa7b2eefb02742c4f`：核心编译完成，Transformers from_pretrained 的真实 meta Tensor 缺失；同时观察到 NPU 被发现为 cuda 的设备接口问题。 |
| 候选 RNG / launcher / 确定性断点 | `8e7848e5471eeadd752fa2165603c841dda6c5b5d4aa92fffc65057119da9fad` 缺 NPU RNG checkpoint API；`917b4148d36a288bb6ef894c592a30e638aae350696cd2883939e577b6d5e547` 缺公开 torch.distributed.run；`e289dddff5e921c6ef5cb1f533a44259d27f1bb30fd0a6b0013b30be6eb67bfd` 确定性 setter 是空操作，CANN 独立查询仍为 0。 |
| 两卡 HCCL 微检 | `13538b5180ab2cc6a9a4924a2f56859ddbaca88eae1a73187f58ad23646b9608`：原生 4 项通信通过；候选真实 HCCL 初始化/all-reduce/broadcast 通过，all-gather 因 compat 未分派至已有 HCCL 算子失败。候选已执行阶段 fallback=0，整体微检失败。 |

原生完整 checkpoint 包含 Swift adapter、验证未变的完整 base、optimizer/scheduler、RNG、
trainer step 与数据位置证据。为让 HF Trainer 正确识别 Swift adapter checkpoint，证据工具
生成标准 safetensors shard index，映射到实际已有 adapter 文件；没有修改下游库。
恢复由公开 `resume_from_checkpoint` 完成，插件只观察并核验初始恢复状态及续训轨迹。
未开启 `full_determinism` 的原生对照出现约 `9.54e-7` 的 loss-only 差异；严格比较如实失败，
没有放宽容差。双方公开训练协议已锁定 `--full_determinism true`，之后原生严格比较通过。

上述断点对应的核心 meta/RNG、ACL 确定性、compat NPU facade/launcher 与 HCCL dispatch
修改正在整合验证，不能据源码存在宣称支持。原始证据分别保存在 `$TASK_STATE/delivery/`
的 `runs/`、`inference-runs/`、`api-probes/`、`hccl-runs/` 和 RNG 汇总目录。
真实多机仍缺第二台 Ascend 主机；没有模拟第二主机或把元数据合同检查写成多机通过。

## 公开入口基础能力与当前断点（2026-09-22 核验）

同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 加各 manifest 记录的 dirty source。
`delivery/foundation-runs/6ed8ba92ad53ecaa5dba198ecbfbfcb6bc85ce60ec3d3e2d545e1711fba767d7`
通过五项真实 ACL/meta 检查（5 executed / 5 passed），并通过真实 NPU 默认 SFRL pool 峰值生命周期、reset-live 及原样 Swift 内存查询。fallback_policy=error、fallback_count=0。
meta 覆盖无存储构造、默认梯度状态、Parameter/clone/detach、assign 与 to_empty；不包含 CPU 模型计算。
峰值由 NativeRuntime 的真实分配/释放事件维护，范围仅直接 raw backend→默认 SFRL pool，排除 workspace、驱动及 HCCL 内存；单可见设备执行，不声明多设备隔离已验证。

已保留的公开 Swift candidate 前置失败：
- `0ad826750685f89973b11e3b85b706641ce42d9fff18cd19d8ca71c2ba1667e0`：Transformers eager 导入仍要求可导入的 `torch_npu.npu_fusion_attention`。compat 仅发布明确 UNIMPLEMENTED 的失败入口，未增加融合注意力能力。
- `67e1933e026844841bf6ccc826fae9ce0d6f4ddb91cb5ada03ce629dd5e738e7`：base 模型加载成功，device_map 为 npu:0；数据预处理 Manager 因 shim 覆盖短 TMPDIR 而报 AF_UNIX path too long。使用已有 JITTOR_TORCH_KEEP_TMPDIR=1 的启动环境选项解决，不修改下游。
- `ffea416466d0f1411b8942cde8999e3847552d488017f7b490932e070cf3d609`：进一步完成数据预处理及模型/adapter 准备；打印模型时 native extra_repr 假定 compat LayerInitializer 有 __code__ 失败。此运行未进入有效训练步，不算 L4。

运行键独立保留；串行复用已完成的 67e193 JITTOR_HOME 仅复用编译产物，训练进程全新，原失败不会升级为通过。内容依赖校验未关闭。

维护回归 `06f9c8c5fb489b5995381d7e692ec6935c6f7f239909e7e2fb959ec1066bdd65` 实际执行 8 项，8 通过、0 skip，覆盖 meta、设备身份、确定性与默认 SFRL 峰值生命周期；严格 ACL 回退策略为 error、计数为 0。修复了仅构造、尚未使用的 SFRL 描述符导致峰值账本永久拒绝的问题。此前 `a13164f914a39b8bfa24aca09e508afb9ec268317b60e4aab9b917bcfd0105c5` 的 7 通过/1 失败记录原样保留，不重标为通过。

compat 默认 `extra_repr` 修复的元数据定向检查 `4feb1222a3f5d30cc4d56cf61a2802c6c0684105bc03d4a3d57d5d240710997b`：3执行/3通过/0 skip，不构造张量或执行模型。

| 新增候选基础能力证据 | 已验证范围 |
| --- | --- |
| RNG 汇总 `0169ea0215bcfc4a8a5e0546b4956af195df9df70d8ceec392b092c851e34978` | rand/randn/非零 dropout 各自的进程内恢复、保存、全新进程恢复，共 9 阶段全部通过；另一个坏 payload 检查通过，拒绝后状态及后续随机数逐位不变。仅单可见 NPU，不推出多设备原子恢复或完整训练 checkpoint 通过。 |
| truth reduce `91c31989aa3adefd8a1b3ba97a4368efd5fa0ae45d498d3853e93d8fadb42546` | 真实 NPU truth-reduction dispatch、keepdims 与 isin 定向检查 1 执行/1 通过。 |
| bool setitem `52a48eabd0fe9239dcfc2310f4b3adef9815f5e6412385e07aef633ab86b58f8` | 真实 ACL 布尔切片赋值及精确设备数据检查 1 执行/1 通过。 |

这些候选检查均保持真实 ACL 驻留、error 策略和 fallback_count=0。RNG 各阶段仍是独立新进程；串行独占借用已完成运行的原编译缓存不改变运行键、产物校验或失败结论的口径。

候选公开训练仍未通过 L3/L4。`eafd149a7d2b1d87c48ee283c132a69d07972808986ece68b9a06f807fe5c0b3` 的 optimizer.param_groups 泄漏内部 native Var、`119b8a297e14660cc11f25ced08d84362afd13a67ebc608107be09671d96938d` 的恢复后 step dtype/device 失败均保留。修复后的真实 NPU optimizer 检查 `0266c5b109b13a5430ab6ae11f22147211862936a344785b93645dda5515d76a` 通过；与原生 `8b135db2331a8087687de74f961f09575a95569ec99d1754d9eb907afa9069c3` 的比较 `4472300f7709b3c7f67f3777fcbdd1f38cd3c8f58f8ec360a395f68f8ead87f8` 通过，30 个浮点值最大绝对差 `1.49e-8`，跨运行时比较按原定容差，双方同进程新 optimizer 恢复后的继续更新各自精确一致。维护回归 `1dedc18ab4b82958aa923d665e6ec24441e6a6abcf47ae697cb8b0afb4cc30aa` 为 1 通过/0 skip；这些不替代全新进程训练恢复。

公开训练 `04ff0f3c96bae2adee8bddd7ede4560fdc131b7d1af6f0fe5e3c6d65fa20ed6c` 随后因 scheduler.state_dict 含函数对象失败。LambdaLR/MultiplicativeLR callable 状态保存恢复、initial_lr 与真实初始化/get_lr 上下文修复的 7 项纯元数据检查，原生 `ece21d60b3ac1600c62287ce05179a792d00f7c5a5990467fa154edefc6afc95` 与候选 `c758d788387e501fad0a600b9fb09cbb255ded9e14b73f800e6ec33ab9064754` 均 7 通过/0 skip。未构造模型或张量；公开训练仍需新运行验收，不能将该失败重标成功。

独立单 NPU 公开推理已通过 L0/L1/L4：原生 `01b7fedc778aea38328103c079416bbd1bcafdfb0d30041dcb7cb499f3ce7179`、候选 `7af2b00b150da0d59f07d0ce2ccf62d8f2b12011f568114c9b2a8072206ea532`，审计键 `862472f6603a4ec4e8df0e970701464d49525f2a3d5967034bb1772a7c6d8d8b`。v3 协议按实际运行时分类持久参数与派生非持久状态；锁定模型及 adapter 加载完整检查、4 个请求的输出结构与 41 项张量快照通过，logits/cache 最大绝对差 `1.1920928955078125e-7`，包含 EOS 的真实 greedy token ID 精确一致。候选真实 ACL、error 策略、fallback_count=0；原生无统一计数器，保持 unknown/null。L2/L3 训练项不适用于该推理轨道，L5 未运行。此前 `77481e34fdd42326b13a82307a672e0da77010693ca29f7e203192b0251b9f64` 的 BOOL 切片赋值失败及旧比较失败保留，不以新协议改写旧运行结论。

真实多机启动合同元数据检查 `d8bcd38e2bc5800ca8aec9a58516fa468aee0b3504af288b27a67d65596efb66`：6项通过，只使用标准库；实际单主机资源检查退出78且未启动worker。真实多机仍资源阻塞。

## 最新增量证据（2026-09-22 核验）

以下各项保留自己的 manifest、dirty source 和失败产物；同步基线仍为
`61294cd14673ba60f4072542a2e73ea2c8f8c509`。运行完成、定向回归通过和完整轨道验收
分别记录，不以局部修复结果推定公开训练或恢复已通过。

| 项目 | 已验证结果及边界 |
| --- | --- |
| 两卡真实 HCCL | `fbb22a3a15f2e2aebe38f8b2da34c4f7b05291a92dbed4e9c0cf1fc369b0722c` 的 `attempt/comparison.json` 通过：两 rank 实际执行 all_reduce、broadcast、all_gather、barrier；候选严格 error 策略、fallback_count=0，原生 counter unknown/null。它只证明单主机两卡通信，不证明分布式 Swift 训练、checkpoint 或公开 launcher 通过。此前 `13538b…` 的 all_gather 失败保留。 |
| 合并维护回归 | `3de8eff27405164f16960255b784fdf19de0665925a344e90bbf1346325406cc`：11 collected / 11 executed / 11 passed / 0 skip，error 策略、fallback_count=0；postflight 确认源码及包装器未变。覆盖 meta、设备身份、确定性、默认 SFRL 峰值、truth reduction、bool 赋值及 AdamW 公开状态/恢复。 |
| 布尔索引及梯度 | `9f47403bd65f5872e15f8549f4388bbfa63ca3c8dcc0820018a8c1737e32c553`：真实 ACL `test_npu_single_bool_index_with_full_slices` 1 执行/1 通过/0 skip，error 策略、fallback_count=0。该定向结果不代替完整训练回归。 |
| 公开训练的当前失败 | `4a789c193c61b32689e362f4a232b1ae01c9eaf3e749d8fa68db444c0df099c2` 已执行两次实际 optimizer 更新，step 2 保存后 `on_save` 观察器拒绝 `parameters/value2` 非 NPU 驻留，整体退出 1。不能把已经生成 checkpoint 文件视为完整保存/恢复通过。 |
| 前两步离线诊断 | `93cc0163388ddd787a57a29f89e429168e9f39d1074fca6cdd80363695b9b4dc` 只核对该失败运行已存在的两步轨迹：29 初始参数、8 可训练参数，键/形状及实际输入等精确结构检查未发现差异；loss、logits、梯度、参数、更新和 optimizer 有逐类误差统计。报告明确 `numeric_pass_fail=not evaluated; no predeclared cross-runtime tolerance`，没有数值通过判定，也不授予 L3/L4。 |
| 序列化驻留修复回归 | `d98d50adf6b11ebe82aa6cc1c758d91b2bfa8062ac8b8791b7525bea20cbf3c1`：真实 NPU `test_npu_serialization_preserves_live_state_dict_residency` 1 执行/1 通过/0 skip。在 error 策略和 forbid_backend_fallbacks 下执行，保存的计数证据为 `fallback_delta=0`；不把增量字段误写成未记录的绝对计数。仍须重新执行公开训练保存、全新进程恢复与续训。 |
| 真实尺寸原生公开推理 | `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26`：约 1.1B FP32 模型、Swift q/v LoRA，经真实公开 CLI 完成 4 请求/16 forwards/16 greedy tokens，各 prompt 521 tokens，退出 0，完成文件及快照哈希齐全、postflight 通过。只建立原生正确性执行证据；候选对拍和独立稳态性能尚未通过。观察器含同步/D2H，运行耗时不是 L5 结果。 |

原始证据分别位于 `$TASK_STATE/delivery/` 的 `hccl-runs/`、`maintenance-runs/`、
`api-probes/`、`runs/`、`partial-training-comparisons/`、`real-size-inference-runs/`。
真实尺寸 fixture `fc8a8deae25e7fcc7e97ae0cb6a390926a95c9c2a3c3f36a4064f2622ddedc30`
复用历史锁定初始参数，原生 NPU 构造并导出；独立计算的非持久 buffer 未注入。
它的 tokenizer 词表为 32000，不能复用 tiny case 的 128 词表解释输入。
这些新增结果不改变 tiny 公开推理已通过、公开训练 L3/L4 未完成、四轨 L5 未通过及
真实多机资源阻塞的总体边界；CPU 模型测试仍未运行，也不声明 CPU 支持。

## 原生单机多卡公开参考（02:10 更新）

2/4/8 NPU 的原生参考均经过真实 `swift.cli.main → torch.distributed.run`，每 rank
记录实际 HCCL、设备绑定、完整模型/梯度/优化器/RNG 和 checkpoint。全局 batch 8，
每 rank microbatch 1，分别累积 4/2/1 次；三步控制运行及 step 2 保存后的全新进程续训
均执行。严格恢复比较包括 raw microbatch loss/logits、全部可训练梯度、参数/update、
优化器/调度器、RNG 及实际下一批输入，未放宽容差。

| 原生规模 | 控制运行 | fresh restore | 严格恢复比较 |
| --- | --- | --- | --- |
| 2 NPU | `05f8013d9a5b0396940e15e8da1cf56b58eb77e5d63adf2fcdc5cc60d0d51058` | `6b799fa1a1357e2886309127f217d19245f4496b2276e08524676c4a0742184b` | `7c5d52f528a321c033515f6fd9d124e1fb86b5c368f78f1288c0cd5f0d703d22`：通过 |
| 4 NPU | `e30f43651fe2511c12d0c9f47f559edf777f9eee8b16e46d56a34885f86b3daa` | `daedc1022a4d40ece9c7f490aa98d8ed4ea19eb39a805e11fc47df6ab4ffa30b` | `0f18b4ff9fa535d9dec9b5e799eb54c653ad3a17deaedbdb56b36ddeb0a74117`：通过 |

八卡控制运行 `f384f5d90bf6815aacb2ebaaf7581c12d46eb6c6d5664cc3fcfb5dfbfae2e42f`、
恢复运行 `9befd9d3dee6ec45af64caa5d38d871a73e9935da2a48d466f7b114c18579f85` 均通过；
严格恢复比较 `86cebe5de67d423cf9edcb4ac2fc42e0be211465f4573909459d6f882bdf291c` 通过。
八卡前置 HCCL 微检 `6978c37ee93ea34ee38e21855d215a0876d851b5bb3daac39cca0b1650962d3d`
在所有 rank 完成四类通信。八卡窗口独占设备，候选计算在 worker 全退出后才恢复。

实际分区另行对照单卡的固定 token 输入：rank r 在第 s 个全局步骤、第 k 个本地
microbatch 消费记录 `8*s + world*k + r`（从 0 计），控制运行共同覆盖 0..23，
续训覆盖 16..23。各 rank 同步后的梯度和参数一致。这是输入分区与同规模恢复证据，
不是跨 world 的数值等价结论，也尚未覆盖同步前本地梯度。

启动时须使用 CANN 9 接受的 `HCCL_DETERMINISTIC=true`，并采用短的作业本地
`TMPDIR` 避免 AF_UNIX socket 路径上限。公开数据集 barrier 先初始化 communicator，
之后 Transformers 构造 Trainer 时将环境变量写为 `1`；起止记录保留实际变化，未修改
下游或截获写入。成功结论不推广到后写值之后新建的其他 process group。此前错误值
与过长临时路径的失败分别保留在 `012da5ed…`、`0cdb7f45…`，不记为通过。

详细证据见 `$TASK_STATE/delivery/native{2,4,8}-public-result.md` 与各 run 的归档。
候选单机多卡对拍仍未完成；真实多机继续受资源阻塞。

## 保留的真实尺寸测量

已有测量完成 1,100,611,584 参数的 FP32 Llama / Swift LoRA 训练，563,200 个可训练参数，batch 1/seq 512、eager attention、HF32 关闭。3 次预热后测量 10 次完整 zero_grad/forward/loss/backward/AdamW/device-sync；没有把 D2H 快照放进计时。177 项 loss/最终参数/梯度产物通过精确 key/shape/dtype 与数值比较，性能产物归一化误差容差为 **0.02**。时间统计已由原始样本重算。

| 指标 | 原生 torch_npu | Jittor ACL |
| --- | ---: | ---: |
| 中位训练步时间 (s) | 0.1469488514 | 0.6504897987 |
| 最小时间 (s) | 0.1463247468 | 0.6423129011 |
| P10 / P90 (s，线性分位) | 0.1467983386 / 0.1471413851 | 0.6438713219 / 0.6545085962 |
| 输入 token/s（512/中位时间） | 3484.206 | 787.099 |
| loss token/s（511/中位时间） | 3477.400 | 785.562 |
| 结束时 allocated/live (B) | 4,475,036,160 | 4,477,397,504 |
| 结束时 reserved (B) | 8,023,703,552 | 11,530,141,696 |
| 预热后峰值 allocated (B) | 7,316,457,472 | 未提供 (None) |
| 预热后峰值 reserved (B) | 8,023,703,552 | 未提供 (None) |

候选中位耗时为原生的 **4.42664 倍**。这组数字只证明所列 case 和测量协议的历史结果，不表示性能领先、达到额外速度阈值或四轨 L5 已验收。候选首个预热步约 **123.7077 秒**，冷启动成本单独保留，不混入稳态中位数。全部 10 个样本及 3 个预热时间保留在原始运行报告中。

原生内存来自 `torch.npu` allocator，峰值在预热后重置；候选来自 `jt.core.device_memory_used/reserved(0)` 同步边界样本，峰值未知，不能据此比较峰值节省。候选真实 ACL 身份、warmup/final 及每步 loss 驻留、error policy 和实际 fallback 计数 0 均通过。被比较的最终运行已在同步前创建并物化返回的 detached loss，避免 lazy 值尚未驻留造成错误证据。

## 已验证修复与维护边界

1. 核心两处 atomic 计数直接流输出在 GCC 10 下产生歧义，改显式 `.load()`；相同 translation-unit 检查及后续完整核心构建通过。
2. accelerator 静态初始化触发 sync_all，访问未构造的全局 fetch 队列。两条队列改由 NativeRuntime 首次访问构造，保留 pending/deferred 生命周期和 cleanup 顺序；结构检查及真实 NPU 回调次序、重复 sync 不重放、第三次 fetch 回归通过。不据此宣称任意回调重入/退出竞态安全。
3. 维护的 shim deploy 补齐缺失 Torch distribution 元数据；未私造兼容包。metadata API 2.11.0 与默认模块 Jittor 版本不同，属于现有发布策略。
4. compat `_ip` 改读取真实 requires_grad，修复 host-to-NPU copy_ 在 no_grad 中解冻参数。8 组合 Parameter/Linear、host/NPU、冻结/可训练及 connected inplace 梯度回归通过。没有泛化到所有 `.data` 或启用梯度时的冻结目标写入。
5. runner 显式将 from_numpy 输入迁至 NPU，保留核心对混合设备的拒绝。真实 Embedding 回归通过。
6. `torch.get_num_threads()` 改委托核心 OpenMP 查询，修复 CPU 数量 192 误报为线程数的问题。真实 OpenMP 2→3→5 及委托检查通过；setter/interop 未因此获得实现。

5 项真实 NPU 回归、17 项证据 metadata 及 2 项线程检查已有通过记录。已有门禁按白名单执行：结构目录包含 CPU 前向/梯度/对拍，不能整个运行并称纯结构。选中的 C++ ownership/OpenMP 检查是元数据/结构验证，不是 CPU 模型测试。

**已有定向门禁：** `checks/36933c7f2428b9a58b5a6a2dbf00fef7c9688b54d728b6de4b80a24e1a1c3ccb`：81 collected / 81 executed / 81 passed / 0 skipped，17.80 s（不含首次编译）；真实 NPU 回归期间 `fallback_policy=error`、核心计数 0。覆盖受影响的 fetch 生命周期、复制/原地梯度、设备迁移、线程查询、多步/性能证据合同、当时的文档及后端结构检查；当时布局检查和 `git diff --check` 通过。这是原基线记录，不代表当前同步提交或本次新验收合同已经重跑门禁。

## 可追溯证据

下列键分别位于 `$TASK_STATE/runs/` 或 `$TASK_STATE/comparisons/`。原始路径、脚本名及产物内历史等级标签不改写；下表按工作负载识别，验收等级以上文统一刻度为准。

| 角色 | 完整运行/比较键 |
| --- | --- |
| 单步 oracle | `169c603c50d5e964bd72ecc42b93add81d6f85219a51baad58025727c6d61050` |
| 单步 candidate | `7fe94c94b5c193995015dd68d4b090047378cf50dbefdd9d39a68ab73bcde52b` |
| 单步 comparison | `7cd7db6a32a6bbe9d64e3bcd004cc00a2c6df85e619a0f6dcbd4e0327dc2829e` |
| 三步训练 oracle | `26cd043896ff6ec1444c5326ebb3ca695a7e65798e06e77916bc4763e45e1c4e` |
| 三步训练 candidate | `6edc0ea84b3dd7a0002adbb6553817ec3ea8794ed3dcc6f88417157e75e5a024` |
| 三步训练 comparison | `9f611125ddda3f8cff0e8bbd73b4d97b727d6d6f710c7312d8a8f2520cb6f85a` |
| 真实尺寸测量 oracle | `b6b279d397337080bb94a6321a68d57ab0f5a1df4d348d0f369725c7f0c154db` |
| 真实尺寸测量 candidate | `8adf387f8708d531c0e2514c2fc216032551e083f327669a355a585bf8028e44` |
| 真实尺寸 measurement comparison（修正容差字段后最终结果） | `b8e312b7b1d7b4dc38342ee7fb689f7658bb2ba71e62c7ca8c361c28e09ecf9d` |

单步 oracle/source 指纹为 `e0ec299638c5c4126473c618dfa4cbc4934cf291632c03b7a0200c88f67a4d54`，候选为 `7d419fee6bce4c4665e601e8eaf73d76430c4dbaaf2d9eb116be45b40d108d78`；保留原生参考与修复后候选各自 manifest，不声称源码相同。三步训练双方 source 指纹均为 `f77b84977411919a22af8dcbebd7264a24df6a858c96aeed91da2ad29c688f1b`。真实尺寸测量双方 source 指纹均为 `e9eb4f23ba9728c7dbb8d32c87fba60f50b73c0e965bd844c29a029fe1acf215`。比较 manifest 还哈希实际比较器和全部输入产物。

## 历史用例重放接口

先检查运行键和产物完整性：没有相关变化就复用，不能为了重述等级重复模型计算。`TASK_STATE` 指向本次未版本化证据目录；`ORACLE_RUN`、`CANDIDATE_RUN` 取上表对应完整运行目录。所有 Python、编译和测试均通过 `srun --jobid=720 --overlap` 执行；登录节点仅做 Git、文本与编排。更改源码前停止同一工作树已有模型/编译。

```bash
# CASE 取本报告三个显式 case 之一；新键确需运行时先 oracle，再加载完整 weights 运行 candidate。
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" torch npu "" "$2"' \
  bash "$TASK_STATE" "$CASE"
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" jittor npu "$2/result.weights.npz" "$3"' \
  bash "$TASK_STATE" "$ORACLE_RUN" "$CASE"

# 历史脚本名不代表当前等级；compare-l3.py 比较的是三步训练，现归 L2。
srun --jobid=720 --overlap "$TASK_STATE/host-oracle-python" \
  "$TASK_STATE/compare-l3.py" "$ORACLE_RUN" "$CANDIDATE_RUN"
srun --jobid=720 --overlap bash "$TASK_STATE/run-final-gate.sh"
```

单步、三步、真实尺寸对应的历史比较器分别为 `compare-l2.py`、`compare-l3.py`、`compare-l4.py`。包装器保存精确命令及 manifest；失败退出码、缺产物、非有限值、缺梯度、dtype/shape 差异及驻留不符都拒绝。相关源码或协议改变时生成新键并保留原证据。这些内部 case 的重放不代替四轨真实公开入口验收。

## 公开训练对拍及恢复断点（04:25 更新）

新 `public-training-parity-v2` 协议在运行前绑定分类容差、比较器及说明文件；
初始序列化参数、optimizer step 和组控制状态精确比较，Adam moments 按预声明容差比较。
候选 `f976d62b228464b1bfb5db14d048875eba67d2b43e64e68140a83fc3e392a332`
与原生 `9a6afcfa43e316f04b2bb1b8f505e123a4a25fe78adf52925d3f67610ab40395`
均完成真实公开 Swift 三步训练及 checkpoint-2/3 保存。离线运行键
`16f3b0b36ad4e1756300462c4866cc205d06e7cb3f74c9585a90236342d5fa5a`
对拍通过：24 个固定 microbatch、全部可训练参数梯度、更新及优化器状态；候选记录
ACL、error 策略、fallback_count=0。输入为整数 token，输入梯度不适用。
这补齐公开训练的 L1/L2 证据；因 L3 尚未通过，不宣布训练 L4 完整通过。

原生 fresh-process 恢复 `a3f7f0bfc7c5396dd6e6ceb83d123ed99f5cc6d3e42fb39956b48c6d3b907c36`
经离线键 `24f86a62135fc2cc0598a3f028d64f672a477b3a5e0bcf089844640263922969`
证明第 3 步与不中断参考精确一致。候选恢复
`474f9c6f6709ab4096738503bbf4db0d13ab2de772a7ca1924b27a382848b2d9`
在恢复 host RNG 控制状态时报 `RNG host engine state mismatch`，整体失败。
独立宿主控制解析复现 `ccfdf21578f79a10171c75b3919c4d10fd6ba71ef7399fed1685edf4b1b5e300`
确认原解析拒绝有效状态、显式消费分隔空白后接受；它不是 CPU 模型验证，也不代替修后 NPU 恢复。
此前 `0169ea…` 九项 RNG 结果覆盖 NPU counter replay，并未覆盖这条 host engine 恢复分支。

单卡 hook 微检查 `8f182ec2b3e2aafa1e2ddb31506af90fbfb5800b4d57efca2533d96564f48b95`
中无 hook 的两步训练通过，但注册 hook 使同一 Parameter 的真实 leaf 标志由 true 变为 false，
观察组失败；后续两卡 hook 阶段未运行。不得将 plain 阶段或 HCCL 通信通过改写为分布式训练通过。

## RNG、真实尺寸数值及 leaf hook 增量（05:00 核验）

以下结论限定各运行 manifest 所绑定的实际源码，不将已通过的单项检查推广为完整恢复
或分布式成功。CPU 模型计算仍不在范围；RNG host 控制状态解析不属于 CPU 模型验证。

| 项目 | 实际证据与结论 |
| --- | --- |
| host RNG 恢复修复 | `25d0c76911f74f826502852e94bf65cb4a80b1054a44eaabed3b00aaacf0f32a` 通过：真实 checkpoint 的 torch.load / torch.random.set_rng_state 控制路径、自身状态往返和三类坏 payload 拒绝原子性均检查；ACL RNG 状态不变，error 策略、fallback_count=0。维护回归 `7fdf95ddd29a44cc56bb238194ebbcd29bf1af480cdb8534e95001eb21bc74f2` 的 `test_cpu_state_restore_does_not_reseed_acl` 为 1 执行/1 通过/0 skip，初始计数 0 且 fallback_delta=0。完整 fresh-process 训练恢复仍须重跑，不能据此声明 L3。 |
| 真实尺寸公开推理执行与失败 | 原生 `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26` 与候选 `d47623e80206f235e482634be57b0937a845f9f855874acdc1a9aa074f5f22b7` 均完成约 1.1B 公开推理、4 请求/16 tokens。但严格比较 `d69c8471117551bf55972e2eee631354f3f23c67d45cda161de5ffaa7ed4b4a3` 退出 1：首个 logits 快照在固定 rtol=1e-4/atol=1e-5 下 65/32000 元素超差，最大绝对差 `2.849102e-5`。执行成功不等于 L1/L4 数值通过，L5 不得启动。 |
| 全量离线数值诊断 | `0cf881e2d2e1dfb73f87b56eb24448e8a3ac3f9a42437c07e72c44175af1f775` 只分析现存快照：1618 个张量中 1220 个未通过原严格阈值；序列化初始状态精确、public responses/tokens 精确，元数据未发现差异。非持久 inv_freq 有 18/32 个数不同、最大差 `5.960464477539063e-8`，自身仍在预定计算容差内；最早已观察到的 KV 差异为 layer0 keys，最大差 `9.083747863769531e-5`、218 元素超差。没有逐层 activation，因此不能把“最早可见 KV 层”写成根因层；未放宽容差。 |
| Rotary 原语定位 | `328cd35fc81e78cdd4a1f08f1e3564bd1c52aff3c2ac9c1ad00e8cada76996df` 的双方真实 NPU 角色均执行完成；离线比较 `7b88d9dcaba71c25d6ce0e89393e55e8c442d9b2482f5261e12dc4b1b87e2161` 对 arange/cast、指数、pow、倒数、position matmul、频率拼接、cos/sin 等全部 13 个阶段逐元素精确一致。候选 count0、原生 counter null。该结果仅排查锁定原语路径；不能据此宣称完整模型数值失败已解决，也不能在没有复现时修改 pow/rotary 算子。 |
| 原生 leaf hook 与优化器提交 | `11f793b313d9be9aa75b3146d24f7fc324182dc768e59343c9d693c40782cc00` 已完成核心构建，注册身份及前两次 backward 的 hook/local-gradient 检查通过，但 SGD step 使 Parameter 成为 `grad_fn=fused_sgd` 的非 leaf，整体失败。共享优化器提交边界修复后，`453a803e6b68df16f1b3d62f7e9f4cf4c6c7440a6bacc2343210462491211475` 的原定维护方法 1 执行/1 通过/0 skip，error 策略，初始 count0 且结束 delta0；覆盖 leaf 身份、顺序/替换/移除、局部与累积梯度、更新后继续 hook，以及错误 shape/dtype/device 替换拒绝。非 leaf hook 沿用旧机制，实际两卡 hook、完整公开训练及恢复未由此通过。 |

优化器矩阵的前置失败也原样保留：`6e9911e55790fbcf2e2d541a0613ebd917d88c13ca32c43c53c6d2f66a5b2296`
因对无 momentum 的无状态 SGD 索引不存在的 state 失败；测试改为允许该明确无状态类别。
`0044d0890b76e35635af134bacea67b9e2152208d3a02d08b55eeda27d60fb56`
在 SGD 构造参数 `fused=False` 处失败。现有 shim 构造器不接受该关键字，属于明确 API 缺口；
覆盖便携提交路径可使用现有 parameter-group `fused=False` 开关，不因此宣称构造器兼容。
修正后的真实 ACL 矩阵 `7793dd80adc83474a50f90a097f4ad55ad5edabb50d4632f22e473bc5ac1f96c`
已通过：1 个维护方法、7 个 factory 各 2 次实际更新、0 skip；error 策略、初始 count0 及
结束 delta0。明确断言每类完整 state 键集合，覆盖 SGD 自动/便携 momentum、Adam、
AdamW 便携/fused、RMSprop、Adan 的 leaf/hook/冻结参数/状态策略及移除后第三次 backward。
该矩阵不掩盖 SGD 构造关键字缺口，不代替真实分布式训练或完整恢复。

新增原始文件分别位于 `$TASK_STATE/delivery/api-probes/`、`real-size-inference-runs/`、
`real-size-drift/runs/`、`rotary-primitive-probe/` 以及 `$TASK_STATE/evidence-audit/`。
总体保持：tiny 推理已通过其适用正确性/公开入口；真实尺寸推理数值失败；公开单卡训练
已存在三步 L1/L2 对拍，但候选完整 L3 恢复未通过，故训练 L4 未完整验收；分布式训练未完成，
真实多机资源阻塞，所有轨道 L5 均未通过。

## 单卡公开训练完整恢复（05:12 核验）

共享优化器提交与 RNG 解析修复后的新控制运行，原生
`37ea72fd6cf1bbb2b9911a1d473f666967845935308f9f4262ffe8252e4b0477`
与候选 `7dee183aee0fb4117c0ec3365036e952c5e5ecb000ce5de7ad3048703ec459d1`
均完成公开 `swift sft` 三步训练。严格跨运行时比较
`393d90698d94c489d15a344e5989bd330d54ca469321e1a38e8ec35caf1de784`
通过，未改变既定容差。

候选以全新进程从 checkpoint-2 恢复的运行
`5c9a9d5b7073b04c6005ab0560669a380f21feeb7ad3473280ebbc2d2cb12e4b`
完成 step3；精确续训比较
`3f33178fb0239e408463b362858926a9a198efa49ff19d2916d2b03e0bfc939e`
通过，比较覆盖 8 个 loss、8 个 logits、16 个梯度快照、96 个优化器数组、120 个参数数组、
62 个状态数组及 29 个参数更新，并核验完整 checkpoint 文件哈希、恢复状态与 RNG。
候选实际 ACL，fallback_policy=error、fallback_count=0，实际处理 8 个续训 microbatch。
原生 fresh-process 恢复 `c81dc2e430d5b4c61222d492088ed90789a015603663089e62db61e100a5ae66`
也由 `981578a35a71f30d868547c2aab970e881e2241b78c479eb47b1a1b231a34ffd`
通过同运行时精确续训比较。

这补齐锁定 tiny、整数 token 输入、普通单卡训练 case 的完整 checkpoint 恢复与公开入口证据。
输入 token/label 为整数，不把其不可微梯度虚构成通过；不推广到浮点输入或其他模型配置。
单卡训练 L5 仍未运行，分布式公共训练及真实尺寸推理的剩余门禁分别验收。

## 维护回归、结构范围与多机准备（05:26 核验）

维护的真实 ACL 组合回归
`abffd41cb89d077885ce7ae2f15ead5845c93e299da66b86488ac8acf27d9219`
为 15 执行、15 通过、45 个 pytest 阶段均通过、0 skip，error 策略且初始/结束 fallback_count=0。
覆盖 meta、实际 NPU 身份、确定性、显存计数、布尔归约/写入/索引、优化器公开状态与恢复、
序列化不改变现场驻留、leaf hook 及七优化器提交路径。原生 full structure 命令不是纯元数据：
`3fa2c39b4b6f9d271d9e75c3a4e70cd66c61d0a97fd1eafadc145d2d017a1220`
布局检查通过，但 structure pytest 在约 15% 时已有七项未分类失败，之后进入 native collect-only
编译；审查发现后续含范围外 CPU 数值用例，故仅中断该 pytest。未取消作业 720，没有将中断
或 skip 记为通过。其原因尚未逐项定位。

审查后的结构子集与布局检查
`ed81a9b938a7fa6f0afa7e37c0362325f94823a0fb819e81e65f873bce47d59d`
通过：38 执行/38 通过/0 skip，另有 1 个显式不选择的独立 C++ 编译项，不计为通过。
子集限定 holder 源码归属、fallback 策略、ACL 结构边界和文档结构；不声明全仓结构通过。

双卡原生公开 v3 证据已加入每 microbatch 的八个 LoRA 本地梯度以及完整输入可微性清单。
控制 `040da30a6c84e9a5f7adfcc5fa0b937dbd14ffc21224ef8c57a9dc67955fff98`
与新进程恢复 `e355fd1965719d7d51e04f864c506210aad3be82f53a59bfb75eb92b489c976c`
由 `c3304c115c84bec9b1490e0cf63b4599947c3c4ec824812e7aebfc71a8a3ebb9`
通过各 rank 精确续训比较；候选公共双卡尚未通过。v3 比较器的 20 项纯 NPZ/元数据正反例
`63b74f983ce11b558ad9276c0704cfcdfbf25b592f4a1ab502d4096e7615c787`
通过，不能替代真实分布式梯度次序验证。

真实多机独立实现已准备在 `$TASK_STATE/delivery/multihost-proposal/after-v2`，重基当前
v3 观察协议，包含真实 Slurm 主机核验、全局 rank 私有缓存、跨主机整池租约、HCCL 微检
及真实公共 launcher、完整 checkpoint 共享路径和逐节点退出核验。元数据检查
`de2c9f4b228cad66a8e7eceeeffbdbe0f9301c6e66eb8d04148ad13833d690a3`
通过，实际 allocation 仍只有一台主机，runtime_execution=false；没有模拟第二主机，
未声明多机 HCCL、训练或恢复通过。解除阻塞需要两个实际 Ascend 主机，每主机至少两张设备，
并实测驱动/CANN/网络和共享文件系统锁一致性。

## v3 完整训练证据与 v4 推理断点（05:50 核验）

以下是各 manifest 所绑定源码/协议的结果，更新前文对应轨道的当前状态；历史失败仍保留。

**单卡公开训练 v3。** 新协议逐 microbatch 记录八个 LoRA 参数 hook 的实际梯度贡献，
并列举全部位置/关键字 tensor 输入的 dtype、shape、requires_grad 和适用性。
原生 `e016f2f04b70328477872ccc53dee1fa4ac7884707a10a93b5850a0bb26b8798`
与候选 `5a590c596d9fb3dba46a573fd54757ddaeb805a079267e3dc7914467bf0a8c05`
由 `7c43d97d69cff0cb25aab81666d1b824a4e5f1f6cb818f70ce23f5f5e5fda8df` 严格对拍通过。
候选 fresh-process 续训 `11ce99d47af1386ac5c4cc9358d92adb6c4795782e9359f2bad292d740234622`
经 `417badbe996de264b994a43dcf47acdf0299f37b64f95699e86a2ad4dca56653` 精确通过；
原生续训 `6025abcb44db8ca606160c16ab8fa37ed3999a98f6125ede65875fda533a70fd`
经 `988b9465c2ab6a861329f58a0c98999f9183d011da9ee6035d80ed03136e8876` 精确通过。
输入是不可微整数/bool，明确 N/A；协议拒绝未覆盖的浮点或 requires_grad 输入，不能声明浮点输入梯度支持。
该结果补强锁定单卡 case 的 L0–L4，未运行 L5。

**原生多卡与候选双卡必须分开。** 原生四卡公开控制
`d147e1bbfccd8a26fff48de2d2a949f66d88bfedeca54e536970b829b269f71a`、恢复
`23a371f2f63ee8fd4c247e2b9cc575fe8dd9b6ebcb3bd2ae0fc436c1317f7d59`
由 `273675978e0423805b86f2df4c9b607cca99c40806b8603b43a915bf01df1d9f` 通过逐 rank 精确续训。
原生 2/4 卡控制及恢复的固定 global-batch 分片审计四项均通过，依次为：
`298dc9f900042898b05bf2be54333545bd1800964679a4d77d4c46e7a597d9d4`、
`34389fd62a1cba88f8bc35542e1b5f96278d4179f783ae8ebdf1a3d2c3171ff6`、
`ca4b33148c9c81c5e04e7e39efc92f4027c6888a7f379f8f19663f9ac81dd694`、
`9fa8efbb844a13ae9eaedc3c309ca3770034b0a84ffda6b44b6ff6faed79b520`。
这些是 oracle 证据，不代表候选多卡验收。

候选双卡 `938ecba806bf8d01c9a7e6de21692617786de9f0d2b4d4147adbc8e9a669aa8b`
完成各角色私有缓存预热及实际 HCCL 双 rank 通信检查，但真实公开 Swift/Accelerate 入口
在导入 `torch.distributed.algorithms.ddp_comm_hooks` 时失败。兼容命名空间修复已应用，
新键 `54683642cdd439e0b5341703117fbbc5900dcd7957dd83c87226663e35354e5e`
已越过导入断点，但在公共 Trainer 的 token 计数 gather 中因裸 `scalar[None]` 被 ACL 拒绝而失败。最小索引入口修复已应用，待新运行验证；尚无候选多卡训练/梯度/恢复通过结论。

**真实尺寸推理仍严格失败。** 显式共同 NPU 初始化 v4 的原生
`146b00bb6140fd8076c3bb16f651d97cf94cc51d40a00eb6b77c76d25c8c1f72`
与候选 `796e39e26587c85945fd4e0e493a59cba3a07173a07606c5b6547455b0cae1b1`
完成执行，但严格比较 `782de967bf7d8a903f9425b8ff8d225612732117f799ae299aa6c1460f4cd284` 失败。
全量离线诊断 `7f63b78b4f92bb96b1c795ea8f2eca0f5b1ba74430384fe9083453f7b37ba827`
记录 1618 个 tensor 中 34 个超出原定 rtol=1e-4/atol=1e-5；初始参数/buffer 精确，
公开响应与 tokens 精确，候选 fallback0，均不能抵消 KV/logits 数值门禁失败。
逐层 activation 诊断 `ccf303a5175cba78e1f4e6cddd1c906c67db6dedaf8022cd957197cefca66059`
确认双方观察前后各自 prefill logits 逐位一致；首个非零差异在 layer0 RMSNorm 输出，
最大绝对差 `9.5367431640625e-7`，该点本身尚无超差；首个已观察严格超差在 layer3 residual 输出。
这是定位证据，不是 Rsqrt 或其他算子根因定论，不放宽验收阈值，不启动 L5 性能。

**默认设备 API。** `fe7a1203de38cbd308b3c60b7b207b384d262f0f13d293b5e0d9eef92a973c72`
真实复现 `set_default_device('npu:0')` 后 getter 错报 `cuda:0`。修复位于 compat 的默认设备
报告路径，复用 ACL 类型解析并遵守实际 factory 上下文，不修改模型或 observer 掩盖错误。
定向真实 ACL 维护回归 `3650bcf86c25136e038a93482eb96394b4168afbb017c5db902e0e02b4a55376`
为 1 执行/1 通过/0 skip、初始 fallback0 且 delta0；CPU/meta 分支只查元数据，不执行 CPU 模型。

截止本节：候选双卡新运行、候选四/八卡完整 public 路径均未完成；真实多机仍缺至少两台主机。
所有轨道 L5 未通过。原生 torch_npu 无通用 fallback 计数器，继续记录 null；
候选 error+0 与真实 NPU 驻留证据仅对各自已经实际验证的运行生效。

## 2026-10-07 增量：Qwen3 SDPA 训练失败与公开推理通过

本节同步到 upstream `2.0-refactor=9e37dc025bc2e3e94f59e04a8fa5004e934e6b83` 后记录；集成基线为 `a53220d37640458b8fb48e51d1877791990dd963`。原始产物未版本化，位于 `$TASK_STATE/runs/qwen3-sdpa-public-r3-20261007/` 与 `$TASK_STATE/runs/qwen3-public-infer-nativeckpt-r1-20261007/`。

| 单 NPU 轨道 | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3 tiny 公开 SFT，FP32、Swift LoRA、bool-mask SDPA、batch 2、3 步 | passed（实际公开构造） | passed（同权重 forward） | **failed**（第二次 AdamW 更新/参数轨迹超出预置阈值） | not-run | not-passed（L2 前置失败） | not-run |
| Qwen3 tiny 公开推理，FP32、Swift LoRA、greedy 4 token | passed | passed（完整加载参数、logits、token ID） | not-applicable（推理） | not-applicable（推理） | passed（`swift.cli.main infer`） | not-run（tiny 且无稳态协议） |

**公开 SFT 失败仍按原结论保留。** 原生 job 1608 与严格候选 job 1609 都真实执行 3 步。候选使用 ACL、`fallback_policy=error`、`fallback_count=0`；两侧实际 bool attention mask SDPA、batch、参数、forward logits、梯度和 AdamW moments 均在 NPU。固定容差文件 `public-parity-tolerances-v1.json` 未改：forward logits 最大差 `2.16e-7`、梯度最大差 `6.61e-8`；优化器更新 step index 1 失败，`lora_A` 对应最大更新差 `4.0270388e-6`，最终参数差 `4.0754676e-6`。诊断表明一个近零梯度误差被 AdamW 归一化放大；未调宽阈值、未改源码。optimizer scalar step 状态按原生语义在 CPU，其余 moment tensor 驻留 NPU；若准入要求每个标量状态也必须在 NPU，此项另外不满足该更严门槛。详细逐项证据见运行 manifest 与 `comparison.json`。

**独立公开推理通过。** 新运行键 `qwen3-public-infer-nativeckpt-r1-20261007`，oracle job 1613 先于严格候选 job 1614；候选之后由 worker 比较 job 1617 对拍。两侧从相同 Qwen3 base 和同一份**原生** SFT checkpoint-3 adapter 加载，使用四条固定请求、FP32、SDPA、greedy 生成 4 token。公开 `swift.cli.main infer` 两侧均成功；全部加载参数、16 组实际前向输入、16 组 logits 和四组生成 token ID 精确一致，37 份 snapshot 完整，logits 最大差为 0。候选 launcher/child 都报告 ACL、严格 error 策略、fallback 计数 0；两侧模型参数、forward 输入与 logits 的审计设备均为 `npu:0`。原生 fallback 通用计数器 unavailable，记录为 unknown。候选首次构建使用隔离 JITTOR_HOME；推理 job 1614 的墙钟包含 JIT 编译，不能作为 L5。比较器 job 1615 因系统 Python 缺 NumPy、job 1616 因读取 Python 字面量时误用 JSON 而失败，均在比较前失败并保留；job 1617 使用 oracle Python 与 `ast.literal_eval` 后通过。这是 Qwen3 tiny 推理路径证据，不覆盖官方大权重、其他模型/tuner、训练 checkpoint 恢复或性能门槛；推理通过不改判同模型族训练 L2 失败。

完成推理对拍时，Slurm 作业 1613–1617 均已退出。个人 fork 远端仍在 `a4fa5b2801bef077b54d9bf669c08c1b3c6b4b37`；已获授权的普通推送再次因未配置 HTTPS username 凭证而失败，没有读取或保存 token，也没有 force push。该认证待办独立于验证结论。


## IA3 两卡公开 SFT optimizer/device 审计（2026-10-08）

基线为 integration HEAD `fbe0b1d1b29d3f35279c71c3591d7388c16a4b19`，upstream `2.0-refactor` 已同步到 `0b8850c3d1c48a284c81f0395bb05647f521a218`；工作树开始时 clean。远端个人 fork 的跟踪 SHA 上次确认为 `a4fa5b2801bef077b54d9bf669c08c1b3c6b4b37`；本轮针对 fork 的 live refspec fetch 挂起后中断，未改写跟踪引用。作业、插件与原始张量证据未版本化，位于 `$TASK_STATE/runs/ia3-world2-device-audit-20261007/`、`ia3-world2-device-audit-r3-20261007/` 和 `ia3-world2-fused-optimizer-audit-20261007/`。

| 运行键/作业 | 结论 |
| --- | --- |
| `ia3-world2-device-audit-20261007`，native fresh job 1626；candidate job 1627 | 原生 torch_npu 每 rank 35 参数、4 输入、2 forward outputs、14 梯度均在对应 NPU；42 个 optimizer tensors 中 28 moments 在 NPU、14 scalar step 在 CPU。候选 1627 在 Jittor 首次编译中于 20:22:28 达 1h walltime，未产生张量证据，保留 timeout。 |
| `ia3-world2-device-audit-r3-20261007`，native resume job 1628；条件 candidate job 1629 | job 1629 的两个 rank 均记录 ACL/HCCL world=2、`fallback_count=0`；参数、实际 batch、forward/compute outputs、梯度和 28 个 moment tensors 在对应 NPU，但 14 个 scalar step tensors 在 CPU。该设备审计没有保存 forward/optimizer state 数值，因此不宣称该运行完成跨运行时数值对拍。job 1628 原生恢复 step 3→4 的 optimizer tensors/rank 均在 NPU；缺候选恢复设备配对。 |
| `ia3-world2-fused-optimizer-audit-20261007`，native job 1630；candidate job 1631 | 公开 `swift.cli.sft`、IA3、FP32、真实两 rank/HCCL、单步。原生 torch_npu 与候选 ACL 的每 rank 35 参数、输入 4、forward outputs 2、compute output 1、14 可训练参数、14 梯度及 42 optimizer state tensors 均位于相应 NPU。候选 process end 与回调记录两 rank `fallback_count=0`；原生通用计数器 unavailable/unknown。 |

fused profile 在预先锁定的 FP32 容差下：两 rank 的 14 个初始 IA3 参数精确相等，4 组 batch 精确相等，22 个冻结参数/buffer 精确相等且训练前后未变；14 梯度最大绝对差 `3.725290298461914e-9`，pre/post/final 可训练参数快照最大绝对差均为 0。两侧每 rank 均为 14/14 非零梯度与 14/14 非零更新。worker 数值报告为 `$TASK_STATE/runs/ia3-world2-fused-optimizer-audit-20261007/numeric-audit-v3.json`（job 1665）。

此结果仍失败于严格序列化门禁：原生 `optimizer.pt` 为有效 ZIP（27141 bytes），候选文件为 11612 bytes 且 `zipfile.is_zipfile` 返回 false；独立 worker `torch.load` 报 `Invalid magic number`。候选 optimizer checkpoint 的 state 数值因此未能读取/对拍。插件同时只记录 forward output 的 device/shape/dtype 元数据，没有保存 logits/hidden 数值，所以 forward/output 数值比较未执行。optimizer checkpoint 与 forward 值缺失都不能从参数/梯度/更新轨迹间接推定通过。比较器 job 1660 因 final 路径拼错失败，1661 因无法读取候选 optimizer 文件失败；1665 将可比快照写入 JSON 后，其 stdout 摘要代码 KeyError 退出，报告中的单项实质失败是候选 checkpoint 归档格式。

因此本证据只补强单机两卡 IA3 的 L0–L2 设备观察与部分数值轨迹，不授予严格 L2、L3 或整体 L4；不覆盖恢复候选、推理、其他 tuner/model、L5 或多机。下一断点是用修正后的观测协议保存 forward 数值和 optimizer state 数值，并先修复/定位 candidate optimizer checkpoint 格式；按验收顺序 native oracle 在前。


## IA3 fused AdamW 双卡完整恢复增量（2026-10-08）

运行键 `ia3-world2-fullstate-fused-l3-r3-20261008` 绑定集成代码基线 `a8dbba3984931a699aa814c31bcec7edf09fcb73`、upstream `7a18abf295668d9b19da5fa1657f5606e84b65a0`。Slurm 原生 torch_npu continuous/resume 在候选之前完成；严格 ACL 候选 Slurm 1713 通过公开 `torch.distributed.run → swift.cli.sft` 完成 world-size=2 四步训练，1714 在全新进程从 checkpoint-3 恢复并完成第 4 步。

比较协议的逐次结果完整保留在 `$TASK_STATE/runs/ia3-world2-fullstate-fused-l3-r3-20261008/` 及两个 comparator 目录。job 1715 的首个 checkpoint comparator 因把 shim continuous-vs-resume optimizer tensor tolerance 错设为 0 而报 28 项差异（最大 `4.66e-10`）；协议 v2/job 1721 将该项改回预先锁定的 optimizer tensor tolerance `1e-7`，checkpoint 224 项零失败，但 trace 比较仍把跨 runtime resume 初始可训练参数错误要求精确相等，产生 8 项最大 `1.1921e-7`。协议 v3/job 1724 只将该项改为已预锁的参数轨迹 tolerance `1e-6`，其它精确与容差规则不变；checkpoint 224 项和 trace/device 298 项均零失败，Slurm exit `0:0`。两次比较器失败结果都未修改或删除。

trace comparator 同时检查连续训练与恢复轨迹中的每 rank batch、前向输出、梯度、更新前后参数、最终参数和 optimizer 数值/标量；checkpoint 比较检查完整键、shape、dtype、有限值、adapter/optimizer/scheduler/trainer 状态及两 rank RNG。候选每 rank runtime 为 ACL/HCCL、`backend_fallback=error` 且 fallback 计数 0；batch、forward、trainable 参数、梯度和 optimizer tensors 的物理设备记录均为对应 NPU。原生通用 fallback counter 不可用，记为 unknown。工作负载满足适用 L0–L4；不授予其他模型、tuner、数据/精度组合或 L5 性能通过。

2026-10-08 的 Slurm structure job 1718 在 NPU worker 完成：`tests/structure` 1389 passed、6 skipped、24 warnings，严格 execution 要求开启；`tools/check_repo_layout.sh` 通过。该门禁完成后报告矩阵按上述结果更新。


## ORPO 双卡公开训练与恢复增量（2026-10-07）

既有 ORPO 失败键和 rank 样本顺序错位结论继续保留。本轮只采用修正 RNG/CPU Generator 语义后的新运行 `orpo-world2-rngfix-20261007`：锁定 tiny LLaMA、Swift LoRA、FP32、两条偏好样本、seed/data_seed=42、HCCL world-size=2、四步公开 `swift.cli.rlhf --rlhf_type orpo`。原生公开轨迹 oracle job 1390 在先；候选 job 1406 完成 continuous steps 1–4 和 checkpoint-1..4，两个 rank 都报告严格成功及 fallback=0。

跨实现比较 job 1407 对每 rank 四步 batch、21 冻结参数、1 buffer、28 梯度/更新、adapter、optimizer、scheduler 和 trainer；`failure_count=0`，batch/frozen/buffer 精确，28/28 可训练张量每步都实际更新，最大梯度差 `2.61e-8`、更新差 `5.60e-8`、参数差 `1.24e-7`。恢复 job 1408 在全新双 rank 进程从 checkpoint-3 继续到 step 4；job 1409 将其与候选 continuous 轨迹比较，`failure_count=0`，数据/冻结状态精确，梯度最大差 `1.49e-8`、更新最大差 `1.11e-8`，adapter 差 `1.11e-8`、optimizer 差 `3.73e-9`，scheduler/trainer 与双 rank RNG 状态精确。

复核还发现 candidate logs 明确报告 Jittor `1.3.11.0`，与当前目标 Jittor 2.0-refactor 不同；故该运行在目标矩阵所有等级都记为 `not-run`。即便只看它的历史数值运行，它也不满足全部 L1–L4 设备门禁：`trajectory.py`/`recovery_trajectory.py` 仅将 batch、冻结参数、buffer、梯度及参数快照搬到 CPU 保存，没有记录原始 `.device`；也没有保存 model forward 输出值。日志中的 model `hf_device_map={'': npu:<rank>}` 与 `fallback=0` 不能代替对每 rank batch、forward、gradient、optimizer state 的物理驻留核验。故该行 L1 首先失败，L2–L4 保持未通过；需要新的带 device/forward observer 的双侧运行键才能关闭。数值比较与原失败历史仍按原样保留。详细原始结果和各作业日志位于 `$TASK_STATE/runs/orpo-world2-rngfix-20261007/`，CPU Generator 先行 oracle/回归及原修复说明见同目录 manifest 与 `cpu-generator-parity-20261007` manifest。


## 目标 2.0-refactor 的 DPO/ORPO 全审计调度与失败记录（2026-10-08）

运行时守卫原先依赖共享文件 `current-job-id`。DPO candidate job 1761 与 ORPO analysis job 1787 并发时，1787 覆盖了 1761 写入值；DPO 完成全部 289 个 `jittor_core` 编译单元和 ACL backend 编译后，shim 子进程按 fail-closed 规则拒绝 job identity。作业在训练前退出，没有候选 batch、forward、gradient、参数更新或 optimizer 状态证据，因此该键不授予目标 L0–L4。失败日志与 manifest 保存在 `$TASK_STATE/runs/dpo-world2-2x-full-audit-20261008/`。

已将外部运行时守卫改为读取 `current-job-id-$SLURM_JOB_ID`，并要求 Slurm job ID 合法、存在该 job 的独立 marker，且 marker 与当前 job 一致；所有子进程仍须处于 Slurm overlap step。DPO 新键复用已成功且先运行的原生 torch_npu oracle job 1783（连续训练与 fresh-process resume），候选使用独立 JITTOR_HOME、运行缓存、launcher 状态和运行 ID，并串行复用已构建的 ccache 对象；Slurm 1797 正运行，1798 依赖候选、1799 依赖恢复完成。原始失败运行键与日志保留。

ORPO identityfix01 job 1790 在启动 `launch.sh` 时因文件缺少可执行权限退出（exit 13），模型没有启动；该键保留为失败。全新 jobguardfix01 键修正权限和 per-job marker：原生 job 1800 在 2026-10-08 20:45:55 至 20:48:18 于 worker 成功完成，候选 job 1801 于 20:48:18 在 `cscg-hw01` 启动。2026-10-08 21:00 CST 复核时，候选仍处于 Jittor 首次编译阶段，rank 日志推进到 87/289；尚未产生候选模型训练、设备审计或回退计数证据。恢复 job 1802 依赖候选成功，比较 job 1803 依赖恢复完成。不能把原生 oracle 成功提升为候选通过。相关包、checksum、Slurm 日志和 manifest 均未版本化，位于 `$TASK_STATE/runs/`。

2026-10-08 21:22 CST 再次检查同一运行键时，job 1801 仍在 `cscg-hw01` 运行。共享 `jittor_core` 已完成 289/289；候选随后进入 Jittor Python 绑定扩展编译，worker 上可见活跃 `cc1plus`，rank 0 的独立编译缓存仍有新对象写入。rank 0/1 launcher 日志仍只有 `ORPO_DEVICE_AUDIT_START`，没有候选模型构造、batch、forward、gradient、optimizer 或 fallback 结果；因此候选验收仍未通过任何目标级别。job 1802/1803、DPO 定向复现 1821 和结构门禁 1823 仍依赖前序作业；不重跑或替换这些运行键。

2026-10-08 21:42 CST 对同一 job/运行键复核：job 1801 仍为 `RUNNING`（已运行 53 分钟，4 小时上限），worker 上 torchrun 父进程和两个 rank 子进程仍存活；rank 0 的隔离 ccache 在 21:41:49 继续生成新 manifest/cache 统计，说明首次扩展编译仍有进展。两 rank 观测日志和 shim 汇总日志仍停留在 launcher 开始标记，没有新增候选模型、batch、前向、梯度、优化器或 fallback 结果。Slurm accounting 当前不可用（`sacct` 报 storage disabled）；队列中 1802、1803、1821、1823 仍按既有依赖等待。故保持原运行继续，不因编译耗时重新提交；本次状态检查不构成任何候选 L0–L4 通过证据。

2026-10-08 21:54 CST，job 1801 已越过首次 JIT/ACL 扩展编译，进入真实公开 `swift.cli.rlhf --rlhf_type orpo` 两 rank 训练初始化：rank 0 配置显示 `Train: 0/4`，日志出现 HCCL all-gather；trace 目录已有 rank0/1 的 trainable initial、frozen initial、buffer initial 和 `batch-step0.npz`。这只证明入口已开始执行且 step-0 batch 已被采集；尚无前向数值、梯度、优化器更新/状态和 fallback 完整窗口。设备事件与最终回退计数由回调在 train-end 才落盘，因此目前仍不能通过 L0–L4。worker 同时输出 `torch.utils.checkpoint` 在 shim 下仅直接执行函数、不会重算激活的警告；该警告已保留，待结果中评估其对该工作负载内存/性能解释的影响。

2026-10-08 21:56 CST，同一运行键的 rank0/1 均新增 `forward-values-step0-call0.npz`（各 4388 bytes），候选的 step-0 模型前向已实际返回并保存数值。两 rank 日志继续有 HCCL all-gather；尚无 step-0 梯度、optimizer state、更新后参数或完整 fallback/device 清单文件。故暂仅记录候选前向已执行，尚不能判定其设备驻留、数值对拍或训练层通过；比较仍依赖后续原生 oracle/候选结果与 job 1803。

2026-10-08 21:59 CST，候选两 rank 均已产生 step-0 gradient、pre/post 参数和 optimizer-values 快照，且 rank0/1 均已记录 step-1 batch；公开训练日志显示保存 `checkpoint-1`。job 1801 仍在运行并继续编译/执行 ACL kernels，未完成四步轨迹。由于 per-rank 原始 device 事件和最终 fallback 汇总要等 callback train-end 写出、原生对照及 job 1803 比较尚未发生，这只是候选单步执行证据，不授予 L2/L3/L4 通过。

2026-10-08 22:00 CST，ORPO candidate job 1801 以 Slurm `COMPLETED 0:0` 结束。两 rank 的 step0–3 batch、forward-values、gradients、pre/post parameters 和 optimizer-values snapshots 均存在，且输出目录有 checkpoint-1..4。rank0/1 的 `events.json` 均记录 Jittor shim/ACL、HCCL world-size=2、strict `fallback_policy=error`、起始及最终 fallback count=0、4 步日志及 final state；`devices.json` 的参数、compute input/output、forward output、gradient、optimizer-state 等事件在 rank0 仅出现 `npu:0`、rank1 仅出现 `npu:1`。这是 candidate 单侧的四步公开 ORPO 与设备驻留证据；native torch_npu oracle job 1800 已先运行并有对应 rank trace，但数值跨实现比较尚待 job 1803，故不据此判 L2/L4 对拍通过。全新进程恢复 job 1802 已按 `afterok:1801` 启动，下一步从 checkpoint-3 继续；L3 仍待恢复轨迹和比较结果。

## DPO Bool-mask 探针的过期基线修正（2026-10-08）

只读检查发现待运行 job 1821 的脚本在 oracle 前要求工作树 HEAD 精确为 `4d1c8c617d1c3f5b44bebb38f16065bc59c61355`，而当前基线为 `79eadd9c5ab225474c3253bb2c306b755646212f`。两 SHA 间 `src/`、`python/`、`backends/` 无差异，只有结果报告有变更，但原 SHA 守卫仍会阻止该脚本启动 oracle。保留 1821 原提交和 r3 运行目录，不覆盖其记录。

最初新增 r4/job 1825 依赖 `afterany:1823`，并将精确 SHA 守卫更新到 `79eadd9`；提交本报告后，守卫又会因新的文档提交而失效。故在其启动前取消 1825，保留其作业历史和 r4 运行目录，再新增唯一运行键 `getitem-boolmask-fused-acl-repro-20261008-r5`，job 1826 同样依赖 `afterany:1823`。r5 固定代码基线 `79eadd9`，在 worker 上要求该基线为当前 checkout 的祖先且 `src/`、`python/`、`backends/` 无差异，同时记录 worker 实际 HEAD；这样允许后续仅文档的提交，不会掩盖源代码变化。r5 复用 r3 probe 与已完成 r2 编译缓存，先执行 torch_npu oracle，再顺序执行 `backend_fallback=error` 严格 ACL candidate，并断言设备驻留与零 fallback。job 1823 完成后才会启动 r5，不与现有 ORPO/恢复/比较 JIT 链并发。探针仅用于定位 `roll → bool mask → leading-row slice → masked select` 的 ACL 算子断点，不代表 DPO 公开训练通过。r3/r4/r5 脚本、manifest 和 SHA256SUMS 未版本化，分别位于 `$TASK_STATE/runs/getitem-boolmask-fused-acl-repro-20261008-r3/`、`r4/`、`r5/`。

## 本轮终态复核（2026-10-08）

ORPO `orpo-world2-2x-full-audit-20261008-jobguardfix01` 已由 native job 1800、candidate job 1801、恢复 job 1802 和 comparator job 1803 完整闭环。公开入口为 `swift.cli.rlhf --rlhf_type orpo`，tiny LLaMA、Swift LoRA、FP32 fused AdamW，world-size 2；连续四步并从 checkpoint-3 在新进程续到 step 4。比较器共 146 项、0 失败。native torch_npu 和 candidate ACL 两边 rank 0/1 的初始/最终参数、batch 对应计算输入、前向输出、梯度和 84 个 optimizer state 张量均分别在 `npu:0`/`npu:1`；candidate HCCL、`backend_fallback=error`，连续训练和恢复阶段 fallback 均为 0。native 通用 fallback counter 不可用，记录 unknown。此锁定配置的 L0–L4 通过，L5 未运行。

历史 DPO 公开候选 job 1797 曾在首步 ACL fused graph 的 `reindex/void` 处失败，不能用窄布尔选择探针替代该图；该根因由后续 int64 Roll 修复处理，旧失败不改判。r5/job 1826 的 native oracle 已运行，但 candidate 与结构门禁 job 1823 都在 Jittor 导入前因未设置 `CCACHE_DIR` 而退出；这是运行脚本环境错误，不是算子验收结果。后续全新 DPO 公开训练与恢复由本报告下方 2026-10-09 专节记录，仅对该锁定配置授予 L0–L4。Slurm 当前仍保留 1798 (`DependencyNeverSatisfied`) 与 1799（等待 1798）两项终态不可达的依赖作业；不把它们描述成仍会执行的验证。

结构门禁 job 1832 在 NPU worker 上运行至结束：1385 passed、8 skipped，另有 `test_process_mode_contract.py` 两项因子进程冷启动超过默认 600 秒而失败，child collector 明确提示提高 `JITTOR_TEST_CHILD_TIMEOUT`。仅复制编译产物到另一 `JITTOR_HOME` 的 job 1858 在测试前因路径相关缓存键变化而重建 `jit_utils` 并以退出码 3 结束，不作为测试结果。随后 job 1860 在原 Jittor/ccache 路径上把 child timeout 设为 1200 秒，仅重跑该失败文件，4 项全部通过（20.90 秒）；因此定向覆盖已通过，但 job 1832 的全量结构门禁仍记为未全绿，不宣称全量门禁通过。原始日志在 `$TASK_STATE/runs/ms-swift-report-structure-timeout-retry-20261008/pytest.log` 和 `$TASK_STATE/runs/ms-swift-process-mode-contract-retry-20261008/pytest.log`。

2026-10-09 的提交前全量结构门禁首次作业 1907 在 NPU worker 上完成 1387 passed、8 skipped，但严格会话审计将缺失的 `pytest-xdist` 依赖计作 2 项 `other` skip 并以退出码 1 结束。作业 1910 的补依赖脚本因在 batch shell 而非 Slurm overlap step 执行 pip，在测试前退出；两次结果均保留且不算门禁通过。新运行键 `ms-swift-dpo-report-structure-xdist-step-20261009` 的 job 1911 在 overlap step 将 `pytest-xdist==3.6.1` 及其依赖安装到独立目录，并在真实 NPU worker 重跑全量 `tests/structure`：1389 passed、6 skipped、0 other skips，24 warnings，用时 569.24 秒；`tools/check_repo_layout.sh` 通过。该全量结构门禁通过，原始作业日志、安装记录和隔离缓存均未版本化，位于 `$TASK_STATE/runs/ms-swift-dpo-report-structure-20261009/`、`ms-swift-dpo-report-structure-xdist-20261009/`、`ms-swift-dpo-report-structure-xdist-step-20261009/`。

本次仅更新结果文档后的门禁也保留：job 2142 在 Jittor 首次重建 `jit_utils` 后按设计退出 3，未进入测试；新运行键 job 2143 在真实 NPU worker 完成 `tests/structure`，1388 passed、6 skipped、1 failed，布局检查通过。唯一失败是 `test_process_mode_contract.py` 子进程收集 `_NATIVE_TARGET` 时，内部连续 3 次遇到 `jit_utils was rebuilt and cannot be reloaded`（exit 3），未触发其进程模式断言；worker log 位于 `$TASK_STATE/runs/report-l3-update-structure-r2-20261009/`。这不改变已通过的 1911 基线结果，也不把当前提交门禁说成全绿。

## DPO 掩码融合图的精确重现（2026-10-09）

运行键 `dpo-mask-select-exact-repro-20261008-r1`（worker job 1862）以两个 chosen/rejected label batch 重现公开路径 `cat → roll → labels != -100 → origin[:num_examples][loss_mask[:num_examples]].mean()`。原生 torch_npu oracle 先在 NPU 完成，labels、mask、origin、selected、loss 在拷贝回主机前均为 `npu:0`；选中值为 `[0,1,2,4,5,6]`，均值为 `3`。随后 candidate 使用 ACL 和 `backend_fallback=error`，在 `getitem_op.py:304` 计算 `slices.sum().item()` 时失败。严格门禁报告融合序列 `array → reindex → broadcast_to → unary.cast → binary.not_equal` 中 `reindex/void` 未注册；fallback 被拒绝，没有把该失败转为 CPU 执行或通过。原始 oracle、候选栈和脚本位于 `$TASK_STATE/runs/dpo-mask-select-exact-repro-20261008-r1/`，候选日志为 `candidate-1862.jsonl`。

该精确路径确认公开失败至少发生在生成 `labels != -100` 的掩码图，而不仅是后续 bool masked-select/getitem。r1 日志没有记录 ReindexOp 的索引表达式，但源码表明本 case 的 `roll` 对每个维度构造带取模规范化和边界条件的索引表达式；这不是 Expand/identity。通用 `ReindexOp` 仍支持任意索引表达式、overflow 条件和额外索引张量，不能通过无条件名称映射到 `Expand` 处理。随后确认 ACL 专用 `Roll` 仅因 int64 dtype 白名单漏项而未被选中，修复与验证见下节。

## int64 Roll 根因修复与精确链路回归（2026-10-09）

代码检查发现 `python/jittor/ops/shape_ops.py::roll` 先通过 `tensor.roll` dispatch 选择后端 kernel；ACL 已有 `RollACL`，但 `backends/acl/kernels/tensor.py::_roll_acl` 的 dtype 集合漏掉 int64。此前 DPO 的 int64 labels 因此走通用 `ReindexOp`，再与 `labels != -100` 融合，触发未注册节点。CANN 9.0 所用 `aclnnRoll` 在当前设备上可执行 int64：公开 CANN API 对应设备和支持类型见 [aclnnRoll 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-math/aclnnRoll.md)，本轮 NPU 实测也已通过。

提交 `ac515b406c80e33528c6cc8f953ff0497f466474` 将 int64 加入 `_roll_acl` 支持集合，并增加 CPU/加速器值回归与严格 ACL NPU 设备/fallback 回归。运行键 `dpo-mask-select-exact-repro-20261009-r2`、job 1865 在源码基线 `ac515b406` 上先运行原生 torch_npu，再运行 `backend_fallback=error` candidate；完整精确链路成功，选中值 `[0,1,2,4,5,6]`、均值 `3` 一致。Candidate 的 labels、mask、origin、selected、loss 均在 NPU，`fallback_count=0`。原始证据目录 `$TASK_STATE/runs/dpo-mask-select-exact-repro-20261009-r2/`。这只验证具体 Roll 与掩码操作；当时 DPO 公开 L4 训练/完整恢复仍待新公开运行键验证。后续完整结果见下节，不追溯改判任何旧运行。

## DPO 公开入口四步训练与完整恢复（2026-10-09）

目标工作负载为 tiny LLaMA、Swift LoRA、FP32、`adamw_torch_fused`、固定 DPO 数据/seed、真实单机双 NPU HCCL；公开入口由 `torch.distributed.run` 启动真实 `swift.cli.rlhf --rlhf_type dpo`。运行键 `dpo-world2-2x-full-audit-20261009-int64roll-fix01` 使用先行 native torch_npu oracle job 1783（复用 `dpo-world2-2x-full-audit-20261008` 的连续四步与 checkpoint-3 新进程恢复证据；数据及 adapter 输入哈希一致），随后 candidate job 1887 在 ACL strict mode 连续训练至 checkpoint-4，job 1888 在新进程从 checkpoint-3 恢复并续至 step 4。worker 记录源代码基线为 `ac515b406c80e33528c6cc8f953ff0497f466474` 的后代 `bd3ba23ece5bbcedf581600a081f044c5fb7bc19`，且 `src/`、`python/`、`backends/` 与修复提交一致。

Comparator-only job 1905 使用新键 `dpo-world2-2x-full-compare-20261009-fix03`，没有重新运行 native/candidate 模型。比较协议累计 170 项检查、0 失败：连续跨运行时 78 项、恢复跨运行时 22 项、两侧各自恢复轨迹 22+22 项、两侧 step-3 恢复初值同运行时精确检查各 6 项、每侧/阶段/rank 设备审计 8 项、跨运行时 checkpoint 4 项及同侧完整恢复 checkpoint 2 项。最大绝对差 `1.1920928955078125e-07`，在原预锁容差内；没有在观察结果后放宽阈值。两 rank 的 runtime context 均为 HCCL/world-size=2，rank 与 NPU 绑定为 0→`npu:0`、1→`npu:1`。候选 policy 为 `backend_fallback=error`，每 rank 连续及恢复阶段计数均为 0；参数、batch/计算输入、forward output、gradient、optimizer-state 与 buffer 物理驻留均通过。native 为原生 torch_npu，通用回退计数器不可用，记录 `unavailable/unknown`。该锁定配置 L0–L4 通过；L5 真实尺寸稳态性能未运行。

旧失败保持原样：job 1797 的首步 `reindex/void` 失败促成 int64 Roll 根因修复；比较 job 1889 在数值比较前因 comparator 对 fallback 整数调用 `.get()` 而退出；job 1902 暴露 observer frozen/buffer 文件名映射错误及将跨 runtime 恢复初值误设为逐位比较；job 1904 的 16 项失败均为恢复阶段 frozen/buffer 路径漏用映射函数。修正比较器后 job 1905 全部 170 项通过。对应原始日志、trace、checkpoint、manifest 和比较报告均在 `$TASK_STATE/runs/dpo-world2-2x-full-audit-20261009-int64roll-fix01/`、`dpo-world2-2x-full-compare-20261009-fix01/`、`...-fix02/`、`...-fix03/`；不把比较器失败误记为模型数值失败，也不删除失败原件。


## LLaMAPro 单 NPU 训练与 checkpoint 恢复断点（2026-10-09）

运行键 `llamapro-world1-full-audit-20261009-jittor2-fused-r1` 绑定基线 `82263a2abb537626c8e2c3289e950942ea132889`、tiny LLaMA、Swift LLaMAPro、FP32、fused AdamW、固定数据与 seed、HCCL world-size 1（单进程单 NPU）。原生 torch_npu job 1917 先完成，随后严格 ACL candidate job 1922 完成公开 `swift.cli.sft` 三步；worker comparator job 1926 对 30 项进行比较，0 失败。两端 loss 轨迹一致；candidate 以 `backend_fallback=error` 运行且 fallback=0，记录中的参数、输入 batch、forward 输出、gradient、optimizer state 均物理驻留 `npu:0`。跨实现最大绝对差为参数 `5.97909e-7`、optimizer state `2.7940e-9`、logits `8.9407e-8`、gradient `1.4901e-8`、optimizer update `5.9605e-7`，均使用预先锁定容差。故该特定公开训练配置的 L0–L2、L4 通过。

L3 独立遵循 oracle-first。native 新进程恢复 job 1928 失败，尚未启动候选恢复。checkpoint-3 将 adapter 存于 `default/adapter_model.safetensors`，但没有根目录模型 shard/index；Transformers 4.57.6 将 SwiftModel wrapper 当成普通模型，并在 `_load_from_checkpoint` 中调用 `load_sharded_checkpoint` 搜索根目录 index 后退出。它是该锁定恢复路径的 native oracle 失败，不能用 adapter 权重加载代替 optimizer/scheduler/trainer/RNG 的完整恢复。原始证据在 `$TASK_STATE/runs/llamapro-world1-full-audit-20261009-jittor2-fused-r1/` 与 `...-resume-r2/`；resume-r1 的无效 checkpoint 路径及 1918/1919、1924/1925 的启动/比较器错误均保留在其独立目录。L5 未运行。


## Swift Adapter 双 NPU 训练与 checkpoint 恢复断点（2026-10-09）

运行键 `adapter-world2-full-audit-20261009-jittor2` 基于 Jittor 2.0-refactor，固定 tiny LLaMA、Swift Adapter `adapter_length=16`/GELU、FP32、fused AdamW、公开 `swift.cli.sft` 三步、单机双 NPU/HCCL。native torch_npu job 1930 先完成；严格 ACL candidate job 1931 随后完成，job 1935 在真实 NPU worker 上完成 2092 项比较、0 失败。预锁容差下最大绝对差为 logits `4.76837158203125e-7`、gradient `5.960464477539063e-8`、parameters `4.470348358154297e-8`、updates `4.470348358154297e-8`、optimizer state `7.450580596923828e-9`。每 rank 参数、真实 batch、前向值、梯度、优化器状态、frozen tensors 均记录在 rank 对应 NPU；HCCL ranks 0/1 分别绑定 `npu:0`/`npu:1`；候选 `backend_fallback=error` 且 fallback=0。此锁定配置 L0–L2/L4 通过，L5 未运行。

完整恢复仍遵循原生先行。job 1936 的 native fresh-process oracle 在两 rank 失败：checkpoint 把 Swift Adapter 权重保存在 `default/adapter_model.safetensors`；运行时对象为 `swift.tuners.base.SwiftModel`，而 Transformers 4.57.6 的 `_is_peft_model` 只识别 `PeftModel`/`PeftMixedModel`，所以 Trainer 将其送入全模型加载分支并在 checkpoint 根目录查找 `pytorch_model.bin.index.json` 或 `model.safetensors.index.json` 后退出。候选恢复未启动；不能以 adapter-only reload 替代完整 optimizer/scheduler/trainer/RNG 恢复。比较器 job 1932（bool 减法）、1933（脚本语法）、1934（把汇总日志计成 step）失败均保留；修正后的 worker job 1935 比较 2092 项、0 失败，执行 stdout 与归档 JSON 哈希已逐项核对，artifact path 偏差在 run manifest 中注明。原始运行与恢复断点位于 `$TASK_STATE/runs/adapter-world2-full-audit-20261009-jittor2/`、`adapter-world2-full-compare-20261009-jittor2-fix02/` 至 `fix04/`、`adapter-world2-resume-audit-20261009-jittor2-r1/`。

## Swift Adapter 双 NPU L3 完整恢复复核（2026-10-09）

针对旧 Adapter L3 native checkpoint 分支错误，另用公开 `swift.cli.sft` 进行 native-first 完整恢复复核。保留的原始三步 checkpoint-3 在新进程恢复后续训至 step 4：native torch_npu job 2082，strict Jittor ACL job 2091；两侧各自另有公开 CLI 四步不中断参考，native job 2135、candidate job 2136。恢复和不中断运行均为 world-size 2、真实 HCCL，每 rank 使用固定分区和同一锁定输入。candidate 全阶段启用 `backend_fallback=error` 并记录起止计数 0。

worker comparator job 2139 使用 compare-only key `adapter-world2-resume-audit-20261009-jittor2-compare-r6`，检查 11,637 项、0 失败。每 rank 的 runtime/provider、rank/NPU/HCCL、checkpoint 完整文件、参数与真实 batch、forward、梯度、更新、optimizer 张量设备及数值均通过；恢复首步与连续第 4 步的 batch 精确相等。adapter、optimizer、scheduler、trainer/global step 和 rank RNG 的恢复状态与对应 runtime 连续运行一致；训练浮点轨迹采用预先锁定的类别容差，未调整阈值。最大绝对差：logits `1.1920928955078125e-7`、梯度 `5.960464477539063e-8`、参数/更新 `4.470348358154297e-8`、optimizer `1.1175870895385742e-8`。

协议失败保留且不重判：compare-r4 的 92 项失败来自它要求每一步重复记录只在 step 0 产生的 initial-parameter 事件，以及把同 runtime 恢复/连续的浮点快照错误按 bitwise exact 比较；compare-r5 的数值检查 11,637 项通过，但结果元数据仍写入已失败的旧连续任务号 2132/2133，因此不作为交付报告。compare-r6 独立运行修正 provenance，引用 2135/2136 并复验通过。原始日志、checkpoint、逐 rank 快照和比较产物均未版本化，位于 `$TASK_STATE/runs/adapter-world2-resume-audit-20261009-jittor2-r2/`、`...-r4/`、`adapter-world2-uninterrupted-audit-20261009-jittor2-r2/` 以及 `adapter-world2-resume-audit-20261009-jittor2-compare-r4/` 至 `...-r6/`。此结论只适用于锁定 tiny LLaMA/Swift Adapter 配置；L5 真实尺寸性能仍未运行。

## Swift Adapter 单 NPU 公开推理（2026-10-09）

锁定 tiny LLaMA、已验证的原生/候选 Adapter checkpoint-3、Swift Adapter length=16/GELU、FP32/eager、4 条固定 prompt、greedy 4 token、公开 `swift.cli.main infer`。原生 oracle job 1940 运行键 `adapter-world1-infer-audit-20261009-jittor2-r4`；strict ACL candidate job 1975 使用 candidate-only 运行键 `...-r6` 并复用同配置原生结果；worker comparator job 1976 检查 2461 项、0 失败。比较精确检查输入和 token ID、响应结构、键/形状/dtype/有限值、所有前向输出和每次前向后的完整参数值与设备；参数 atol=`1e-6`，logits atol=`1e-5`/rtol=`1e-4`，均为预先锁定阈值。最大绝对差分别为参数 `4.470348358154297e-08`、logits `1.1920928955078125e-07`。两侧 16 次前向的输入、logits、tokens 均记录在 `npu:0`；全部 29 个参数在每次前向后都在 `npu:0`。Swift 4.5.2 `AdapterModule.forward` 会在第一次收到激活时将 8 个 Adapter 线性参数从 CPU 延迟迁移到激活设备，因此记录了前向前 CPU 状态和迁移后每次前向设备，不把构造前快照混作计算阶段驻留。候选 runtime identity 为 Jittor shim/ACL，`backend_fallback=error`，完整推理过程起止 counter 均为 0。故此锁定公开推理配置 L0/L1/L4 通过；训练 L2/L3 不适用，tiny case 的 L5 未运行。

## ms-swift 模型 registry 盘点（2026-10-09）

只读盘点运行键 `ms-swift-registry-inventory-20261009-r2` 在 Slurm job 2042 的 `cscg-hw01`、Ascend 910B3 worker 上运行原生环境导入与设备探测；读取的是安装版 ms-swift 4.5.2 的 `swift.model.MODEL_MAPPING`（registry 源文件 SHA-256 `78e73070ab92a352ad1ebf34580d2e090a460c95e4661c8722203e50310bf9a9`），共 222 个 model type，其中 118 个 `is_multimodal=true`，无 T5 model type。注册表含 BERT/ModernBERT 编码器及 reranker，也有 118 个多模态 model type。job 2041 的首版盘点在 JSON 序列化 `ModelKeys` 对象时失败并保留；r2 已完整写出 222 条 registry JSON，worker 脚本最后的可选格式化命令因该环境没有 `python` 可执行名而返回非零。原始 JSON 由控制端 JSON::PP 成功解析并校验条目数。此项仅证明公开安装包暴露哪些 registry 类型及无 T5 注册，不执行模型构造、前向或 Swift CLI，也不为任一 L0–L5 授予通过。原始结果、模块路径、版本、设备与失败栈位于 `$TASK_STATE/runs/ms-swift-registry-inventory-20261009-r1/` 和 `...-r2/`。

原始证据位于 `$TASK_STATE/runs/adapter-world1-infer-audit-20261009-jittor2-r4/` 和 `...-r6/`。run r1 的探针钩错 SwiftModel.forward，r2/r3 的路径与协议缺陷、r5 的 fallback 计数缺口和 comparator 部分结果均保留在各自运行目录，不用于本结论；没有重新运行已完成的 native r4 oracle。

## BERT encoder / Swift Adapter 严格恢复对拍（2026-10-09）

在真实 Ascend 910B3 worker 上复核 Swift Adapter direct API 的 BERT sequence-classification 代表面。配置为 hidden=64、2 层、FP32、固定 batch、AdamW 三步，oracle 与候选共享完整初始参数、输入及 adapter。原生 torch_npu jobs 2144/2145 先完成连续三步和独立进程恢复，job 2147 对 95 个状态键、105,206 个值做 exact internal resume 检查。候选使用真实 ACL 和 `backend_fallback=error`。

最初严格比较 job 2157 首先发现 BERT 非持久整数 buffer `position_ids` dtype 不符：native 为 int64，shim 为 int32。根因是 compat 工厂未覆盖 Jittor `arange` 的 int32 默认值，而 Torch 整数边界 `arange` 默认 int64。修复位于 `compat/torch/installers/factories.py`：仅纯 Python 整数边界且调用方未指定 dtype 时显式传 int64；浮点边界和显式 dtype 保持原逻辑。对应 worker 回归 job 2162 在 Jittor 2.0.0 / ACL worker 上 1 passed、0 skipped；测试在 CPU 与可用加速器执行，冷编译串行完成。

修复后严格 ACL candidate job 2165 的连续训练与新进程恢复均成功，损失与 oracle 相同，候选 fallback 起止均为 0；每步参数/buffer、输入、输出、梯度、optimizer tensor 均记录为 NPU 驻留。job 2166 comparator-only 诊断确认初始参数与 buffer（含 int64 `position_ids`）逐项 exact，输入、logits、loss、梯度和 optimizer 状态符合原协议，且两侧各自连续 step 2 与 fresh resume 逐位 exact。跨运行时却在 AdamW 更新上失败：配置仍为 `eps=1e-8`，step 0 的两个 `linear2` 更新最大差分别为 `6.243179e-6` 和 `4.089903e-6`，超过预锁 `atol=1e-6, rtol=1e-5`；梯度最大差 `1.66e-9`、optimizer moments 最大差 `1.66e-10`，loss 精确相同。差异符合近零梯度在过小 epsilon 下被归一化放大的现象，不以放宽容差处理。故该 direct API 配置可记 L0/L1 通过、L2/L3 失败；公开 CLI L4 与稳态 L5 未运行。

后续运行键 `encoder-bert-full-audit-20261009-jittor2-r7-eps1e-6-native` 的 native-first job 2167 已通过：原生 torch_npu 连续三步与 fresh resume 均在 Ascend910B3 执行；内部 exact 检查 95 键通过，loss `0.7017301917076111`，每步模型、输入、前向、梯度、buffer 和 optimizer tensor 均实测 NPU 驻留。

严格 ACL candidate 与 full comparator job 2168 随后通过，使用同一 `AdamW eps=1e-6`。oracle 与 candidate 各自 continuous step2 / fresh resume 均为 95 键、105,206 个值 exact 一致；跨运行时三步轨迹、初始完整参数与 buffers、以及恢复 step 共有 95 键并全部通过既定 dtype/shape 与数值门禁。各步骤最大绝对差：更新 `1.313e-7`、参数 `2.006e-7`、梯度 `4.427e-10`、optimizer tensors `5.554e-11`、logits `1.886e-8`、loss `5.961e-8`；初始参数和 buffers exact。候选运行全阶段 `backend_fallback=error` 且 fallback start/end 均为 0，所有模型、batch、forward、梯度与 optimizer tensor 均为 NPU 驻留。故锁定的 BERT encoder / Swift Adapter direct API case 的 L0–L3 通过。该运行不是公开 CLI，L4/L5 仍未运行，也不推广至其他模型、tuner 或真实尺寸工作负载。

所有运行脚本、快照和日志均未版本化，位于 `$TASK_STATE/runs/encoder-bert-full-audit-20261009-jittor2-r1/` 至 `...-r8-eps1e-6-candidate/`，比较诊断在 `...-r6-numeric-diagnostic/`。

## arange dtype 与 NPU 设备语义回归（2026-10-10）

针对 BERT `position_ids` 暴露出的 Torch 整数 arange 默认 int64，新增了纯整数边界时传入 int64 的工厂适配，并将测试设备名改为真实的 `npu`。原生 torch_npu oracle job 2181 通过 NPU `.is_cuda == False` 与 float64 arange 检查；job 2185 在 `npu:0` 上通过 9 种 arange 语义（浮点 start/stop/step、NumPy 浮点 step、tensor 边界/step、float64、float16、NumPy int64 stop）。原生侧 fallback 计数不可得。

严格 ACL job 2186 使用 `backend_fallback=error`，float64 之前的 6 个 float32 arange 构造断言通过；float64 执行在 CANN 9.0 `Expand` workspace 查询失败，运行时列出的支持类型不含 DOUBLE。失败之后的 float16 与 NumPy int64 stop 未执行。候选 fallback 计数为 0，但整体失败，不能记为兼容通过。为验证 DOUBLE 临时增加的 ACL dtype 映射与白名单代码已撤回，不能宣称该 ACL 路径支持 float64 arange。

另一个严格 ACL 回归 job 2184 中，NPU `.is_cuda` 断言通过，随后 median 梯度触发未注册 fused `reindex_reduce/add`，严格模式捕获 11 次 fallback 尝试，整组结果为 33 passed、10 failed。该证据说明当前 median backward ACL 缺少该 fused kernel；没有声称 median 梯度兼容。job 2180 首次整组回归还记录了 `.is_cuda` 语义错误与异步失败；job 2182 编译中止，未运行测试。这些失败均保留在原始运行记录中。

以上是窄范围兼容回归，不赋予 ms-swift 功能面 L0–L5 通过。运行目录：`$TASK_STATE/runs/torch-npu-device-semantics-oracle-20261010-r1/`、`torch-arange-int64-acl-regression-20261010-r5/` 至 `...-r7/`、`torch-arange-full-semantics-oracle-20261010-r1/` 与 `torch-arange-full-semantics-acl-20261010-r1/`。
