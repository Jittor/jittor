# ms-swift Jittor Torch 兼容验收合同

本文件是本任务验收状态的唯一口径。它适用于 ms-swift 的完整适配面，而不是单个
`ms_swift_lora_llama` case。飞书 L0-L5 含义固定如下；旧报告或通用 skill 的同名等级不能
直接换算，必须按原始证据重新判定。环境约束与运行键见[环境合同](environment.md)。

## 状态与等级分开记录

每个格子记录两项：公开入口是否实际启动、完成并产出；以及适用 L0–L5 哪些层已按本合同验收。入口完成但功能语义不符属于“已运行、未通过”，不是“未运行”；前置层失败时后续层保持 `blocked`，并写明曾执行的事实。C0–C5 问题分类与 L0–L5 验收等级分别记录，不互相推断。性能诊断数据允许帮助定位，只有全部适用正确性层和公开入口通过才可标 L5。

## 适配面总矩阵

验收前建立“功能面 × 模型/流程 × 设备 × 执行拓扑”矩阵。至少包含入口/config、模型族、
Swift tuner、训练流程、optimizer/scheduler/AMP、checkpoint/resume、推理/generate、
datasets/tokenizers/collator、safetensors/export、transformers/peft/accelerate 实际调用、
分布式和性能。每一格只能是 `passed`、`failed`、`not-run`、`resource-blocked` 或有证据的
`not-applicable`，并附首个断点、运行键和证据位置。

`ms_swift_lora_llama` 是 smoke/首个断点，不是范围终点。只有版本或配置盘点证明入口不存在
时才可记 `not-applicable`；没有资源、依赖或时间只能记阻塞/未运行。组合 case 只能证明
ms-swift 对该依赖的实际调用路径，不能推出 Transformers、PEFT、Datasets 等库独立全量支持。

## 四个执行拓扑

下列四项是拓扑维度，不是完整适配面；每项还要与适配面矩阵交叉记录。

| 轨道 | 必须验证的实际工作负载 |
| --- | --- |
| 单 NPU 训练 | 同权重同数据的前向、梯度、优化器更新、多步轨迹、完整 checkpoint 恢复及公开 Swift 训练入口 |
| 单机多 NPU 训练 | 以 2 NPU 验证实际 HCCL、固定数据划分、全局 batch、梯度同步、每 rank 更新/恢复及公开分布式 launcher；4/8 NPU 仅在用户要求或故障定位需要时另列运行键 |
| 真实多机多 NPU 训练 | 至少两个真实主机且每主机至少两张 NPU，同步与恢复覆盖全部 rank，由真实公开 Swift launcher 启动 |
| 单 NPU 推理 | 真实模型及 adapter 加载、前向结构/logits、确定性 greedy tokens、公开 Swift 推理入口 |

四轨分别记录各层 `passed`、`failed`、`not-run`、`resource-blocked` 或有理由的 `not-applicable`。
单机不同卡数分别记结果。训练轨道不得把 checkpoint 或分布式必需项标为不适用。
推理轨道的 L2 训练/梯度和 L3 训练恢复标为不适用，模型及 adapter 加载仍是 L1/L4 必验项。
推理可以在适用前置层均通过后推进 L4/L5，但不能据此声称训练层通过。当前仅有单主机的
作业不能满足多机轨道；保持资源阻塞，不以单机模拟替代。

## 每层硬门禁

每个适用层、两侧每个 rank 都记录并检查这些条件；它们不是留到 L5 的附加验收：

- **身份与设备**：oracle 确认原生 `torch` 与 `torch_npu`；candidate 确认 Jittor shim
  和实际 ACL/NPU provider。记录模块来源、实际 NPU 绑定，检查参数、计算输入、输出、
  梯度和优化器张量的 NPU 驻留。数据准备及序列化的显式主机副本不充当计算驻留证据。
- **分布式真实性**：所有分布式层记录实际 HCCL backend、process group、world size、
  global/local rank、hostname 与 NPU ID；启动声明不能替代 worker 实际观测。L0 尚未
  初始化 process group 的状态必须写清，并在分布式构造微检中补齐后才能推进。
- **零回退**：candidate 全程 `fallback_policy=error`，捕获完整工作阶段的计数并断言
  `fallback_count == 0`。策略未启用、计数缺失、观察窗口不完整或任一次回退均失败。
  原生 torch_npu 若无已验证的通用 counter，明确记 `unavailable/unknown`，不得填零；
  仍检查其身份、驻留、已知 CPU fallback 告警，观察到实际计算回退即失败。
- **结构与训练语义**：返回结构、参数/buffer/梯度/优化器状态键集合和形状须一致，dtype
  与锁定配置一致；缺键、多键、缺失必需梯度、NaN/Inf、错误广播或 device 错配均失败。
  初始参数与持久 buffer 来自同一锁定文件；整数状态和数据身份仍精确检查。训练中
  跨实现的 logits、loss、梯度、更新量及浮点状态误差须保存并用于定位，但不单凭数值
  超容差判 L1–L3 失败。推理仍按预声明的 logits 容差和 greedy token 精确对拍。
  更改验收协议须生成新运行键，历史失败与原始证据不追溯改判。
- **证据完整性**：当前层尚未产生的张量或状态标明未产生，不伪造通过；一旦产生即执行
  对应检查。skip、空结果、仅 import 成功或只有 rank 0 日志都不能证明完整验收。

## 飞书唯一 L0-L5 刻度

| 层 | 含义 | 必须证据 |
| --- | --- | --- |
| L0 | 导入与构造 | 真实 Swift import、模型/tuner/运行组件构造、锁定配置及每层硬门禁 |
| L1 | 前向与推理 | 同权重同输入前向；训练要求输出结构、shape、dtype、有限值和可反向 loss，跨实现浮点误差仅诊断；推理仍比较 logits 与确定性 greedy tokens |
| L2 | 训练与梯度 | 必需梯度存在且有限、优化器真实更新与状态结构有效；两侧在预先锁定的可学习数据和训练窗口内各自证明 loss 下降 |
| L3 | 完整 checkpoint | 保存和全新进程恢复完整训练状态；步数、数据游标、优化器等不重置，恢复后继续训练且 loss 保持下降趋势 |
| L4 | 公开 ms-swift 端到端 | 对应轨道真实公开 CLI/launcher 跑通；训练满足同配置收敛与状态验收，推理满足输出对拍及全部适用前置证据 |
| L5 | 稳态性能 | 适用正确性与公开入口先通过，再按锁定真实工作负载测量稳态性能 |

以下各节定义该刻度的具体执行要求，不另建替代分级。

### L0-L1：构造、前向和推理

两套隔离解释器使用同一份锁定模型结构、初始权重、全部 `named_parameters()` 与
`named_buffers()`、数据和随机种子。transferable state 加载必须严格校验完整键、形状
及 dtype，不能只传 LoRA 参数而默认 base model 恰好相同。

维护的 `compat/tests/torch/_ecosystem_cases.py::_ms_swift_lora_llama` 可作最小训练
构造/前向断点；它的通过只证明已执行部分。推理独立检查模型与 adapter 的实际加载状态，
比较输出容器/键、logits，以及固定 tokenizer、prompt、最大生成长度、EOS/padding 和
其他生成配置下的确定性 greedy token ID 序列；不能只比较解码文本或 token 数。

### L2：训练、更新和分布式

两侧锁定优化器及超参数、全局 batch、梯度累积、各 rank 固定数据分区、loss reduction
与随机状态。运行前选择非平凡的可学习固定数据、训练步数和 loss 统计口径。两侧各自
末段平均 loss 必须低于初段平均 loss，不能将单步偶然下降或三步烟测当作收敛；已接近
loss 下界时用新运行键选择可学习样本。逐步保存 loss、全部可训练参数及适用输入梯度、
参数更新、更新后参数和优化器状态。缺失必需梯度、非有限值、无实际参数更新、状态
结构错误或候选计算回退仍失败。跨实现梯度、更新、loss 和浮点状态误差保留供根因定位，
不作为训练通过的固定阈值；不能只看最终一个 loss。

`ms_swift_lora_llama_adamw3` 是可复用的三步定位用例，不能替代公开训练入口。
分布式先用小张量在真实 HCCL 上做 broadcast/all-reduce/barrier 微检，检查参与 rank、
结果与完成情况；随后进入真实模型梯度同步、全局样本语义和优化器步骤。每个 rank 保存
同步前后可审计的梯度/参数及状态摘要；两侧各自验证通信完成和应一致的 rank 状态，
跨实现浮点差异只作诊断。单卡、单机和多机证据不能互相代替。

### L3：完整 checkpoint 与继续训练

在固定训练步保存模型（含 base/adapter 的完整可恢复关系）、全部参数与 buffers、优化器
及调度器状态、global step、梯度累积边界、各 rank RNG、数据 sampler/游标和适用的
混合精度 scaler/分片状态。adapter-only export 或只保存权重不算完整 checkpoint。

终止原 worker，在全新进程中按同一拓扑加载并继续预先固定的训练窗口。
检查每个 rank 的完整键、shape、dtype、global step、学习率/优化器步数、下一批数据
和 RNG/采样器状态，证明没有重置状态或错误重放数据；继续训练后 loss 须保持有限并
在固定窗口内呈下降趋势。分别保存 oracle/candidate 不间断与恢复轨迹供诊断，但不要求
两套运行时的浮点权重、梯度或逐步 loss 逐值相等。仅能加载权重或从头重新训练不算恢复。

### L4：公开 Swift 入口

使用锁定版本实际提供的公开 ms-swift CLI/launcher，记录完整命令、解析后的配置和
worker 证据。训练入口覆盖所声明的训练轨道及 checkpoint 保存/恢复；推理入口覆盖
模型+adapter 加载和相同 greedy 协议。launcher 子进程必须继承正确 runtime、NPU 绑定、
离线配置与严格回退策略，不能只检查父进程。

手写循环、内部 tuner case、通信微检或包装成类似 CLI 的脚本都只是定位工具。缺少真实
公开入口运行时 L4 保持未通过；训练按上述收敛、状态、设备和零回退门槛验收，推理继续与原生公开入口做确定性输出对拍。

### L5：正确性之后的稳态性能

仅在该轨道所有适用正确性层与 L4 通过后测量，使用锁定的真实目标尺寸、精度和配置。
先完成编译及预热，再至少测量 10 次；报告样本、统计方法、有效样本/token 数、延迟、
吞吐和原生/candidate 比值。tiny case 的计时不能代表真实工作负载性能。

训练计时包含前向、反向、优化器更新和实际设备同步；惰性执行须物化保留 loss 与更新
后的参数。分布式按所有 rank 完成后的步时间和全局有效样本计吞吐，报告各 rank 差异。
推理固定 prompt/生成长度及缓存配置，分别报告适用的 prefill、decode 和总延迟。

显式 case `large_ms_swift_lora_llama_1b_train` 可用于约 1.1B、FP32、batch 1/sequence
512 的合成训练定位；它本身不证明公开入口、真实数据收敛或其他轨道性能。精确 allocator
峰值与同步边界 live/reserved 分开报告；指标不可观测就记不可用，不填零或计算伪峰值比。

## 证据与复用

每次运行保存两侧 manifest、命令、锁定配置、精确 SHA/dirty diff、运行键与原始证据位置。
每 rank 至少具有 `hostname`、`global_rank`、`local_rank`、`world_size`、NPU ID、runtime/
backend、fallback 观测、梯度、优化器与 checkpoint 状态；该阶段不适用字段写明理由。
大张量放快照并记录摘要，不能用只含 rank 0 的最终报告代替全 rank 证据。

结果按“轨道 × 实际规模 × L0-L5”记录，附首个失败层、阻塞原因和解除条件。资源不足保持
未验证，性能不能覆盖正确性失败。原始产物完整且运行键未变时复用；变更后仅复验受影响
部分，保留旧基线和原始运行键。报告不复制安装流水或废弃方案，不把历史 tiny case 成功
提升为完整 checkpoint、多机或公开入口通过。
