# 一周内 Torch 兼容发布收口计划

- 状态：发布验收规格；本文记录范围与门槛，执行状态以整改看板及 dated results 为准，本计划本身不构成发布通过。
- 用户确认：2026-10-20；目标一周内，最迟 2026-10-27，Linux CPU/CUDA，包含单卡与单机多卡。
- 用户质量目标：日常训练、微调、推理和状态流转适配良好，尽量没有明显速度、显存劣势；特殊与极端场景可明确不支持。
- 审阅基线：`c136d38e7465d09c3c1745b402f14a61919b985c` 加保留工作树；70 个未提交路径未清理、未 stash。已重新 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为 HEAD 祖先，无需 merge。
- 维护者：cgq / Jittor Torch 兼容层维护者。复查触发：候选源码、依赖、默认优化配置或发布范围变化。
- 唯一状态源：[整改看板](../architecture/refactor-board.md)。本文规定范围与门槛，实际执行结果写入 dated results，不在这里另建动态状态表。

## 现状判断

现有证据已覆盖大量公开模型、算子/梯度、优化器及恢复协议，足以作为候选版基础；还不能从 tiny 用例通过外推日常完整工作流成熟，也不能承诺普遍接近 PyTorch 的性能。当前优先事项是把已有实现与本地修复收敛为可复现的候选提交，补齐真实任务、常用组合、默认性能和安装后的验证。无需在这一周完成 HF 全计划的所有排列。

本轮文档检查时，在修正 MKL 环境变量后的同一源码上完整复跑 `tests/structure`，结果 **1395 passed、3 failed、2 skipped**，164.21 秒。三项失败分别为未跟踪 onboarding 文档的 toctree、对应生成 MANIFEST 差异与 dirty `py_converter.h` 的 USER_CHECK 计数；没有做干净 HEAD 归因。相对链接、布局和 diff 格式检查通过。审阅快照、源码文档 hash 与本轮结构日志位于未版本化 `$JITTOR_LAB_ROOT/_state/release-readiness/20261020-r1/`；该检查不替代候选的训练、性能或安装后验收。

现有证据应分别使用：

| 范围 | 已有证据 | 发布时需要补充的证据 |
| --- | --- | --- |
| 文本模型推理 | 固定 Transformers 4.56.2 的 17 个 public checkpoint FP32 CUDA 对拍，含 tokenizer/cache/generation；属于历史基线 | 候选 wheel 上复验代表模型，补常用低精度、变长批处理和 prefill/decode 性能 |
| 通用训练 | CNN/ViT/GPT-2/UNet 紧凑模型 CPU/CUDA 多步轨迹；部分真实规模性能数据 | 候选提交上的真实数据、多 epoch、累积尾窗、评估、保存和新进程续训 |
| HF 组合链 | tiny Trainer/LoRA train/eval/save/reload；FP16/BF16 与 scaler 的定向轨迹 | 真实分类和指令子集上的 Trainer/Accelerate 组合，普通默认入口与 AMP |
| 显式精确恢复 | workers=2 及 workers=4/persistent，两个 runtime 各自在 CPU/CUDA 的 12 vs 5+7 均 800/800 exact | 保留已声明首 epoch 范围；普通多 epoch 恢复另验，不能把拒绝配置当支持 |
| 原生单机 DDP | 72 项矩阵中 69 PASS、2 FAIL、1 UNSUPPORTED；有实际训练、状态恢复、时间线和显存证据 | 审核未提交实现与候选基线一致性，核对当前普通环境的默认传输与恢复 |
| Torch/HF 单机多卡 | 历史 Accelerate 两 rank 与 FSDP2 局部证据 | 当前候选的公开 Torch DDP/Trainer 路径，2/4 rank、累积、尾批、保存恢复；原生 DDP 通过不能替代 |
| 安装与维护门禁 | 有历史 wheel/sdist 门禁；当前 core/structure 仍有失败记录 | 整理具体失败、修复常用路径；候选 core/smoke/structure 及干净环境安装后验证 |

主要来源：[HF 进度](../results/2026-10-13-hf-stack-validation-progress.md)、[Stateful Trainer](../results/2026-10-17-trainer-stateful-resume.md)、[17 模型报告](../results/transformers/2026-09-11-transformers-refactor-17-model-l4-cuda.md)、[通用训练](../results/2026-08-26-common-network-training-trajectories.md)、[原生 DDP 后续](../results/2026-10-14-native-ddp-overlap-followup.md)。这些来源绑定各自提交，不汇总为当前候选的通过率。

## 本次发布承诺与可后置范围

优先承诺 Linux CPU/CUDA、常用 FP32 与 CUDA BF16/FP16 AMP、单卡及单机 NCCL DDP。下游首期钉住已经使用的 Transformers 4.56.2、Accelerate 1.10.1、PEFT 0.17.1；独立 PyTorch 2.6.0+cu124 用作现有 oracle。用户默认路径与数值等配置路径分别记录。不得将 shim 声明的 Torch API 版本解释为该版本全部 API 一致。

常规训练必须覆盖多 epoch、数据集大小不能整除 batch/累积次数、梯度裁剪、scheduler、train/eval 切换和新进程保存恢复。LoRA 的保存、加载及至少一类常用 merge/reload 流程，普通模型/配置/tokenizer 的保存加载，以及 batched generation 都属于日常链路。随机 worker 增强的逐位恢复可以是明确的 opt-in 能力；默认恢复按照独立 PyTorch 的相同配置验证其状态和数据推进契约，不能要求 PyTorch 自身没有保证的 worker 逐位一致，也不能让普通恢复静默丢失参数、optimizer、scheduler 或 scaler。

本周可以后置多机、70B 长训练、极长上下文、复杂高阶/complex128 自动微分、任意第三方扩展、所有库版本组合、DoRA/IA3 全矩阵以及 NPU/ROCm 的新增承诺。FSDP2 先保留实验性，不用来替代本次必须通过的单机 DDP；若发布要承诺 FSDP2 节省显存，则额外要求当前候选与原生 PyTorch 的同模型峰值和恢复验收。非常规并发参数写入可明确拒绝或不支持，但普通单线程模型加载、参数复制和 EMA/参考模型独立性不能归为极端条件。

## 优先消除的日常风险

按 verify-then-fix，用当前候选最小复验定性，不直接照抄旧账本：

1. `load_state_dict` 后两个模型独立性、`no_grad/copy_` 后叶子梯度、普通 SGD 的 `foreach=False/fused=False` 等参数。检查 CUDA 上更新一个模型是否会改动另一个模型；遇到静默错必须修复并回归。
2. 核心门禁中的 Linear 构造、boolean RHS 梯度和 arange dtype 等记录，先复验并判断是否触及日常路径，不能因为报告标为既有就自动豁免。文档可达性、生成 MANIFEST 与错误分类计数也要在候选完成收口。
3. 首次/重复编译和普通 CUDA 训练的崩溃记录，包括历史 CUDA random SIGSEGV 与卷积 liveness underflow。用候选的冷/热进程重复短跑确认；通过一次 smoke 不能关闭根因，仍可重复触发的正常路径崩溃阻止相应范围发布。
4. 数据加载的真实吞吐、worker 异常传播和 epoch 切换。compat worker 为线程、Python/NumPy RNG 隔离可能串行化增强；正确性通过不能替代含真实数据预处理的端到端性能。
5. CUDA RNG 旧文档与新实现不一致。当前源码已有 Philox 状态格式，近期有 mixed RNG 通过证据；应按当前 fixture、源码 hash 与依赖判定支持范围，不能继续笼统宣称 uniform-only，也不能外推所有随机历史。

问题来源见[活跃问题](../../agent/manuals/known-issues.md)及 HF 进度；以上清单是待核对风险，除已有报告明确通过/失败的 fixture 外，本轮未重新复现。

## 最小发布工作流矩阵

复用已有 harness，增加必要真实工作流；先单卡最小配置定位，再到代表尺寸和 2/4 卡，不做全笛卡尔积。

| 工作流 | 功能验收重点 | 性能/显存代表格 |
| --- | --- | --- |
| CNN 分类 | CIFAR-10 固定子集，SGD momentum，BatchNorm，真实 transform，2 epochs，尾批与保存恢复 | 单卡 FP32/AMP；DDP 两卡，四卡作 scaling 复验 |
| ViT/BERT | ViT 常规训练与推理；BERT SST-2/MRPC 固定子集的 Trainer 训练、评估、best checkpoint、保存加载 | 单卡常用 batch/序列；CPU 用缩小工作流验正确性，不以 tiny 计时代表 GPU 性能 |
| Llama/Qwen LoRA | 约 0.5B–1.5B 固定权重和指令子集，AMP、G=2/4、clip、scheduler、跨 epoch、fresh-process checkpoint、adapter reload；至少一格 gradient checkpointing | 单卡 BF16 为主，FP16 含 scaler 正常增长/overflow 定向格 |
| Causal LM generation | 上述模型加载后生成，padding/变长 batch，greedy/常用采样、KV cache | 常用短/中序列，prefill、decode、TTFT、TPOT、batch=1 与吞吐格 |
| Seq2Seq | 小型 T5 的 train/eval/generate/save/reload，验证 collator 和 target masking | 一个常用单卡配置，按该发布承诺决定是否扩展性能格 |
| 公开单机 DDP | Torch/Accelerate/Trainer 实际入口，BERT 或 causal LM 的 2/4 rank；累积/no_sync、非整除尾批、评估、checkpoint 与新进程恢复 | matched PyTorch DDP；每 rank 峰值与最慢 rank 时间；原生与 compat 各自报告 |

真实任务可用固定子集与 100–300 updates 控制预算，多 epoch 使用可完成的子集。确定性短轨迹沿用原冻结容差；有 dropout 的真实任务检查 loss/任务指标与更新统计，使用匹配的至少两个 seed 估计重复性，不要求两个 runtime 的 dropout mask 相同。首次 JIT 串行；并行任务隔离编译缓存，不并行写同一工作树。

## 速度与显存目标

“无明显劣势”应落成每个常用代表格的目标，不能用整体平均数掩盖单个常见模型的明显退化。本轮建议发布目标为：

$$
R_t=\frac{T_{\mathrm{Jittor}}}{T_{\mathrm{PyTorch}}}\le 1.10,
\qquad
R_m=\frac{M_{\mathrm{Jittor,peak}}}{M_{\mathrm{PyTorch,peak}}}\le 1.10.
$$

这是待执行的发布目标，未改变已有数值容差或 nightly 阈值。10%–20% 差距需要给出具体范围、归因与发布取舍；超过 20% 的常用格优先修复，不能宣传已满足“没有明显劣势”。匹配模型和配置下 PyTorch 能运行而 Jittor OOM，或稳定后显存持续增长，也视为相关日常范围未达标。另测至少一个接近容量边界的 batch/序列格，避免很小模型的固定开销遮蔽容量差异。

先在相同 dtype、batch、序列、有效 token、梯度累积、attention/optimizer 算法及 TF32 策略下比较，再补用户默认设置。不能为了赢速度单侧降低精度、缩小 batch、关闭训练功能或只给一侧启用优化。手动配置 FlashAttention、cast、fused optimizer 等得到的结果要标为推荐配置证据，并复验常规安装是否能稳定进入该路径。

按[HF 计划共同协议](2026-10-13-hf-stack-validation-plan.md#33-速度内存与任务指标)，分开记录冷启动/import/JIT、模型加载、第一步、热态训练、评估、保存和恢复。预热至无编译与内存稳定，至少 10 个样本、至少 3 轮 PT/JT 交替测量，报告 median/p95 与离散度；既有工具的 min 字段继续称 min，不能改称 median。训练步必须完成真实更新，计时边界同步；正确性导出/profiler 与性能 run 分开。

显存分别报同设备 allocator live peak、reserved peak，另报进程 NVML 峰值与 CPU RSS；分布式报每 rank 与 max-rank。分阶段观察加载、optimizer state 初始化、训练、生成 KV cache、保存/恢复，并检查稳定后增长。采样峰值不能冒充严格 allocator high-water，Jittor 同步 vs readiness 的差值也不能冒充 Jittor vs PyTorch。

共享 GPU 可以先跑正确性与诊断。性能需要相近且可记录的负载；若交替配对波动已大于拟判断的差距，先保留为诊断，安排有界稳定测量窗口，不能宣称已通过。不要为了性能测量无限等待整卡空闲，也不修改其他用户进程。

现有优先排查线索来自不同旧基线，需当前候选复测：[ViT 约 1.33x、ConvNet 约 1.08x、UNet 约 0.79x](../results/2026-08-26-common-network-training-trajectories.md#real-scale-performance-follow-up)；[默认 GPT-2/Llama 约 1.13x/1.22x，显式优化路径更接近](../results/2026-08-26-transformers-training-normalization.md)；[原生 DDP 普通 AdamW 约 1.319x、融合配置约 1.009–1.036x](../results/2026-10-12-native-ddp-transformer-performance.md#真实训练与资源)。这些不是当前候选的性能验收，不能跨硬件、日期和精度合并统计。

## 七天执行顺序与发布条件

| 时段 | 主要产出 | 退出条件 |
| --- | --- | --- |
| 第 1 天 | 盘点保留的本地修复，形成候选提交；最小复验常用接口与现存门禁失败 | 候选源码/依赖/hash 固定，常用路径没有已知静默错；不通过丢改动或 stash 制造干净状态 |
| 第 2–3 天 | 单卡真实分类、LoRA、Seq2Seq、生成；多 epoch/尾批/AMP/保存恢复 | 公共入口完整执行，与独立 PyTorch 同协议验证，实际 CUDA/fallback=0 |
| 第 3–4 天 | 公开单机 DDP 2/4 rank，原生受影响回归 | 数据覆盖、同步、状态恢复、副本一致性通过；环境传输故障单列，不拿 SHM 通过宣称默认 P2P 通过 |
| 第 4–5 天 | 代表尺寸配对速度/峰值显存测量及瓶颈修复 | 每格有匹配基线和可信统计，常用明显退化得到处理；修复后复验受影响正确性 |
| 第 6 天 | clean wheel/sdist、安装环境、自检、core/smoke/structure 与 30 分钟代表训练/生成稳定性检查 | 无仓库 PYTHONPATH 或个人路径依赖，支持路径的失败闭环，日志完整 |
| 第 7 天 | 冻结 RC，复验最终变化，整理支持表/限制/性能表/复现命令 | 每项发布声明能指向候选源码与证据；未达项缩小宣传或范围，不能凭截止日期标 PASS |

候选安装验证必须在 lab 新环境、仓库外运行，分别验证 core、compat、adapters 发行物，防止源码路径掩盖漏包。检查生成 MANIFEST、资源、CUDA 工具链/运行库前置，核对独立 oracle 不是 shim；版本范围只承诺实际验收的组合。普通用户不应依赖维护者的个人环境脚本或手动 preload 才能完成公开承诺路径。

发布阻断项：支持范围内静默错误、可复现正常流程崩溃、普通模型/optimizer/scaler 恢复错误、公开单机 DDP 未通过、安装包漏资源，以及仍声称“无明显劣势”却没有配对性能/显存证据。特殊场景可以明确报不支持；普通多 epoch、变长/尾批、AMP、保存加载不可整体移出范围来凑通过。

本计划制订阶段只完成证据审阅、范围确认和计划落盘，没有重跑上述新增验收、启动正式训练或进行发布上传；后续执行结果见看板及 dated results。具体实验按现有资源规则有界执行，发布上传仍在工作成果可审阅后处理。
