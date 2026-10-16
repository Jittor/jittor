# Hugging Face 组合工作流兼容性与性能实验计划

状态：增量执行中；截至 2026-10-17 的结果见 [HF stack 验证进展](../results/2026-10-13-hf-stack-validation-progress.md)。更新日期：2026-10-17（Asia/Shanghai）。维护者：cgq / Jittor Torch 兼容层维护者。同步目标为 upstream `origin/cgq_transformers`；本轮已 fetch `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为验证基线 `48de31b4157fa128c5e1e19b052fefa3be7ed098` 的祖先，无需 merge。验证输入为该基线加保留工作树，不能把整个 dirty tree 视为单一发布提交；显式 stateful 子项的源码和产物 hash 见[验收报告](../results/2026-10-17-trainer-stateful-resume.md)。先前 CPU/Trainer 基线为 `fe4e738338de5e275471da125816968af7d945fc`，相关修复和结果文档提交 `2a5b0123af8e33defac2e478cb0a0993de8dfa1d` 属于当前历史。其他未提交 native DDP/优化器/文档工作保持独立。复查触发条件：下游库版本、Torch frontend、autograd、RNG、device placement、序列化、AMP 或分布式实现变化。

目标是证明 Transformers、PEFT、Tokenizers、Safetensors、Accelerate 和 FSDP2 在真实组合工作流中可用，并提供与独立 PyTorch 可比较的结果、时间和内存证据。Tokenizer 指 Hugging Face `tokenizers` 及 Transformers 的 tokenizer/processor 交接。FSDP2 是 Torch 分布式 API 的兼容面，不是第七个需要独立版本号的下游包。

本文是实验设计入口，实际整改状态仍只维护在[整改看板](../architecture/refactor-board.md)。实验配置、脚本和结果 JSON 留在 lab；已复现问题按[下游适配流程](../../agent/skills/downstream-library-adaptation/SKILL.md)分流，验收结论写 `refactor-wip/results/`，不把本文变成第二个问题看板。

本轮已执行范围仍以 tiny/定向证据为主，不等同于完整组合验收：已有 `ENV-01/02`、`TOK-02`、`SER-01/SER-05`、`PEF-01`、`ACC-05`、`INF-01`、`TRN-01/02` 和 `ACC-04/E2E-01` 子项外，`TRN-03/04` 新增 tiny Llama LoRA 单卡 BF16/FP16 CUDA 轨迹及 scaler fresh-process 恢复；`INF-05` 新增混合 q/k/v dtype 的 SDPA autocast 定向回归。真实数据链、Trainer 恢复的扩展矩阵、DDP/FSDP2 组合、稳态性能和大分片仍未完成，详见结果报告边界表。

## 1. 推荐先跑什么

先完成 `ENV-01～04`，然后按下列批次执行。每个复杂组合先跑 tiny 配置，再跑公开 checkpoint，最后才扩展尺度或测性能；前置正确性失败时暂停该条性能比较。

| 批次 | 主要实验 | 要回答的问题 | 停止扩大的条件 |
| --- | --- | --- | --- |
| A：数据与权重闭环 | `TOK-01～05`、`SER-01～05`、`INF-01～04` | 两侧是否真的吃到相同 token、mask、权重和 cache | 输入 hash、device、forward、加载键或 alias 不符 |
| B：单卡完整训练 | `TRN-01～06`、`PEF-01～04`、`ACC-01～06`、`E2E-01～04` | 是否更新了全部预期参数，累积、AMP、保存与恢复是否真的生效 | 梯度缺失、冻结参数变化、更新次数错误、非有限值或恢复不一致 |
| C：真实多卡组合 | `DST-01～04`、`FSD-01～07`、`E2E-05～06` | DDP/FSDP2 是否兼容数据尾批、LoRA、共享权重及续训 | 任一 rank 错误、collective 顺序不符、错误 replica/shard 或退出挂住 |
| D：范围扩展与性能 | `INF-05～07`、`PEF-05～06`、`SER-06`、`TOK-06`、`PRF-01～05` | 长上下文、非文本、其他 PEFT 方法及真实尺寸是否可比 | 内存预算不足、数值越界、性能样本受并发污染 |
| E：独立能力声明 | `ACC-07`、`FSD-08`、`E2E-07～08`、`PORT-01～03` | offload、SHARDED、多机、其他后端和版本声明是否成立 | 无合法 oracle/硬件或上游本来拒绝该组合；明确记录边界 |

最先值得投入的八条组合是 `E2E-01～06`、`ACC-05` 和 `FSD-02`。其中 `E2E-02` 能贯穿六个库的大部分单卡接口，`E2E-06` 进一步覆盖最容易暴露交互缺陷的 LoRA + FSDP2 + AMP + checkpoint。不要一开始就跑 70B 长训练：0.5B 单卡、tiny 双卡和 1～3B 双卡已经能定位多数语义问题。

## 2. 基线、已有证据和不能外推的边界

本轮直接读取两套 venv 的包元数据，得到如下共同基线。原生环境另做了 `import torch` 身份断言，确认来自其 venv 的真实 PyTorch，`torch.version.cuda == '12.4'` 且 CUDA 可用。模型 runner 已进一步在子进程中检查实际导入文件、frontend 归属、device 和 fallback；下表仍是环境入口基线，具体执行证据见结果报告。

| 项目 | Jittor 侧 | PyTorch oracle 侧 | 约束 |
| --- | --- | --- | --- |
| Python | 3.11.3 | 3.11.3 | 记录 ABI 与实际 `sys.executable` |
| Jittor / jittor-torch | 1.3.11.0 / 1.3.11.0 | 不安装 | 记录源码 SHA；distribution 版本不能代表当前源码 |
| Torch distribution / API | shim 元数据 2.11.0 | 2.6.0+cu124 | 两者不是同一 Torch 版本；新 API 另建版本矩阵 |
| Transformers | 4.56.2 | 4.56.2 | 两侧实现和模型 revision 相同 |
| PEFT | 0.17.1 | 0.17.1 | 固定适用的 task type / base architecture |
| Tokenizers | 0.22.1 | 0.22.1 | 固定 tokenizer JSON、special tokens 与 chat template |
| Safetensors | 0.6.2 | 0.6.2 | 比张量语义，文件整体字节不要求一致 |
| Accelerate | 1.10.1 | 1.10.1 | 固定 launch / plugin / scheduler 配置 |
| NumPy / Datasets | 1.26.4 / 3.6.0 | 1.26.4 / 3.6.0 | Datasets 仅作为数据来源；独立保存输入 manifest |

环境入口分别是 `$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh` 和 `env_pytorch.sh`。本轮实际 lab 是用户指定的仓库同级 `jittor-lab/`；文中使用变量，避免绑定个人主目录、设备编号或机器地址。可见硬件为 NVIDIA A800 80GB PCIe；这只用于估算配置梯度，不表示这些卡已空闲或本轮获得多卡性能窗口。

已有证据只作为选择实验的依据，不在当前 SHA 上自动登记 PASS：

| 来源 | 已有范围 | 本计划补齐什么 |
| --- | --- | --- |
| [17 个文本模型报告](../results/transformers/2026-09-11-transformers-refactor-17-model-l4-cuda.md) | 固定 public checkpoint、FP32、CUDA 的 forward/cache/generation；部分 round-trip | AMP/纯低精度、长序列、多步训练、Trainer/PEFT/分布式组合与稳态性能 |
| lab `transformers_compat/VALIDATION_HISTORY.md` 的 2026-09-12/13 章节 | 视觉、音频、图文/视频公共 processor；部分 VLM 全梯度及 round-trip | 多模态真实训练链、AMP/LoRA、数据加载和性能；不同模型按实际范围验收 |
| [PEFT Qwen 报告](../results/2026-09-14-peft-qwen-compatibility.md) | Qwen2 适用的 25 类 PeftType 及 rsLoRA/DoRA、真实 0.5B FP32 LoRA、adapter 互通 | 其他 base/task、真实低精度、多卡、长上下文及完整组合续训 |
| [Accelerate 能力报告](../results/2026-09-17-accelerate-maturity-repair.md) | 部分 AMP、累积、尾批和 local 两 rank FULL 保存/恢复 | Trainer/PEFT/FSDP2 组合、真实数据、容量语义与性能 |
| [FSDP2 生命周期报告](../results/transformers/2026-09-07-fsdp2-memory-lifetime-fix-cuda.md)与[内存边界](../results/2026-09-08-fsdp-mesh-optimizer-lifetime.md) | 元数据生命周期修复、有限尺寸数值与内存证据 | 逐 rank shard 语义、长期稳定性、真实峰值、FULL/SHARDED 区分和规模收益 |
| lab 的 tokenizers / safetensors 记录 | tokenizer 核心及 Pegasus 交接；safetensors 基础/adapter 读取 | 多 worker/大词表/长文本，完整模型 round-trip、大分片和错误契约 |

需要优先复验的风险有：SHARDED DCP、uneven-input Join、真正无 storage 的 meta 与容量释放式 offload、非连续 tensor 的 safetensors 拒绝，以及 Jittor optimizer pickle 与原生 PyTorch 的互通。模型/adapter safetensors 能互读，不等于 optimizer `.pt` 能互读；FULL tensor state 变换能工作，不等于实现了 DCP reader/writer。

CUDA RNG 特别容易被旧文档误导。[活跃问题总账](../../agent/manuals/known-issues.md)和 2026-09-17 报告记载 mixed normal/float64 历史不能完整 capture；lab `Accelerate/LOCAL_WORK_LOG.md` 的 2026-09-21 章节记载后续 Philox 修复。但其实现提交 `a0cebbf9b09854cb961c6815554e8ec6a2a11568` 不是本轮源码基线的祖先。因此 `ACC-05` / `ENV-04` 必须在实际 checkout 上重新确认，不能用其他分支日志覆盖当前结果，也不要为了跑计划自动合入该提交。

旧报告对 L0～L5 的编号与下游 skill 的等级含义并不完全相同。本计划统一按“功能、数值、训练、恢复/互通、设备、性能”六轴验收，不用一个等级数字代替覆盖明细。

## 3. 两侧可比的共同实验协议

### 3.1 输入、权重、设备与随机性

所有数值 case 使用同一个不感知 runtime 的模型/工作流定义，由 oracle 先导出初始参数、buffers 和输入，Jittor 再加载。随机种子相同不能代替相同初始权重；训练还要固定 optimizer groups、参数顺序、学习率、epsilon、weight decay、betas、scheduler、clip、loss reduction 和数据迭代顺序。AdamW 缺省值和 `fused/foreach` 选择必须显式记录。

CPU 与 CUDA 各用独立进程。CPU 必须在 import 后显式设置 `use_cuda=0`；CUDA 必须验证实际输入、输出、全部可训练参数和梯度驻留，必要时用 driver pointer 查询佐证。每个 rank 绑定逻辑 local rank；物理映射由外部 `CUDA_VISIBLE_DEVICES` 指定，不把机器设备编号写进配置。Jittor 侧启用 `backend_fallback='error'`，每个 case 的 fallback 增量必须为 0；正常 Rust tokenizer、文件 I/O 和 DataLoader 的 CPU 工作不计作模型计算 fallback。

先做确定性协议：dropout=0、固定 batch、关闭 TF32、明确 eager/math attention，不开 compiler。然后独立打开 dropout、AMP、fused attention 等功能。开启随机功能时，跨 runtime 比分布/行为和数值包络；**精确恢复比的是各 runtime 自己的连续运行与 fresh-process resume**，不要求 PyTorch 与 Jittor 的 RNG 序列逐位相同。需要逐步精确前向对拍时，可把 dropout mask/采样 uniform 作为外部 fixture 注入，另保留未经注入的公共 API 功能试验。

### 3.2 数值与训练验收

正向训练/推理验收的输出、loss、梯度、更新后参数和 optimizer state 都必须 finite；先比较 key/shape/dtype/expected gradient 集合，再比较数值。故意注入Inf/NaN、全忽略labels或上游定义特殊全mask行为的负向case，预先登记预期异常/非有限结果，单列错误契约，不计作正常训练PASS。冻结参数应不更新，unused 分支和合法零梯度由 oracle 明确登记，不能把“所有梯度非零”当通用条件；LoRA B=0 初始化时首步 A 梯度为 0 是合法行为。输入梯度只对可微输入检查，token IDs 不做梯度检查；涉及 `inputs_embeds`、pixel/audio 张量时检查对应梯度。

对每类张量同时报告最大绝对误差、归一化误差和近零单独结果，防止一个大幅值张量掩盖小张量错误。既有 ecosystem gate 的全场 floor 口径照原样执行，不悄悄改容差；新增组合用下面的逐元素约束，并在 oracle 重复性测量后、运行 Jittor 前冻结阈值。

$$
e_\infty=\max_i|x_i-y_i|,\qquad
e_{\mathrm{norm}}=\frac{\|x-y\|_2}{\max(\|y\|_2,\epsilon)},\qquad
|x_i-y_i|\le a_d+r_d|y_i|.
$$

以下是新增短轨迹 case 的**候选起点**，不是已经验证的统一保证；矩阵同时保存实际最差张量、绝对误差及阈值。低精度梯度有真实 AMP opmath 差异，不能只看到“都 finite”就通过。

| 协议 | forward/loss 候选 `atol / rtol` | grad 候选 `atol / rtol` | 使用限制 |
| --- | --- | --- | --- |
| FP32 CPU/CUDA，严格数学 | `1e-5 / 1e-4` | `1e-5 / 1e-3` | 先在 tiny fixture 定位；特殊归约另建预先声明的阈值 |
| FP16 AMP | `1e-3 / 1e-2` | `2e-3 / 2e-2` | 比 unscale 后梯度；必须检查 overflow 和 master/state dtype |
| BF16 AMP | `5e-3 / 3e-2` | `1e-2 / 5e-2` | AMP 与纯 BF16 权重是不同协议；不能混报 |
| 同 runtime 保存/重载 | 权重和整数状态 exact；算术沿用对应 dtype 阈值 | 无随机确定性恢复优先 exact | RNG resume 自己对自己；跨后端/版本不自动要求 exact |

训练比较除参数本身外还要比较更新量 $\Delta\theta=\theta_{t+1}-\theta_t$，报告绝对误差及每层更新范数；不能让相近的初始大权重掩盖错误更新。10-step 确定性 case 要逐步对 loss、grad、参数和 optimizer state；100～300-step 真数据 case 比验证 loss、任务指标和更新统计，多 seed 给出方差。真实 AdamW 的近零梯度会放大端到端轨迹差异，同梯度 replay 可用于定位优化器公式，但不能代替真实训练验收。

变长 causal LM 同时保存有效 label token 数。参考训练把所有有效 label 的 loss 总和除以全局有效 token 数；如果 Trainer/Accelerate 的原生默认 reduction 是 microbatch mean，则两侧公共工作流必须使用同一默认语义，再单独与 global-token-normalized 手写参考比较。不得把梯度累积或尾批造成的 loss 权重变化误判为 kernel 差异，也不得为了对拍偷偷修改其中一侧的应用语义。

### 3.3 速度、内存与任务指标

分开记录下载/准备、import/首次 JIT、加载、第一步、预热后稳态、评估、保存和恢复；稳态比较排除下载与首次编译，同时保留冷启动用户成本。每个性能格先运行约 5 个预热 step，直到编译/内存稳定；预热上限 20，仍不稳定则标记未达到稳态。至少 10 个计时样本，至少 3 轮 PT/JT 交替的同卡比较，报告样本数、min、median、p95 和离散度。已有工具的 min-over-repeats 字段保持原名，不当作 median。

计时区间前后同步，Jittor 必须实际执行每次计算：训练检查每步参数/optimizer step，独立 forward 吞吐实验保留各次输出并用有界窗口同步，防止惰性图只执行最后一次。同步延迟模式与 queued 吞吐模式分别报告；不能在一个 runtime 每步 `.item()`，另一个整段才同步。数值导出、profiler、逐梯度 host readback 与高频外部采样都在独立诊断 run，性能 run 保留等价轻量校验。

$$
R_t=\frac{T_{\mathrm{Jittor}}}{T_{\mathrm{PyTorch}}},\qquad
\text{有效训练吞吐}=\frac{\sum\text{非 }-100\text{ 的 label tokens}}{\text{训练耗时}},\qquad
E_N=\frac{Q_N}{NQ_1}.
$$

`R_t<1` 表示该协议下 Jittor 更快。推理报告 prefill tokens/s、首 token 延迟 TTFT、decode tokens/s、每 token 延迟 TPOT 与 request p50/p95；模型加载与 tokenizer/processor 时间另列。TTFT 定义为输入开始处理至首个输出 token 完成，另给只含模型的 TTFT；吞吐同时列 input/generated/valid-label token 计数，不能混用。Seq2Seq 给 target tokens/s，视觉给 images/s，音频给 audio seconds/s 或 real-time factor，任务指标选择 accuracy/F1、ROUGE、WER/CER、perplexity 等并固定评估实现。

GPU 内存两侧各报每设备 live allocated、pool reserved 及对应 whole-run peak，另报第一步和稳态 step 的峰值。PyTorch 使用其 allocator 统计；Jittor 使用当前源码的 per-device runtime 计数器和真实 high-water API，缺少对应 API 则标记该 peak 未测，并使用独立进程分阶段测量。不能用 host+device 的全局 peak、定时线程采样或 `total_cuda_used` 冒充同口径 live peak；reserved 只与 reserved 比。分布式列每 rank 与 max-rank，不把所有卡之和写成单卡峰值；补充 RSS、CPU/disk offload 字节、checkpoint 大小及 I/O 时长。

性能验收先报告差距和归因，PR 不断言墙钟。建议把 `R_t<=1.10` 作为后续协商的“接近原生”目标，并将 `1.10～1.30`、`>1.30` 分级排查；它是规划目标，不覆盖已有 nightly 配置阈值。语义对等档固定 attention/math、optimizer、TF32/AMP；另开用户默认档允许各自默认实现，但必须写实际路径。原生 `torch.compile`、FA2、bitsandbytes/QLoRA、DeepSpeed 等是独立扩展协议，不能单侧启用后称等配置性能比较。

## 4. 固定模型、数据与资源梯度

### 4.1 模型与训练默认配置

tiny 主要做断点和严格数值，public checkpoint 做真实组合。模型 revision 从现有下载 manifest 解析为完整 SHA，权重文件另计算 hash；没有 revision 的模型名称不能进入正式比较。优先复用现有本地 snapshot，下面列表是候选，不表示已全部缓存或可下载。

| 配置 | 固定模型/尺寸 | 默认输入与训练参数 | 用途 |
| --- | --- | --- | --- |
| T-ENC | BERT，layers=2、hidden=128、heads=4、intermediate=512、vocab=4096 | B=2、S=64，10 steps，FP32、dropout=0，SGD `lr=1e-2` | 分类、MLM、token/QA heads，tiny grad/更新 |
| T-DEC | Llama / Qwen2 / GPT-2，各 layers=2、hidden=128、heads=4、vocab=4096；Qwen2/Llama KV heads=2 | B=2、S=64/257，10 steps；AdamW `lr=1e-4, betas=(0.9,0.999), eps=1e-8, wd=0.01` | GQA、padding、cache、LoRA、AMP、DDP/FSDP2 |
| T-S2S | T5，encoder/decoder layers=2、d_model=128、heads=4、d_ff=512、vocab=4096 | B=2，source=64、target=32，10 steps；显式 pad/eos/start IDs | cross-attention、labels shift、Seq2Seq adapter |
| R-ENC | `google-bert/bert-base-uncased` 或已缓存的同族真实 checkpoint | B=8、S=128/512；分类 100～300 updates，AdamW `lr=2e-5` | Trainer、DataLoader、AMP 与完整恢复 |
| R-DEC | 优先 `Qwen/Qwen2.5-0.5B`，已有 revision `060db6499f32faf8b98477b0a26969ef7d8b9987` | micro B=1/2，S=512/2048；LoRA r=8/32、alpha=2r、q/v 或全部线性层，`lr=2e-4` | 单卡六库组合、greedy/sample 与模型权重闭环 |
| R-S2S | 已缓存 `google/pegasus-xsum` 或小型 T5/BART；Pegasus 已有 revision `8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b` | B=2，source=256/512、target=64/128；`lr=1e-4` | 摘要、Seq2SeqTrainer、beam 与 checkpoint |
| R-SCALE | 合法可用的 Qwen/Llama 1～3B → 7～8B；70B 为最后可选项 | S=2048 起，AMP BF16、AdamW；G 根据目标 global batch 设置 | FSDP2 真实分片/扩展、容量与通信收益 |
| R-MODAL | 已缓存 ViT、Whisper/Wav2Vec2、CLIP、SmolVLM2 或 Qwen2.5-VL | 视觉 B=4、224²；音频 B=1/2、5/15/30s；VLM B=1、1/2 图或 2/4 帧 | processor、多模态 loss/梯度、LoRA 和生成 |

所有 tiny config 显式补齐 attention backend、max positions、special IDs 和 task head，避免继承上游巨大默认值。public 默认 clip norm=1、scheduler=constant、dropout=0，另在 `TRN-02/ACC-05` 打开 scheduler/dropout；绝不为了“跑得更快”改变用户指定更新频率或训练目标。

### 4.2 数据集、指标与持久样本

| 任务 | 小规模真实数据候选 | 固定规模 | 指标与 fixture |
| --- | --- | --- | --- |
| 文本分类 | GLUE SST-2/MRPC 或现有合法本地分类集 | train 1024 / eval 256 | accuracy/F1；文本、label、token IDs 与样本顺序 |
| Causal LM / SFT | WikiText-2 或已有 No Robots/指令数据；另准备固定中英混合对话 | train 512 / eval 128，token 长度分桶 | token-weighted NLL/perplexity；prompt/completion label mask、chat template |
| 摘要 / Seq2Seq | XSum 或已有摘要集 | train 256 / eval 64 | teacher-forced loss、ROUGE、固定生成参数；source/target IDs |
| 视觉分类 | 本地公开许可图像子集 | train 256 / eval 64 | accuracy；图片 hash 与 processor 输出 |
| 语音识别 | 本地许可音频或 LibriSpeech 小切片 | 16/64 条、5～30s | WER/CER；采样率、音频 hash、feature/mask |
| 图文 / 视频对话 | 本地许可的图文对和短视频帧 | train 32 / eval 16 | task loss、greedy token、结构化 processor 输出 |

划分、revision、样本 ID、字段规范、license/access 条件、原始文件 hash 和 token 后 fixture 都进入数据 manifest。分类/总结任务的多 seed 质量结论只适用于该小规模验证，不宣称达到完整 benchmark 分数。每个任务同时准备非整除样本数、最后一个小 batch、全 padding/空文本/超长输入和未知特殊 token 的负向 fixture。

### 4.3 资源与时长预算

| 资源码 | 资源与推荐规模 | 初始工作量预算 | 扩大条件 |
| --- | --- | --- | --- |
| C | CPU，tiny 或 tokenizer/I/O fixture | 1～10 分钟/case，不含首次编译 | ENV 通过；线程数和 worker 数固定 |
| S | 单张加速卡，tiny / 0.5B / BERT，先 B=1 | 10～30 分钟/case | 正确性先过；独立峰值测量确定预算 |
| M | 双卡，tiny → 1～3B | 20～60 分钟/组合 | 每卡剩余容量与通信路径足够，首次 JIT 已串行预热 |
| L | 四卡，3～8B 或二维 mesh | 30～120 分钟/组合 | 双卡通过，有同配置 oracle 与每 rank 证据 |
| H | 6～8 卡及大模型/长上下文 | 只先做 12-step 短验证；完整训练另设预算 | 真实参数、梯度、state、activation 和 gather 峰值估算后确认可行 |
| V | 其他设备或多机 | 先 tiny，时长待硬件标定 | 两侧原生后端可用，独立环境和通信基线成立 |

这些是安排工作的预算，**不是当前设备上的耗时/显存实测**。对于 FP32 参数 + FP32 梯度 + 两份 FP32 Adam state，持久部分约 $16P$ 字节，混合精度再看实际 master copy；FSDP2 不能简单把整个峰值除以 world size，必须加激活、通信 buffer、最大的 unshard group 与初始化副本。先测 1～3B 的逐阶段峰值，再确定 7B/70B 是否值得启动。

正确性短验证可在显存充足的共享卡上运行并记录负载；用于速度结论的 matched runs 需要负载稳定且不与本任务其他 benchmark 重叠。后续资源不足时可按用户规则设计等待队列：配置与 baseline hash 冻结、启动前复查容量、只管理自身 PID/进程组；队列等待不计作实验通过。本轮没有创建或启动队列。

## 5. 实验目录：每行都是一个值得独立验收的场景

下面列出 **70 个场景族，其中 8 个端到端组合**，不是一次展开所有选项的笛卡尔积。每行写出的“默认”先跑，其他选项按第 7 节策略增量扩展。P0=先跑，P1=主要覆盖扩展，P2=新能力/昂贵或硬件门禁。所有新执行记录初始均为 `NOT_RUN`；既有报告仅填 `prior_evidence`。

### 5.1 环境与入口

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| ENV-01 / P0 / C | 分别 source 两个环境；先 metadata，再 runtime identity；普通 import 与源码入口各一次 | Python/库版本、`torch.__file__`、`jittor.__file__`、namespace owner、backend、API/distribution 版本 | oracle 真实且独立；shim 使用目标源码和独立 Torch frontend；两个环境不交叉污染 |
| ENV-02 / P0 / C→S | 同一 tiny tensor forward/backward；显式 CPU 与 CUDA 独立进程 | 实际 storage/device、输出、输入梯度、fallback 计数 | 所宣称 device 真计算且 fallback=0；CPU 必须真的关闭 CUDA |
| ENV-03 / P0 / C→S | 同 fixture 新进程，离线 snapshot，冷/热 cache 各一次 | 输入/权重 hash、import 顺序、首编译、输出、重复执行与退出码 | 重跑无联网、无残留环境依赖；完成打印后进程正常退出 |
| ENV-04 / P0 / C→S | 当前 checkout 做版本条件分支探针：FSDP2 API、RNG capture、显式 CUDA Generator、DCP | 与当前 oracle 共同 API 的路径及真实行为；独立保存不存在/不支持项 | API 2.11 元数据不被当作实装能力；历史修复以当前复现为准 |

### 5.2 Tokenizer、collator 和数据加载交接

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| TOK-01 / P0 / C | WordLevel fixture + public BPE/WordPiece/SentencePiece 各一个；中英、Unicode、空/未知 token，单条/批量 | IDs、mask、offset、word_ids、special tokens、decode 与 JSON reload | 整数/字符串/字段 exact；损坏配置或非法参数明确报错 |
| TOK-02 / P0 / C→S | R-DEC，B=1/2/8，S=1/63/257；left/right padding、truncate、`pad_to_multiple_of=8`，`return_tensors='pt'` | BatchEncoding dtype/CPU 起始位置、`.to(device)`、position/mask 与模型输出 | token/mask exact；交接张量实际迁移，无隐式 FP32 token IDs |
| TOK-03 / P0 / C→S | classification/MLM/causal/seq2seq collator；MLM 固定 mask fixture，另测随机 mask | `labels=-100`、target shift、padding、有效 token 分母、全部 collator 字段 | masked labels 和损失分母与 oracle 一致；随机任务另验 RNG 行为。状态：causal LM tiny 轨迹的 `labels=-100`、shift 和两种累积归约已在 CPU 通过，microbatch mean 另在 CUDA 通过；MLM/Seq2Seq/random mask 未跑 |
| TOK-04 / P0 / C→S | R-DEC 指令 SFT：chat template、assistant-only mask、EOS；不 packing→packing 两档 | 特殊 token、样本边界、labels、position、训练 loss | packing 明确采用同样跨样本 attention 策略；模板或 mask 差异可追踪 |
| TOK-05 / P0 / C→S→M | DataLoader workers=0→2→4，shuffle、persistent worker、spawn；样本数 17/101，drop_last False/True，恢复在中 epoch | 全部样本 ID、epoch/worker seed、尾批、跨 rank 补齐/去重、sampler cursor、顺序 | 确定性轨使用预保存 batch 顺序；公共 loader 轨按声明的顺序/覆盖契约验收，异常及时传播 |
| TOK-06 / P1 / C | 32k/128k 词表，短/512/8192 token 文本，B=1/64/256；worker=1/2/4，固定 Rust 线程数，30min soak | records/s、有效 tokens/s、RSS、FD、输出 hash、队列等待及重复初始化 | tokenizer 本体只报 CPU 性能；无增长/丢样本/死锁；GPU 交接另测 |

### 5.3 Safetensors 与模型/adapter 文件互通

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| SER-01 / P0 / C→S | FP32/FP16/BF16/int64/bool、scalar/empty，metadata；`safe_open/get_tensor/get_slice` 和 `load_file(device=...)` | keys/shape/dtype/value、请求切片范围、真实 placement | lossless tensor exact；不要求文件整体 SHA 相同；不支持 dtype 明确报错 |
| SER-02 / P0 / C→S | T-DEC/R-DEC `save_pretrained`，单文件→小 `max_shard_size` 强制分片，PT→JT→PT 和 JT→PT→JT | index/total_size、无缺/重 key、config/tokenizer/generation_config、fresh forward | 模型全张量互通，首次 forward finite；明确记录 tied weight 的保存/re-tie 行为 |
| SER-03 / P0 / C→S | T5/GPT-2 tied embeddings，shared storage、clone、transpose/slice/stride-offset；save_file 与 save_model 分别验 | alias map、写回语义、连续/非连续拒绝、save_model metadata/reload | 与各公共 API 的 oracle 错误/共享语义一致；物化后“能保存”不能冒充 stride 兼容 |
| SER-04 / P0 / C→S | PEF-01 的两个 adapter 与 `modules_to_save`，不同 device 保存/reload，auto PEFT load | 所有 adapter key/值、base revision、trainable mask、合并后输出 | adapter 双向可用、基座不被误写；不把 adapter 互通外推 optimizer pickle |
| SER-05 / P0 / C | 截断 header/payload、非法 offsets、metadata、缺 shard、重复/缺 key、版本错误；临时输出原子提交 | 失败类型/阶段、已有模型未部分覆盖、未留下“成功”checkpoint | 对损坏输入明确拒绝，无 OOM 式无限 header 分配或宽泛异常转 skip |
| SER-06 / P1 / C→S | 连续 fixture 64MiB→256MiB→1GiB、多 shard；单进程→2 workers，CPU load→CUDA 首次访问 | payload hash、load/save GB/s、RSS、requested-read 字节、live/pool peak、首访问 | 冷进程与 warm/page-cache 分开；不要 drop 全机 OS cache；并行读写无 FD/内存增长 |

### 5.4 推理、cache、attention 与生成

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| INF-01 / P0 / C→S | T-ENC/T-DEC/T-S2S → R-ENC/R-DEC/R-S2S；`AutoModel*`、labels、`return_dict`、hidden states/attentions、`inputs_embeds`；另测classification/feature extraction `pipeline` | forward/loss/所有可微输入与参数梯度、任务输出结构、加载键、pipeline单条/批量预处理和后处理 | 先 FP32 定向严格对拍；覆盖分类/MLM/token/QA/causal/seq2seq 公共任务入口 |
| INF-02 / P0 / C→S | decoder B=1/2、prefix=64/257、追加1/4/32 tokens，use_cache False/True；cache position/mask，支持的 Dynamic/Static/legacy cache 分开 | full vs cached 同 runtime 与跨 runtime logits、KV shape/dtype/length、batch reorder | 每步 finite；padding/position 与 cache 扩展准确；不支持 cache 类明确拒绝 |
| INF-03 / P0 / S | R-DEC/R-S2S：greedy、beam=4、EOS early stop、min/max new tokens、pad==eos、多个 stop IDs；text-generation/summary `pipeline`与streamer另档 | token/score、有效长度、beam reorder、结束样本 mask；生成后cache、streamer事件与最终文本 | 无边界tie时greedy/beam exact；tie必须保存score margin，不能泛化豁免；streamer取消/线程退出按公共契约验 |
| INF-04 / P0 / C→S | temperature=0.7、top-k=10/50、top-p=0.9、combined；重复 seed 与外部 uniform fixture；B=2、new tokens=32 | processor 后概率/支持集、采样分布、同 runtime 可重现、random state 推进 | seed 相同跨框架 token 不强求 exact；有固定 uniform fixture 时比较决策，top-p tie 单独验 |
| INF-05 / P1 / S | R-DEC，FP32 eager/math→SDPA math→可用 fused；AMP FP16/BF16 和纯半精度独立；causal/masked/all-masked row、GQA、非连续输入 | backend hit/miss 与 fallback reason、forward/backward、实际 op dtype | forced fused 失败不能退成 math PASS；all-masked row 按上游行为单列，训练 case 仍要求 finite。状态：混合 `q/k=float32, v=bfloat16` 的 autocast SDPA 前向及 q/k/v 梯度 CPU/CUDA 定向回归通过；完整 fused/GQA/mask/dtype 矩阵仍未完成 |
| INF-06 / P1 / S→M | R-DEC，S=512/2048/8192（再选32768），B=1/2/8；prefill→128 decode tokens，变长请求 | logits/cache、TTFT/TPOT、真实执行 token 数、各阶段峰值与 OOM 上限 | max position/RoPE 配置固定；对超出模型声明上下文只做明确拒绝/专门扩展协议 |
| INF-07 / P1 / S | R-MODAL：图像两尺寸、音频5/15/30s、VLM图像/两帧视频；公共 processor→device→loss/greedy | 全部 processor 字段、padding、时间/图像 token、梯度/生成和模态 task 指标 | 禁用 lab-only stub/workaround 来冒充公共路径；每模态固定输入和对应原生依赖 |

### 5.5 训练、优化器与 AMP

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| TRN-01 / P0 / C→S | 三种 tiny 模型，10 steps；SGD→AdamW；`zero_grad(set_to_none=True/False)`，非默认参数组 | 每步 loss、全部 grads、更新量、buffers、optimizer moments/step 与 grad identity | 更新真实发生；冻结/unused 正确，未用某轮 grad 不得沿用旧值。状态：tiny Llama LoRA FP32/AdamW 固定数据 10-step CPU 及有效 token 4-step CPU/CUDA 子项通过；SGD、其他模型和 zero-grad/参数组矩阵仍未完成 |
| TRN-02 / P0 / C→S | 累积 G=1/4；固定 global tokens，对比大 batch；clip=1、constant→warmup-linear；最后不满累积窗口 | backward/optimizer/scheduler 次数、pre/post clip norm、LR、tail 更新与 loss 分母 | G 倍 loss scaling、尾窗和 scheduler 次数正确；变长 token 另对齐 reduction。状态：tiny causal LM G=2、padding label 的 microbatch mean 与 global-token mean CPU 子项通过；CUDA microbatch mean 子项通过；CUDA global-token、G=1/4、大 batch 等价、clip/tail/scheduler 矩阵仍未完成 |
| TRN-03 / P0 / S | FP32 master + autocast FP16/BF16；纯 FP16/BF16 另档；标准 Linear/attention/normalization→R-DEC | 实际计算/参数/grad/state dtype、finite、精度与显存 | BF16 不是默默变 FP16，AMP 参数和 optimizer state 不被错误转换。状态：tiny Llama LoRA、AdamW、CUDA 4-update BF16/FP16 autocast 子项通过；不代表纯低精度权重、其他架构或显存矩阵通过 |
| TRN-04 / P0 / S | FP16 GradScaler：finite→注入 Inf/NaN→finite，G=4、clip；增长/回退、两个 optimizer | overflow step 无参数变化、scheduler skip、scale/tracker、只 unscale 一次 | 合法状态推进，非法重复操作明确拒绝；恢复后同 runtime 下一步一致。状态：tiny Llama LoRA、CUDA、G=1 的 Inf 注入/backoff/recovery 及 fresh-process scaler restore 通过；G=4、clip、NaN、多 optimizer 和增长间隔矩阵仍未完成 |
| TRN-05 / P0 / C→S | 同模型 gradient checkpointing off/on，dropout=0→非零；use_cache 配置兼容；重入选项按支持面分开 | 全梯度、参数轨迹、forward 调用/实际 recompute、activation peak、step time | 输出对了但无重算不能算 checkpointing；同时报告耗时成本和内存收益 |
| TRN-06 / P0 / S | R-ENC 或 R-DEC 真数据，先10步，再100～300 updates，seeds=0/1/2；保存/评估定期执行 | train/eval loss、任务指标、参数更新统计、稳定内存、失败 batch | 与同预算 oracle 的数值/质量包络一致；loss 下降不代替参数与梯度检查 |

### 5.6 PEFT 跨模型、方法与 adapter 生命周期

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| PEF-01 / P0 / C→S | T-DEC→R-DEC，LoRA r=8、alpha=16，q/v→all-linear；bias none/all，r=32；FP32→AMP | trainable keys/count、A/B 全梯度、冻结 base hash、更新和 adapter fresh reload | 预期梯度齐全，合法首步零 A 不能误判；base 与 adapter dtype 策略可审计 |
| PEF-02 / P0 / C→S | BERT sequence/token/QA、T5 Seq2Seq、GPT-2 Conv1D、Llama/Qwen GQA LoRA | target discovery、fan_in_fan_out、wrapper signature、task head/modules_to_save | oracle 支持的 task/base 组合逐项对拍，不能从 Qwen causal 外推 |
| PEF-03 / P0 / C→S | add/set/disable/delete/load adapter，两个名称切换，weighted merge、merge/unmerge/unload，safe merge 注入 NaN | 不同 adapter 输出差、disable 恢复 base、反复切换 state、合并/解合并误差 | 禁止 no-op；拒绝非有限 safe merge 且原状态不被部分写入 |
| PEF-04 / P0 / S | R-DEC，LoRA/rsLoRA/DoRA，FP16/BF16 AMP，G=4、gradient checkpointing、AdamW 20-step + fresh resume | 幅值参数/范数、trainable dtype、全部梯度、冻结 base、恢复后 step | 低精度真实权重组合完整，而非只 tiny forward；DoRA 独立记录更新轨迹 |
| PEF-05 / P1 / C→S | 已有25个Qwen适用类型做最小回归，再选 AdaLoRA/IA3/prompt/prefix/OFT/BOFT/VeRA 100-step 方法特有行为；T5 选支持子集 | rank allocation、task id、prompt cache、projection reconstruction、sparse/routing/solve grads | 适用性先由 oracle 确认；Qwen 不适用 ADAPTION_PROMPT 不计失败；VeRA 不完整上游保存语义显式登记 |
| PEF-06 / P1 / S | R-MODAL 上视觉/语言 projection 或语言层 LoRA，B=1，10→100 steps | processor→PEFT wrapper、跨模态 loss、全部 adapter grads 与 save/reload | target modules 真实命中；`PeftMixedModel` 公共 API 不支持 save 时只测其合法范围 |

### 5.7 Accelerate、Trainer 与恢复

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| ACC-01 / P0 / C→S | 同一10-step脚本 direct Torch API→Accelerator.prepare(model,opt,loader,sched) | 数值轨迹、包装前后 param identity、实际 step/zero_grad 与 dtype/device | 包装不漏 step、不改参数归属；性能用独立run拆 wrapper 开销 |
| ACC-02 / P0 / S | `mixed_precision=no/fp16/bf16`，G=1/4、clip、tail、两个 optimizer/scheduler | sync_gradients、LR/step、scaler lifecycle、state_dict | 与相同 native Accelerate 工作流一致；尾窗/overflow行为显式定义 |
| ACC-03 / P0 / C→S→M | 数据分片、split_batches/dispatch_batches 支持组合，样本数17、batch4、even_batches True/False | 每 rank 样本、重复补齐、gather_for_metrics 去重、loss/accuracy分母 | 不重复计尾样本；不支持 uneven Join 时明确拒绝而不是挂起 |
| ACC-04 / P0 / C→S | `Trainer`/`Seq2SeqTrainer`：train/evaluate/predict，eval/save每5步，best model、callback、gradient accumulation | callback事件/次数、PredictionOutput、eval metrics、保存的best权重、公共args | 默认配置先在oracle成立；同参数/输入沿途对拍，不能退回私有手写流程当Trainer通过。状态：synthetic BERT sequence classification Trainer 4-step CPU/CUDA 子项通过，G=2、eval/save 每2步、callback、best checkpoint 和 adapter reload 均核对；Seq2SeqTrainer 与计划中的每5步周期仍未跑 |
| ACC-05 / P0 / C→S→M | 连续12 updates vs第5步保存退出+fresh process恢复；固定batch→dropout+shuffle+normal/uniform/f64 RNG，含显式Generator、FP16溢出 | model/optimizer/scheduler/scaler、Python/NumPy/Torch RNG、sampler/data cursor、恢复后实际更新 | 各runtime内部后7步序列/轨迹一致；unsafe snapshot显式失败单列；仅恢复epoch/seed不能称精确续训。状态：此前 Accelerate FP32/随机 dropout 子项和 FP16 overflow/scaler resume 通过；本轮 Trainer tiny BERT/LoRA 固定顺序 CPU/CUDA case 均从 step 1 保存并 fresh process 恢复，样本 ID 从 4 接续，PyTorch/Jittor 各自 1+3 与连续 4-step 的 216 项状态 exact match；每设备跨 runtime 216 项在 `2e-5/2e-4` 容差内通过，最大绝对差 CPU `4.30e-8`、CUDA `4.47e-8`。新增 Trainer `RandomSampler + dropout=0.2` CPU/CUDA case，两边每个 runtime 内部仍分别 216 项 exact match，sample ID 顺序连续；workers=0，混合 Python/NumPy/normal/uniform/f64 RNG 与显式 generator 未覆盖。新增 workers=2 worker RNG 隔离测试 `12 passed`，但 Jittor CPU workers=2 continuous-vs-2+2 fresh resume 在 step 3 起 Python/NumPy transform 首先偏移，`78/272` 项失败，因此默认入口历史子项 `NOT_PASS`；worker RNG 隔离不等于 checkpoint 保存 generator、worker-local RNG、cursor 和预取 batch 状态。2026-10-17 显式 stateful 入口已在下一行的 CPU/CUDA 12 vs 5+7 完成 workers=2 随机增强精确恢复；mixed RNG history 全矩阵仍未完成。最小复现发现 `LambdaLR` checkpoint 将 Transformers `functools.partial` 原样写入导致 `weights_only=True` 恢复失败，现已按 PyTorch 格式保存 callable 状态字典，scheduler owner 文件 `7/7 passed`。报告与产物在 `$JITTOR_LAB_ROOT/_state/hf-stack-validation/trainer-resume-{cpu,cuda}-20261014-r1/`、`trainer-resume-stochastic-{cpu,cuda}-20261014-r1/` 和 `trainer-resume-workers2-cpu-20261015-r1/`。|
| ACC-06 / P0 / S | 保存到CPU后fresh CUDA load；param groups顺序/名称、无optimizer state、strict/non-strict、损坏state | moment实际device/dtype、step、加载键、恢复首步、原子失败；尝试跨runtime optimizer文件 | 同runtime合法恢复数值一致；跨runtime可读性单独判，原生拒绝pickle不能隐藏 |
| ACC-05-stateful / P0 / C→S | 显式 `stateful_trainer_class()`：tiny BERT/LoRA FP32、RandomSampler、dropout=0.2、workers=2/prefetch=2、Python/NumPy transform、sampler/loader generators；12 updates vs 第 5 步退出 + fresh process 7 updates | prepared loader cursor/prefetch/worker RNG、generator 状态、全部 trainable gradients、model/optimizer/scheduler、真实 device/fallback；状态损坏与配置变更拒绝 | **PASS（2026-10-17，声明范围）**：独立 PyTorch/Jittor CPU/CUDA 四格各 800/800 exact；单进程单设备、首 epoch、无累积尾窗。loader 50 项、adapter 41 项、真实 CPU 门禁 6 项通过。普通 loader 不留历史，stateful 默认限制 100000 实际样本索引；默认 Trainer 历史 NOT_PASS 保留，跨 epoch/worker=4/Trainer persistent/AMP/分布式仍开放。见[验收报告](../results/2026-10-17-trainer-stateful-resume.md) |
| ACC-07 / P2 / C→S→M | `init_empty_weights`、low_cpu_mem_usage、device_map、CPU/disk offload；先能驻留的小模型，随后容量诊断 | 构造storage、pointer、加载/前向后GPU释放、CPU/RSS/disk占用、tied weights与hook移除 | 功能正确与真正容量释放分别打分；is_meta标记不等于无storage；超显存正式模型须先过storage语义 |

### 5.8 DDP 与分布式真实数据

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| DST-01 / P0 / S→M→L | world=1/2/4，T-DEC→R-DEC，固定global batch，G=1/4、no_sync，30 updates；SGD/AdamW | global loss、mean-reduced grads、每rank参数/buffers/optimizer、update count、launcher退出 | 与单卡同有效batch及独立PyTorch DDP一致；no_sync减少collective次数并在边界同步 |
| DST-02 / P0 / M | uneven最后batch、sampler set_epoch、worker2、padding补齐；有unused branch及shared/tied params | rank输入/有效token分母、grad存在性、collective序、metric去重 | 默认even_batches先通过；支持外的动态unused/Join路径明确拒绝，不误标通用DDP |
| DST-03 / P1 / M→L | R-DEC + LoRA + AMP FP16/BF16 + G=4 + checkpointing，shared frozen base，20 updates | 冻结不变、全部adapter梯度、no_sync/clip/scaler、replica equality、fresh resume | 同时使用这些功能时仍正确，避免每个开关单独过但组合失败 |
| DST-04 / P1 / M | 自有任务中注入一个rank错误/超时、非法rank映射、worker异常；subgroup/broadcast object支持面 | 全rank退出码、bounded timeout、遗留子进程、临时checkpoint原子性 | 不吞rank失败、不挂住其他rank；只清理本任务进程，不干扰外部工作 |

### 5.9 FSDP2、分片内存与 checkpoint

| ID / 优先级 / 资源 | 实验配置 | 必须比较或观察 | 通过条件 / 定位价值 |
| --- | --- | --- | --- |
| FSD-01 / P0 / S→M | T-DEC，world1→2，root only→每block fully_shard，reshard_after_forward True/False，10 updates | full/shard逻辑shape、local storage、grad shard、unshard/re-shard序、SGD/AdamW state | 数值与独立PyTorch FSDP2/单卡参考一致；storage真正归本rank而非full slice保留 |
| FSD-02 / P0 / M | tied embedding/head、shared module、frozen parent、frozen base+LoRA、param-group lr不同，连续20次变更/forward | 唯一logical param/alias关系、只更新一次、parent-child shard ownership、optimizer映射 | 共享参数无重复更新或悬空引用；冻结LoRA基座实际不参与错误reduce/update |
| FSD-03 / P0 / M | FP32 master+BF16/FP16 compute，reduce dtype FP32/低精度分档；G=4、no_sync、clip、checkpointing | 真正dtype、全局norm归约、累积/scale、rank shard更新，inf overflow | 与oracle相同mixed-precision policy；不双算replicated axis或误clip局部范数 |
| FSD-04 / P0 / M | FULL model+optimizer state，rank0-only与各rank合法模式；连续12步vs5+7 fresh resume | keys/值/optimizer state、全rank调用次序、gather内存、CPU load后placement与下一步 | FULL save/load真实完整，恢复后更新；保存时collective次序正确，文件格式互通单独测 |
| FSD-05 / P1 / M | SHARDED DCP save/load，model+optimizer；先当前拒绝契约，能力补齐后同world size→2→4 reshard | reader/writer/planner、chunk/global metadata、每rank文件、原生Torch可读与后续更新 | 当前未实现则SUPPORTED范围不含此项；不得生成私有假分片文件冒充DCP，跨world size另验 |
| FSD-06 / P1 / L | 2D mesh=(replicate2,shard2)，命名轴/非顺序rank/subgroup，合法hybrid FSDP2 | group成员/通信顺序、shard/replica参数、global norm、样本分配与故障 | 与oracle同mesh；replica axis不重复范数，错误mesh配置构造期拒绝 |
| FSD-07 / P0 / M→L | R-SCALE 1～3B起，20 warm steps→100～300 steps，root/per-block两档，DDP vs FSDP2 | init/load/unshard/fwd/bwd/step/save全阶段peak，每ranklive/pool、liveness、step/s、collectives | 数值先过；无持续live增长；报告真实capacity与峰值收益，不能用Var快照替代peak |
| FSD-08 / P2 / H→V | 7～8B/可选70B、prefetch/overlap开关、CPU offload合法模式、多机2节点 | 准备/通信/计算timeline、重叠比例、step time、全rank数值、checkpoint容量 | 同步基线先建立；异步API名称不是overlap证据；无硬件标未验证，多机不继承单机PASS |

### 5.10 用户会实际使用的组合实验

这些条目是最终验收锚点。前面的定向实验解释失败原因，以下组合证明用户工作流完整；组合 PASS 仍不能替代未执行的单项边界。

| ID / 优先级 / 资源 | 可复现默认配置 | 贯穿路径与额外验收 |
| --- | --- | --- |
| E2E-01 / P0 / S | R-ENC + SST-2/MRPC固定子集；B8/S128，FP32→BF16 AMP，workers2，G2，100 updates；第50步保存fresh resume | tokenizer→collator→Trainer/Accelerate→AdamW→eval→safetensors；与oracle比较loss、accuracy/F1、回调、best checkpoint和完整恢复。状态：synthetic tiny BERT + tokenizer/PEFT/Trainer 单卡 4-step smoke 已在 CPU/CUDA 通过；真实 SST-2/MRPC、BF16、workers2 和第50步 fresh resume 仍未完成 |
| E2E-02 / P0 / S | Qwen2.5-0.5B + 指令子集；S512→2048，microB2/G4，LoRA r8 q/v，BF16 AMP，checkpointing，100 updates；20+80 resume | 六库单卡链：chat template/label mask、全部LoRA梯度、冻结base、scaler或BF16无scaler语义、adapter export→原生PT greedy；另开sample功能run |
| E2E-03 / P0 / S | R-S2S + 摘要子集；source256/target64，B2/G2，Seq2SeqTrainer，FP32→BF16，100 updates，beam4 | tokenizer/collator labels shift→cross attention训练→generation eval→ROUGE→model/tokenizer save→fresh resume；缓存和beam不能只测decoder-only |
| E2E-04 / P0 / S | R-DEC causal LM全量微调，S512/B1/G4，AdamW，50 updates；同shape LoRA另档 | 区分full finetune与LoRA内存/更新集合；effective tokens/s、eval NLL、model safetensors跨runtime、optimizer各runtime内resume |
| E2E-05 / P0 / M→L | E2E-02的DDP，world2→4；固定global batch8，world2 microB1/G4、world4 microB1/G2；tail batch+workers2，100 updates | launch→分片数据→PEFT/AMP/no_sync→gather eval→rank0 adapter save→全部rank fresh resume；与独立PyTorch DDP及单卡global batch对照 |
| E2E-06 / P0 / M→L | E2E-02换FSDP2 plugin；先tiny双卡12 updates，再0.5B/1～3B 50 updates；FULL checkpoint，BF16、LoRA、G4、checkpointing | wrap顺序、tied embedding、冻结base、optimizer param groups、FULL model/optimizer与adapter提取；比较全局梯度、各rankshard、恢复后更新和真实峰值 |
| E2E-07 / P1 / S→M | R-MODAL：ViT分类或Whisper ASR全量/LoRA二选一，再SmolVLM2/Qwen2.5-VL图文LoRA；10→100 updates | 公共processor/真实数据/PEFT wrapper→AMP→Trainer/Accelerate→save/reload→accuracy/WER或generation；torchvision/torchaudio等依赖另登记 |
| E2E-08 / P2 / S→V | 同模型不同部署条件：干净wheel安装、离线重启、合法低显存CPU/disk offload；可选长任务继续训练 | 从源码开发环境迁移到用户环境；验证部署包身份、缓存资源、恢复、真实capacity；`torch.compile`/量化作为单独扩展，先确认oracle及兼容ABI |

### 5.11 性能和稳定性专项

| ID / 优先级 / 资源 | 实验配置 | 必须报告的匹配结果 |
| --- | --- | --- |
| PRF-01 / P1 / S | R-DEC，B1/2/8，prompt128/2048/8192，new32/128，FP32→BF16；prefill-only/decode-only/end-to-end | actual tokens、TTFT/TPOT、prefill/decode throughput、p50/p95、cold/warm、KV peak；先单因素扩大，不全组合 |
| PRF-02 / P1 / S→M→L | BERT/Qwen/Seq2Seq各一锚点；direct→Accelerate→Trainer，full→LoRA，FP32→AMP，G1→4 | forward/backward/optimizer/loader/clip/scheduler时间、有效tokens/s、step time、峰值；wrapper与kernel差距分别定位 |
| PRF-03 / P1 / S→M→L | 同模型DDP/FSDP2各world1/2/4，固定global tokens的strong scaling；固定local batch的weak scaling另档 | speedup、$E_N$、max-rank耗时、collectives bytes/time、init/save cost、每ranklive/pool与通信拓扑；两个scale协议不混合 |
| PRF-04 / P1 / C→S | 已过正确性的TOK-06/SER-06；tokenizer/processor→H2D→forward端到端pipeline，workers0/2/4 | CPU threads/Rust threads、RSS、数据等待、H2D、I/O吞吐、模型利用率；预token fixture用于拆分CPU瓶颈 |
| PRF-05 / P1 / S→M | E2E-02/E2E-06，30min soak或300 updates，checkpoint每50步；adapter切换/反复eval，退出再重启一次 | 稳态live增长斜率、pool high-water、RSS/FD/Var liveness、失败batch、checksum、每段resume一致；reserved plateau不等同泄漏 |

### 5.12 后端、版本和发行物矩阵

| ID / 优先级 / 资源 | 实验配置 | 准入与验收 |
| --- | --- | --- |
| PORT-01 / P1 / C→S | 当前pin不变，源码入口→干净安装core/compat/必要adapter wheels；一条INF、SER和E2E tiny | 验证安装后实际路径、runtime资源、离线cache和公共入口；当前Dockerfile不自动当作CUDA/六库验证环境 |
| PORT-02 / P2 / C→V | 当前pin，新增CUDA/ROCm/NPU目标；先CPU共享协议，再与同device独立PyTorch/torch_npu对拍一条INF/TRN/PEF/E2E | 一后端一进程/cache；无真实oracle时只记降级证据，fallback=0；FSDP/NCCL/HCCL另外hardware gate |
| PORT-03 / P2 / C→S | 保留当前venv，新建候选Transformers/PEFT/Tokenizers/Safetensors/Accelerate版本对；一次只升级一个变量，再做兼容组合pin | pip resolver不替换oracle；adapter version gate必须fail-closed；新Torch版本只在独立oracle环境，不覆盖当前基线 |

## 6. 结果产物与通过判定

每个展开后的实例用 `case_id = 场景族/模型/dtype/device/world_size/variant`，如 `E2E-06/qwen05b/bf16/cuda/2/full-gc-g4`。每次重试新建 `run_id`，保留失败产物与原始退出码；实例数、场景族数、已经执行数分别统计。一个场景族仅在其明确声明的必跑实例全部执行后才能发布覆盖结论。

| 字段组 | 必须记录 |
| --- | --- |
| 来源 | case_id/run_id、父依赖、owner、时间、源码SHA/dirty diff hash、环境脚本hash、库版本与实际导入origin |
| fixture | model/data/tokenizer revision、config、权重/输入hash、trainable/alias/expected-grad清单、样本ID/batch次序、loss reduction |
| runtime | backend/device实际断言、precision/AMP/TF32、attention路径、optimizer实现、world/rank/mesh、通信拓扑、fallback、host线程和worker |
| correctness | forward/loss/grad/输入grad/更新/state最差张量和误差、finite、缺失键/梯度、冻结状态、step/LR、metrics、generation/cache |
| restore | checkpoint文件列表/hash/格式、保存步、恢复后真实更新、RNG/sampler/scaler、same-runtime与cross-runtime结果分别写 |
| performance | phase/min/median/p95/repeats、valid tokens/items、warm/cold、live/pool/first/steady峰值、RSS/I/O、共享负载与profiler独立run |
| outcome | functional/numerical/training/restore/device/performance分轴结果、exit code、日志/最小复现/trace路径、边界原因 |

使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED_ENV`、`UNSUPPORTED_EXPLICIT`、`INCONCLUSIVE_RESOURCE`。oracle 也不支持的组合记录为适用性边界，从适用分母中有证据地排除；shim 对oracle支持路径明确拒绝仍是功能覆盖缺口，不能计作PASS。负向测试单独可PASS，含义仅为“拒绝契约正确”。资源不足/无卡/没装库不算算法失败，也不能算通过；不使用宽泛异常捕获把编译/数值/rank失败降成skip。

实验目录建议为 `$JITTOR_LAB_ROOT/hf-stack-validation/{configs,fixtures,scripts,results}/`，运行状态与原始日志为 `$JITTOR_LAB_ROOT/_state/hf-stack-validation/<run_id>/<runtime>/<rank>/`；benchmark与unittest、各device/并行rank的JIT cache隔离，首次编译串行。稳定case提炼为 tiny门禁，复用 `compat/tests/torch/_ecosystem_cases.py` / harness；有意义的长期回归进入nightly，不把真实大模型训练塞进core。

大型模型/数据下载遵守用户约定，使用 HF mirror 和用户指定的集中Hub目录，具体绝对路径放lab配置而不写进可移植仓库规则。下载时固定完整revision，评测阶段只读snapshot并开离线；日常日志、脚本、checkpoint和临时文件不写入下载中心。两套env脚本现有`HF_HOME`会指向topic cache，执行者应在source之后按实际已缓存snapshot或集中cache显式配置，防止意外重复下载。

## 7. 用最少实验覆盖最多交互

先以 `T-DEC + FP32 + eager/math + B2/S64 + G1 + 单卡 + 无checkpointing + 固定batch` 为控制配置，每次只改一个因素，完成API断点定位；每个新加速后端都先通过共享CPU协议。再做下列必须组合，不依赖随机pairwise抽样刚好选中：

1. LoRA + BF16 AMP + G4 + checkpointing + FULL resume；其中FP16另加overflow分支。
2. LoRA + tied embedding + FSDP2 per-block wrap + 非默认optimizer groups。
3. 变长token + padding + tail microbatch + G4 + scheduler + DDP metrics去重。
4. dropout + mixed RNG + shuffle/worker seed + 中epoch fresh-process resume。
5. Seq2Seq cross-attention + labels shift + beam reorder + adapter save/load。
6. 多模态processor + AMP + LoRA + 全adapter梯度 + safetensors fresh reload。

上述交互完成后，再用pairwise覆盖 B、S、dtype、padding、attention、G、checkpointing、optimizer、worker、world size、wrap policy 等维度；候选组合先经oracle适用性验证。典型阶梯是 B=1→2→8，S=64→257→512→2048→8192，world=1→2→4；不是每个模型都要跑全部梯度。

一个实用的周内顺序是：A/B批先tiny和0.5B单卡，C批先tiny双卡和0.5B LoRA，再把1～3B FSDP2峰值/恢复跑完整；之后才决定7B、非文本更多模型、30min soak和多机。具体总工期在首批实测后估算，不以本文预算冒充排期承诺。定位失败时固定输入/weight/seed，先比较单步、单层和state变换，再修对应owner，最后复跑受影响组合。

## 8. 现在已有的可复制入口与待实现部分

以下命令复用仓库现有工具，**本轮没有运行这些模型实验**。默认从仓库根执行；每个CPU/CUDA任务独立shell。先指定 `JITTOR_REPO_ROOT` / `JITTOR_LAB_ROOT`，并由调度者选择 `CUDA_VISIBLE_DEVICES`。示例不提供机器设备编号。

### 8.1 不运行模型，先列出现有case

```bash
export JITTOR_REPO_ROOT="$PWD"
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --list-only
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo peft --list-only
```

### 8.2 现有双解释器CPU正确性门禁

```bash
export JITTOR_REPO_ROOT="$PWD"
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
parity_state="$(mktemp -d "$JITTOR_LAB_ROOT/_state/hf-stack-cpu-XXXXXXXX")"
export JITTOR_HOME="$parity_state/jittor-home"
export TMPDIR="$parity_state/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
export REAL_TORCH_PYTHON="$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python"
export JITTOR_REQUIRE_REAL_TORCH=1
export JITTOR_TEST_REQUIRE_EXECUTION=1
export JITTOR_ECOSYSTEM_TF32=0
export PYTHONPATH="$JITTOR_REPO_ROOT/compat/shim/resources:$JITTOR_REPO_ROOT/python"
python -m pytest -q compat/tests/torch/test_ecosystem_parity.py \
  -k '(transformers or peft) and not CUDA and not NPU'
```

`--device cpu` 的runner或该CPU gate在import后明确关闭CUDA，不用env脚本默认的`JITTOR_CUDA=1`推断实际device。CPU一次覆盖共享语义后再做CUDA。同文件的CUDA选择器为 `-k '(transformers or peft) and CUDA'`，使用新的state/cache且先串行预热；验收时须检查执行数与device记录，不能只读pytest退出码。

### 8.3 已有四轴工具的CUDA锚点

```bash
export JITTOR_REPO_ROOT="$PWD"
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
# 先由调用者 export CUDA_VISIBLE_DEVICES，避免继承env脚本的默认卡。
source "$JITTOR_LAB_ROOT/transformers_compat/env_jittor.sh"
verify_state="$(mktemp -d "$JITTOR_LAB_ROOT/_state/hf-stack-verify-XXXXXXXX")"
export JITTOR_HOME="$verify_state/jittor-home"
export TMPDIR="$verify_state/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
export REAL_TORCH_PYTHON="$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python"
export JITTOR_ECOSYSTEM_TF32=0
# verify_repo把环境传给两侧runner，不清理oracle的PYTHONPATH。
# 此处只指向core源码，不能放顶层torch facade，否则oracle会导入shim。
export PYTHONPATH="$JITTOR_REPO_ROOT/python"
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases transformers_llama,transformers_bert \
  --device cuda --repeats 10 --out "$verify_state/out"
```

这个工具覆盖现有case的输出/梯度、device、fallback、时间及其内存pass；tiny性能不能替代R-DEC真实推理/训练性能。当前工具在Jittor有`get_peak_device_used_memory`时使用真正的live high-water，但其pool仍取周期采样与末值的最大值，可能漏掉瞬态，因此不能直接当作严格reserved peak；需要runtime high-water或独立阶段测量补齐。工具报`ran`不代表本文所有轴已通过。CPU pytest harness会清理oracle路径，四轴工具不做同样清理，两种入口的PYTHONPATH不能直接互换。

### 8.4 已有回归和lab脚本索引

| 范围 | 现有可复用入口 | 使用前检查 |
| --- | --- | --- |
| tokenizer | lab `tokenizers/run_core_matrix.py`、`run_l3_l5.py`、`run_l5_mp.py` | 输出位置、tokenizer文件、文本fixture、worker及线程参数 |
| 文本checkpoint | lab `transformers_compat/{encoder_l4,decoder_l4_20260906,seq2seq_l4_20260906}/` | 冻结脚本SHA、实际revision、当前frontend与finite comparator |
| 多模态 | lab `transformers_compat/vision_audio_multimodal_l4_20260912/run_real_checkpoint_l4.py` 与 `multimodal_extended_l4_20260912/run_extended_checkpoint_l4.py` | 不能继承历史stub/远端代码workaround；按当前公共入口复验 |
| PEFT | [方法门禁](../../compat/tests/torch/test_peft_qwen_methods.py)、[基本LoRA](../../compat/tests/torch/test_peft.py) | 方法可用性、预期gradient例外和真实base；全方法冷JIT不可当快速smoke |
| Safetensors | [frontend测试](../../compat/tests/torch/test_safetensors_frontend.py)、lab `safetensors/noncontiguous_probe.py` | view/alias语义和真实device；禁止用is_contiguous标记代替storage证据 |
| Accelerate | [单进程](../../compat/tests/torch/test_accelerate_single_process.py)、[训练控制](../../compat/tests/torch/test_accelerate_training_control.py)、[AMP](../../compat/tests/torch/test_accelerate_amp.py)、[scaler](../../compat/tests/torch/test_accelerate_scaler_protocol.py) | 每文件实际覆盖与所需device；单库契约不是双解释器端到端 |
| DDP/FSDP2 | [Accelerate集成](../../compat/tests/torch/test_accelerate_distributed_integration.py)、[FSDP2多rank](../../compat/tests/torch/test_torch_fsdp2_multirank.py)、[参数角色](../../compat/tests/torch/test_torch_fsdp2_parameter_roles.py)、[混合精度](../../compat/tests/torch/test_fsdp2_mixed_precision.py) | launcher/rank/cache契约和oracle API；不是文件存在就代表真实NCCL执行 |
| 性能 | [四轴工具skill](../../agent/skills/torch-compat-repo-runbook/SKILL.md)、[Transformers性能skill](../../agent/skills/jittor-transformers-perf/SKILL.md)、[基准口径](../../docs/performance/benchmarking.md) | 旧skill的机器路径/alias入口不得照抄；核对本轮env及独立frontend |

`_training_trajectory_runner.py` 提供选定 tiny PEFT + Accelerate 状态轨迹；`_hf_trainer_resume_runner.py` 提供 Trainer 固定顺序和随机路径，并已接通显式 `stateful_trainer_class()`。该 adapter 保存 prepared loader 公开状态且独立发行；通用 loader 在 compat 提供有界 batch plan、cursor、worker RNG 和 generator 状态。2026-10-17 workers=2 CPU/CUDA 12 vs 5+7 已各 runtime exact 通过，详见[验收报告](../results/2026-10-17-trainer-stateful-resume.md)。默认 Trainer 仍保留原语义。后续按跨 epoch/worker=4/persistent 状态契约 → 真实 SST-2/MRPC → Trainer AMP → 多 rank → 性能推进；通用 E2E 配置生成、phase 计时、per-rank 汇总和真实任务指标仍待实现。

新修复先最小复现、补定向回归，再按变更风险跑core与相关后端/组合；PR前smoke，full只在需要评估整棵树时运行。正式结论必须注明当前提交、执行数量、通过/失败/未执行范围、日志hash及复查条件，避免再次用“某个库已经适配”代替具体工作流的证据。
