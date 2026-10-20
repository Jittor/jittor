# Torch 兼容发布第二三天：单设备真实工作流

- Status: 执行中；尚未完成第二三天退出条件，不构成发布通过。
- Date: 2026-10-20
- Baseline: 首日候选 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`；后续未提交修复以各 run 的源码 manifest 标识，旧冻结点 `e007383133e49c819b54fbd3a876cbefa856a227` 只对应早期实验。
- Remote sync: `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 已为候选祖先；未整合 `origin/2.0-refactor`。
- Candidate: `release/torch-compat-day23-20261020`，本轮运行明确指向独立 worktree。
- Owner: cgq / Jittor Torch 兼容层维护者。
- Review when: 候选源码、依赖、资产、数值协议、设备、Trainer 或 checkpoint 生命周期变化。

## 接续验收状态

2026-10-21用户要求暂停新增修改并推送独立工作树快照；当前分支、运行进程和恢复纪律见[暂停交接](2026-10-21-workspace-snapshot-handoff.md)。

本次恢复再次 fetch `origin/cgq_transformers`，目标 SHA 仍为
`e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，已经包含在候选 HEAD
`9e0e187bb013e71d85111acfd668e46aeff9f9d2` 中。主树的91个未提交路径全部保留。
本节是当前状态入口；下文的 r3、r5 失败对应各自源码冻结点，继续保留为定位证据。
共享卡上有足够可用显存即运行，不要求空卡；本轮结果不提供配对性能或峰值显存结论。

| 验收格 | 当前证据 | 尚需完成 |
| --- | --- | --- |
| HF CPU compact 三任务 | r10修后两端三任务各两epoch、尾批、checkpoint4新进程恢复均零容差通过；公开reload logits exact | 保留compact范围，不能替代真实CUDA |
| HF CUDA 独立 PyTorch 六格 | 三任务双 seed full/resume 已完成且 exact | 仅作独立 oracle，不能替代 Jittor |
| Jittor BERT BF16 | r10双seed full100及checkpoint50新进程恢复均exact；原始checkpoint审计通过；任务loss包络15.426%仍FAIL | 保留原15%任务门槛，继续核对任务比较协议 |
| Jittor Qwen LoRA BF16 | rsqrt/28层定向CPU/CUDA八oracle通过；r18无hook完整logits仍FAIL：7015339/29171712项超原预算 | 暂停新增修改；定位剩余最早差异，未启动JT LoRA full/resume |
| Jittor T5 BF16 | seed821 full/resume完成但strict恢复FAIL771/2088项；原始数组finite；seed823 full运行中 | supervisor已暂停后续stage；实际进程与阶段receipt优先于campaign旧RUNNING状态 |
| 默认 Trainer FP16 | r10真实BERT full/resume/verify均通过；原始2412数组finite、每checkpoint1241状态签名匹配、1648项恢复exact | 保持实际协议范围；不外推更长训练或多卡 |
| Jittor Vision CUDA | r11五格exact恢复FAIL保留；r18新smoke均因仓外wrapper直接读取缺失torch.__file__退出，未进入模型训练 | 暂停工具修正；新cohort仍待完成；不把入口错误当作新数值结论 |

2026-10-21 接续重新 fetch 后，目标远端仍为上述 SHA，已包含于候选，
无需合并；同步记录为未版本化 `20261020-r1/sync-resume-20261021-r12.json`。
候选49个、主树91个未提交路径均保留，运行期间源码和 HEAD 冻结。
当前实时 GPU 可用显存支持已有有界正确性任务，不等待空卡。

本轮最终 CPU 门禁 r10 的实际结果为 core **330 passed、48 skipped、
1 xfailed**，layout 与受影响 structure 通过，完整 structure **1393 passed、
5 failed、2 skipped**。四项失败涉及重复 fixture、跨 test 导入和 collection
时动态执行 runner；另一项 ASV 导入失败源于外部门禁脚本缺少仓库根目录
`PYTHONPATH`。仓外已准备结构修复，待当前源码冻结运行结束后审阅应用并复跑；
五项失败记录保留，不能将 core 通过写成完整维护门禁通过。原始结果位于
未版本化 `20261020-r3/gate-final-{core,structure}-cpu-r10/`。

独立 PyTorch 的 r10 BERT 双 seed、默认 FP16 和 T5 seed821 的新
full/resume/verify 已通过；BERT 双 seed 与 FP16 的原始 checkpoint/state
审计另保存于未版本化 `20261020-r3/hf-resume-acceptance-r10/`
`pytorch-completed-evidence-r10-part1.json`，SHA256
`7947575e00a9c717aaef55adb4a68e7b5ad69f0d76a97cd748673eead9b99130`。
该审计使用独立 PyTorch 在 CPU 读取 retained checkpoint，不是无框架读取，
也不替代 Jittor 的新基线恢复结果。

BERT r7 两个新进程的输入、logits/loss 与四个 RNG 边界均 exact，201个
raw gradient 实际都在 `cuda:0`，fallback=0。其中只有
`bert.encoder.layer.6.intermediate.dense.bias` 的第1269个元素不同：
`-0.00010204315185546875` 与 `-0.00010156631469726562`，绝对差
`4.76837158203125e-7`；该参数公开梯度为 FP32。末端1642项中该 bias
及其两项 AdamW moment 共3项不 exact。修前稳定 Linear bias provider 仅准入
FP32，BF16 走通用 broadcast backward 归约；固定输入和上游梯度的真实 CUDA
最小复现已确认 BF16 cancellation 的公开 dense 路径在16次执行中不稳定，
独立 PyTorch 同协议逐位稳定。该修前失败保留；修后定向证据及 r8 进展见下文，
不以差异小或范数接近关闭恢复失败。完整 capture 与核对结果在未版本化
`$JITTOR_LAB_ROOT/_state/release-day23/20261020-r3/hf-resume-diagnostic-r7/`。

验收测量合同新增 optimizer group 数量、完整恢复协议、实际 device、零 fallback
和已加载 state 证据检查，24项纯 Python 合同通过。公开默认 pre-hook 收到
模型 kwargs 时错误传入三个参数的签名问题已有实际失败日志及最小修复；独立
PyTorch 的 CPU/CUDA hook 各5项、混 dtype cat 二阶导数各2项通过，Jittor
定向复验继续执行。旧 cat CUDA 一阶/前向回归12项通过，不能替代新增二阶验证。
独立 oracle 索引为未版本化
`$JITTOR_LAB_ROOT/_state/release-day23/20261020-r3/gate-oracles-r1/index.json`。

## 2026-10-21 r17 当前修复验证

重新fetch的目标远端仍为`e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，已是主树与两个候选HEAD的祖先；同步凭据为未版本化`20261020-r3/sync-resume-20261021-r17.json`。主树91个、原候选49个未提交路径全部保留；独立修复树的源码与HEAD在验收期间冻结。

修复树当前CPU定向回归93 passed、CUDA九组定向回归106 passed，均零失败、零fallback，包含cuDNN 2D/3D确定性重复forward/双梯度、公开cuDNN flags、native rsqrt、Torch rsqrt autocast、日常state、BatchNorm、混dtype RMSNorm、Vision合同与scheduler。证据分别为`20261020-r3/gate-fix-focused-{cpu,cuda}-r16/`；这些是定向回归，不能替代完整模型验收。

独立PyTorch/Jittor CPU/CUDA的rsqrt与28层公共RMSNorm replay共八个oracle均已通过，真实device与JT fallback0已核对，证据在`20261020-r3/fix-validation-r16/`。r13的CUDA autocast dtype错误断言被独立PyTorch最小复现纠正：out-of-place rsqrt提升reduced input至FP32，in-place/native保持其原dtype；数值预算未调整。追加修复源码凭据为同目录`autocast-followup-source-receipt-r17.json`。

新增生产修复前的r14完整structure为1398 passed、2 skipped。生产修复后的r16完整structure为1396 passed、2 failed、2 skipped，失败已定位为新rsqrt头文件的MANIFEST登记与新增complex拒绝边界的计数登记；确认runtime/cache holders退出后两文件补丁已应用，凭据在`structure-followup-r17/`；最终r17完整structure为1398 passed、2 skipped，layout通过。r16 core最终330 passed、48 skipped、1 xfailed。旧oracle与当前源码差异恰为MANIFEST和结构ledger两文件，r18清单保留明确source_transition，没有伪称旧oracle属于新fingerprint。

存储r14转存已完成：14个结束的独立PT r10 full/resume目录共44,452,046,764字节逐文件SHA256校验后保留在数据盘，原路径symlink保持可访问；索引为`20261020-r1/archived-completed-oracles-r14.json`。本轮没有删除checkpoint、日志或活跃缓存。

## r10 观察器修正及完整复验

本次恢复同步记录为未版本化 `20261020-r1/sync-resume-20261020-r9.json`。
远端仍为 `e5cffe1a` 且已是候选祖先，候选 HEAD 与49个未提交路径保留，
主树91个未提交路径保留；已有 r10 runtime 期间没有整合源码或改变 HEAD。

旧梯度健康观察器采用 FP32 平方求和，在固定原始梯度下自身存在顺序不确定性。
新算法 `fp64_pairwise_256_v1` 只改变诊断测量：在原 device 上以显式 FP64
固定加法树求平方和，末端只读取一个标量，不改变生产 clipping、optimizer 或
Trainer 循环。没有 round、mask 或放宽恢复零容差。报告与比较器显式要求算法名
及梯度阶段，正常正 clipping 标为 `postclip_unscaled`；FP16 无正 clipping
的测量配置在入口拒绝，缺字段的旧报告也拒绝作为新恢复证据。

固定输入 probe 在独立 PyTorch/Jittor 的 CPU/CUDA 四端，各13组、每组16次
重复均 exact；包含6组真实梯度和7组边界输入，最大23,440,896元素。实际
目标 device、FP64测量和 fallback=0 均核对通过。独立 NumPy 的同一加法树与
四端结果 exact，与通用 FP64 求和的最大绝对差5.684e-14。25项报告/比较器
纯合同通过。只读证据审计24项通过，索引为未版本化
`20261020-r3/hf-observer-norm-diagnostic-r2/evidence-audit-r1.json`，SHA256
`7a31d812a383e3dd5003ba8adb2b3097769b382906d353e1fea94beb1c8e293d`。
AST确认该函数无通用 sum/reduce 或全梯度 host 搬运；尚未逐 kernel 建立
完整生成代码溯源，不扩大为所有 CUDA kernel 的源码审计。

修后 CPU compact 三任务已以 r10 完成两端复验：各8次更新、checkpoint4
新进程恢复，完整状态与样本/更新/梯度摘要在零容差下通过。PyTorch
BERT/LoRA/T5各238/68/242项，Jittor各236/66/240项；跨runtime数量差异
不解释为遗漏的同runtime状态。所有运行实际CPU、fallback=0，公开reload
logits exact，LoRA额外完成adapter/merge/reload及变长生成生命周期。
外层核对当前完整源码snapshot与r10冻结点一致，各CPU stage另保存dirty
Python/MANIFEST hash；运行中的核对与结束后审计记录分别保留，不追溯
伪称旧launcher本身已有每stage完整原生源码guard。汇总索引为未版本化
`20261020-r1/hf-cpu-both-evidence-r10.json`，SHA256
`300356f65c04765cddb056d09559c6ab6c659832a678f136fa0ba88a2389cb29`。

新完整 HF 冻结计划为未版本化
`20261020-r3/hf-resume-acceptance-r10/acceptance-plan-r10.json`，SHA256
`a30ecb075a5a5a4e623828452ddeb74e549defeea5b945f92d053479d83d8f40`。
独立 PyTorch 与 Jittor 分卡执行，每 runtime 内部串行；共享卡显存足够即运行。
Jittor 当前只执行 BERT、默认FP16和T5，不以 Qwen 的局部 SiLU 通过扩展完整
训练。逐格完成后另核实 strict 结果，启动状态不计作验收通过。

SiLU修后的Qwen r7完整初始前向实际已执行并 **FAIL**。原门槛仍为
`atol=.005, rtol=.03`：全部logits有8,671,719项超门槛、maxabs3.203125；
有效位置有3,694,620/22,030,720项超门槛、maxabs.4375。loss为
Jittor2.694406986、PyTorch2.690021753，loss通过不能替代logits。实际
CUDA、fallback=0。产物为未版本化 `20261020-r1/hf-qwen-cuda-initial-bf16-cross-r7.json`，
SHA256 `bff97e24029ee019fb1be2e307398bb5a38963019956daac6f10fee6549cf759`。

旧 r1 诊断在 generic FP32 full-sum 的 `synthetic/65537` 处曾报
`code_op.cc CHECK(0 == cudaGetLastError())`，生成代码hash为
`68d21e54e5de2890`；单独同步最小复现16次未重现。新 pairwise probe
通过65537仅覆盖新测量算法，不能关闭旧 generic CUDA 异常根因；继续保留
原日志并纳入后续稳定性复查。共享 GPU 的上述计时均不构成发布速度/显存结论。

## 本轮新增的定向修复

恢复同步记录为未版本化 `20261020-r1/sync-resume-20261020-r8.json`：远端仍为
`e5cffe1a`，已是候选祖先；主树91个未提交路径保留，候选同步时47个未提交路径保留。
没有 stash、reset、clean 或 push。

低精度 SiLU 的原组合式 `x * sigmoid(x)` 会先把 sigmoid 舍入回 FP16/BF16。
固定输入和上游梯度的真实 CPU/CUDA 修前对拍确认 BF16 forward 有350/1495项
不同、最大差0.03125。核心便携路径现将低精度输入提升至 FP32 计算，并只在
最终输出恢复原 dtype。修后 BF16 forward 和 input gradient 在 CPU/CUDA 均与
独立 PyTorch exact；FP16 的5803项中 forward/gradient 各有1项一 ULP 差异，
FP32 最大前向差约9.54e-7，未宣称所有 dtype 逐位一致。证据为未版本化
`20261020-r1/qwen-silu-jittor-{cpu,cuda}-after-r1/report.json`；完整 Qwen 门槛仍须另验。

同一源码的 `20261020-r3/gate-focused-both-r4/summary.json` 已完成 CPU/CUDA
各90项定向回归：低精度 SiLU 2、native hooks 31、native weightnorm 5、compat
hooks 11、混 dtype cat 二阶梯度2、scheduler 39，合计180 passed、0 skipped，
各 suite fallback=0。scheduler 覆盖公开 LinearLR 递推、手改 LR、显式 epoch、
checkpoint metadata、共享 step metadata 和 SequentialLR 活动 LR 轨迹；
没有把该结果扩展为 SequentialLR 子 scheduler checkpoint 对象互换通过。
前序外部 harness 的 native/compat namespace identity 及 oracle preload 错误日志保留。

CUDA Linear bias provider 已扩展至 FP16/BF16，partial/final 均以固定顺序
FP32 累加，再转换回输入 cotangent dtype；准入要求 input/bias 同 dtype。
固定真实 `(2,46,3072)` 输入与上游梯度，各16次重复的 provider/公开 dense
FP16、BF16、FP32 结果均自身 exact。原生 CPU 一阶/二阶/交叉梯度通过，
CUDA **5 passed、7 deselected**，覆盖分块边界、非连续 cotangent、二阶
adjoint、重复 cancellation 和 FP32 master bias。实际 CUDA、fallback=0。
通用 broadcast-add 归约仍可能不确定，此修复仅覆盖准入的 Linear bias。
定向汇总为未版本化 `20261020-r3/hf-bias-reduce-diagnostic-r1/diagnostic-summary-r2.json`，
源码 manifest SHA256 为 `24fb05b5c376bfa3778d54676a860647661a003c61648f5f1bc3903c14198ba8`。

修后 r8 的 checkpoint50→51 双 fresh-process capture 共2747项、201个
首 microbatch raw gradients 全 exact，末端1642项 state 亦全 exact；实际
`cuda:0`、fallback=0。但原 strict trajectory gate 仍有一项 observer norm
失败：`0.9999999297577934` 对 `0.9999999278790046`。该值来自验收观察器
逐参数 `float().square().sum()`，且 callback 在 clipping 后触发；不能称为
剪裁前梯度范数。固定梯度的归约复现与确定测量方式正在验收，不放宽零容差，
也不把这两次单步 capture 标为完整 Trainer full/resume 通过。产物为未版本化
`20261020-r3/hf-resume-diagnostic-r8/`。

CPU `pow` 的 codegen 在混合整数/浮点输入时直接使用 `std::pow` 的 double
结果，使融合 reciprocal 跳过声明 FP32 中间结果的舍入。修复后显式转换至
输出 dtype，最小 fused/materialized 频率64/64 exact；原生 binary 回归
CPU/CUDA 均10 passed、0 skipped，并核实实际 device 和 fallback=0。
公开 Qwen 加载确实先在 CPU 初始化 `inv_freq` 再搬 CUDA；修后与独立 oracle
的频率差异从24/64降至1/64，cos exact、sin 最大差 `3.725e-9`。完整 BF16
r6 logits 仍未通过原 `atol=0.005, rtol=0.03`：有效位置
`4,137,888/22,030,720` 个元素超门槛、最大绝对差0.4375；loss 通过。
这项算子修复不构成完整 LoRA 验收。证据索引在未版本化
`$JITTOR_LAB_ROOT/_state/release-day23/20261020-r1/qwen-pow-fusion-evidence-index-r3.json`，
SHA256 为 `fe8b9a340fe61ae147cdecc1b30710fa96c3fdfcb38db5cb656b7814c25015d4`。

Qwen 的混合 dtype RMSNorm 替代路径曾把 BF16 activation 与 FP32 weight 的输出错误地保留为 BF16；独立 PyTorch 和模型原始 forward 均为 FP32。compat 现对混合 dtype 拒绝该替代路径，继续执行用户 forward，保持 native serving kernel 的原契约。CPU **4 passed**、真实 CUDA **6 passed** 的定向回归覆盖类型提升、前向及输入/weight 梯度。实际第0层 Q/K norm 修后与独立 PyTorch 完全一致，但整模型 BF16 logits 的原逐元素门槛仍失败，不能据局部修复关闭整模型数值问题。较大随机 RMSNorm 输入的两次直接 forward 所产生的 weight 梯度也可能受到通用 CUDA 归约非确定性影响，原 exact probe FAIL 保留，未放宽既有阈值。

另已确认稳定 CUDA Embedding provider 在 Torch 模式下用 `str(dtype)` 检查原生名称，遇到 `torch.float32`/`torch.int64` 会拒绝；原 provider 也排除了 padding 参数。改用核心 `var_dtype_name`，并复用既有 padding stop-gradient mask 后，真实 CUDA 探针从两条路径各4个不同梯度变为各1个，provider/gradient provider 实际调用均为8次。compat CPU **4 passed**、CUDA **6 passed**，原生 CUDA **10 passed**，覆盖重复和分段归约、padding非零前向、signed indices、二阶梯度和原有边界。它仍不构成全局 deterministic 算法保证，完整 BERT 恢复需在新源码重验。

公开 CUDA GradScaler 的独立增长/overflow fixture 及 fresh-process restore 均通过：有限更新观察到 scale `16→16→32→32→64`，注入 Inf 后 `64→32` 且模型和 optimizer 状态逐位不变；新进程公开加载 model/optimizer/scaler 状态 exact，并继续 `32→32→64`。实际 projection 为 FP16、master/momentum 为 FP32、fallback=0。这是明确缩短 growth interval 的定向合同，默认 Trainer scaler 使用原参数另验。上述证据索引为未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261020-r1/source-fix-directed-evidence-index-r1.json`。

## 范围与证据位置

执行[发布收口计划](../testing/2026-10-20-torch-compat-release-gate.md)的第二三天：CPU 紧凑工作流与 CUDA 真实模型，覆盖分类、LoRA、Seq2Seq、变长批生成、多 epoch、尾批和累积尾窗、AMP、评估、best checkpoint、公开保存加载和独立进程普通恢复。公开 Torch/HF DDP、配对速度/显存、安装包和默认长时稳定性属于后续门槛。

主树分支为 `integrate/cgq-transformers-native-multicard-target@2690b03e3ff489134c093eaf5c4a8fd2af53fd5c`，开始时 91 个未提交路径全部保留。没有 stash、clean 或清空 index；本轮代码在 `$JITTOR_LAB_ROOT/worktrees/torch-compat-release-day23-20261020/`，原始证据在未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261020-r1/`。`preflight.json`、原始 patch、`acceptance-protocol.json` 与 addendum 固定输入、同步 SHA 和阈值，原始失败日志保留。

## 资产与协议

固定真实资产为 BERT-base-uncased、Qwen3-0.6B、CommonGen 微调 T5-base、CIFAR-10、Rotten Tomatoes 和 Alpaca。Rotten Tomatoes 替代当前离线缓存缺失的 SST-2/MRPC，不能称作 GLUE 验收；T5 是实际 base 模型。CPU compact 明确缩小模型结构，保留真实 tokenizer/vocab，不能代表真实规模性能。`fixtures/assets.json` 固定模型 revision、文件 SHA256、源数据 schema 与子集 indices；资产清单 SHA256 为 `de5e32d93b220658111067254cb78d2ac10300f09409da1031787be4138a9eb6`。

主 CUDA 协议为 seeds 821/823、两 epoch：BERT/T5 各 197 train、33 eval、B=2/G=2 共100更新；Qwen LoRA 为101/17、B=1/G=2 共102更新；CIFAR ResNet18 width64 为3137/65、B=32/G=2 共100更新。Jittor 首次 JIT 串行，实际计算设备与 fallback=0 单独核对。共享 GPU 只提供正确性证据，墙钟是诊断耗时。

## 已复现问题与验收边界

BatchNorm 的 Torch frontend 曾继承 native 非持久 `int32[1]` counter，保存5次 forward 后漏存，独立进程恢复仅 `0→2` 而非 `5→7`。修复限定 owned frontend 的 counter 元数据为持久 scalar int64，不改变 native checkpoint 格式；必须重新生成完整训练 checkpoint，不能用数据游标补造遗漏状态。

公开 `from_pretrained` 的 `torch.empty(device="meta")` 曾被实际 placement 拒绝。沿现有 metadata-only 契约修复时仍需明确：native placeholder 有真实 storage，功能通过不代表 meta/offload 节省显存。

Transformers 4.56.2 普通 Trainer 在恢复跳过 microbatch 后，以局部 step 和完整 epoch 长度判断尾累积窗，N=13/B=2/G=2 从 checkpoint5 恢复会漏一次更新；独立 PyTorch 已复现。验收拆为有尾累积的 epoch 边界恢复，以及 N=11/B=2/G=2 的 epoch 中途恢复，不改 oracle 或复制私有训练循环。任意位置与剩余尾累积组合保持上游限制，不能据其他格通过外推。

N11 的第二轮中途 checkpoint4 又独立复现了另一上游问题：Accelerate 1.10.1 `skip_first_batches` 重建 `DataLoaderShard(iteration=0)`，迭代时把 Trainer 已设置的 sampler epoch1 重置成 epoch0。更新数虽为6且加载 state exact，数据游标与50项 optimizer 数组仍不同；这属于日常恢复限制，必须公开说明，不能称为已支持任意中途恢复。正向中途格改为首轮 checkpoint1→6，仍完成两 epoch、使用默认 sampler、保留单样本尾批；独立 PyTorch 与最终修复源码的 Jittor 三任务均 exact（BERT 238/234、LoRA 68/64、T5 242/238项），逐 epoch 样本覆盖与完整游标 exact、CPU/fallback=0。原失败和最小 loader 复现保留于 `hf-cpu-mid11-*r1.json`，协议补充为 `hf-cpu-mid11-protocol-addendum-r2.json`。

独立 PyTorch CNN CUDA 默认算法在同初值、同输入下重复完整运行已于第2步出现1ulp loss 差异，最终371项不逐位一致；显式确定性配置下，100更新完整运行与 checkpoint61 新进程恢复至100的1206项精确一致。默认重复性与确定性恢复分别记录，数值阈值未因默认算法的非确定性而放宽；完整的独立 PyTorch vision 五格结果见后文。

## 当前已完成的功能证据

| CPU 紧凑工作流 | 独立 PyTorch 恢复 | Jittor 恢复 | Jittor 评估 loss |
| --- | --- | --- | --- |
| BERT 分类，N13/B2/G2，两 epoch、8更新 | 238项 exact | 234项 exact | 0.69084847 |
| Qwen LoRA，同一数据规模 | 68项 exact | 64项 exact | 11.92768288 |
| T5 Seq2Seq，同一数据规模 | 242项 exact | 238项 exact | 11.10298729 |

两侧保留默认 Trainer sampler；完整跑与恢复跑各自比较全部最终模型/optimizer/scheduler/scaler state、样本游标、逐 microbatch loss、更新数、学习率和 unscaled 梯度范数，atol=rtol=0。两侧 state 数量不同来自 scheduler 等私有状态格式，未将两种 runtime 的私有结构直接相等作为要求。上述三类初始完整 logits/loss/input probe 也在冻结 FP32 前向门限内通过跨 runtime 对拍。Jittor 计算实际 CPU、fallback=0，均观察到真实单样本尾批和单 microbatch 尾累积窗；公开保存重载 logits exact，LoRA adapter/merge/reload 和两类文本模型变长批 cache/generation 完成。

CNN CPU 为 width8 的紧凑工作流。修复 counter 后，两 epoch 共10更新与 checkpoint6 新进程恢复至10，在独立 PyTorch/Jittor 分别为306/304项 exact；这只建立各自 runtime 的恢复证据。跨 runtime 的原完整状态 strict 比较仍有65条失败记录，包含 momentum 数值、scheduler metadata 键集合和 binary64 学习率差异，未标为跨 runtime 数值通过。额外定位 probe 直接保存 raw gradient、clipped gradient 和实际 momentum，验证本地 clipping、SGD 更新与状态映射公式；main workflow 则只保存梯度范数和 finite-count 摘要，不能将这两种证据混用。

CNN 首窗误差已定位到 `layer1.1.bn1→ReLU` 的单个近零输入。实际两侧34次 ReLU、4,456,448个输入元素仅一个 mask 异号：坐标 `(19,0,2,11)`，PyTorch 为 `+1.47317024e-7`，Jittor 为 `-1.77099480e-7`，其余 mask 全同、两侧 ReLU 本地公式误差为0。独立 PyTorch 诊断仅将该点 backward 项置零，按原逐元素 `atol=1e-5, rtol=1e-3` 重新核算，raw-gradient 失败由11/62个 tensor、835个元素降为0，整体相对 L2 差从 `1.30036e-3` 降至 `2.64393e-6`。保存的首窗训练 loss 与原 PyTorch exact；该 probe 未保存 raw training logits，不能额外声称已对这些 logits 做逐位复核。这个受控反事实只定位该首窗的非光滑点敏感性，不作为生产修改或验收替代，也不证明后续全部 momentum 差异均由同一点引起；原短梯度11项失败与长状态65项失败保留。十步低学习率 fixture 的 accuracy 约为随机水平，不能作为任务收敛证据。只读归因见未版本化 `cnn-mask-cause-attribution-r2.json`。

ViT CPU 是 hidden32/depth2/heads4 的 compact 补充，两 epoch 共10更新与 checkpoint6 fresh resume 在独立 PyTorch/Jittor 分别为202/200项 exact。跨 runtime 的模型参数、momentum、logits/loss 以及梯度范数/finite-count 摘要满足原完整轨迹门限；artifact 没有 raw gradient 数组，ViT 也没有需比较的模型 buffer，不能据此宣称逐元素短梯度或 buffer 数值已验。原完整状态 strict 仍有7条失败记录：1条键集合差异记录涉及两个 scheduler serialization metadata 字段，另6条为 binary64 学习率字段差异，最大绝对差约 `1.0842e-19`，精确位序距离为1–4 ULP。独立语义审阅的188项非LR公共字段通过，但额外≤2ULP的LR诊断仍有4项失败，未标为所有状态通过；同 runtime 恢复仍要求完整状态 exact。原始 strict 与补充诊断分别保留于 `vit-cpu-cross-compare-r2.json`、`vit-cpu-lr-independent-ulp-r1.json` 和 `vit-cpu-semantic-cross-r1.json`。

最终 meta factory/loading、tied parameter/buffer lookup 与 BatchNorm checkpoint 定向回归在真实 CUDA 上合计 **18 passed**，全部执行、`use_cuda=1`、`fallback_count=0`。meta 最小公开加载 probe 也通过，物化后 `cuda:0`；placeholder 仍是有实际 storage 的功能契约。

BERT-base 的 Jittor BF16 两更新 CUDA smoke 已完整通过：参数/optimizer state 为 FP32，实际 projection 有 BF16，模型 native dispatch 为 CUDA，公开重载 logits exact，fallback=0；初始6项 logits/loss/input probe 满足冻结 BF16 门限。随后 Qwen3-0.6B 的公开 `from_pretrained` 在识别 `lm_head.weight` 时失败，矩阵按 fail-closed 停止。根因是 lookup 依赖默认去重后的参数名，且 `remove_duplicate=False` 未传给 native iterator；已同步修复参数和 buffer 的全名称枚举/类别查找。真实 Qwen CPU 公开加载与短 forward、tied parameter/buffer identity 回归均通过。最终源码CUDA smoke也已完整执行两次更新、adapter/merge/merged reload、变长batch greedy/sampling和cache对照，均fallback=0；初始BF16跨runtime数值另记未通过。

独立 PyTorch 真实 CUDA 主矩阵六格均完成：BERT 两 seed 各100更新、checkpoint50恢复，各1646项 exact，best accuracy均为26/33、loss为0.48360559/0.58682531；Qwen LoRA 两 seed 各102更新、checkpoint51恢复，各932项 exact，loss为2.45664406/2.45659447，其中seed821开启 gradient checkpointing；T5两seed各100更新、checkpoint50恢复，各2090项 exact，loss为3.05906415/3.04997396。T5在8个生成token限制下的exact match均为0，仅证明工作流执行。每格全部游标、loss、梯度摘要和最终state在零容差下恢复，全部真实CUDA、fallback=0。独立 safetensors 审计确认BERT201/201、LoRA112/112、T5257/257个训练tensor均实际改变，LoRA base冻结。最终oracle索引为未版本化 `hf-cuda-oracle-acceptance-r1.json`，SHA256 `2075f9546e968b9559b5f2bd6c18e8b8355bea08b813869e4d3a9eb20524bb69`；这些结果不替代 Jittor CUDA 验收。

独立 PyTorch 的默认 FP16 BERT compact 两 epoch 格已完成：8次 attempted optimizer windows 对应7次成功更新和1次合法 overflow skip；第2次窗口 scale 从65536降至32768，model/optimizer/scheduler 的前后 signature 精确不变。checkpoint4 新进程恢复到8时，1650项最终 state/metadata、样本推进、loss、nullable 梯度范数和 scaler/skip/update 计数全 exact。harness 保留默认 scale 与 Trainer 循环，正常 overflow 不再误判为异常；只在 skip、scale 和状态不变均满足公开契约时接受。另有独立 public GradScaler `growth_interval=2` fixture 覆盖增长和注入 inf，不能称为默认 Trainer 的2000步增长证明。

LoRA 合并采用独立 PyTorch FP32 merge 预校准的单独预算：完整 logits 最大绝对差≤1e-3且相对全局尺度≤5e-5、loss 绝对差≤1e-5、greedy tokens exact。原先普通 elementwise reload 门槛会拒绝独立 PyTorch 的合法浮点重排，负向日志保留；预算在真实 Jittor merge 前冻结。普通 public reload、merged artifact 自身的 fresh reload、初始前向与梯度门槛均未放宽。

Qwen3-0.6B 的 Jittor BF16 初始完整 logits 与独立 PyTorch 对拍未满足原冻结门限：有效145个token的22,030,720个logits中4,508,679个元素超出 `atol=0.005, rtol=0.03`，最大绝对差0.4375；padding位置另有更大差异。loss相差约0.0028646虽在门限内，不能替代完整logits验证。原始FAIL保留。分层capture的末端与无hook原始probe完全一致，首层embed与Q/K/V projection也完全一致；最早dtype断点是q_norm/k_norm，独立PT输出FP32、Jittor输出BF16。已定位兼容层按类名识别的RMSNorm fused shortcut不区分input/weight混dtype，而native serving kernel固定返回input dtype；正在最小复现后收紧shortcut准入，让原模型forward执行Torch提升规则，未扩大到该任务主训练。

独立公开 GradScaler CUDA exercise及fresh restore也已通过：FP16 projection、FP32 master/momentum，finite scales为16/32/32/64，注入inf时64→32且model/optimizer exact不变，公开加载后继续32/64。独立checkpoint审计确认finite阶段weight64/64与bias4/4元素真实改变。默认Trainer scale仍为65536/2000，不能混作同一协议。

清理仅涉及本轮已经完成exact resume和生命周期验证的独立PyTorch重复权重，先登记hash再删除，累计释放47,795,146,716字节；初始权重、失败产物、日志和验证摘要保留。最终CUDA矩阵、维护门禁、源码与证据hash将在执行结束后补齐；当前未完成项不标PASS。

三类Jittor真实CUDA两更新smoke现已全部执行完毕。T5初始完整BF16 logits/loss/input probe满足原门限，公开reload最大logits绝对差4.19617e-5仍满足原elementwise预算，生成cache与reload token exact。Qwen分层hook版与原无hook初始完整logits/loss exact，允许继续用分层结果定位，但不改变原数值FAIL。checkpoint签名哈希前新增numeric finite检查，NaN/Inf负向和finite SHA不变合同16/16通过；只增强测量拒绝条件，不修改训练、sampler、AMP或数值门槛，Jittor主矩阵用r3重新冻结。BERT双seed100更新主矩阵、scaler和默认FP16恢复串行推进中。

BERT主矩阵seed821的Jittor BF16完整两epoch100更新已执行，best checkpoint100的loss为0.4698745310、accuracy26/33；真实CUDA/fallback=0、FP32 master/state、尾batch和累积尾窗、公开reload logits exact。checkpoint50的新进程恢复仍在验，不能以完整运行替代恢复结论。

## BERT CUDA 恢复阻断：r3 独立离线复核

本节更新前述 BERT seed821 恢复状态；第二三天仍为**执行中**，普通 CUDA 恢复尚未通过。r3 完整运行的两轮197个样本均恰好覆盖一次，198个 microbatch、100次成功更新且无 skip；独立读取初始化与最终 safetensors，201/201个参数 tensor 均实际改变，最大绝对变化0.0012619942。公开 best checkpoint 按 `eval_accuracy` 最大值选择100步，accuracy为26/33、loss为0.4698745310，最终保存模型与所选 checkpoint 的201个 tensor 全 exact 且 finite。该完整运行的生命周期证据不能替代恢复证据。

从同一 checkpoint50 新进程恢复到100步的首次运行与再次重复均执行完成，`resume_loaded_state_exact=True`；相应检查覆盖已加载的 model/optimizer/scheduler/scaler，不包含 RNG。两次恢复与完整运行的数据游标、shape、device、版本/初始化、更新计数、学习率和精度 metadata 一致，均实际 CUDA、fallback=0、BF16 projection、FP32 master/optimizer state。三个 checkpoint100 的 Python、NumPy、CPU、CUDA RNG 原值全部 exact，CUDA Philox 终点均为 `seed821/offset342542112`，checkpoint50 为 `seed821/offset170572800`；RNG 端点相同仍不能证明中间每个 draw 或 dropout mask 相同。

零容差比较中，完整运行对首次恢复、首次恢复对再次恢复、完整运行对再次恢复三组均为 **603/1642项不一致**，其中201项模型参数与402项 optimizer 动量数组不同，scheduler/scaler 及其余 metadata 相同。两次 fresh resume 本身也不能复现，因此已有同一 checkpoint 重跑非确定性的直接证据；这不证明恢复实现没有其他缺陷，不能将其接受为噪声或放宽门限。

逐 microbatch 复核显示，完整运行与首次恢复从 step51 首个 microbatch（sample IDs14/136）开始，梯度范数即相差 `1.02164699e-7`，但其转成 FP32 后为0 ULP；前8个 microbatch 的 loss 仍 exact，step55 首个 microbatch 才由0.357421875变为0.3583984375。恢复段99个 microbatch 的梯度范数均不 exact，39个 loss 不同，loss 最大绝对差0.0048828125。两次 fresh resume 同样从 step51 有约 `1.02196912e-7` 的范数差，step55 首次出现 loss 差，42/99个 loss 不同，最大绝对差0.00634765625。范数来自各参数 FP32 平方求和，再在 host 以 binary64 累加并开方；其标量 ULP 不能解释为逐元素梯度 ULP，且 `on_pre_optimizer_step` 的范数是在 Trainer clipping 后观测。

原始 safetensors 与受限 NumPy unpickler 读取的 optimizer/scheduler 还按冻结的 FP32 state `atol=1e-5, rtol=1e-4` 独立比较，以下结果全部 **FAIL**；键、参数 ID、shape、dtype、非 tensor metadata 和 scheduler 精确相同，所有读到的原始数值均 finite。

| 比较 | 模型超阈元素 / tensor | 模型最大绝对差 | optimizer 超阈元素 / 数组 | optimizer 最大绝对差 |
| --- | --- | --- | --- | --- |
| 完整运行 / 首次恢复 | 122913 / 89 | 0.0002613206 | 81029 / 172 | 0.0016390830 |
| 首次恢复 / 再次恢复 | 17460 / 58 | 0.0002400361 | 69152 / 166 | 0.0020356318 |
| 完整运行 / 再次恢复 | 304676 / 101 | 0.0002366826 | 171983 / 180 | 0.0029251892 |

源码审计发现待验证的具体分流：native `embedding` 的通用 `weight[input]` 在反向生成 `setitem(..., add)`，CUDA 最终使用 `atomicAdd`。已有稳定 sort/segment CUDA embedding provider 的准入要求 `padding_idx is None`，而 BERT 的 word embedding 使用 `padding_idx=0`，因此不能套用该稳定路径；position/token-type embedding 需按各自条件区分。公开 `torch.use_deterministic_algorithms` 目前为 no-op，`are_deterministic_algorithms_enabled()` 恒为False，不能用该开关声称已排除 atomic 顺序问题。此时只确认源码候选原因，尚未运行最小 probe 证明其是本次轨迹分叉的根因。

退出该阻断至少需要：先以固定 weights、重复 indices、固定 upstream gradient、无 dropout 的最小 CUDA probe 比较 `padding_idx=None/0`，并以非重复 indices 作对照；若最小复现证明并修复该路径，再从相同 checkpoint 捕获 step51 第一 microbatch 在 clipping/optimizer 之前的全部 raw gradient，与相同源码新完整运行的对应点比较，同时保存完整输入、logits/loss 和前后 RNG bytes。现有 r3 artifact 只有梯度标量摘要，无法补做首个不同梯度 tensor 的逐元素定位；原始失败、checkpoint 与日志继续保留，不将 task metric envelope 替代恢复验收。

本节纯离线证据位于前述未版本化 run 根目录：`hf-bert-cuda-jittor-seed821-full-offline-audit-r3.json`、`hf-bert-cuda-jittor-seed821-resume-offline-diagnostic-r3.json`、`hf-bert-cuda-jittor-seed821-resume-vs-repeat-offline-diagnostic-r3.json`、`hf-bert-cuda-jittor-seed821-full-vs-repeat-offline-diagnostic-r3.json` 与 `hf-bert-cuda-jittor-seed821-resume-localization-offline-r3.json`。最后一份索引 SHA256 为 `f41c092c96dc1b5aba39e808645a508f7dd4f02a9449469f9668ca8c3470fb21`，包含三组 strict 结果及源码证据 hash。`hf-cuda-two-seed-task-metrics-r2.json` 的三个任务均仍为 INCOMPLETE，尚缺其余 Jittor 主矩阵；本次未启动 Jittor/JIT/CUDA，也未删除产物。

## Vision CUDA 矩阵准备与独立 oracle

独立 PyTorch vision 五格均完成两 epoch、100次更新，以及 checkpoint61 新进程恢复至100的零容差比较；每格实际 CUDA、fallback=0、FP32 master 和公开 save/reload logits exact。以下数量包含完整最终状态和逐更新轨迹摘要，不能读作相同数量的 raw gradient 检查。截至本节复核时，五格 Jittor CUDA smoke 与主矩阵均为 **NOT_RUN**，没有将独立 oracle 或已有 CPU 结果记作 Jittor CUDA 通过。

| 模型与 seed | 独立 PyTorch 完整运行 / 恢复 | 实际训练 logits dtype | Jittor CUDA |
| --- | --- | --- | --- |
| CIFAR ResNet18 width64，FP32，821 | 100更新 / 61→100，1206项 exact | FP32 | NOT_RUN |
| CIFAR ResNet18 width64，FP32，823 | 100更新 / 61→100，1206项 exact | FP32 | NOT_RUN |
| CIFAR ResNet18 width64，BF16，821 | 100更新 / 61→100，1206项 exact | BF16 | NOT_RUN |
| Compact ViT，FP32，821 | 100更新 / 61→100，1102项 exact | FP32 | NOT_RUN |
| Compact ViT，BF16，821 | 100更新 / 61→100，1102项 exact | BF16 | NOT_RUN |

Compact ViT 固定 image32/patch8、hidden32/depth2/heads4，`width` 参数不会扩大其结构；两格只补充 Transformer vision 工作流，不代表完整规模 ViT 验收。全部五格已由纯离线审计逐项核对 fixture 参数/缓冲区 keys、shape、dtype 和初始参数 hash、实际输入及 target SHA、样本顺序、每 epoch 覆盖、尾批/尾累积窗、全部保存状态 finite、FP32 SGD momentum 和全局非零参数更新。main 的每步 norm/finite-count、末端参数快照和 logits dtype，仍不能证明逐元素 raw gradient、每步每参数均发生非零更新，或所有中间 conv/attention/MLP projection 的 AMP dtype。

证据均位于未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261020-r1/`：seed821 CNN FP32 使用 `cnn-cuda-pytorch-seed821-det-resume-compare.json`，其余格使用 `vision-cuda-matrix-r1/*-pytorch-exact-resume.json`；只读审阅汇总为 `vision-evidence-audit-r2.json`、`vision-offline-audit-contract-checks-r1.json`，compact ViT 独立 oracle 索引为 `vision-cuda-matrix-r1/vit-independent-pytorch-summary.json`。原 CPU strict FAIL、反事实诊断和同 runtime exact 恢复保持独立，不以任务 loss/accuracy 包络替代。

后续入口为同一 run 目录中的 `vision_jittor_serial_campaign.py`，精确命令与边界见 `vision-serial-campaign-instructions-r1.md`。root 完成本轮源码修复后运行 `prepare --campaign vision-serial-campaign-r1`，再按 case 执行 `execute --campaign vision-serial-campaign-r1 --stage smoke --case <case>` 和 `--stage main --case <case>`；调用解释器为 `$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python`。smoke 保留原初值、数据、精度和100更新 LR schedule，只在第2次更新停止；main 才执行100更新及61→100 fresh resume。每次新 launch 前核对源码/输入冻结 hash 并记录实时 GPU/进程与磁盘快照，不要求空卡，不重复启动已有 label，也不自动删除或重试失败。

同 runtime 恢复仍要求全部容差为0。完整轨迹跨 runtime strict 沿原 runner 的 `max(atol, rtol × 参考数组最大绝对值)` 规则：loss 默认 `atol=2e-5, rtol=0.004`，参数/optimizer 为 `2e-5/0.003`，buffer/logits 为 `5e-5/0.004`，accuracy、LR、scheduler/scaler 仍为 exact。新的 strict 报告使用 `*-cross-runtime-strict-r1.json`，不覆盖旧 matrix 较宽的完整轨迹诊断。冻结 FP32 短前向 `atol=1e-5, rtol=1e-4` 与逐元素短梯度 `1e-5/1e-3` 是另一个门槛，不能由 main 摘要补证；BF16 短前向 `0.005/0.03` 和短梯度 `0.01/0.05` 同样需要各自直接产物。cross-runtime strict FAIL 原样记录后继续独立 case；same-runtime exact-resume FAIL 则停止该 case 的进一步扩展，仍可继续剩余独立 case。调度状态机19项纯 mock 检查已通过，未启动任何 runtime；共享卡结果不用于速度、显存或 DDP 发布结论。


## 2026-10-21 接续 r14：恢复、数值分层与存储

本轮重新获取 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，仍为候选 `9e0e187bb013e71d85111acfd668e46aeff9f9d2` 的祖先，无需 merge。主树91个、原候选49个未提交路径保留。同步凭据为未版本化 `20261020-r3/sync-resume-20261021-r14.json`。HF r10/Vision r11继续冻结原候选源码与HEAD。为并行验证修复，建立独立 `release/torch-compat-day23-fix-20261021` 工作树；先复制原候选全部 tracked/dirty/untracked 文件并逐文件核对SHA256，再仅在新树应用结构补丁。原候选及现有训练没有被改写或重复启动；拷贝和补丁应用索引在 `20261020-r3/fix-worktree-r14/`。两树的结果分别绑定源码，不能把新树准备状态算作原候选修复通过。

HF r10新增完成：Jittor BERT seed821/823均100更新，checkpoint50 fresh-process恢复零容差通过；seed821为1644项 exact。默认真实BERT FP16的full/resume/verify也已通过。Jittor T5仍在运行，不能外推通过。独立PyTorch的BERT双seed、默认FP16、T5双seed和Qwen LoRA双seed整组均PASS。`hf-completed-evidence-r12/completed-full-part4-r14.json`（SHA256 `303171da8830d34482ad64d3a838070533333941370d472db3f119dd54db1c81`）实际审计JT BERT823、PT T5823及PT Qwen821的全部保留checkpoint原始数组、更新、覆盖、FP32 master/state、BF16 projection及公开reload，均通过且实际CUDA/fallback=0。旧part3检查器误要求batch1的Qwen出现短microbatch，原失败报告保留；r14按冻结batch配置推导尾批/尾窗是否应出现，Qwen批量1无短microbatch，短尾批覆盖由BERT/T5/Vision另证，不把缺失覆盖改称已测。

恢复通过仍不等于任务门槛通过。新的JSON/NumPy双seed汇总 `hf-resume-acceptance-r10/task-metrics-partial-r14.json` 实际判定BERT任务包络FAIL：PT mean eval loss=0.5352154523、JT=0.4526533633，绝对相对差15.4259539%，超过冻结15%门槛；mean accuracy分别0.7878788/0.8030303，绝对差0.0151515，通过0.1门槛。JT本次loss更低，不能据此宣称质量退化或提升，也不能把原包络FAIL改为PASS。其余任务缺JT完成证据，保持INCOMPLETE。汇总工具r14修正batch1尾批期望，不改loss/accuracy容差、样本或seed。

Vision r11五格两更新smoke全部通过离线artifact审计，实际CUDA/fallback=0，公开reload logits exact、FP32 master。main已真正执行：CNN FP32 seed821 full/resume运行与artifact审计通过，same-runtime恢复245/1206项失败；ViT FP32 seed821恢复97项失败。CNN第62更新是恢复后首步，其loss exact但gradient_norm差9.5367431640625e-7，第63步loss也分叉。最终参数/SGD momentum/BN running buffers有差异；样本/输入/游标/scheduler一致。对应 `vision-final-r11/*-same-runtime-exact-resume.json` 保留，其他独立格继续。2D cuDNN backward的硬编码benchmark=true是待验证线索，不能预先归因为恢复根因；ViT同时失败还需要独立排查公共reduction/clip路径。完整跨runtime strict与短梯度门槛没有因此关闭。

Qwen BF16新增MLP捕获已在两端真实CUDA完成，各自完整logits可由捕获运行exact复现，输入与六组MLP权重hash相同。SiLU修后第0层全部attention/MLP与残差输出exact；第1层input norm有23691个FP32值微差，仅1个在BF16 cast跨舍入边界，随后Q/K/V差异扩大。独立固定输入CUDA原型以vector4四累加器、ascending warp shuffle和rsqrtf，在28层捕获输入上mean/rsqrt/output/cast全部exact；产物 `20261020-r3/qwen-norm-kernel-diagnostic-r12/`，仅为算法诊断，不是Jittor公共修复或整模型通过。native rsqrt与共享FP32 RMSNorm provider补丁正在仓外审阅；原r7整模型logits门槛仍FAIL，未放宽容差，也未启动JT Qwen full训练。

存储先保留证据再释放系统盘副本。r12已将6个结束诊断目录18,431,409,499字节逐文件SHA256校验后转存至数据盘，原路径为可访问symlink，索引 `20261020-r1/archived-diagnostics-r12.json`。r14继续对全部结束的独立PT r10 full/resume目录执行相同校验和无进程引用检查；实时进展见 `20261020-r1/archived-completed-oracles-r14.json`，未完成前不报告最终释放量。活跃JT目录、缓存、初始化权重和其他用户进程未清理。
