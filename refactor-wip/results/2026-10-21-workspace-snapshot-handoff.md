# 2026-10-21 工作区快照与本机接续交接

## 2026-10-26 19:36 +0800 当前继续入口

Day2–5尚未完成，唯一状态源仍为[看板](../architecture/refactor-board.md)，直接证据见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。main HEAD30085673、91项原dirty保持；再次fetch目标e5cffe1a已为八相关树祖先，无需整合，未stash/reset/clean/整合2.0-refactor。以下旧窗口/PID均仅历史快照。

T5 row-mean r2已bootstrap64s、真实CUDA17pass/0skip、两fresh native各32默认3冻结回放五阶段exact/finite和独立PT原局部预算PASS，source efe22515/core6dfe0645。原r1两失败保留，新隔离修正包括实测para0 launch宽度和profiler Count聚合；未带回main。原双seedstrict771/2088 FAIL不变，受影响CPU和新默认3 step51尚未执行成功。当前串行窗口已交`20261026-t5-r2-cpu-regression-r1/execute.py`一次900s/15原测试；每次先读其root-status/receipt和实际/proc，不重复启动。step51后继仅准备于`20261026-t5-ordered-mean-r2-1854/step51-default3-prepared-r1/`，硬前置包含CPU PASS，没有full100。

性能r6 PT/JT各19update、原rawmemory/allocator/NVML/真实CUDA trace及原机械comparator均PASS；最终source/tool守卫和组空/holder核验19:35:14完成，main完整冻结已解除。原收据绑定HEAD30085673/source bdae0ec0，后续文档提交不得重绑旧结果。共享1×8单轮hot ratio1.356只为诊断，无正式三轮/代表尺寸/容量验收。GC r3私有CPU14+解析PASS，但新add/sum一阶策略负例FAIL；r4四文件静态修正/17回归已准备，排T5CPU后且仍不得直接公共接线。Vision bootstrap2在300s预算超时未eval，剩1734.724s；绝对cache路径导致重编/重链已证实，v4单次cold-aware900+3×180工具只准备。原Vision和BERT任务FAIL保持，BERT两update只离线准备，不重训。

## 2026-10-26 18:40 +0800 当前继续入口

Day2–5 尚未完成，唯一状态源仍为[看板](../architecture/refactor-board.md)，数字、身份和证据边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。主树已本地提交 `6f61c185`、未 push，91项 dirty 保留；再次 fetch 的目标仍 e5cffe1a，六相关树均已含其历史，无需 merge。Vision r2 默认3、无hook 连续61→67/save64及fresh64→67均执行成功、实际CUDA/fallback0/core exact，但原零容差769字段仍FAIL1项final/eval/logits，max_abs5.96e-8；最终40参数/40momentum及scheduler exact，history只有统计字段，没有raw-grad/每步参数证据。两artifact audit PASS，失败后无cost/重试，原五格FAIL不变。后继eval replay只离线准备，不能把短窗口写成完整模型通过。

性能small/profile r5以完整main/source/tool冻结执行：独立PT一轮全部阶段rc0；JT越过zero_grad(None)且生成10个hot provisional，随后因独立memory阶段缺NVML observations在18:38:20 rc1退出，无final row/attribution/pair结论。原r5收据/source guard已完成，main冻结解除；旧失败与cache保留，性能代理只离线定位schema/采样合同，不自动重跑。串行firstJIT窗口现交T5 row-mean v6 bootstrap（900秒bound），候选source3290d483…104be、工具manifest e80bb722…6e514；v6补worker/core/GPU与跨进程数组守卫，不改生产五文件。后续须先真实17项CUDA测试零skip、两native32回放/独立PT比较、受影响CPU再到有界默认3 step51，不能根据这段快照重复启动。每次先读 `20261026-t5-ordered-mean-1554/runs/bootstrap/attempt-01/{process,receipt}.json` 与实际 `/proc`/cache。

GC代理在新隔离r2树写真实私有CPU保存值/重算切片和正负回归，仍未获得功能执行结果，公共False/RNG/AMP/CUDA/默认fusion均OPEN；不得将旧边界VJP或语法编译代替原GC门槛。BERT15.425953905%原任务FAIL、正式性能三轮/容量/稳定窗口仍未闭环。主代理当前T5，Vision仅离线eval计划，性能仅离线NVML失败定位，GC仅编码/静态编译；首次JIT串行、各自cache隔离，无full100、无无限waiter。原17:52“未提交/待bootstrap”以下均为历史快照。

## 2026-10-26 17:52 +0800 公共优化器闭环与后继窗口

Day2–5仍未完成；最新证据以[本日结果](2026-10-26-torch-compat-release-day2-5.md)与唯一看板为准。主树HEAD37021507，目标远端再次fetch为e5cffe1a、六相关树均含其历史，无需merge，dirty/失败/cache全保留。zero_grad(None)原生两行修复已最小复现、独立PT CPU/CUDA对照及最终设备回归：core328pass/46skip/1xfail，CUDA8pass/0skip/exit0，CPU扩展34pass/1CUDA环境skip/exit0。GradScaler只改旧测试生命周期；新增测试是真实CPU/CUDA两个方法，计数门槛不变。六份既有core hash未变，所有自有优化器组退出；最终审计入口20261026-perf-1311/zero-grad-unused-fix/FINAL_RELEASE_AUDIT.json。旧7pass但gate exit1等失败保留。本批维护结果未提交，不替代模型验收。

串行firstJIT窗口现交Vision r2 CPU bootstrap，78文件/11脚本计划39f14347…49bab7d，CPU15真实测试+15格矩阵、后继默认3无hook短程全字段exact；原五格FAIL，累计余3241.1228秒不扩。T5 row-mean候选仅离线绑定通过。Qwen原False GC需要非重入原图重算，原型不能别名凑过；公共GC、CPU RNG/AMP/live-memory及CUDA未闭环。BERT原15.425953905%仍FAIL；Day4–5新small准备须在main收口后重绑，不重启旧r4/r5。每次启动重读实际process/receipt/cache/source，首次JIT串行、失败停下保留；18:08本批structure1398pass/2accelerator skip/exit0、完整source guard通过，layout/MANIFEST/相对链接/diff检查通过；Vision CPU15测试/15格矩阵均PASS，现仅释放CUDA bootstrap-model1。下一步精确partial staging/本地中文提交，无push。

## 2026-10-26 16:44 +0800 本机活跃工作入口

Day2–5继续执行且尚未完成，以[看板](../architecture/refactor-board.md)为唯一状态源，完整证据与边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。主树本地提交37021507已完成，无push；全部原有dirty和历史失败保留。Vision r3原生16pass/0skip及两新进程32×3冻结梯度/独立PT原预算通过，但尚未执行默认3无hook模型恢复与成本门禁，原五格FAIL不变。GC外部原型真实CPU解析梯度/一次重算正例和闭包反例已完成；公共模块GC、RNG/AMP/live-memory/CUDA未验，Qwen原候选门禁不转给主树。T5隔离row-mean候选仅准备，无新的默认配置通过或full100。BERT原双seed任务FAIL未变。

性能small新r3的PT执行完成，JT在首update后zero_grad(set_to_none=False)因group grads包含None而崩溃，16:41:26自然退出/source guard通过；完整main freeze37021507/c51928…099caab只用于这批旧回执，已解除供公共语义最小复现和修复。性能代理现持串行CPU→CUDA复现窗口，禁止用改objective、造pooler梯度或从性能缓存跑正确性测试绕过。Vision后继模型与GC controls均仅离线准备，T5候选首次JIT尚未排入；下一阶段必须按实际source/process/receipt重新分配窗口，不依据旧PID重启。旧r2 metadata失败与GC r1 CPU flag失败均保留，所有候选采用新身份，未覆盖旧输出。

## 2026-10-26 15:26 +0800 历史工作快照

Day2–5继续执行，尚未验收完成；本节覆盖下方历史PID、WAITING和“未启动”快照。当前唯一状态源仍为[看板](../architecture/refactor-board.md)，完整数字、源码指纹和收据哈希见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。本逻辑任务已同步目标远端e5cffe1a，为相关HEAD祖先，无需整合；未stash/reset/clean或整合2.0-refactor。主树本批验证绑定395949d7，原有dirty工作保留。

Qwen独立候选双seed full102/resume51及原strict/task/artifact gates均通过，所有正式worker自然退出，不重复训练；GC真实重算仍OPEN，源码审计和显式参数pure-function最小probe仅准备。T5第一次有界新campaign是1800s supervisor timeout（非SIGSEGV），第二次step51配对rc0，2803数组和404动态事件exact但静态mode快照不同，原FAIL保持。BERT配对初始forward/201梯度按原预算通过，只是0更新诊断；from_pretrained meta装载前置失败和r3公共constructor精确装载差异有独立收据，不替代原任务loss FAIL。

主树CUDA唯一oracle skip已另补1pass/0skip（独立PT2.6、native入口、真实CUDA/fallback0）；core/定向结构已完成，完整structure1398pass/2skip（accelerator类别、246.84s），source unchanged。本批提交仅包含四个修复/测试文件、WarpSoftmax清单三行及本批文档，简明中文本地commit、不push。混合清单其余12行、Day4–5旧报告、发布规格及其他原有dirty改动不暂存。BERT全部文件/HEAD冻结已于最终guard后解除，旧PT/JT收据继续绑定原身份，不重绑到新commit。

随后启动已准备的 `20261026-vision-native-validation-r2/`，先原85-path source下复现copy向量，再隔离95-path候选实编、native回归、两新进程32重复及独立PT；seed和新runtime核心哈希分别记录。仅已知import-only bootstrap rc3允许显式第二interpreter，保留首失败，累计70分钟上限，数值失败不重试，无full100或矩阵。GC先CPU解析梯度/closure-negative，再决定RNG/AMP/存储与CUDA。性能v3仅准备，须等新冻结和独立performance cache窗口；当前共享资源不满足真实容量边界，旧r5不重启，不无限等待。

## 2026-10-26 14:03 +0800 活跃工作增量

Qwen 两更新 smoke 已完成且执行/生命周期通过（原rc3 FAIL保留，report SHA `57a0f0491a5794d95f96615920cc668d8f645a1cb27a392e3f4103fe93925de6`），真实CUDA/fallback0、112 adapter tensors实际变化，独立87文件候选冻结不变。full102/resume51待T5首次JIT窗口后协调，不能由smoke代替。配置GC仍为true，但shim明确直接执行wrapped function、不重计算、不省activation，至少一格GC功能验收仍未闭环。

T5 `20261026-t5-step51-fixed-1320/` continuous于14:01:40启动，最近核验PID3077840/start_ticks160371398，使用修复树85文件冻结和修后warm cache。先检查实时身份、execution/status与checkpoint/trace；若未通过不启动resume，严禁用r66/r70产物。该目录已备新cohort全部输入/RNG/梯度/event诊断比较器，旧reference/signature comparator保持独立，原771/2088 FAIL不变。主树已补WarpSoftmax结构引用清单（6项AST契约通过），core/受影响pytest仅已准备独立warm cache副本，尚未运行。Qwen与main源码并不等价，不能把独立候选完整前向/训练结果当main发布通过。Vision原生局部确定性实现正在独立设计/候选准备，尚无编译或修复验收。

## 2026-10-26 13:31 +0800 当前续接入口

本机任务已继续，仍未满足 Day2–5 退出条件。以下早晨 r5 WAITING、旧 HEAD/PID、暂停与未启动记录均为历史快照；r5 原始收据已确认 08:09:08 +0800 超时（4024 样本、无正式测量），旧 PID 已不存在。远端重新 fetch 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为四个工作树 HEAD 的祖先，无需整合；主树开工 HEAD `395949d7`，保留所有 dirty/untracked，未 stash/reset/clean 或整合 2.0-refactor。

新进展：Qwen 公共 softmax 修复已在冻结独立候选上通过原无 hook 完整 BF16 前向门槛（全部29,171,712 logits exact、loss合原预算、真实CUDA/fallback0），7项定向回归通过，已按基线逐字节核对后将两文件改动带回主树；这只关闭前向前置项。新 LoRA smoke2 正在独立缓存冷编译，首次rc3重载前置退出已保留，新label有界续试尚无训练结果。T5真实旧core的空线程范围最小复现达到同一崩溃偏移，修复后9项CUDA回归通过；新step51 continuous/resume准备在 `20261026-t5-step51-fixed-1320/`，必须先核验实际进程/产物，禁止用r66/r70不存在的checkpoint恢复。Vision广播反向默认level3非确定已最小复现，level4同输入稳定但仍只是诊断，没有全局改默认或原模型恢复PASS。BERT原任务gap继续FAIL，旧11缺失是跨任务原件，不代表r10 BERT/FP16六checkpoint审计缺失。

当前唯一看板和详细证据见[看板](../architecture/refactor-board.md)、[本日结果](2026-10-26-torch-compat-release-day2-5.md)。仓外新增证据根为 `20261026-continuation-1257/`、`20261026-t5-reduce-repro-1315/`、`20261026-t5-reduce-fix-1310/`、`20261026-qwen-causality-1300/`、`20261026-vision-1305/`、`20261026-vision-para-ab-1316/`、`20261026-perf-1311/`，均在原 release-day23 主题下。性能旧r3仍仅共享负载诊断，近容量未覆盖且旧计时工具有额外缺口；新工具合同/PT smoke和未执行v3不构成配对性能验收。不新建无限等待或覆写旧产物。

- 状态：2026-10-21 用户已明确恢复本机开发与实验。Day 2–3 仍未通过；当前进程与产物状态以下述恢复核对为准，Day 4–7 尚未验收。
- Owner：cgq / Torch 兼容层维护者。
- 源码基线：主工作区 `2690b03e3ff489134c093eaf5c4a8fd2af53fd5c`；Day 2–3 两候选 `9e0e187bb013e71d85111acfd668e46aeff9f9d2` 加各自保留的本地修改。
- 同步：目标 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 已在上述三个 HEAD 历史内。本次不融合主分支，不整合 `origin/2.0-refactor`。
- 复查条件：候选源码、依赖、工作树、进程或验收协议变化。暂停记录保留为历史凭据，活跃状态以实际进程及产物为准。

## 本机恢复核对

2026-10-22 再次 fetch 并核对 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`；该提交仍是主工作树与两个验收 worktree HEAD 的祖先，无需整合。本轮没有合并或整合 `origin/2.0-refactor`。主树 HEAD `2690b03e3ff489134c093eaf5c4a8fd2af53fd5c` 保留94个 dirty/untracked 路径；原候选与修复树 HEAD 均为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，分别保留51与82个路径。三树源码、index 与本地改动均未覆盖。

旧暂停 PID 2363937 当前不存在；本轮没有重复启动full训练。T5 seed821/823 strict verify仍各FAIL 771/2088项；r31首microbatch配对捕获200/208事件输出不同，最早可见为encoder block 0 FFN dropout，actual functional dropout均为training=True，但trace缺dropout输入和wi/activation输出，根因仍待定位。Qwen r18完整BF16 logits仍FAIL；r22 PT/JT层捕获确认q_proj和q_norm输入/权重exact，最早差异在q_norm输出的4/393216个BF16边界值，随后完整logits仍FAIL。Vision r18的`torch.__file__`入口失败保留；r22b四格full/resume和artifact audit均完成，实际CUDA/fallback0；三格same-runtime exact分别FAIL 104/239/148项，第四格same-runtime exact通过但cross-runtime strict FAIL177项，验收仍FAIL。BERT BF16双seed loss包络15.425953905%超过原15%门槛，继续FAIL。

存储迁移审计见 `$JITTOR_LAB_ROOT/_state/storage-cleanup/20261021-r3/archive-migration-r1.json`。已校验的完成态 native DDP 归档保留在 `/data6`，主机原路径为 symlink；本轮未删除验收日志、checkpoint 或 warm cache。

## 当前恢复诊断入口

T5 r31配对诊断完成并FAIL，报告在仓外 `20261020-r3/t5-resume-diagnostic-r29/paired-trace-r31.json`。其capture和checkpoint身份只绑定r28/r29新cohort，不能替代或覆盖旧r4/r6恢复证据；下一诊断要新增只读输入/中间层capture并使用全新输出目录，不重复full100。

Qwen `qwen-layer-capture-r22/` 的PT/JT capture和离线比较已完成，manifest SHA `ce6872e7e6fc2c21008d488c058af7f610a303d1a9b74a77ec5a62a2e3759ea0`，比较器SHA `607d32e29b355e1494ea1c533f16cbfb70304a30755a19f20612db3ff5b6c435`。实际同一A800 UUID、`cuda:0`、fallback0；q_proj与q_norm输入/权重exact，首差异在q_norm输出4/393216元素，max abs0.01953125，完整logits仍超原预算。不要以诊断capture关闭r18 full forward门槛，不启动LoRA full/resume。

Vision新增 `vision-recovery-audit-r22/recovery-audit-r2.json`（SHA `d8cd2393ffb31cb74bbdef2143bce6b4ad67172e9b48bc912df922e8a27005d2`）恢复核验r19b CNN FP32 seed821，26/26证据检查通过、同runtime exact通过、跨独立PT strict仍FAIL160/1206；原receipt保留。新cohort `vision-deterministic-cuda-r22b/` plan SHA `e509680f901d4ad991a9ff8367c1b1e849f641d336e6380d80fcdfe7a01eeaf8`。四格full/resume与artifact audit均完成：same-runtime exact差异数为vit-fp32-821 104、cnn-bf16-821 239、vit-bf16-821 148、cnn-fp32-823 0；cnn-fp32-823跨独立PT strict仍FAIL177项。四格实际设备`cuda:0`、fallback0；阈值保持原值且所有失败保留。BERT保留原best-checkpoint loss gap15.425953905% FAIL。详细证据见[本轮结果](2026-10-20-torch-compat-release-day23.md#2026-10-22-r31r22b-增量)。

用户已授权空间不足时清理lab无用文件；最近根盘余182GiB、数据盘余6TiB，当前run根与新增cache在数据盘，暂未删除文件。保留checkpoint、失败证据、日志、已绑定cache和所有worktree修改；空间仍足够完成本轮有界诊断。

## 新机器从哪里开始

推荐 clone `wip/cgq-day23-fix-20261021`。它包含当前单卡适配修复、测试及交接，属于开发快照，不能视作发布候选已验收。完整主工作区保存在 `wip/cgq-workspace-20261021`，其中原生 DDP 等保留工作与修复候选并未相互融合；需要这些变更时先比较两个分支，不把一个分支当成包含另一个分支的所有修改。

每个快照以对应工作树 HEAD 为父提交，保存该工作树全部 tracked/untracked 非忽略文件的当前内容及删除；不更改现有工作树 HEAD、index 或源码，不 stash/reset/clean。不同工作树的冲突版本分别保留。本机索引与各路径内容有独立核对记录，远端提交 SHA 以最终 `git ls-remote` 为准。

| 快照分支 | 保存的来源 | 用途 |
| --- | --- | --- |
| `wip/cgq-day23-fix-20261021` | `release/torch-compat-day23-fix-20261021` | 推荐新机器适配底座；含最新 rsqrt/RMSNorm/cuDNN 等修复 |
| `wip/cgq-workspace-20261021` | `integrate/cgq-transformers-native-multicard-target` | 主工作区所有保留改动 |
| `wip/cgq-day23-original-20261021` | `release/torch-compat-day23-20261020` | 冻结 HF r10 训练与旧 Vision/Qwen 数值证据对应源码 |
| `wip/cgq-peft-base-20261021` | detached `b67236dc` | 历史 PEFT 基线的两处结构修改 |
| `wip/cgq-accelerate-c5-20261021` | `accel-c5-distributed` | 历史 Accelerate 隔离树的两处修改 |
| `wip/cgq-ddp-integration-20261021` | `verify/cgq-ddp-on-2.0-latest` | 历史 DDP/FSDP2 整合树的保留修改 |
| `wip/cgq-fsdp2-rehearsal-20261021` | detached `22c098dc` | 历史 FSDP2 rehearsal 的保留修改与删除 |

仓外 `$JITTOR_LAB_ROOT` 的模型、数据、缓存、原始日志和 checkpoint 不进入源码 Git。这些实验产物继续保存在本机；clone 能取得源码、测试、公开文档与本交接，不能取得本机缓存或复用本机绝对路径脚本。新机器按环境文档建立自己的 lab 与环境，下载大模型/数据遵循当地用户规则。Day 4–7 留在本机，新机器具体任务由用户另行指定，不自动领取。

## 必须阅读的入口

按顺序阅读根 `AGENTS.md`、[协作手册](../../agent/manuals/collaboration.md)、[开发上下文 skill](../../agent/skills/jittor-dev-context/SKILL.md)、[当前上下文索引](../../agent/manuals/project-context.md)，再看[新机器 onboarding](../../docs/development/2.0-refactor-onboarding.md)、[环境规则](../../agent/manuals/environment.md)、[源码架构](../../docs/development/source-architecture.md)、[Torch 兼容原则](../architecture/torch-compatibility-principles.md)。执行下游适配前阅读[下游适配 skill](../../agent/skills/downstream-library-adaptation/SKILL.md)。整改状态以[看板](../architecture/refactor-board.md)为唯一源；本文件不另建任务看板。

工作目录外建立 `$JITTOR_LAB_ROOT`，并设置独立 `HOME`、`JITTOR_HOME`、`TMPDIR` 与编译缓存。首次 JIT 串行；并行任务隔离缓存；独立 PyTorch oracle 不安装 shim；声称 CUDA 时必须实际 device 正确且 fallback=0。不要复用其他机器的编译产物。

## Day 2–3 暂停时的实测结论

详细协议与历史失败见[第二三天报告](2026-10-20-torch-compat-release-day23.md)和[发布收口计划](../testing/2026-10-20-torch-compat-release-gate.md)。下面列出本机恢复所需的最新增量；它们均绑定各自运行快照，不能宣称已在主工作区或快照提交上重新执行。

| 项目 | 最新完成结果 | 未完成边界 |
| --- | --- | --- |
| 修复树维护门禁 | focused CPU 93、CUDA 106 全通过，fallback=0；core 330 passed/48 skipped/1 xfailed；最终 structure 1398 passed/2 skipped，layout 通过 | rsqrt 新头文件 MANIFEST 与 complex 错误计数的两处登记已修正，旧失败记录保留；未跑 clean wheel 或完整 smoke |
| rsqrt / 28层 RMSNorm | 独立 PT/JT × CPU/CUDA 八个定向 oracle 均 PASS | 不能代替完整 Qwen/LoRA 验收 |
| BERT BF16 | r10 双 seed full100 与 checkpoint50 新进程恢复 exact，实际 CUDA/fallback0 | 任务 best-model loss 均值差 15.425953905% > 原 15%，包络仍 FAIL；JT 观测 loss 更低不代表改进 |
| 默认 BERT FP16 | full/resume 与1648项严格比较通过；三 checkpoint 原始2412数值数组全部 finite，每份1241状态签名匹配；7次真实更新和1次合法overflow skip | 仅为已测短协议，不外推更长训练或分布式 |
| T5 BF16 seed821 | full100/resume均完成，真实CUDA/fallback0；三checkpoint共3084数值数组finite且保存签名匹配；reload/cache/generation完成 | 原零容差恢复失败771/2088项，step51首microbatch梯度范数已不同，原因尚未定位 |
| Qwen BF16 r18 | 新独立 PT oracle exact 匹配历史参考；JT 无hook完整前向实际CUDA/fallback0完成，loss在预算内 | 全部29171712 logits中7015339项超原 `.005/.03` 预算；有效位置maxabs .3125，padding2.25，完整前向仍FAIL；未启动JT LoRA full/resume |
| Vision | 原r11五格full/resume完成但exact恢复均FAIL；cuDNN定向修复与重复前向/双梯度回归通过 | r18新五格在模型训练前因仓外wrapper读取合法缺失的 `torch.__file__` 而全部退出；属于测量入口错误，不能计作新模型数值失败或通过；r19修正准备未完成 |

公开 reload 原合同是逐元素 `allclose(atol=2e-5, rtol=2e-4)`。旧 r14 审计误加 `max_abs==0`；新 r17 审计已恢复原合同，原数据与旧记录保留。未保存完整 reload logits，不能离线重算 relative 项，也不称作 exact reload。这不影响 T5 训练恢复的独立真实 FAIL。共享 GPU 的墙钟和占用不提供发布速度/显存结论。

## 本机运行和防重复启动

产物根为未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261020-r3/`，原资产与部分旧运行位于同主题 `20261020-r1/`。暂停时 T5 seed823 full 仍在运行：campaign supervisor PID2363937已单独SIGSTOP，训练PID2492435及其timeout/acceptance子进程保持运行，以便当前full阶段正常写完证据；后续resume/verify不会由暂停的supervisor继续启动。PID可能失效或被复用，恢复时必须核对 `/proc/<pid>/cmdline` 和进程starttime，不能凭旧PID直接发信号。

暂停凭据位于 `../push-handoff-20261021-r1/pause-supervisor.json`，包含真实command与starttime。`hf-resume-acceptance-r10/jittor-campaign-status-r10.json` 因supervisor暂停可能仍写RUNNING；判断当前full是否结束应同时读 `jittor-t5-823-full-r10.execution.json`、阶段report和实际进程。确认本机用户要求恢复后，再对同一supervisor使用SIGCONT或按已完成stage恢复调度，禁止重复启动。

HF运行仍使用原候选和旧模型cache `transformers_compat_jittor`。Qwen r18、Vision r18与全部r16/r17门禁已退出，CUDA unit cache目前无本任务运行。所有agent已停止新增工作；仓外r19诊断/入口修正若有半成品只能按文件实际状态检查，不能视作准备或验证通过。

关键凭据目录：`sync-resume-20261021-r19.json`、`gate-fix-focused-{cpu,cuda}-r16/`、`gate-fix-final-cpu-r16/`、`gate-fix-structure-cpu-r17/`、`fix-validation-r16/`、`structure-followup-r17/`、`hf-final-audit-r17/`、`qwen-full-fix-cuda-r18/`、`vision-deterministic-cuda-r18/`。HF终审manifest SHA256为 `fa2f71550272e5c4848520e5694bd3f936d1f708f28cf37c91a666b1f3803247`。

已完成存储转存：14个结束的PT oracle目录共44452046764字节逐文件SHA256校验后移到本机数据盘，原路径symlink可访问，索引为 `20261020-r1/archived-completed-oracles-r14.json`。不要重复转存、清理活跃cache或删除证据。

## 2026-10-22 r32/r23 增量

T5 r32 新 cohort 已完成 continuous/resume step51 capture；两侧输入与起始参数签名彼此 exact、实际 CUDA/fallback=0，但 continuous 顶层为 eval、resume 为 train，首个差异为 functional dropout 输入；正式 comparator 因与旧 r6 signature 不同而 FAIL。旧 seed821/823 strict FAIL 771/2088 不变。Qwen r23 最小 CUDA replay 使用冻结 r22 输入与 manifest，pow2、输入和权重 exact，首差异为 mean reduction（1046/3072，max abs 约 1.49e-8），随后 rsqrt/cast 翻转 4 项；完整 logits 仍 FAIL，未启动 LoRA full/resume。

本轮磁盘复核：根盘约 179 GiB 可用，/data6 约 6 TiB 可用；r23 cache 与产物均在 /data6。未删除 checkpoint、失败证据、日志、已绑定 cache 或 worktree 修改。

## 本机恢复顺序

先核对进程和当前full产物，保留所有失败与原冻结点；再检查r19工具准备，修正Vision测量入口后创建新cohort，不覆盖r18。Qwen完整前向仍FAIL，应先定位最早剩余差异；T5从step51首microbatch定位，避免盲目重跑全部训练。BERT任务门槛继续保留。完成修复后再按同输入/权重/seed/阈值复验，当前不称Day2–3完成。Day4–7、公开Torch/HF单机DDP、配对性能/峰值显存、安装包与稳定性仍留本机后续执行。

## 2026-10-23 Day2–5 暂停续接

用户于 2026-10-23 要求暂停本轮工作，仅更新本交接；本轮没有启动训练、benchmark 或测试，没有清理文件，也没有修改源码或看板。下一次会话目标为先收口 Day2–3，再完成 Day4–5。Day6–7 不属于本次续接目标。Day2–3 的 T5、Qwen、Vision、BERT 失败边界以上述报告及 2026-10-22 r32/r23 增量为准，不因性能诊断结果改变。

### Day4–5 已有 BERT 诊断产物

`$JITTOR_LAB_ROOT/_state/release-day23/20261020-r3/day45-bert-pilot-r3/` 保存固定权重、FP32、batch 8、sequence 256 的 BERT CUDA forward/backward 入口诊断。`day45-bert-protocol-r1/` 使用修复树 `release/torch-compat-day23-fix-20261021`，HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，运行时源码状态 SHA256 为 `8bbc50428e4e0616e15f584fb65c391217fba5348daada3675aa39db0ea3ac7e`，目标为 A800 UUID `GPU-1f10b7e3-00af-a430-1ad8-1c7d7b85e7ac`、可见设备 7。其 10 个 PT 与 10 个 JT 新进程样本报告热态计时中位数分别为 `0.0813136 s` 与 `0.0957233 s`，JT/PT 中位数比 `1.17721`，p95 比值 `1.02550`，进程墙钟中位数比 `1.39078`；PT/JT 计时 CV 分别为 `26.87%` 与 `16.48%`。详见该目录的 `summary.json`、`samples.json` 和 `manifest.json`。

该结果不是 Day4–5 发布性能验收：计时范围仅含同步 forward+backward，没有 optimizer update；protocol 复用了 `day45-bert-pilot-r3/jittor-home` correctness cache；没有 allocator/NVML 峰值显存记录，也没有接近显存容量边界的配对格。运行时共享 GPU 负载很高，因此这些数字仅作诊断，不能形成速度结论或与 correctness pilot 绑定的性能结论。不要覆盖或删除这些产物；新测量应使用独立性能缓存和新目录，保留原始失败/诊断证据。

### 2026-10-23 本机只读状态

主工作区分支 `integrate/cgq-transformers-native-multicard-target`，HEAD `2690b03e3ff489134c093eaf5c4a8fd2af53fd5c`，dirty/untracked 共 94 条；原候选分支 `release/torch-compat-day23-20261020`，HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，共 51 条；修复树分支 `release/torch-compat-day23-fix-20261021`，HEAD 相同，共 82 条。所有现有修改继续保留。此处没有重新 fetch；此前交接记录的 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 仍须在新会话按仓库规则重新 fetch 核对。不要自动整合 `origin/2.0-refactor`。

按 `ps` 的任务关键词扫描，没有发现匹配 `day45`、`release-day23`、`ecosystem_runner`、`jittor_core` 或 BERT train/bench 的活动进程；这不是完整 GPU 进程归属审计。当前 `/` 约有 175 GiB 可用，`/data6` 约有 6.0 TiB 可用；目标 UUID 当前显存占用约 14.7 GiB、GPU 利用率约 98%，因此当前快照显示卡仍忙，但本轮没有任务进程。未做空间清理；只有空间实际不足时再清理经核验无用且无活跃 holder 的 lab 文件，checkpoint、失败证据、日志、绑定 cache 与 worktree 修改必须保留。

修复树的 `refactor-wip/architecture/refactor-board.md` 当前仍写 Day4–7 尚未启动；该状态未反映上述 pilot/protocol 诊断产物。本轮按用户要求只更新交接，未改看板。恢复工作时由看板维护者依据原始 manifest/receipt 将诊断产物与未通过的正式验收区分后更新唯一状态源，不得把现有诊断改写为通过。

### 新会话执行入口

在 `/home/cgq/jittor` 原工作区继续，不重新 clone。先依次阅读根 `AGENTS.md`、`agent/manuals/collaboration.md`、`agent/skills/jittor-dev-context/SKILL.md`、`agent/manuals/project-context.md`、本交接、发布验收规格、Day2–3 证据报告、`refactor-wip/architecture/refactor-board.md` 和下游适配 skill；随后按当前实际状态重新核对全部 worktree、目标远端 SHA、进程身份/阶段 receipt、GPU、缓存 holder 与磁盘，再决定恢复或续跑。保留所有 dirty/untracked 修改；禁止 stash/reset/clean、远端快照覆盖本地工作或自动整合 `origin/2.0-refactor`。本轮没有授权变更其他机器。

Day2–3 优先项：T5 从 step51 的 dropout 输入及 encoder block 0 FFN `wi`/activation 边界定位 continuous/resume 差异，不重复 full100；Qwen 从 r23 mean reduction 的最早差异继续，完整无 hook BF16 logits 达原门槛前不启动 LoRA full/resume；Vision 检查 r19 半成品，修正 `torch.__file__` 读取入口后创建新 cohort 并保留 r18 失败；BERT BF16 双 seed loss gap `15.425953905%` 仍超过原 15% 门槛。复用未变化的定向门禁和原始 checkpoint audit，不放宽阈值、不换 seed、不删比较字段。

Day4–5 按发布规格做真实训练更新的 PT/JT 配对测量：代表尺寸和接近容量边界配置使用匹配权重、输入、device、精度、optimizer 与运行时设置；性能 cache 独立于 correctness/unittest cache，首次 JIT 串行；预热后至少 10 个样本、至少 3 轮 PT/JT 交替，测量边界同步，报告 median、p95、离散度及加载/首步/热态拆分；分别测 allocator live/reserved peak 和 NVML 进程峰值，记录真实 CUDA device 与 Jittor fallback=0。运行期间冻结对应源码/HEAD。若共享 GPU 干扰统计，结果标为诊断并保留门槛未通过，不把 GPU utilization 当作启动条件或性能结论。定位到常用路径退化后先处理，再复验受影响正确性门禁和性能格。最终更新唯一看板和日期化结果，并保留 Day2–3 各自独立的 FAIL 状态。


## 2026-10-25 Day2–5续接

本轮重新 fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81；主树 HEAD 为 95d0600ba7d4888082b3a0f71bac9f2df8acd5dc，两个验收 worktree HEAD 均为 9e0e187bb013e71d85111acfd668e46aeff9f9d2，dirty/untracked 修改均保留，未整合 origin/2.0-refactor。当前复核根盘约余 108 GiB、/data6 约余 6.0 TiB，未清理任何 checkpoint、失败证据、日志、绑定 cache 或 worktree 修改。

Qwen r24 已在 GPU UUID GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068 串行完成 PT/JT reduction replay，双方 actual_device=cuda:0、fallback_count=0。冻结 r22 输入和 manifest 不变；pow2 exact，但 native mean、sum/N、CUDA scalar 除法及 epsilon 变体均保留 1046/3072 项差异，max abs 1.4901161193847656e-8。完整 BF16 logits 仍 FAIL，未启动 LoRA full/resume。

T5 r36 使用新 cache 和 GPU UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5，continuous 与 fresh-process resume 均完成 step51 capture，sample_ids=[14,136]、actual_device=cuda:0、fallback=0，97个冻结源未变。正式 comparator 仍 FAIL：新 pre-forward 参数签名与保留 r6 reference 不同；该 fail-closed 检查保留。独立 trace 对拍显示 continuous 顶层/encoder.block.0 为 eval、resume 为 train，最早配对张量差异在 encoder.block.0.layer.1.DenseReluDense.wi 输入，1449项、max abs 1.430511474609375e-06；此前 dropout/attention 事件 exact。该结果只定位 mode/trajectory 边界，不关闭历史 771/2088 strict FAIL。

Vision r18 缺失合法 torch.__file__ 的失败、r19 getattr 修正及 r21/r22b 失败均保留；BERT BF16 双 seed loss gap 15.425953905% 仍超过 15% 门槛。Day4–5 的 day45-bert-release-r3 已完成真实 AdamW update、独立性能 cache、代表尺寸三轮及合法近容量两轮配对，包含 median/p95/离散度、加载/首步/热态、allocator/NVML peak 和 fallback=0；但 summary.json.status=DIAGNOSTIC_SHARED_GPU，不能视为发布速度/显存通过。完整结果见 [2026-10-25结果报告](2026-10-25-torch-compat-release-day45.md)。


### 2026-10-25 T5 r36完成

r36 continuous 与 fresh-process resume 均完成 step51 capture，sample_ids=[14,136]、actual_device=cuda:0、fallback=0，97个冻结源未变。正式 comparator 仍 FAIL：新 pre-forward 参数签名与保留 r6 reference 不同；该 fail-closed 检查保留。独立 trace 对拍显示 continuous 顶层/encoder.block.0 为 eval、resume 为 train，最早配对张量差异在 encoder.block.0.layer.1.DenseReluDense.wi 输入，1449项、max abs 1.430511474609375e-06；此前 dropout/attention 事件 exact。该结果只定位 mode/trajectory 边界，不关闭历史 771/2088 strict FAIL，也未升级为发布通过。

### 2026-10-25 Vision r25 smoke与续跑

r25 重新冻结五个原始 Vision case，focused CPU/CUDA gate identity 与计划一致，并使用独立 JITTOR_HOME/cache。首个 cnn-fp32-821-smoke execution receipt 为 PASS（returncode 0），artifact audit 全部通过：实际 cuda:0、fallback_count=0、完成2次真实参数更新、62个可训练参数均变化、保存/重载 logits exact。该结果仅证明修正后的读取入口和 CUDA smoke 可执行；stage all 随后在同一 cohort 中完成。r25 wrapper 的 Jittor cache 独立，HOME/TMPDIR/XDG_CACHE_HOME 指向保留的 r19 warm-unit 目录；该边界已在完整 receipt/log 审计中确认并保留。

随后 r25 stage all 已收口为 COMPLETED_WITH_FAILURES_RETAINED。五个 case 的 smoke/full/resume 与 artifact audit 均完成，实际 cuda:0、fallback_count=0；cnn-fp32-821 same-runtime exact PASS 但 cross-runtime strict FAIL 160 项，vit-fp32-821 same-runtime exact FAIL 100 项，cnn-bf16-821 same-runtime exact FAIL 245 项，vit-bf16-821 same-runtime exact FAIL 143 项，cnn-fp32-823 same-runtime exact PASS 但 cross-runtime strict FAIL 177 项。所有 receipt、comparison、checkpoint 和日志保留，Vision 仍 FAIL，不能把真实 CUDA 更新写成发布通过。

### 2026-10-25 T5 r38 dynamic trace 增量

r38 使用全新独立 cohort/cache，continuous 与 fresh-process resume 均完成 step51 capture，sample_ids='[14,136]'、actual_device='cuda:0'、fallback_count=0，97 个冻结源未变。两侧各有 404 个 trace 事件；hook 时 training/is_train/is_training 和 functional dropout training=true 均一致，事件顺序、kind、shape、device 一致；前 21 个归档张量（包括 encoder block 0 dropout 与 FFN wi 输入/输出）exact，首个归档张量差异为 event 21 的 encoder.block.1.layer.1.DenseReluDense.wi 输入，1377 项、max abs '3.5762786865234375e-07'。静态 543 模块表在 capture 后为 continuous=eval、resume=train，只能作为 capture 后状态记录，不能证明前向时 mode 不同。正式 comparator 仍因新 pre-forward 参数签名与保留 r6 reference 不同而 FAIL；历史严格恢复 FAIL 771/2088 保持。resume 日志的 tied-weight 缺失 key（encoder.embed_tokens.weight、decoder.embed_tokens.weight、lm_head.weight）如实保留。该诊断仅缩小了首个张量差异边界，不关闭 T5 或发布门槛。


### 2026-10-25 r4 bounded wait

A new /data6/cgq/jittor-lab/_state/release-day23/20261020-r4/day45-bert-release-r4/ was created without replacing r3. Its manifest and scripts bind repair worktree HEAD 9e0e187bb013e71d85111acfd668e46aeff9f9d2 and the independent performance cache. The waiter requires target GPU UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5 to remain at utilization <=10 percent and free memory >=60000 MiB for five consecutive five-second samples. Current free memory is 70498 MiB, utilization 100 percent, streak 0, and no r4 measurement process has started.


Latest r4 live poll: waiter PID 1746490 remains alive with no launch receipt; target GPU utilization is 100 percent, free memory 41199 MiB, and qualifying streak 0. Do not restart while this PID is live.


### 2026-10-25 r4 wait terminal state

The r4 waiter ended with `WAIT_TIMEOUT_NO_RELEASE_WINDOW` at 2026-10-24T21:56:34.266419+00:00 UTC after 1800 seconds. No launch receipt or measurement child was created; utilization stayed at 100 percent, and the source HEAD/dirty SHA remained unchanged. Day4-5 is still not a release pass.

### 2026-10-25 Qwen r26 reduction diagnostic

r26 reused frozen r22 input/manifest in a new output/cache. PT/JT both ran actual CUDA 0 with fallback_count=0 and unchanged candidate HEAD/status. Pow2 was exact; variants retained 1038–1046 mismatches, while four-chunk summation retained 662. The full no-hook BF16 logits gate remains FAIL; no LoRA full/resume was launched. Artifacts: `/data6/cgq/jittor-lab/_state/release-day23/20261020-r3/qwen-rmsnorm-reduction-r26/`.

## 2026-10-26 Day2–5续接

本轮重新核对主工作区与两个验收 worktree：主树 integrate/cgq-transformers-native-multicard-target@95d0600ba7d4888082b3a0f71bac9f2df8acd5dc，dirty/untracked 95 项；原验收树 release/torch-compat-day23-20261020@9e0e187bb013e71d85111acfd668e46aeff9f9d2，51 项；修复树 release/torch-compat-day23-fix-20261021@9e0e187bb013e71d85111acfd668e46aeff9f9d2，83 项。重新 fetch 的 origin/cgq_transformers 为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81；未 stash/reset/clean，未整合 origin/2.0-refactor。当前 / 可用约 101 GiB，/data6 可用约 6.0 TiB；未清理文件。

Qwen r44 在 /data6/cgq/jittor-lab/_state/release-day23/20261026-r44/qwen-full-nohook-r1/ 完成 PT/JT 原始无 hook BF16 初始完整前向。两端输入、样本、权重和冻结 source manifest 对齐，PyTorch 2.6.0+cu124 与 Jittor shim 侧都为实际 cuda:0、fallback=0。execution receipt 为 FAIL_RETAINED，PT/JT worker returncode 均为 0，full_training_launched=false；比较器只在 logits 键失败。原始 atol=0.005, rtol=0.03 下 7,015,339/29,171,712 个 logits 超预算，max abs 2.25；有效位置 2,756,985/22,030,720 个超预算，max abs 0.3125。loss 差 0.0038883686 在冻结预算内。修复 worktree HEAD 与 77 个冻结 source 文件 hash 均与本次运行一致。完整无 hook logits 仍 FAIL，因此没有启动 LoRA full/resume。比较文件 SHA256 为 d604c30238a3a6c27d2adf5a283ed38af05d0396599088dd3019d46cf60eb770，执行 receipt SHA256 为 eadd2e696d142817bc357b8249319f7c109f5de39ee7224e947fd3f97a34deee。

Qwen r46 在冻结 r22 输入 `[2,96,16,128]` 上继续定位 r23 mean 首差异。依据 PyTorch 2.6 `Reduce.cuh`/`MeanOps`，每个 lane 顺序累加 `lane+0/32/64/96`，再按 warp offset 1/2/4/8/16 合并并除以 128。PT 与 Jittor shim 在真实 CUDA `cuda:0` 上显式计算该树，均与 PT 原生 mean 3072/3072 exact；当前 Jittor 原生 mean 仍有 1046/3072 项差异、max abs `1.4901161193847656e-8`。两边 fallback_count=0，冻结 input SHA 为 `f2df7b930b0f2227dbea25b0a95bd3bd6bc2dfe811a328d0fe7389c0e089cd13`，candidate HEAD/status hash 为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2` / `a061f1cd85aa91037f50720ce2a50a5211843f16feeed7809871ae724398c41e`。这是 reduction-only diagnostic，不是源码修复或全模型通过证据；r44 logits FAIL 仍保留，未启动 LoRA。结果与数组在 `/data6/cgq/jittor-lab/_state/release-day23/20261026-r46/qwen-reduction-tree-r1/`。

Qwen r50 继续作为当前 unhooked full-logits release gate：原始 `atol=0.005, rtol=0.03` 下 6,690,070/29,171,712 项失败，最大绝对差 1.953125；未启动 LoRA。r54 在冻结候选 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`、85 个 dirty source 文件 fingerprint `d8b57cb5da41cbbf1a59c1310cc1204d79a0b9394aa66b4f1774cdec0bd927dd` 上完成 PT/JT CUDA capture，reference input/logits exact、fallback=0。source-formula replay 的 `position_ids` exact，`inv_freq[37]` 一项差异（PyTorch `0.0003398208355065435`，Jittor `0.00033982080640271306`），随后 `freqs` 81/6144、`sin` 162/12288 项不同；`cos` exact。首个 public module 差异是 RoPE sin，162/12288 项、max abs `3.725290298461914e-9`；layer 2 `self_attn.o_proj` 有 59/196608 个 BF16 项不同、max abs `0.0009765625`。64 项独立 CUDA probe 中候选 `pow`/倒数与 host `math.pow` float32 rounding 均 exact，而独立 PyTorch 分别有 4/64 与 3/64 项差异；这是 backend 浮点差异证据，不等于修复或将 Jittor 判错。r54 comparator SHA256 `022b806e18f8bb0bbc9c70758b72ca228fd3eddfbb128a6e779281f328f08ff6`，产物在 `/data6/cgq/jittor-lab/_state/release-day23/20261026-r54/qwen-rope-replay-r1/`。r50 gate 仍 FAIL，源码未变、未启动 LoRA。

Day2–3 其余状态未变：T5 r36/r38 只把 step51 差异缩小到 mode/FFN 轨迹边界，正式 comparator 与历史双 seed strict 恢复仍 FAIL 771/2088；不重复 full100。Vision 已从 r19 的 wrapper 入口修复推进到 r25 五格 full/resume；所有 real CUDA/fallback=0 与真实更新证据保留，但 same-runtime 或 cross-runtime comparison 仍有失败。BERT BF16 双 seed loss gap 15.425953905% 继续超过 15% 门槛。

Day4–5 的 r3 代表尺寸三轮 JT/PT median ratio 为 1.409、1.510、1.341，near-capacity 格只有两轮（0.746、1.132）；实际 AdamW 更新、allocator/NVML peak、cuda:0 与 fallback=0 均有记录，但共享 GPU 负载使 summary 状态为 DIAGNOSTIC_SHARED_GPU，且 near-capacity 少于要求的三轮。r4 等待 1800 秒后以 WAIT_TIMEOUT_NO_RELEASE_WINDOW 结束，没有启动任何测量进程。r3 与 r4 均保留。

新的 Day4–5 waiter 位于 /data6/cgq/jittor-lab/_state/release-day23/20261026-r45/day45-bert-release-r5/，复用验过 hash 的测量脚本，独立 perf-cache/ 与新输出目录。修复树绑定 HEAD 9e0e187bb013e71d85111acfd668e46aeff9f9d2、dirty source SHA 4f28ab7c26e2ede3239cbad0ae0336c92d492dcfcf87d792c342850314625cee、权重 SHA 86fa7113367fd0d231a76571ca1b73a1b24166af33558fee960636ce56337ed7。waiter PID 2616242 启动时间为 2026-10-26 02:09:02 +0800；设置 6 小时上限，要求同一 GPU 连续 5 个 5 秒样本满足 utilization ≤10%、free memory ≥60000 MiB。02:34 +0800 最近核对仍为 WAITING，state 有 120 个采样点，最后样本 free 50422 MiB、utilization 99%，没有 launch/completion receipt。队列中每格计划 10 个热态样本、预热 4 次、PT/JT 交替三轮，near-capacity 格也要求三轮；首次 JIT 串行。后续必须重新读取 `/proc`、receipt 与 GPU，不把当前 waiter PID 或 wait_state 当作永不过期凭据。

当前没有 Day2–5 训练/验证进程；r44 与 r46 cache 的 `jittor.lock` 均无 holder。唯一本任务活动进程是 r5 waiter；不发信号给其他用户任务。共享负载下 r3 的速度差异只能作为退化调查线索，不能据此直接归因或声称已经修复常用路径。

2026-10-26 04:35 +0800 复核：r54 capture/comparator 与 structure pytest 均已退出，r52 cache `jittor.lock` 无 holder；任务没有运行中的训练/测量进程。PID 2616242 仍是启动于 02:09:02 +0800 的 r5 `wait_and_launch.py`，状态 `WAITING`，最新样本 GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5 free 49,409 MiB、utilization 100%、连续合格样本 0；没有 launch/completion receipt。不要重启或向该 waiter 发信号；后续恢复仍需重新核对 live process、GPU、state 与 receipts。

2026-10-26 04:54 +0800 增量：主树当前 `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c`，dirty/untracked 96 项，含本轮三处文档更新；原验收树与修复树仍分别为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`、51/83 项，Qwen 诊断树同 HEAD、85 项。重新 fetch 的 `origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为主 HEAD 祖先；未 stash/reset/clean，未整合 `origin/2.0-refactor`。`/` 可用 99 GiB，`/data6` 可用 6.0 TiB；没有清理文件。

Qwen r56 在冻结 r54 source identity 上将 PyTorch 捕获的 `inv_freq` 送入 Jittor CUDA replay，`freqs/cos/sin` 逐位 exact，fallback=0。r58 无 hook BF16 控制前向替换同一 buffer，source fingerprint、输入和实际 CUDA 均通过核对，forward hook=0、fallback=0；控制 probe SHA256 与 r50 原始 Jittor probe 完全相同，原 logits gate 仍有 6,690,070/29,171,712 项超 `atol=0.005, rtol=0.03`。没有公共源码修改，也没有启动 LoRA。r55 与 r57 的 harness preparation 错误分别保留在各自 cohort receipt 中，不计作模型结果。

Day2–3 其余证据没有变化：T5 r36/r38 与历史 strict recovery 仍 FAIL 771/2088，不重跑 full100；Vision r19 是 preflight 停止、没有模型启动，r25 五格比较仍 FAIL；BERT 双 seed loss gap 15.425953905% 超原 15% 门槛，原始 checkpoint audit 保持原件、未重跑。

r5 waiter PID 2616242/启动时间 02:09:02 +0800 经 `/proc` 与命令行复核仍存活；04:53:39 最新状态为 WAITING，目标 UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` free 49,409 MiB、utilization 100%、合格 streak=0，无 launch/measurement/completion receipt。r5 source HEAD/status SHA 未变。GPU 共享负载尚不满足性能窗口；r3 只保留诊断结论，未声称 Day4–5 通过。

2026-10-26 06:07 +0800 增量：重新核对主树为 `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c`，99 项 dirty/untracked；原验收树和修复树均 HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，分别 51/83 项；Qwen 树同 HEAD、85 项。`origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 已 fetch 且为主 HEAD 祖先，无需合并；未整合 `origin/2.0-refactor`。无运行中的 pytest/run_test_suite；`/` 约 99 GiB 可用，`/data6` 约 6.0 TiB。

T5 r65 level 3 对同一冻结输入重复 32 次出现 29 个 variance hash、13 个 output hash；level 4 的 reduction/output 32 次逐位稳定，但与 continuous/resume FFN wi 输入仍分别差 4,227/5,604 项，最大差 `9.5367431640625e-7`。r66 continuous PID 2788927 于 05:56:01 +0800 在 GPU UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d` 启动，冻结检查通过（HEAD `9e0e187…`、51 项、fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`），当前仍在首次编译 CUDA 算子；它是 r65 JIT home 上唯一 `jittor.lock` holder，continuous status 尚为 `started`，尚无 step51 trace/checkpoint receipt。不要重复启动 continuous；成功后只按 r66 脚本启动同 GPU/cache/参数的 fresh resume，再运行原 comparator，保留其历史签名 FAIL。

Qwen r61 在冻结 r60 attention-softmax 输入上，PT/JT 各自复现同 runtime 捕获，但交叉仍有 56,251/294,912 项不同（最大差 `3.5762786865234375e-7`），actual CUDA、fallback=0、85-file source identity 不变。r67 分阶段 probe 已准备并做语法检查，未启动；等待 r66 首次 JIT 退出后再串行运行。r50/r58 full logits 继续 FAIL，禁止启动 LoRA。

Vision r19 state 为 `STOPPED_PREFLIGHT_NO_NEW_LAUNCH`，没有模型运行；r25 已完成五格 smoke/full/resume 和 artifact audit，但比较失败仍保留。BERT 双 seed loss gap 15.425953905% 继续超过 15% 门槛，原 checkpoint audit 未重跑。Day4–5 r5 waiter PID 2616242 于 02:09:02 +0800 启动，截至 06:12 仍 WAITING，GPU0 最新 free 49,409 MiB、utilization 100%、合格 streak=0，无 launch/measurement/completion receipt；其原始 6 小时窗口、性能缓存与 source identity 不变。共享 GPU 下 r3 仍只作诊断，Day4–5 未通过。

2026-10-26 06:12 +0800 增量：T5 r66 continuous PID 2788927 已退出，session 返回 signal 11，Jittor 核心 PC 符号为 `plan_reduce_thread_order`。仅生成 `initial-probe.npz`（SHA256 `2608a604101809e8098bb9de9b9e3b6dfe53a16fff438dea7bf84e41f60b62a8`），没有 checkpoint-50、step51 trace 或 resume prerequisite；candidate HEAD/status 冻结校验仍通过。失败收据为 `/data6/cgq/jittor-lab/_state/release-day23/20261026-r66/t5-step51-para4-r1/continuous-execution-failure.json`，cache 和 initial probe 均保留，不从该 attempt 启动 resume。

Qwen r67 staged softmax probe 复用 r60 PT 输入与 r53 frozen manifest，独立 cache 位于 `/data6/cgq/jittor-lab/_state/release-day23/20261026-r67/qwen-softmax-stages-r1/`。PyTorch side 于 06:09:33 UTC 在 actual `cuda:0` 精确复现捕获输出，85-file source fingerprint 不变；Jittor PID 2803257 于 06:09:50 +0800 开始该 cache 的首次 core 编译，06:12 时显示 118/228，尚无 stage report。r50/r58 无 hook full logits 仍 FAIL，未启动 LoRA。

2026-10-26 06:24 +0800 续接：再次检查主树与两个验收 worktree，主树 `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c` 有 96 个 dirty/untracked 路径；原验收树与修复树均为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，分别有 51/83 项。Qwen 诊断树同 HEAD、有 85 项。`git fetch origin cgq_transformers` 后 SHA 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是上述 HEAD 的祖先；没有 stash/reset/clean，也没有整合 `origin/2.0-refactor`。磁盘 `/` 余 98 GiB、`/data6` 余 6.0 TiB。无测试进程，GPU 当前均处于 99–100% utilization，空闲显存约 23.8–58.9 GiB。

Qwen r67 Jittor stage replay 已完成，actual `cuda:0`、fallback=0、冻结 85-file identity 不变。maximum/shifted/exponentials exact；首差异为 denominator 1290/3072、max abs `1.9073486328125e-6`。r68 的显式 96-wide lane-tree reduction 在 PT 与 JT 都 exact 对齐 PT native sum；Jittor native sum 仍有 1290/3072 项差异。r69 在新输出目录顺序复用 r67 cache，actual `cuda:0`、fallback=0、source identity 未变；offline 数组比较确认 r69 native primitive 与 r67 Jittor primitive exact，tree-normalized output 与 PT explicit primitive exact；PT primitive 与 PT fused softmax 自身有 53,939/294,912 项差异。r69 的 `native_fused_*` 字段无效，因为它对已 exponentiate 的输入调用了 softmax；完整报告与脚本保留但不作为模型证据。该结果支持 native reduction 顺序是 explicit primitive 首差异原因，但不解释 full logits FAIL。r50/r58 无 hook BF16 logits 仍 FAIL，LoRA full/resume 未启动。

T5 r66 signal 11 收据仍只有 initial probe，没有 step51 trace/checkpoint，不得从该 attempt 启动 resume。Vision r25 五格 comparison FAIL，BERT BF16 双 seed 15.425953905% gap 仍超过 15%。Day4–5 r5 waiter PID 2616242 经 `/proc` 核对仍是原 `wait_and_launch.py`，目标 UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` 最新样本 49,409 MiB free、100% utilization，连续合格数为 0，无 launch/measurement/completion receipt；它保持存活至原 08:09 +0800 六小时期限。r67/r68 cache 无 holder，r69 已退出。Day4–5 仍未通过，r3 共享负载结果只作诊断。

### 2026-10-26 06:47 +0800 增量续接

重新核对主工作区与原验收、修复和 Qwen 诊断 worktree：主树 `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c` 有 96 个 dirty/untracked 项；三个验收 worktree 都在 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，分别有 51/83/85 项。`git fetch origin cgq_transformers` 得到 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，该 SHA 是四个 HEAD 的祖先；未 stash/reset/clean、未整合 `origin/2.0-refactor`。无 repository pytest/run_test_suite 进程；`/` 余 98 GiB、`/data6` 余 6.0 TiB，没有清理文件。

T5 r70 continuous capture 仍由 PID 2815893（06:28:59 +0800）运行，子进程 PID 2816187（06:29:10 +0800）。冻结校验为 `FROZEN_SOURCE_PASS`，候选 HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`、51 项、fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`。独立 JIT cache 的 `jittor.lock` 由这两个进程持有。初始 probe SHA256 与失败的 r66 一致（`2608a604101809e8098bb9de9b9e3b6dfe53a16fff438dea7bf84e41f60b62a8`）；截至 06:47 仍在 UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d` 上编译 float32 `setitem` kernel，状态为 `started`，无 step51 trace/checkpoint/completion receipt。不要重复启动或在有效 checkpoint/trace 出现前启动 resume。

r5 waiter PID 2616242 经 `/proc` 与命令行确认仍为 02:09:02 +0800 启动的原任务。最新样本 06:47:20 +0800 为目标 UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` free 49,409 MiB、utilization 100%、合格 streak 0/5；其固定门槛是 util <=10%、free >=60,000 MiB、连续 5 个五秒样本，六小时期限至 08:09:02 +0800。没有 launch/measurement/terminal receipt。Day2–3 状态不变：Qwen r50/r58 logits FAIL且未启动 LoRA，T5 r66失败保留、r70未到step51，Vision r18失败保留且r25 FAIL，BERT 15.425953905%仍超15%；Day4–5没有新的发布有效测量，r3只作共享负载诊断。

2026-10-26 06:49 +0800 终态增量：r70 continuous PID 2815893（06:28:59 启动，child 2816187）已退出，执行 session 返回 1，收到 signal 11。`jittor_core.so+0x4c47e0` 经 `addr2line` 确认为 `jittor::plan_reduce_thread_order(...)`，无行号信息；与 r66 报告的崩溃函数相同，但根因尚未定位。冻结 source check 仍 PASS（HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`、51 项、fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`）。只有 initial probe，SHA256 与 r66 完全相同；`continuous-capture/status.json` 仍为 `started`，无 step51 trace、checkpoint-50、fallback 或完成收据。r70 独立 cache 已无 holder，cache 与新增 `continuous-execution-failure.json` 保留；不得从该运行启动 resume，也不应未定位崩溃便重复同一 capture。

r5 waiter PID 2616242 仍为 02:09:02 +0800 启动的原进程，06:49:08 +0800 最新状态 WAITING，目标卡 free 49,409 MiB、utilization 100%、qualifying streak 0/5，没有 wait/launch/measurement receipt。固定窗口门槛与 08:09:02 截止时间不变。此时无其他 task-related 训练或验证进程；`/` 余 98 GiB、`/data6` 余 6.0 TiB。Qwen full logits FAIL且未启动 LoRA；Vision r18失败保留、r25 FAIL；BERT gap 15.425953905% 仍超过15%；Day4–5没有发布有效测量。

2026-10-26 07:09 +0800 验证增量：`bash tools/check_repo_layout.sh` 通过；Torch-mode `JITTOR_TORCH_SHIM=1` 下 `tests/structure` 为 1396 passed、4 skipped（1400 collected、1398 executed，650.77 秒）。第一次导入受 Anaconda `libstdc++` 缺少 `GLIBCXX_3.4.30` 影响，随后显式 preload 系统 `libstdc++` 并用独立 JIT/TMP 路径重跑成功。该结构门禁不改变 Torch compatibility 发布状态。当前 r5 waiter PID 2616242 仍 WAITING，07:09:34 +0800 最新样本为 free 49,409 MiB、utilization 100%、qualifying streak 0/5，无 receipt；它保持原截止时间 08:09:02 +0800。无其他 task pytest 进程。
