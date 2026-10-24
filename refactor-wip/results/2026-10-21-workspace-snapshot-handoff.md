# 2026-10-21 工作区快照与本机接续交接

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

本轮重新 fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81；主树与两个验收 worktree 的 dirty/untracked 修改均保留，未整合 origin/2.0-refactor。根盘约余 113 GiB、/data6 约余 6.0 TiB，未清理任何 checkpoint、失败证据、日志、绑定 cache 或 worktree 修改。

Qwen r24 已在 GPU UUID GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068 串行完成 PT/JT reduction replay，双方 actual_device=cuda:0、fallback_count=0。冻结 r22 输入和 manifest 不变；pow2 exact，但 native mean、sum/N、CUDA scalar 除法及 epsilon 变体均保留 1046/3072 项差异，max abs 1.4901161193847656e-8。完整 BF16 logits 仍 FAIL，未启动 LoRA full/resume。

T5 r36 使用新 cache 和 GPU UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5，continuous 与 fresh-process resume 均完成 step51 capture，sample_ids=[14,136]、actual_device=cuda:0、fallback=0，97个冻结源未变。正式 comparator 仍 FAIL：新 pre-forward 参数签名与保留 r6 reference 不同；该 fail-closed 检查保留。独立 trace 对拍显示 continuous 顶层/encoder.block.0 为 eval、resume 为 train，最早配对张量差异在 encoder.block.0.layer.1.DenseReluDense.wi 输入，1449项、max abs 1.430511474609375e-06；此前 dropout/attention 事件 exact。该结果只定位 mode/trajectory 边界，不关闭历史 771/2088 strict FAIL。

Vision r18 缺失合法 torch.__file__ 的失败、r19 getattr 修正及 r21/r22b 失败均保留；BERT BF16 双 seed loss gap 15.425953905% 仍超过 15% 门槛。Day4–5 的 day45-bert-release-r3 已完成真实 AdamW update、独立性能 cache、代表尺寸三轮及合法近容量两轮配对，包含 median/p95/离散度、加载/首步/热态、allocator/NVML peak 和 fallback=0；但 summary.json.status=DIAGNOSTIC_SHARED_GPU，不能视为发布速度/显存通过。完整结果见 [2026-10-25结果报告](2026-10-25-torch-compat-release-day45.md)。


### 2026-10-25 T5 r36完成

r36 continuous 与 fresh-process resume 均完成 step51 capture，sample_ids=[14,136]、actual_device=cuda:0、fallback=0，97个冻结源未变。正式 comparator 仍 FAIL：新 pre-forward 参数签名与保留 r6 reference 不同；该 fail-closed 检查保留。独立 trace 对拍显示 continuous 顶层/encoder.block.0 为 eval、resume 为 train，最早配对张量差异在 encoder.block.0.layer.1.DenseReluDense.wi 输入，1449项、max abs 1.430511474609375e-06；此前 dropout/attention 事件 exact。该结果只定位 mode/trajectory 边界，不关闭历史 771/2088 strict FAIL，也未升级为发布通过。
