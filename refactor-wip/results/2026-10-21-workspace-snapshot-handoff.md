# 2026-10-21 工作区快照与本机接续交接

- 状态：用户要求暂停新增修改与实验启动，先保存交接并推送工作区快照；Day 2–3 尚未通过。
- Owner：cgq / Torch 兼容层维护者。
- 源码基线：主工作区 `2690b03e3ff489134c093eaf5c4a8fd2af53fd5c`；Day 2–3 两候选 `9e0e187bb013e71d85111acfd668e46aeff9f9d2` 加各自保留的本地修改。
- 同步：目标 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 已在上述三个 HEAD 历史内。本次不融合主分支，不整合 `origin/2.0-refactor`。
- 复查条件：候选源码、依赖、工作树、进程或验收协议变化。本文记录暂停时刻，活跃状态以实际进程及产物为准。

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

## 本机恢复顺序

先核对进程和当前full产物，保留所有失败与原冻结点；再检查r19工具准备，修正Vision测量入口后创建新cohort，不覆盖r18。Qwen完整前向仍FAIL，应先定位最早剩余差异；T5从step51首microbatch定位，避免盲目重跑全部训练。BERT任务门槛继续保留。完成修复后再按同输入/权重/seed/阈值复验，当前不称Day2–3完成。Day4–7、公开Torch/HF单机DDP、配对性能/峰值显存、安装包与稳定性仍留本机后续执行。
