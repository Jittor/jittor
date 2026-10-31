## 2026-11-01 06:55 +0800 handoff：wheel 差异域已归类，逐项审阅待继续

主树 HEAD c0db719e9，origin ahead/behind 140/0；102 项既有 dirty/untracked porcelain SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211 未变。基线重建到候选的 wheel 差异为 69 adds、301 changes、2 removals；旧固定 archive 的比较数有 302 changes，差额是 WHEEL 元数据哈希。新归类确认变更集中于 jittor/src 与 jittor/backends，CUDA/ACL、NN、profiling、AMP/optimizer、distributed 均在范围。CPU runtime 复验通过，但 accelerator skipped；ACL 未构建，DDP 两 rank 未运行；allowlist 仍 0 项批准。分类、现有测试映射和限制见 Day2–5 报告顶部。

### 下一步

继续逐项审阅 wheel migration 清单并把功能文件关联到测试和 device/dtype/layout 边界；保持 allowlist 不变，直到每项有证据。完成多 rank DDP checkpoint/resume 与 held-out 任务质量门。Day4–5 等 GPU2 UUID/GRES 与设备 cgroup 独占租约，随后按三轮交替、每轮每侧十个热样本、分位统计和内存峰值/OOM/recompute protocol 验收。

## 2026-11-01 06:37 +0800 handoff：wheel 基线源码已恢复；逐项迁移审阅继续

工作区 /home/cgq/jittor，branch integrate/cgq-transformers-native-multicard-target，HEAD 72a08b2237ff4164fe5f5786a28cb70fb5d2cf6e；origin e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 139/0。编辑前 102 项既有 dirty/untracked porcelain SHA256 为 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，index 为空。

本轮没有运行 GPU 测试或性能任务。R3 wheel 候选的 1095 个源码成员与当前工作树一致；固定 wheel 基线的 1028 个源码成员与 Git 提交 f66e2458a748a5814f28ca4705753ec29b330311 一致。离线重建 wheel 1033 个成员，和固定基线仅 WHEEL、RECORD 生成哈希不同。迁移差异仍为 69 additions、302 fixed-baseline content changes、2 removals；对比重建 wheel 则为 69 additions、301 项成员内容变化（含包元数据）和 2 removals，差一项来自原 wheel 的 WHEEL 生成器元数据。Nox 仍 rc=1，baseline allowlist 没有更改。证据根目录：/data6/cgq/jittor-lab/_state/release-day23/20261101-wheel-baseline-rebuild-r5-prepared/。

两项 ACL 脚本删除在 e7bf82046 有 native Sigmoid、SigmoidBackward 与 Transpose 替代注册，并保留前向/梯度测试；ACL 未构建，因此只记为静态迁移复核。后续先逐域审阅 CUDA RNG/graph/step capture、AMP/optimizer/DDP、profiling/memory、ACL 和核心 C++ 内容变化，并把每项绑定现有测试或明确列为待测；再决定是否更新迁移 allowlist。不要以当前源映射结果替代功能验收。

GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 仍由 yqa PID 3590512 占用约 8640 MiB，计算模式 Default，最新采样 util 约 87%。普通占卡进程无法排除其他用户，未启动 burner。待管理员/调度器分配 UUID/GRES 与设备 cgroup 独占租约后，再执行 Day4–5 正式 PT/JT 对照、近容量/OOM 恢复和真实 activation recompute checkpoint on/off。多 rank DDP、held-out generation、clean CUDA 安装等门仍开放，整体 release Not complete。


## 2026-11-01 06:21 +0800 handoff：声明支持矩阵与 dtype 复验通过；GPU2 仍共享

继续工作目录 `/home/cgq/jittor`，分支 `integrate/cgq-transformers-native-multicard-target`，基线 HEAD `09c4989e90c5cb7bc97c551cba98ad9c0891869d`；origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind `138/0`。本轮在 runtime source fingerprint `ae682fe34dec0cf62e74f5dd3546c8c77dd7d97f23f03f7a4d4b7d11558f2499` 上修改两个测试文件、更新本 handoff 与唯一看板/Day2–5 报告；更新前 index 为空，原有 dirty/untracked 文件均保留。

支持矩阵复验 **54 passed、0 skipped**，其中 CPU 35 passed，CUDA 38 passed；28 个单元明确未验证，包括缺 CuPy 的 `numpy_code` 和 no-probe 项，其他未构建 backend 未覆盖。独立 dtype 覆盖在真实 CUDA 上 **6 passed、0 skipped**。完整执行记录、失败原件、ABI 修复和文件 SHA256 已记入 Day2–5 结果报告；证据根目录为 `/data6/cgq/jittor-lab/_state/release-day23/20261101-current-main-native-contract-matrix-gpu1-r1-prepared/` 与 `.../20261101-current-main-native-contract-matrix-gpu1-r2-prepared/`。矩阵测试分类器修正不改 runtime 行为；支持矩阵并未因此证明所有声明算子均通过。

GPU2 当前仍为共享 `Default`：yqa PID `3590512` 的 compute-app 用量为 8,640 MiB（整卡占用 8,651 MiB），最新采样利用率约 79%。此账户没有 scheduler 命令或免密 sudo 权限，故无法建立系统级独占；没有启动 burner，也没有更改 GPU2 状态。性能测试要保证中途无其他 CUDA client，需要管理员/调度器按 UUID/GRES 与设备 cgroup 预留，并由对照测试自身在整段运行期间持有 context。Day4–5 正式性能/显存仍未验收。

### 下一步

1. 继续处理声明矩阵的 28 个未验证项和未构建后端范围，同时推进 public Torch DDP 多 rank checkpoint/resume、wheel baseline migration 审批及真正 held-out generation 质量评估。
2. GPU2 等现有 yqa 任务结束后，请管理员/调度器建立独占租约；不要用普通 burner 冒充系统预留。取得租约后按三轮以上 PT/JT 交替、每轮每侧十个热样本，完整测 load/首步/热态分位数、离散度、吞吐、allocator/NVML/RSS 峰值、近容量/OOM 恢复及真实 activation recompute。
3. R14 Qwen 两 epoch 训练和 checkpoint resume 的证据边界不变；严格数值对齐、实用训练、支持矩阵、Day4–5 性能/显存、DDP、打包迁移、held-out 任务质量分别判断。整体 Day2–5 release 保持 **Not complete**。

## 2026-11-01 05:47 +0800 handoff：结构门通过；继续推进功能与正式性能发布门

主工作区 /home/cgq/jittor，branch integrate/cgq-transformers-native-multicard-target，HEAD 8c5fe3a6d6346ebab9b237b97fa2f038d8de80ac；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 137/0。此前 dirty/untracked 文件保持原样；本轮两处新增工作区改动只在 tests/structure/test_cleanup_structure.py 与 tests/structure/optim/test_optim_structure.py，没有碰其余 dirty 源码或测试。执行前 status 快照 SHA256 4d5f4225f1f0951d1e4ef57cc1cc7ab184fa54400532b67bf1c1d7f61d9f27f1，index 为空。

布局检查通过；正确 cwd 下完整 tests/structure 为 1395 passed、6 skipped、0 failed（155.90s）。日志位于 /data6/cgq/jittor-lab/_state/release-day23/20261101-package-gates-current-main-r3-prepared/structure-r7/，SHA256 43cc9d1bc7f92cb9bdff45390cc3eee19a409abbba5a65a250ecc6dce891cafe。旧错误 cwd 的结构失败原件留存，不覆盖。该运行是 CPU 结构门，不是 CUDA 或性能证据。

### 继续顺序

1. 维护当前支持矩阵：列明支持的 API、device/dtype/layout、autograd/alias/storage 与 fallback 行为，以及每项实际验证身份和明确不支持边界；保留旧 strict comparator 的失败身份。
2. 在当前候选补齐 held-out generation/任务指标，并审查 Qwen resume archive 的模型、optimizer、scheduler 是否绑定同一原子 checkpoint；已有 train-split capped generation 不可作 held-out 证据。
3. 完成公开 Torch DDP/Trainer/Accelerate 的 2/4-rank 实际路径，覆盖梯度累积、尾批、eval、checkpoint 与 fresh-process resume，记录各 rank 一致性和 max-rank 资源。
4. 审阅 wheel/sdist baseline 的 69 additions、302 changes、2 removals，完成未批准内容分类；在仓库外 clean 环境验证 core/compat/adapters 安装和承诺的 CUDA 路径。
5. 取得 GPU2 UUID/GRES/cgroup 独占租约后，按 Day4–5 正式协议测量；至少三轮 PT/JT 交替、每轮每侧十个热样本，报告 load/首步/热态 median/p95/离散度、吞吐和 allocator/NVML/RSS 峰值，补近容量/OOM恢复及真实 activation recompute 的 checkpoint on/off 测量。GPU2 共享时不启动 burner 或正式性能运行，旧共享负载数据保持诊断用途。

实用训练、结构门、性能/显存、DDP、打包迁移和 held-out 任务质量仍各自独立判断；整体 Day2–5 release Not complete。

## 2026-11-01 05:34 +0800 最终门禁复验

本轮文档更新后布局门禁通过；结构门禁复跑仍为 1391 passed、6 skipped、4 failed，157.73s，rc=1，日志 SHA256 5525c8586ec63067fdddcf6514267bbe0ab5756e07fb13c4ca83df625030ba12。失败分类见 Day2–5 报告。当前工作树是原有 102 项 dirty/untracked 加三份状态文档，index 空；只提交本任务三份文档，不 push。

## 2026-11-01 05:30 +0800 handoff：当前主树 package CPU smoke 通过；结构门仍失败

主仓库 /home/cgq/jittor，branch integrate/cgq-transformers-native-multicard-target，HEAD e46586b26c16c646465b766d8c31dc159aecd00b；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 136/0。开始本轮文档更新前 102 项既有 dirty/untracked SHA256 为 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，暂存区为空；没有 push。

当前源码指纹 ae682fe34dec0cf62e74f5dd3546c8c77dd7d97f23f03f7a4d4b7d11558f2499 的 Qwen3-0.6B LoRA seed821 两 epoch CUDA 实训完成 102 次更新，0 skip、0 fallback、梯度有限，epoch loss 1.205599→1.153534；checkpoint-51 新进程恢复到 102，51 次恢复更新，930 状态检查和 337 项 tensor device map 通过。17 条生成只是 train-split 48-token capped diagnostic，不能作为 held-out 质量通过。证据根目录 /data6/cgq/jittor-lab/_state/release-day23/20261101-qwen-main-r3-practical-r1-prepared/。

当前主树 R3 wheel + sdist 构建、两 wheel 内容审计、独立 CPU 安装 selftest 和 Torch shim CPU backward smoke 均通过；wheel baseline compare 仍有 69 adds、302 changes、2 removals 未批准。仓库布局检查通过；structure gate 1391 passed、6 skipped、4 failed：重复测试定义 device 未审阅、benchmarks.operators 缺失、optim package whitelist 未包含 _state_placement.py，以及外部 Motion324 缺 omegaconf。最终 log SHA256 23dd1cf3860297ebd03973672f334d1d020dbffae7dc959fee831bf509449cf2。

GPU2 仍由 yqa PID 3590512 共享使用约 8.65 GiB，util 约 82%；GPU4/GPU9 也有其他任务。Slurm A800 节点 down，无独占租约；两 rank DDP harness 已修正 fingerprint 但未启动，Day4–5 正式性能/显存 protocol 未执行。下一步先复核结构失败与支持矩阵、推进 DDP 与 wheel baseline 审阅；获取独占 GPU2 后再做稳定配对性能/显存、容量恢复及 checkpoint on/off。整体 Day2–5 release **Not complete**。

## 2026-11-01 03:25 +0800 handoff：主树 Trainer CPU 新进程恢复通过

主树基线 HEAD 262bed6a79cb8c54800a583612e9cbb71686a5dd 已包含 R3 copy_ placement 修复。当前修改了集成 test harness 的可选 cache override，以复用已验证 JITTOR_HOME，未改训练产品代码。完整 stateful Trainer CPU 集成 6 passed、0 skipped；PT/JT 各自连续 4 步与 2+2 新进程恢复的 272 项轨迹严格一致，0 mismatch；JT 恢复 step 2 到 step 4、fallback=0。它是合成 tiny-BERT/LoRA 的 CPU 恢复证据，不代表 held-out 指标、CUDA 或性能/显存验收。

完整产物：/data6/cgq/jittor-lab/_state/release-day23/20261101-trainer-copy-placement-r3-prepared/main-integration-r3/；训练轨迹与报告：/data6/cgq/jittor-lab/_state/trainer-resume/cpu-jwe57ymu/。

GPU2 于 03:25 +0800 仍为共享 Default，utilization 80%，yqa PID 3590512 使用约 8.6 GiB。Slurm 节点仍 down、cgq association 缺失、设备 cgroup 限制未启用；不启动 burner，不把共享卡测量用于 Day4–5。

### 下一步

1. 继续完成声明支持矩阵与 core wheel baseline migration 审阅，并审视 Trainer 多 rank checkpoint 的明确支持边界。
2. 为当前主树/source 身份补真实设备的训练/任务指标证据；保持 GPU2 不动，直到管理员恢复 a800 节点、配置有效 Slurm association 与按 UUID/GRES 的设备独占租约。
3. 独占租约到位后执行 Day4–5 配对性能/显存、近容量/OOM 恢复及 checkpoint on/off 验收。整体 Day2–5 release **Not complete**。
## 2026-11-01 03:14 +0800 handoff：主树 R3 CPU copy_ 回归通过；GPU2 需要管理员独占

工作基线 /home/cgq/jittor HEAD b588c421d59f4d8cf3b4183a15af8c00559ce2c6；本轮只改 src/core/var_holder.cc 和新增的 compat/tests/torch/test_copy_preserving_alias_placement.py，主树 CPU 定向回归 1 passed、0 skipped、fallback=0。主树完整 Trainer 集成及 CUDA copy 尚未验证；候选上的 6 项 CPU Trainer 集成证据仍保留候选身份，不外推。

GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 在 03:13:54 +0800 为 Default，共享利用率 81%、显存用量 8,651 MiB，yqa PID 3590512 占 8,640 MiB。Slurm 的 X12DPG-OA6 节点目前 down，cgq 无 Slurm account association，配置没有 task/cgroup 设备限制；srun --test-only 因账号/分区 association 错误失败。没有写或运行占卡 burner，也没有可靠手段让普通进程阻止其他用户使用该卡。

### 下一步

1. 请管理员恢复 a800 节点、给 cgq 建立有效 association，并提供按 GPU2 UUID/GRES 和设备 cgroup 约束的独占租约；收到前保持 GPU2 不动，不将共享测量当性能验收。
2. 在主树 CPU 路径继续验证 R3 对 Trainer 的集成，再推进 DDP 多 rank checkpoint、支持矩阵、held-out 任务指标与 wheel baseline migration review。
3. 独占租约就绪后，再执行三轮以上交替、每轮每侧至少十个热态样本的性能/显存协议，并补近容量/OOM 恢复、checkpoint on/off。Day2–5 overall release **Not complete**。
## 2026-11-01 02:54 +0800 handoff：隔离 R3 Trainer 候选通过；GPU2 无独占租约

主仓库 /home/cgq/jittor HEAD 751b274adbe0158784a7e3bcb3babdf63be14455，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 133/0；102 项既有 dirty/untracked 状态 SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211 未变，index 原先为空。主源码未动。

隔离候选 /data6/cgq/jittor-lab/_state/release-day23/20261101-trainer-copy-placement-r3-prepared/source 指纹 1731758d3e08197d05a8ff6c89214871e0a1d3839bf6ac9720b4112762d2769c，patch SHA256 d8bd4df49febe9859124f741757049779b2868246979051a1b13251e97f150af。placement 定向用例 1 passed；Trainer CPU 集成 6 passed，PT/JT 各自的连续 4 步和 step 2 新进程恢复到 step 4 都是 272 项严格相同、0 mismatch，JT fallback=0。证据与哈希见 Day2–5 结果报告和 /data6/cgq/jittor-lab/_state/release-day23/20261101-trainer-copy-placement-r3-prepared/integration-r2-review.json。该修正尚未迁入主树；合成 CPU fixture 不证明 held-out 任务、CUDA、显存或 wheel。

GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 于 02:51 +0800 为 Default、99% 利用率、18,081 MiB 已用；外部 yqa 与 cgq V2M4 compute 进程同时存在。没有管理员/调度器独占租约，本轮没有运行占卡程序。要阻止其他作业，需要管理员/调度器提供 UUID/GRES/cgroup 独占预约；普通 burner 做不到这一点。

后续继续非独占 Day2–5 发布门，并审阅是否将 R3 修复合入最终候选。Day4–5 等独占窗口后按固定配置完成至少三轮 PT/JT 交替、每轮每侧十个热样本、分位数/离散度/吞吐与 allocator/NVML 峰值、容量/OOM 恢复及 checkpoint on/off 对照。整体 release 仍 **Not complete**。
## 2026-11-01 02:24 +0800 handoff：当前 wheel CPU smoke 通过；Trainer stateful resume 在权重 copy_ 前失败

主仓库 `/home/cgq/jittor` HEAD `ffa41a6e68837dd09f9d781d5b21c7b37a901857`，分支 `integrate/cgq-transformers-native-multicard-target`；`origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 132/0。102 项折叠 porcelain 哈希 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211` 未变，暂存区为空；`-uall` 展开计数为 105。仅更新三份状态文档，不改其他 dirty/untracked 文件，不 push。

当前源码 fingerprint `1fbdc0fa223b9fb3e016a9aca72f7b95cc6c93f3e10076f3b743d6035e15c343`。CPU-only wheel install smoke 与 37 项 packaging tests 通过，sdist/core/compat/adapter wheel 结构审计完成；clean wheel/sdist 发布门仍因 core wheel 相对 final baseline 的 70 additions、302 内容变更、3 removals 且没有批准 allowlist 而保持开放。细节与包哈希见 Day2–5 结果报告的新章节。

当前源码 Trainer integration 在 PT 侧证明 4 步连续和 2+2 步 fresh-process resume 的 272 项轨迹逐位一致，0 mismatch、fallback=0；整个测试集为 5 passed/1 failed。Jittor 侧在 Trainer 启动训练前加载 LoRA 初始权重时，发现 source explicit CPU placement 与目标 FollowRuntime placement 不同，触发 `Tensor.copy_` placement guard；0 training update。review 收据 SHA256 `04be93c1bbc75cc1e95491184340501fbd535e4f80afbcd346e2357a6b07d0d5`，目录 `/data6/cgq/jittor-lab/_state/release-day23/20261101-trainer-resume-current-main-r2-prepared/`。下一步在隔离候选中修复并加回归，再重跑该功能门；DDP Trainer 多 rank checkpoint 仍未验证。

GPU2 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0` 在 2026-10-31 18:23 UTC 为 Default、100% 利用、18,081 MiB used；yqa PID 3590512 与另一项 V2M4 PID 3812018 共用此卡。没有管理员独占租约，未运行占卡程序。正式性能需要取得 GPU2 UUID/GRES/cgroup 独占时段，随后完成至少三轮 PT/JT 交替、每侧每轮十个热样本的 latency/throughput/memory 统计，并做近容量/OOM 恢复及 checkpoint on/off 对照。已有 R26 median ratio 1.595 仍未过 1.10 时延门。

后续继续补支持矩阵、held-out 任务指标、统一候选训练与新进程恢复、DDP 多 rank checkpoint、序列化互操作契约、clean 安装与 baseline migration review。R14 训练结果仍绑定历史源码与配置；整体 Day2–5 与 Day4–5 release 保持 **Not complete**。完整新证据与失败原件哈希见 [Day2–5 结果报告](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-11-01 01:28 +0800 handoff：R13g 序列化全量回归通过；R14 optimizer 转换通过

主仓库 `/home/cgq/jittor` HEAD `8597b7de66e465a62a1ac65f40cbaad78a93bd2f`，分支 `integrate/cgq-transformers-native-multicard-target`；fetch 后 origin SHA `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 131/0。文档编辑前 102 项既有 dirty/untracked porcelain SHA256 仍为 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`，index 为空；本轮只更新唯一看板、本结果报告和本 handoff，不改主源码，不 push。

R13e/R13f/R13g 绑定隔离候选 `/data6/cgq/jittor-lab/worktrees/torch-compat-fill-zero-runtime-grad-r12-cuda-init-20261101`，HEAD `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，fingerprint `1167ac8ccc2e731fed7b16842c9643b16a60106b97a3fda27224e597971421ce`。R13g 两个序列化测试文件 50 passed、0 skipped、fallback=0，使用 CUDA-enabled Jittor 与 GPU4 UUID `GPU-8bf405a2-a3dd-e203-8dd9-691e9f7b3530`；完整结果与哈希在 Day2–5 报告本节和 `/data6/cgq/jittor-lab/_state/release-day23/20261101-r13g-serialize-full-r13e-prepared/`。

R13f 从 R14 checkpoint-51 旧 JT optimizer 文件读取 336 tensors/136 scalars/121 containers，经 shim 保存为标准 ZIP 后由 stock PyTorch 2.6 读取并完整摘要匹配。它不改变原 checkpoint，也不证明 stock PyTorch 能直接读旧 portable-pickle。失败的 R13c、R13d 和 R13f attempt-01 均保留，详见结果报告。

本轮状态点：GPU2 与 GPU4 都是 Default 且处于共享使用，GPU2 尚无管理员独占租约；根盘可用约 73 GiB，/data6 约 5.5 TiB。R13g 是功能回归，不纳入 Day4–5 性能/显存成绩。Day4–5 仍须至少三轮交替、每轮每侧十个热样本、完整时延/吞吐/内存统计及容量/OOM/checkpoint 配对。更广 Day2–5 的支持矩阵、held-out 指标、最终候选训练/新进程恢复、DDP 多 rank checkpoint、序列化契约与 clean wheel/sdist/migration review 仍开放；overall **Not complete**。

### 下一步

1. 在不触碰旧 R14 原件的前提下，扩展并锁定序列化方向、格式、dtype、device 和 optimizer/model 状态合同；区分旧 JT portable-pickle 的直接 stock-load 限制与 shim 转存后的 ZIP 互操作。
2. 推进不依赖 GPU 独占的 Day2–5 发布门：DDP 多 rank checkpoint、完整支持矩阵、clean wheel/sdist 安装及旧 baseline migration review；held-out 任务指标和统一候选训练/续训需绑定同一候选。
3. Day4–5 需管理员/调度系统按 GPU2 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0` 提供独占窗口；不运行 burner，不把共享 GPU 结果当正式性能结论。

## 2026-11-01 00:10 +0800 handoff：R11h GPU4 有一项 CUDA selector 通过；发布门仍未关闭

主工作区文档编辑前 HEAD `eac64eba9727df20355d62aad90f1b7bb622fa5b`，origin/cgq_transformers `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 130/0。102 项原有 dirty/untracked 保持其 porcelain SHA256 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`；主源码未改。本轮执行与文档变化仅涉及 R11h 隔离候选、其 _state 证据目录及三份状态文档。

R11h 的真实 GPU4 结果需分项解释：R6 证明环境 `use_cuda=1` 在静态初始化期间进入 `sync_all` 会崩溃；R8 的 CUDA fill/zero selector 本身 1 passed、fallback=0，但 runner 的后置 FollowRuntime alias 断言失败，因此总体 receipt 仍 FAIL。R11 诊断中，读回后再 `zero_` 会让 native/torch 指针分开，双方值和后续写入值仍一致；需先确定 `torch.Tensor(existing Jittor Var)` 的构造语义和指针保证，再决定是否修源代码。没有修改 R11h 或主仓库源码。

下一步按以下顺序继续：

1. 审查 `setter_use_cuda → sync_all` 的 C++ 全局初始化路径，设计并实现安全的 CUDA 环境启动修复；用新的源码指纹和隔离缓存验证，保留当前 R6 失败和 R8 单 selector 通过原件。
2. 对照 Torch 兼容层文档/既有测试，明确 `torch.Tensor(existing Jittor Var)` 是复制还是 alias contract；将 R10/R11 的读回后写入序列补成有明确预期的测试，再作最小修复或把地址变化限定为 readback 生命周期语义。值一致不能代替地址合同审查。
3. GPU2 当前仍被 yqa PID 3590512 使用且为 Default；管理员尚未提供 UUID/GRES/device-cgroup 独占租约。不要运行 burner 或以共享 GPU 结果做性能结论。
4. 独占窗口到位后再执行 Day4–5 三轮交替 PT/JT、每侧每轮至少十个热样本、时延/吞吐分布与 allocator/NVML 峰值、容量/OOM/恢复和 checkpoint on/off 对照。
5. 继续关闭整体 Day2–5：最终 API/operator × device × dtype × layout × autograd/alias/storage/fallback 矩阵、held-out 任务质量、最终候选训练/续训、DDP 多 rank checkpoint、序列化互操作、clean wheel/sdist 安装和 baseline migration review。

所有 R5–R11 日志、receipt、worker 与 plan 保存在 Day2–5 报告所列 `/data6/cgq/jittor-lab/_state/release-day23/20261031-r11h-cuda-directed-gpu4-r*-prepared/` 目录；R1–R4 及更早失败原件不覆盖。Day4–5 性能/显存和 broader release 继续 **Not complete**；不 push。

## 2026-10-31 handoff：R11h CUDA runner/cache 初始化未通过；保留 CPU 结果和失败原件

主工作区 /home/cgq/jittor 本轮编辑前 HEAD 7a8552494fc80c63f9db926c15fbcc5cf96159a4，分支 integrate/cgq-transformers-native-multicard-target；fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 129/0。102 项既有 dirty/untracked porcelain SHA256 仍为 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，index 为空。主源码未修改；只更新三份既有状态文档，不 push。

R11h CPU focused 与 Torch core 结果仍分别为 9 passed/1 skipped、227 passed/3 skipped，fallback=0，且只支持其隔离源码 fingerprint 1530199e462b27b9b32ce93b000555bff0b03c28af2f90ab6548c8c37307c770。本轮四个 GPU4 功能启动 attempt 均未进入目标 CUDA selector：R1 使用 CPU bootstrap，R2 启用 CUDA 后在 sync_all 段错误，R3 因 jit_utils 重建要求新进程，R4 的保护器发现 jittor_core 从 2/228 重编译并中止。日志、收据和根因记录见 2026-10-26-torch-compat-release-day2-5.md；这些记录不得覆盖，也不构成 CUDA 通过。

GPU2 当前为 Default，外部 yqa PID 3590512 占用约 8,640 MiB；尚无管理员级独占租约。下一步先审查 R3/R4 的 jit_utils/core cache freshness 与启动顺序，确认 source-bound warm/cold build 方案后再做单次 CUDA 功能尝试。正式性能/显存验收继续等待 GPU2 UUID/GRES/device-cgroup 独占租约；整体 Day2–5 release 保持 **Not complete**。

## 2026-10-31 handoff：R11h CPU focused/core 通过；GPU2 独占前不启动 CUDA/正式性能

主工作区 /home/cgq/jittor 本轮编辑前 HEAD fe8d248a6322e1159381d8ae0fbea97ceb67be09，分支 integrate/cgq-transformers-native-multicard-target，origin/cgq_transformers ahead/behind 128/0。102 项既有 dirty/untracked porcelain SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，index 为空；主源码未改。只更新三份既有状态文档，不 push。

R11h 使用既有隔离 worktree /data6/cgq/jittor-lab/worktrees/torch-compat-fill-zero-runtime-grad-r11-20261031，HEAD 765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe，fingerprint 1530199e462b27b9b32ce93b000555bff0b03c28af2f90ab6548c8c37307c770，source manifest SHA256 0de2c5c034676bfb29b446054d04a7da70a59bf07bda20497eeab97e0d95f844。R11g focused 曾有一项 alias pointer 失败；R11h 仅把 source Var clone 调用改为 backend.Var.clone(args[0])，随后当前 10 项 focused 为 9 passed、1 CPU capability skip、fallback=0；维护的原 Torch core 十文件为 227 passed、3 skipped、fallback=0，214.97 秒。R11h 是隔离候选，不代表主树或最终 wheel。失败与通过产物均保存在 Day2–5 结果报告所列 evidence paths，勿覆盖。

### 建议的继续顺序

1. 等待管理员为 GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 提供 EXCLUSIVE_PROCESS/GRES/device-cgroup 独占租约。15:09 UTC GPU2 快照为 Default、84% 利用率、17,009 MiB 已用；yqa CIFAR100 PID 3590512 可见，Slurm 主机 X12DPG-OA6 为 down。不能用 burner 代替租约。
2. 独占满足后，以 /data6/cgq/jittor-lab/_state/release-day23/20261031-r11h-cuda-directed-gpu2-r1-prepared/launch.py 启动一次 source-bound CUDA 定向回归。launcher 会先查 UUID、exclusive mode、现存 compute PID 和 2807 项源码身份，再用私有 cache 冷编；物理 GPU2 应映射进程 cuda:0，fallback 必须为 0。超时/失败保留原件，不自动重跑。
3. 对 R11h 做功能审阅/迁移决策，再绑定最终整合候选。不要把 R11h 隔离 CPU 结果或 R5/R14 的其它源码训练结果混成最终候选通过。
4. Day4–5 需冻结匹配 JT/PT 源码、wheel/core、模型权重、数据、精度、shape、batch、优化器与设备；稳定独占窗口至少三轮交替，每轮每侧至少十个热样本，报告 load/首步/热态 median、p95、离散度、吞吐、allocator allocated/reserved 与 NVML 峰值。现有 R26 median ratio 1.595 尚未达到 1.10 目标。
5. 分别完成近容量、最大 batch/sequence、OOM 与恢复，以及 checkpoint on/off 配对资源测量；记录重算、输出/梯度/更新一致性、速度成本和显存峰值。
6. 在最终候选上关闭 API/operator × device × dtype × layout × autograd/alias/storage × fallback 矩阵；补预设阈值的真正 held-out 任务指标、训练与 fresh-process checkpoint 恢复、DDP Trainer 多 rank checkpoint、JT/PT 序列化互操作契约、clean wheel/sdist 安装及 baseline migration review。

cuda-directed-r1 的 15:10 UTC preflight 因 compute mode 为 Default，状态 BLOCKED_EXCLUSIVE_MODE，SHA256 f5f6a59a7a03b38c94ec425f1f4c4d3233f1744ca7c63e0e55bc3b63caca471e；它在 Jittor import 前退出，不能算 CUDA 测试。Overall Day2–5 release 继续 **Not complete**。

## 2026-10-31 接续：R5 当前源码实用训练/续训完成；GPU2 独占仍受权限阻塞

R5 在现有 `/home/cgq/jittor` 工作区完成，无 clone 或源码改动。HEAD `d26c31246613c9e6a34da2ee7b6853612c9049a3`，origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 127/0；实验前 102 项其他 dirty/untracked 状态 SHA256 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`。证据目录 `/data6/cgq/jittor-lab/_state/release-day23/20261031-qwen-current-source-practical-r5-prepared/`，快照 manifest SHA256 `af6a7c4e140163820ca26703621974ceec29c5e2850df55335119004f56fe8ff`。

Qwen3-0.6B LoRA seed821 在 GPU4（逻辑 `cuda:0`）完成 102/102 更新、0 skip、fallback=0；epoch loss 1.205599/1.153534，eval loss 2.456431。checkpoint-51 新进程续至 102，完成 51 更新；RNG 检查通过，930 项状态值和设备映射通过（112 个 scalar optimizer step 不参加 tensor 设备映射比较）。该证据没有性能/显存结论或 held-out 生成质量分数。

GPU2 包装器 `/data6/cgq/jittor-lab/_state/release-day23/gpu2-exclusive-run.sh`（SHA256 `5e4da7f88793c6e38e204919a45b7d9463aa21ae2c260330953eb19904902d02`）因当前账户无权设置 `EXCLUSIVE_PROCESS` 未激活。14:43 UTC CIFAR100 进程 PID 3590512 使用 GPU2 8640 MiB；resume 阶段亦有 GPU2/GPU4 共享负载。正式测量需管理员提供 UUID/GRES/device-cgroup 独占租约。

下一步：取得正式独占窗口后执行冻结配置下三轮交替 PT/JT 性能/显存、近容量/OOM 恢复和 checkpoint on/off 配对；同时关闭支持矩阵、held-out 质量、当前候选训练/恢复、DDP 多 rank checkpoint、序列化互操作和 baseline migration review。Overall release **Not complete**。本轮只提交这三份状态文档，不 push；其他既有 dirty/untracked 文件保持原样。

---

## 2026-10-31 接续：R4 clean packaging 结果与 GPU2 独占限制

在现有 `/home/cgq/jittor` 工作区继续，无 clone/reset/stash/clean。HEAD `fa7fae39ad06d05cfbc5545e705c9d64562c2f58`，远端 `origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 126/0。其他既有 dirty/untracked 状态在本轮前为 porcelain SHA256 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`；`/data6` 可用 5.5 TiB，根分区可用约 32 GiB。主工作区源码不变，只更新既有状态文档。

R4 isolated package candidate 位于 `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-package-current-r4-prepared/`。它修复 shim deploy 提示访问 `torch.__file__` 的问题；干净 combined-wheel 安装、部署、精确版本 verify、182-path compat wheel audit、manifest check 与针对性 unittest 1/1 均通过。patched compat wheel SHA256 `9f5ee249b52912a391bf4953210fb9a58fb8aee2a41b1e710fa0ed84c74fedb1`。修复仅存在 R4 隔离候选，未迁入主树。R3 direct-wheel/sdist 对照通过，但 direct-wheel baseline comparison 仍要求 review（69 added/302 changed/2 removed，无 allowlist）；release packaging gate 未关闭。R3 初次 verify 失败和 R4 错 cwd smoke 失败均保留，后续修正重跑通过，没有改动 `/home/cgq/Motion324`。

GPU2 A800 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0` 显示 Default/1 MiB/0%，查询无该 UUID 的 compute PID，但设备节点存在其他作业打开句柄。开启 `EXCLUSIVE_PROCESS` 权限不足，sudo 需要密码，Slurm 节点 `X12DPG-OA6` 为 down。没有运行 burner，因为不能阻止其他进程且会污染基准。性能测试需要管理员提供 UUID/GRES/cgroup 的正式独占租约。

接续优先项：保留 R4 证据并进行 baseline migration review；在获得 GPU2 独占租约及稳定窗口后，完成 Day4–5 三轮交替 PT/JT 时延（每轮每侧 ≥10 热样本）、容量/OOM/恢复和 checkpoint on/off CUDA 资源对照；继续关闭冻结候选支持矩阵、可信 held-out 质量、训练与新进程恢复、DDP 多 rank checkpoint 和序列化互操作契约。当前性能/显存验收及整体 Day2–5 release 均保持未完成。详见 `2026-10-26-torch-compat-release-day2-5.md` 的本轮增量。

---

## 2026-10-31 handoff 增量：R37 配对生成已存档；GPU2 独占受集群配置阻塞

继续使用服务器 `ssh campus-linux` 的既有 `/home/cgq/jittor`，未 clone/reset/stash/clean，也未修改源码或 wheel。R37 运行时 HEAD 为 `2b7dac919764b7b815cf32fff5a42d9e35896c59`，分支 `integrate/cgq-transformers-native-multicard-target`；origin/cgq_transformers 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 125/0。文档修改前 102 项既有 dirty/untracked porcelain SHA256 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`，index 为空；本轮只应提交三份状态文档，不 push。源指纹 `0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50`。

R37 evidence 位于 `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-qwen-seed821-heldout-generation-r1-prepared/`，manifest SHA256 `241d4649ee29f1a4df264321c88dd3935fbf3690ae5f7ca401d76ecbef4b3fb2`。seed821 PT/JT adapters 分别绑定 SHA256 `d48da15c…` / `8669a5fb…`；17 条 greedy CUDA 生成在物理 GPU4、逻辑 `cuda:0` 上都完成，样本对齐通过、14/17 文本相同，JT fallback=0。PT/JT token F1 为 0.208718/0.204534，ROUGE-L 为 0.149341/0.148379，exact match 都为 0。17 条来自 Alpaca train split，未人工复核；JT adapter 训练源码早于当前推理 wheel，故只作探索分数，禁止用来关闭 held-out 质量门。逐样本审计在 `final-review.json`，SHA256 `00c45d4b0e8bf9cddcfd34bea0b9ef834e40ddacbc07b03ca84ddae4ea5037f3`。

GPU2 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0` 检查时空闲，但 compute mode 为 Default；`X12DPG-OA6` 在 Slurm 中 down，当前账户 sudo 需密码，因此不能通过普通占卡进程保证别的用户无法启动 CUDA context。未启动 burner；正式 GPU2 测量要等管理员恢复节点并分配按 UUID/GRES 与 cgroup 隔离的 lease。最终采样 GPU2、GPU4 均 0%/1 MiB，本轮任务已退出；这不是持续独占保证。

记录的准备期意外：一次未在 bubblewrap 内的 Jittor 环境 import 因缺少 `GLIBCXX_3.4.30` 失败，在用户 cache 下观察到一个带该时间戳的 `jittor.lock`，未删除；两次预检启动失败均保留，第三次用 R19 install-target 复用已编译 core 后通过。详见 `preparation-incidents.json`（SHA256 `bc79333f9b7be7cee2d2665de51c851544b4f687d95631322652718b5c4d8f6b`）。

继续目标：不要重跑 seed821 生成、R14 full/preflight/resume 或冷编译。Day4–5 正式时延仍需 ≥3 轮交替且每轮每侧 ≥10 热样本；R21 ratio 1.576、R26 ratio 1.595 未达 1.10。近容量/OOM/恢复与 CUDA checkpoint on/off 仍未通过。更广支持矩阵、可信 held-out 质量、当前候选训练恢复、DDP Trainer 多 rank checkpoint、序列化互操作契约、clean wheel/sdist 与 baseline migration review 仍待完成。整体 release 保持 **Not complete**。

---

## 2026-10-31 续验：R36 证明当前 wheel CUDA checkpoint 重算；共享负载拒绝后仅完成 on 单侧

在既有 /home/cgq/jittor 工作区继续，无 clone 或源码改动。R36 绑定 HEAD 487a82e8d8025c0a7458186e464acc0f76e0ac10、源码指纹 0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50，预实验时 102 项既有 dirty/untracked 状态 SHA256 为 be3ed51168de9922dd80cebe3e134b11967ae95bb5ec4aee986d97f6fa4f071c。R35-r3 的导入审计已由事后审计确认：shim、Transformers、Jittor 与当前 wheel core 绑定正确，CUDA runtime enabled；runner 仅因 compatibility torch namespace 不提供 torch.__file__ 而在写报告时失败，没有加载模型、前向或启动 kernel。R35-r3 不重试。

R36 使用当前 wheel/core、Transformers 4.56.2 LlamaForCausalLM 六层 FP32 随机初始化模型（seed 821，hidden 384，intermediate 1024），固定合成 batch 4 × sequence 512 与 AdamW lr=1e-4，在物理 A800 GPU2（UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0，逻辑 cuda:0）开启 Transformers gradient_checkpointing_enable()。单侧阶段实际观测到六层均为 [no_grad forward, backward recompute]，fallback=0，首个 loss 6.312356、梯度 L2 2.274081、更新 L2 0.331624，随后完成 10 个训练热步。热步 median/p95/std 为 46.529/47.177/0.536 ms；allocator 峰值 allocated 增量 1,403,671,552 bytes、reserved 增量 6,291,456 bytes，NVML 进程峰值 2,160,066,560 bytes。首步 227.745 s 含首次 JIT 编译，不能与热步混读。随机 workload 只支持 CUDA 重算/更新功能证据，不代表任务质量。

round-01-off 启动前复查 GPU2 利用率为 28%（门槛 ≤10%），空闲 81,038 MiB、无可见 compute PID，控制器拒绝启动并停止整个单次 campaign；未重跑。R36 状态为 INCOMPLETE_ADMISSION_REJECTED_AFTER_STAGE1，只有 on 单侧，不构成 checkpoint 显存节省或 on/off 速度结论。证据目录 /data6/cgq/jittor-lab/_state/release-day23/20261031-checkpoint-cuda-resource-r36-llama-pair-prepared/attempt-01/；posthoc audit SHA256 788e4aba7a625fba97ea0b3e1d096f5f086d44ab4352432747980d04fde72c91，on worker result SHA256 f872e43cc84d8433d91112050aa6f5014064c022ddc6c4f58bb38b1ca2e69f33，evidence manifest SHA256 84fbb1f0b49752f83e979897e46e512f18c8016ddd3bff52e5231585d43d64c5。R36 私有 cache 新增 264 项并只改变 lock 元数据；R35 私有保护 cache 前后 manifest SHA256 同为 134fb35833b7400d3a8a0b8acddaaff4278bf8c42a8b66c657f3709861e7e5fc。工作区源文件未改，既有 dirty 状态在实验后仍与上述预实验 SHA 一致。

共享 GPU2 目前没有可由当前账户申请的 Slurm 独占租约：调度器将当前主机 X12DPG-OA6 的 A800 节点标为 down；空闲的 A100 节点 cn2 是另一台机器。没有启动显存占用或持续 kernel 的 burner；它不能阻止非协作任务，且会挤压共享作业。后续性能阶段只能在 GPU2 满足准入时运行，或由集群管理员提供该节点的正式独占 reservation。

Day4–5 仍需匹配配置的 JT/PT 三轮交替正式时延/显存 cohort、近容量与 OOM/恢复，以及至少 checkpoint on/off 三轮配对；更广 Day2–5 仍需支持矩阵、统一候选 held-out 指标与训练恢复、DDP Trainer 多 rank checkpoint、序列化互操作契约、clean wheel/sdist 安装和 baseline migration review。Overall release 继续 **Not complete**。

---

## 2026-10-31 续验：R35 checkpoint CUDA 预检未通过启动绑定；Day4–5 与更广发布门仍开放

R35-r1 的导入探针误用了默认 JITTOR_HOME 和全局 cache，因 libstdc++ ABI 不符在模型启动前失败；原件保留，生成的 48 MiB 全局 cache 未删除或复用。R35-r2 改用 R21 已审计的 bwrap 隔离方式，把 64 MiB R21 私有 cache 复制到新目录并绑定到原绝对 JITTOR_HOME。R21、R19 与 R35 副本中的 CUDA 12.4 core SHA256 均为 `050c612d6b29bb952c4debccc7eb791f1704e3c8a2040723cb352cbda789f986`。但启动器漏传 R21 的 `cache_name` 和 `use_cuda` 配置，Jittor 因而解析到 `default/cfgd436a523`，与绑定的 `jittor_torch_r19_current_wheel_20261031/cfgecf1a63a` 不同；`JITTOR_NO_BUILD=1` 正确阻止了重编，尚未完成 import。没有模型、前向、GPU kernel 或 benchmark。

R35-r2 完整失败原件位于 `/data6/cgq/jittor-lab/_state/release-day23/20261031-checkpoint-cuda-resource-r35-r2-preflight/attempt-01/`，`preflight-failure.json` SHA256 `8460bb3235cc1ed856ca06b1e0eb56a04e8da29c1c78b06275565cc51ec66723`；导入日志 SHA256 `75b6cad4bbfadaede764c6cc907d572a34f093fc6d4b2743fb07c6cf0044ba1e`。仅隔离副本新增默认配置 cache 项并更新锁；R21/R19 core 与 R35-r1 全局 core 的 SHA 均复核未变，预检进程已退出。HEAD 保持 `c442c0de6cec01b47c9a7ffc22d72c4bd3b3f6d1`，本轮前后既有 102 项 dirty/untracked 状态未改。末次 GPU2 为 0% 利用率、81,038 MiB 空闲且无计算进程，但其他设备有共享任务；未启动测量。R35-r1 与 r2 均保留，当前失败不是功能或性能结论。

### 尚未关闭的内容

1. **正式时延门：** R26 当前代表格 JT/PT 热态 median ratio 为 1.595，高于 1.10 目标；R33 hook/分段归因只能诊断。需先有依据地优化并冻结最终源码、wheel、core、模型/权重、数据、精度、shape、优化器和设备，再在稳定窗口（利用率 ≤10%、空闲显存 ≥60,000 MiB、无无关计算进程）完成至少 3 轮 PT/JT 交替，每轮每侧 ≥10 个热样本。报告加载/准备、首步、热态 latency 与吞吐的 median、p95、离散度，并保留逐样本值、设备/fallback、梯度/更新/loss、allocator 与 NVML 峰值。
2. **容量与显存：** 以匹配配置单独测近容量、最大可运行 batch/sequence、OOM 行为和 OOM 后恢复；单一代表格的 allocator/NVML 峰值不等于容量通过。
3. **Checkpoint 资源收益：** 现有 CPU 与窄 CUDA decoder 证据已证明部分路径发生 activation recompute 并产生梯度；仍需同模型、同输入的 CUDA on/off 配对，核验输出/梯度/更新与 fallback，并报告 allocated/reserved 峰值、NVML 峰值和速度成本。功能重算通过不等于省显存通过。
4. **更广 Day2–5 发布门：** 在一个冻结候选上完成声明范围的 API/operator × device × dtype × layout × autograd/alias/storage × fallback 矩阵；补当前候选的训练、held-out 任务阈值与新进程恢复；完成 DDP Trainer 多 rank 保存/续训、JT portable-pickle 与 stock PyTorch 的互操作契约、clean wheel/sdist 安装及 baseline migration review。R14 训练/resume 与 R34 受 token cap 限制的 stock PyTorch 前缀分数均不能替代这些门。
5. **预检续接：** R35 后续新目录必须完整带入 R21 的 `cache_name`、`use_cuda` 和其余绑定环境，先静态核对预期 cache path，再决定是否允许一次 import-only preflight；本目录不重试。

Overall release 继续 **Not complete**。

## 2026-10-31 handoff 增量：R33 归因与 R14 固定长度生成评分完成，发布门仍开放

本轮继续使用 `ssh campus-linux` 的现有 `/home/cgq/jittor`，没有 clone/reset/stash/clean，没有改源码或 wheel。记录起始 HEAD `e5c73b092024e2063ffe0eb01608d2b72f8a9b1c`，origin/cgq_transformers 与远端 ref 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind 为 122/0。开始和结束时 102 项既有 dirty/untracked porcelain SHA256 均为 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`，index 为空；本轮仅更新 handoff、release-day2-5 报告和 refactor-board，提交也只含三文档，不 push。/data6 可用空间 5.5 TiB。

R33 r3 attempt-01 绑定上述 HEAD/source fingerprint、当前 main/compat wheel/core、GPU2 和固定 BERT fixture；3 轮六阶段均通过真实 CUDA/fallback=0/输出有限/权重不变检查。summary SHA256 `ff49769e6021887fefd450eea7cba6c20b700921da7d1c37727de35614ebbc81`，posthoc SHA256 `3062d63e36a95a733c05eb6f50b2459f5075b7c87b11592c12a343074af6b11b`。嵌套 hook 同步使绝对时间与离群点不适用于正式性能验收；Q/K/V median ratio 约 1.54–1.59、attention-output dense 1.529、intermediate dense 1.185、output dense 1.036，当前不足以宣布可修源码瓶颈。r1 GPU 前因 cache lock 审计不符拒绝，r2 因 JITTOR_HOME 挂载位置在 JT0 测量前失败；原件保留。r3 预启动 private clone/保护 cache 快照 SHA256 同为 `fc698085cb9dd8f4feb58d0e3ed38b62d2b5f8eb565a45dfe42aec42fe96aa35`；after 快照比较只有 `jittor.lock` 不同。

R34 绑定已终审 R14 Qwen seed821 checkpoint-102/merged 权重，不重跑 full、preflight 或 resume。训练集 101 条、fixture eval 17 条，instruction/input 交集为 0；eval 原始来源仍标 `train`，因此只称 fixture-held-out。R34-r3 用 offline stock PyTorch 在物理 GPU2/cuda:0 对 base 与 merged 各生成 17 条，greedy、双 EOS、256-token cap。Base token-F1/ROUGE-L-F1 为 0.142914/0.087317，merged 为 0.161602/0.104252；两边 17/17 全触顶，是固定长度前缀分数，不是完整生成质量验收。R34 不提供 Jittor 推理/fallback 结论，推理无梯度、无更新。独立审计复算指标、行数、checkpoint/data 绑定通过，SHA256 `f3ca413ce61457533f4df798ee2dcd88e3ed6656dc3ad1fd0b1e0c7858597d9a`。r1 错误 EOS、r2 的 128-token cap 均保留并标记 superseded；不无限重跑。

R34 证据目录：`/data6/cgq/jittor-lab/_state/release-day23/20261031-r14-qwen-heldout-generation-r34-r3-prepared/`；R33 目录：`/data6/cgq/jittor-lab/_state/release-day23/20261031-bert-inner-op-forward-attribution-r33-r3-prepared/`。收尾采样 GPU2 利用率 0%、空闲显存 81,038 MiB、无本任务 GPU 进程；R34 子进程已退出。

R26 正式 BERT cohort 的 JT/PT median ratio 仍为 1.595，性能/显存正式门没有关闭。接续工作应优先处理可验证的功能/发布缺口与有证据支持的优化，再对冻结候选执行稳定窗口性能、容量/OOM 和 checkpoint CUDA 配对验收。API/operator 支持矩阵、当前统一候选的可信任务质量、候选绑定训练与新进程恢复、DDP Trainer 多 rank checkpoint、JT portable-pickle 与 stock PyTorch 互操作契约、clean wheel/sdist 和 baseline migration review 也未完成。Overall release 继续 **Not complete**。


## 2026-10-31 handoff 增量：当前 wheel CUDA 别名尝试在冷编阶段超时

当前主/compat wheel（SHA256 `2e8f1a4418e3e9aee770843a6fd7d5471e3de12aaad75a8e8e78deee24baa3d9` / `fd7e84f639dba09f9d8293b9d255df09aa7190b890093e614c340cc47a7a7d5a`）的公共 Tensor CUDA 跟随 runtime 别名检查未得到测试结论：两次 runner 缺少 `JITTOR_TORCH_SHIM=1`，在收集前退出；第三次修正环境后冷编到 `jittor_core 92/228`，于 300 秒上限结束，断言未执行。实验私有 cache 新增 677 项，未改源码或既有保护缓存。GPU4 当时由外部进程占用且利用率约 73–84%，该尝试不计功能通过或性能/显存证据；原件分别保留在 `/data6/cgq/jittor-lab/_state/release-day23/20261031-current-main-wheel-public-tensor-follow-runtime-cuda-r1-prepared/`、`...-r2-prepared/`、`...-r3-prepared/`，不自动重跑。

Day4–5 仍需稳定 GPU 窗口下 JT/PT 三轮交替且每侧每轮至少十个热样本，报告 load/首步/热态 median、p95、离散度、吞吐及 allocator/NVML 峰值；近容量、最大可运行配置、OOM/恢复及 checkpoint 开关需单独以匹配 CUDA 配置验收。更广 Day2–5 仍包括支持矩阵、held-out 质量、候选绑定训练与恢复、DDP 多 rank checkpoint、序列化互操作契约、clean install 和 baseline migration review。Overall release 保持 **Not complete**。

## 2026-10-31 接续：R31 停在共享 GPU 准入，R32 确认 wheel 中 checkpoint 真正重算

继续在原 /home/cgq/jittor，未 clone 或修改源码。测量前后 HEAD 3cf20912727973bb645428f3d3c3750d920c3a45，FETCH_HEAD/origin cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 120/0；102 项其他 dirty/untracked 状态 SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211 未变，index 为空。R31/R32 脚本、日志、cache 和审计均保存在 /data6/cgq/jittor-lab/_state/release-day23/，未放入仓库。

R31 只完成一个 current-wheel BERT GPU pair，旁路 227 次模块调用中的嵌套前端上下文后，median 没有改善，且梯度-map hash 不同但幅度未存，故不应形成优化候选。随后 GPU2 在下一阶段准入以 51% utilization 拒绝，完整 R31 保留 FAILURE；不在相同共享窗口重跑。

R32 对照 wheel archive member 和工作区 compat/torch/installers/data.py，确认 SHA256 同为 0a4d1c151a9de5e3d57da60117f156e8641b26febd7c63a4727914fef04c33c8；当前 wheel 已含 _CheckpointRecompute，不再是直接调用 shim。三次 attempt 中前两次分别在导入期 build guard 和 torch.__file__ 记录器处失败，均保留；attempt-03 在 CPU 上运行现有测试 6/6、无 skip，并经独立终审通过。此结果关闭“目标函数是否重算”的 CPU 功能问题，但不关闭 CUDA 显存/速度门。

2026-10-31 10:50 UTC 的 GPU2 采样为 100% 利用、70,559 MiB 已用、10,479 MiB 空闲。下一步待新鲜准入后测最终 wheel checkpoint on/off 的三轮交替 cohort（每侧每轮至少十个 hot sample），并收集 allocator live/allocated/reserved、NVML 峰值及 load/首步/稳态耗时；之后独立测近容量/OOM 与恢复。R26 时间 ratio 1.595 仍失败，R31 partial 不能替代正式时延线。

更广发布门仍包括 support matrix、统一候选 held-out 指标、真实训练与新进程恢复、DDP Trainer 多 rank checkpoint、跨框架 optimizer 格式契约、clean wheel/sdist install 与 baseline migration review。overall release **Not complete**；不要重跑 R14 full/preflight/resume 或已通过的 cold build，不覆盖 R31/R32 失败原件。

## 2026-10-31 最新交接：R27–R29 已定位前向差距线索，Day4–5 与发布门仍开放

在 /home/cgq/jittor 现有工作区继续，无 clone。R29 测量绑定 HEAD 02be05fcaab344735272a8526485330602f32ec0、source fingerprint 0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50、102 项既有 dirty/untracked porcelain SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，无源码/wheel 改动。origin/cgq_transformers 当前 ref 与 git ls-remote 均为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81；HEAD 比该 ref 多 119 个提交，origin 无独有提交。最终 GPU2 空闲 0%、81,038 MiB free；/data6 余 5.5 TiB，/ 显示 33 GiB 可用；R29 进程已退出。

R26 正式 cohort 的真实训练更新/fallback/状态检查通过，但 JT/PT hot median ratio 1.595，未过现行 1.10 时延门；代表 batch 显存 ratio 不是 OOM 容量证据。R27 聚合 trace 没有证明 SGEMM 较慢；R28 12 层 ratio 为 1.499–1.549；R29 第 5 层 attention.output ratio 2.106，attention.self 1.689。R29 runner 在六个 raw phase 全 rc0 后，因 summary 缺少 zero_grad_call 报錯，原 failure 保留；posthoc 数据经独立审计。R28 独立审计另发现原 summary 的每层绝对时间多乘 1000，只有 ratio 和整次同步 forward 时间可直接引用；修正值在 artifact 的独立 audit 中。

下一项先拆 attention.output 内部 dense/dropout/residual/LayerNorm，再决定是否有明确可修的 shim/core 开销；正式性能须绑定新 wheel 和源码，用稳定窗口完成 ≥3 轮交替、每轮每侧 ≥10 热样本，并含 load、首步、热态分布及 allocator/NVML 峰值。随后补近容量/OOM 与 OOM 恢复，以及 activation checkpoint 真实 recompute 的速度/峰值配对。更广 Day2–5 还需同一候选的支持矩阵、held-out 任务分数、DDP Trainer 多 rank checkpoint 恢复、JT/stock PyTorch optimizer 序列化契约、clean wheel/sdist 安装及 baseline migration review。R14 实用 Qwen 证据不替代这些门。Overall release **Not complete**。

## 2026-10-31 最新交接：R26 默认 zero_grad 端到端 cohort 完成；时延仍失败

在既有 /home/cgq/jittor 继续，无 clone、源码改动或 wheel 重建。实验时 HEAD 27d01a6f40d9e7697bbb2558ba1303839eee93a0，origin/cgq_transformers 与远端均为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 118/0；102 项既有 dirty/untracked 哈希 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211 未变。R19/R21 原 cache 未改，测量后无本任务 worker 残留。

R25 分段诊断发现 set_to_none=True 下 JT zero_grad median 从 R24 的 22.216 ms 降到 4.546 ms，PT 从 4.819 ms 降到 1.434 ms。R19/R21 harness 显式传 False，而公开 compat shim 与 PyTorch 无参数默认均为 True，因此完成 R26 默认调用端到端 cohort。R26 使用 BERT B8/S256 FP32、三轮交替、16 warmup + 10 hot/阶段，GPU2/cuda:0。每阶段设备、fallback、梯度、更新、输入/初值配对和稳定性检查通过；每侧 30 hot。

R26 JT/PT median 142.746/89.492 ms，p95 145.052/93.274 ms，标准差 17.597/1.913 ms，ratio 1.59508，仍未达 1.10；三轮 ratio 1.55/1.61/1.49。allocator allocated/reserved 与 NVML 进程峰值 ratio 为 1.09454/1.06319/1.04508，仅代表该 batch。R26 summary SHA256 7fccaf9ba85eda068c15198b872619d2fde7f65ab924d600a1c8632e34b9b7de，独立终审 PASS_FORMAL_COHORT_EVIDENCE_TIME_GATE_FAIL SHA256 7d4838769bdc913601615f3dcea507d907783c948b65312d22f22ba8b314a9dd。R25 summary SHA256 a35832f1a5e28894d4f35a78dd75252826d79425efc3b0074703f44cfd31fa48，终审 PASS_DIAGNOSTIC_EVIDENCE_ONLY SHA256 3eceb2ea1214a622d3782bcb704afdff476079e28fa6928690e0491d380efdda。R25 bwrap cache-path 启动失败和 R26 初版 audit schema 误判均保留；各自后续 attempt 的最终独立审计通过。实验 cache clone 只变更已退出 worker 的 jittor.lock 元数据，R21 source cache 指纹 fc698085cb9dd8f4feb58d0e3ed38b62d2b5f8eb565a45dfe42aec42fe96aa35 未变。

### 当前继续清单

- Day4–5：优化 R26 默认调用下的端到端时延并为新 wheel 候选复测；补匹配配置近容量/OOM 边界；以可审计证据证明 activation checkpoint on/off 的真实重算、峰值显存和速度收益。
- 更广 Day2–5：冻结统一候选，补 API/operator/device/dtype/layout/autograd/alias/storage/fallback 矩阵、held-out 任务指标、真实训练和新进程恢复、DDP Trainer 多 rank 保存/续训、clean wheel/sdist 安装与 baseline migration review。
- 明确 JT portable-pickle optimizer checkpoint 无法由 stock PyTorch torch.load 读取时的互操作支持契约；保留该失败，不否定 Jittor 同框架恢复证据。
- 整体 release 继续 **Not complete**。R14 实用训练结果只绑定自身源码/配置；Day4–5 与更广验收仍开放。

---
## 2026-10-31 最新交接：R24 分段诊断完成，zero_grad/forward 成为优先优化候选

R24 使用 R19/R21 相同源码指纹、wheel/core、BERT B8/S256 FP32 权重/输入与 AdamW default 配置，在 GPU2/cuda:0 对 PT/JT 分别测 zero_grad、forward、backward、optimizer。每侧 30 个热样本，三轮交替，16 warmup/阶段；初始参数和输入配对一致，197 个梯度有限，更新发生，fallback=0。JT median：zero_grad 22.216 ms、forward 77.459 ms、backward 58.732 ms、optimizer 5.143 ms；PT 对照为 4.819、30.461、53.433、9.862 ms。诊断含分段同步边界，不是正式时延门。installed wheel 与工作区 _zero_grad_compat(set_to_none=False) 源哈希相同，其逐梯度 update/zeros_like 与高 zero_grad 计时相符，但尚无消融证明因果。forward 还需拆 attention/GEMM 与执行调度。

R24 summary SHA256 7333eee81407832077499f2f75d75152f857b1da2c1e2c82f47016a543c024eb；独立审计 PASS_DIAGNOSTIC_EVIDENCE_ONLY SHA256 79a5299a4eb61993c0cf7d75a3aec79e10c9b62cd06dae1f9651e8a76af13ca0。一次只读 runtime 初始化失败和一次 GPU admission rejection 均保留；后续阶段在新目录/新 cache 副本完成，原 R21 cache 与 repo 102 项既有 dirty/untracked 状态未变。接下来做有界 zero_grad/forward 消融和 attention/GEMM 归因，再决定是否改源码与构建新 wheel；Day4–5 正式性能、OOM、真实 recompute，以及更广 Day2–5 发布门都仍开放。Overall release **Not complete**。

---
## 2026-10-31 最新交接：R23 只读 trace 聚合完成，尚未找到单步瓶颈

R23 对 R19/R21 已归档的六份 Jittor attribution trace 做只读聚合，没有运行 GPU workload/Jittor，不改代码、wheel 或 cache。GPU kernel event duration 在每份三步 trace 中约 239.06 ms（R19）和 237.02 ms（R21），SGEMM 累计约 199.80 / 199.50 ms；fused AdamW 累计约 5.695 ms/三步（R21）。端到端 Jittor median 约 150.630 / 148.923 ms，PyTorch median 107.903 / 94.494 ms。R21 ratio 变差来自 PT 默认配置的计时更低，而不是两轮 JT median 有明显差别。executor/device-wait trace 事件累计差异大，且 profiling trace 不能充当热样本的 wall-clock 分解，需保持归因未定。

审计 summary: /data6/cgq/jittor-lab/_state/release-day23/20261031-r23-existing-trace-audit-r1/summary.json，SHA256 b3f169b5a844d4ace767543d8f3945fd01770e9367279b8784737e4edad1e4fd。下一步做 profiler 关闭的 PT/JT forward、backward、optimizer 分段诊断，固定 optimizer/模型配置并检查数值和更新等价性；之后若做性能优化，正式结果须绑定新 wheel 身份与三轮匹配热态窗口。Day4–5 的 OOM 边界和真实 activation recompute on/off 对照，以及更广 Day2–5 兼容发布门继续开放。Overall release **Not complete**。

---
## 2026-10-31 最新交接：R22 隔离优化器诊断通过；下一步定位端到端模型计算

仓库在测量时 HEAD 20d573a143f5a585ffa1131832471e070a55091f，origin/cgq_transformers 为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 115/0）；102 项既有 dirty/untracked 状态 SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211 未变。R22 绑定 R19/R21 相同 wheel、core、权重和 199 个参数形状，在 GPU2/cuda:0 对 Jittor 默认与显式 unfused AdamW 做 3 轮 × 每阶段 10 个热样本。只测 optimizer step：默认 median 4.212 ms，unfused 33.596 ms，profile kernel 数 7 对 197；15 步后 max absolute difference 8.15e-10，allclose 通过。attempt-02 独立 audit 通过，attempt-01 汇总器失败的六阶段原件保留。

此结果只说明固定合成梯度下的 Jittor 更新路径，不测 forward/backward，不是 PT 对照，也不关闭 R19 ratio 1.39597 或 R21 ratio 1.57601 的端到端时延失败。优先从已有 trace 区分 BERT 前向、反向、attention/GEMM 与 Python/Executor 调度开销，再选择有依据的优化；之后仍需近容量/OOM、真实 checkpoint recompute 资源对照及广泛 Day2–5 支持矩阵、held-out 质量、DDP 新进程恢复、序列化契约、clean install 和 baseline 迁移审查。R22 artifact: /data6/cgq/jittor-lab/_state/release-day23/20261031-adamw-update-isolation-r22-r1-prepared/。Overall release 继续 **Not complete**。

---

## 2026-10-31 最新交接：R21 默认 AdamW 代表格已审计；时延门未过

R21 在 /home/cgq/jittor 现有工作区完成，绑定测量时 HEAD 62aa8ca7f8c075d22f036d18759c686f70483a00、origin/cgq_transformers e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 114/0）、source fingerprint 0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50。 当时 102 项既有 dirty/untracked、状态 SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211，index 为空；相同 R19/R20 主/compat wheel 和 private core，无源码或 wheel 修改。

BERT B8/S256 FP32 上 PT/JT 均以不传 foreach/fused 的 AdamW 用户默认调用执行，三轮交替、每阶段 16 warmup+10 hot；真实 GPU2/cuda:0、fallback=0、197 个有限梯度、201 个初值映射、参数更新和六阶段 load audit 均通过。独立终审 attempt-03 14/14 checks pass，状态 R21_DEFAULT_ADAMW_AUDIT_PASS_TIME_GATE_FAIL。JT/PT pooled median 148.923/94.494 ms、p95 153.841/97.706 ms、标准差 19.953/3.203 ms，时延 ratio 1.57601 未达 1.10。allocator allocated/reserved 与 NVML 峰值 ratio 1.08066/1.09147/1.07114，只有代表格显存数据，不含容量/OOM。

artifact /data6/cgq/jittor-lab/_state/release-day23/20261031-current-wheel-bert-performance-r21-default-adamw-r1-prepared/；receipt SHA256 cfb894e1866932b024cfc3ff985197f1dec44a01dce23cc7f4d71659c7a4076a，summary adad51aaa394bb613ae0a5bbc64fecb7815b0661573ed9e1665dbb34dbc0b079，终审 6e55d62f378b6d94ad8b74d99c7b2864658b974071cfa2cd2c28f46cb0b5b804。R21 cache 是私有副本，以 bwrap namespace 保留 stamp 绑定路径；R19/R20 原 cache 对照 R20 post-run snapshot 的 1,235 项均完全未变。准备阶段 no-build 复制失败、namespace wrapper/审计脚本误判及误启动 PT-only 中断记录都保留；最终 R3 cohort 自然结束，无 worker 残留。

后续优先级：对已有 trace 做有界性能诊断/优化；补近容量/OOM 边界及 activation checkpoint 实际 recompute on/off 显存/速度对照；在冻结候选上补声明范围 API/device/dtype/layout/autograd/fallback 矩阵、held-out 任务门、DDP Trainer 多 rank 保存和新进程恢复、JT/stock PyTorch 序列化契约、clean wheel/sdist 安装与 baseline 迁移审查。R14 practical train/resume、R19 unfused、R20 fused 与 R21 defaults 各自仅支持绑定的源码和配置，不合并成全局 PASS。Overall release 继续 **Not complete**。

---

## 2026-10-31 最新交接：R20 fused AdamW 敏感性审计完成；Day4–5 时延线仍未达

本轮在 /home/cgq/jittor 既有工作区，HEAD f95ea4cf4a7e1798f02c47b4bffab9dd38380c7f、source fingerprint 0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50 上，用 R19 同一主/compat wheels 与 private core 做 R20。实验前 102 项既有 dirty/untracked 与 status SHA256 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211、空 index 均保持；origin/cgq_transformers e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 113/0。只改变 PT/JT 的匹配 optimizer 配置为 foreach=False,fused=True，目的为敏感性诊断；未改源码或重建 core。

三轮交替、每侧每阶段 16 warmup + 10 热样本完成，GPU2 logical CUDA 0，fallback 0；梯度、master 初值、实际更新和 NVML stage audits 均通过。独立审计 attempt-02 通过（SHA256 0084bf97bf1a1ebfb6b956458ad8e06a174115bba19b151c26cb595e577f3d4d），但 fused cohort JT/PT median ratio 1.68174，时延诊断线失败；allocator allocated/reserved 与 NVML 峰值 ratio 分别 1.08062/1.06150/1.04545。该测量不能替换 R19 官方 unfused 时延失败 1.39597。

Artifact /data6/cgq/jittor-lab/_state/release-day23/20261031-current-wheel-bert-performance-r20-fused-adamw-sensitivity-r1-prepared/。R20 在复用的 private JIT cache 新增 12 个 fused 运行产生的 JIT 产品；终审核对其他既有 cache 项保持、lock 最后 PID 已退出。R20 attempt-01 独立 audit 的 schema 误判原件保留，attempt-02 对同一产物通过，无重跑 cohort。

继续顺序：

1. 用已有 R19 trace 做有界时延诊断/优化。R19 三轮 unfused 测量不重跑；任何候选源码/wheel/支持模式变化后，才以新身份复做规定的 3 轮交替 × 每轮每侧至少 10 个热样本。
2. 另做容量/OOM 边界，以及最终候选上的 checkpoint recompute on/off allocator/NVML 峰值和速度对照。
3. 在统一锁定候选补 API/op 支持矩阵、held-out 任务阈值、DDP Trainer 多 rank 保存与新进程恢复、JT/PyTorch 序列化契约、clean install/wheel smoke 和 baseline 迁移审查。

R14 full/resume、R19 三轮和 R20 sensitivity 绑定各自证据身份；勿重复已经通过的 full、preflight、resume 或 cold core build。当前 GPU2 已回到 0% 使用、81,038 MiB free；其他 GPU 仍有高负载，磁盘 / 可用约 33 GiB、/data6 约 5.5 TiB。三份状态文档已作中文提交；提交后 HEAD 指向该提交，origin/cgq_transformers 仍为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 114/0，远端 ls-remote 一致），index 为空，其余 102 项 dirty/untracked 状态 SHA256 仍为 5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211；未 push。Overall release 继续 **Not complete**。

---

## 2026-10-31 最新交接：R19 当前 wheel 配对性能证据已采集，热态时延线未达

R19 在当前 `/home/cgq/jittor` HEAD `4258239260f2b94c901714cf23d116e9b514b820`、source fingerprint `0382368b8754bf7a90e42382fd66b603ed2d3f323b5e8cda4820425871c7aa50` 上完成当前主/compat wheel 的非 editable BERT B8/S256 FP32 测量。origin/cgq_transformers 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind `112/0`。测量时保留 102 项既有 dirty/untracked（状态 SHA256 `5a6d83a6bce78a372494d8bd340a7a6d685c368b5cbec203a42bbd543c070211`）；当时 index 为空，未改源码，未 push。

三轮 PT/JT 交替、每轮 16 warmup + 10 热样本、3 次 allocator 与 CUDA attribution update 均结束；实际 GPU2/CUDA 设备 0，fallback 0，梯度和参数更新检查通过，NVML stable-load audit 六阶段通过。独立终审确认 artifact 与 summary 完整，但 pooled hot median JT/PT 为 `150.630/107.903 ms`、ratio `1.39597`，未达 1.10 时延诊断线。allocator allocated peak ratio `1.08066`；reserved ratio `1.19237`、NVML peak ratio `1.15966`。该 BERT 代表格没有测容量/OOM，故 Day4–5 整体仍开放。

主要 artifact：`/data6/cgq/jittor-lab/_state/release-day23/20261031-current-wheel-bert-performance-r19-prepared/`。执行 receipt SHA256 `50c2f21ee62846783b7781854d3e59755e5ebcd91e20451b494511adbee6daf0`；summary SHA256 `180358119bc66f91ca30a80aa9820fbcac8cb47397dd6066ec253d875efb21a5`；终审 SHA256 `85a3a2774e1cc5c71fb11ef3e55f2edb8d7560422c4e4f58ccf679c460eb3090`。终审审计通过与性能时延门失败是两个结论。

接手后优先顺序：

1. 基于现有 trace 对当前 BERT 热更新时延做有界 profile/优化诊断；若改源码，创建新身份和 wheel，并复做同协议三轮配对，不放宽 1.10 线。
2. 补 capacity/OOM boundary 和 activation checkpoint on/off 的 allocator/NVML/速度数据。
3. 在统一发布候选上闭合支持矩阵、held-out 任务指标、DDP Trainer 多 rank 保存恢复、JT/PyTorch 序列化契约与 wheel baseline 迁移审查。具体清单见[Day2–5 结果报告](2026-10-26-torch-compat-release-day2-5.md)。

R14 Qwen 实用训练、R19 BERT 性能代表格、旧候选 DDP/checkpoint/生成结果分别绑定各自源码和配置。当前没有证据把这些局部通过合并为完整 release。Overall release **Not complete**。

---

## 2026-10-31 接续交接：主 wheel CUDA smoke 通过；Day4–5 正式测量待当前候选绑定

在 /home/cgq/jittor 现有工作区继续，没有 clone。输入 HEAD 62208754d65007c5e0ae7a21a85ae73563353a9d、origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 111/0）。开工 109 项既有 dirty/untracked、porcelain SHA256 a1448a696968aa28fd29d60e28223958afcb3b9012fa7cb9fdcf3556605c0fb8；R4 smoke 前后同值。其余既有源码与实验产物保留。

本轮关闭了一项实际发布阻断：主 wheel 新增缺失的 CUDA linear Python 资源，主 sdist 排除核心包下的 compat 开发 symlink；修正 manifest/checker 路径。R3 的 sdist/wheel 内容检查通过，直接与 sdist 派生主 wheel 的 1,100 个成员逐字节一致；主包 CPU selftest、准确的组合 wheel CPU 合约通过。随后 R4 对主 wheel+compat wheel 做非 editable 安装，在 A800 GPU2、独立 JITTOR_HOME 上运行实际 CUDA 前向/反向、13 个模块导入、三步 Conv/BN/SGD 更新，以及 compat torch namespace 的 CUDA 梯度与 SGD 更新；两段 fallback delta 均为 0。loss、参数更新、SHA 和 77 次设备采样均写入 R4 artifact。测试只说明该确切构建的窄 smoke 能力。

R4 产物目录：/data6/cgq/jittor-lab/_state/release-day23/20261031-main-package-current-r4-cuda-installed-smoke-r1-prepared/。主 wheel SHA256 2e8f1a4418e3e9aee770843a6fd7d5471e3de12aaad75a8e8e78deee24baa3d9，compat wheel SHA256 fd7e84f639dba09f9d8293b9d255df09aa7190b890093e614c340cc47a7a7d5a，run.json SHA256 bed6853ab86c15f79a2d9e602af4b6931fb1a6164feca76d2dbb8ec449d9387c，日志 SHA256 2e408b9fcdfaa70a5676bc8d294a4862d56d460af4ef7cbf999325665d596c42。默认历史 wheel baseline compare 仍有 69 additions、2 removals、302 content changes 未审批，未重写 baseline。两个失败/非合约尝试（R2 排除嵌套 tools/build，首个 smoke 查询 shim torch.__file__）均留存。

后续按此顺序继续：先针对该确切 wheel/source state 建独立正式性能候选与当前源码 guard，不能复用锁定旧 source fingerprint 的 R18 release review；在稳定设备窗口跑 3 轮 PT/JT 交替 BERT 代表格，每轮每侧 ≥10 热样本并收集首次/热态与 allocator/NVML 数据；再独立完成容量/OOM 边界和 activation checkpoint on/off 显存/速度对照。随后对统一候选补齐 support matrix、held-out 任务指标与新进程恢复、DDP Trainer 多 rank、序列化互操作与 wheel baseline 迁移审查。R14 Qwen 实用训练和当前 wheel CUDA smoke 各自只支持其绑定的源码及配置。Overall release 保持 **Not complete**。

## 2026-10-31 补充交接：R18 BERT 三轮已审计但绑定旧源码；GPU2 readiness 需刷新

旧候选 `df45a565...` / HEAD `e92681d...` 的 BERT B8/S256 FP32 R18 已完成三轮 JT/PT 配对并经独立终审验证（SHA256 `08d7958485ff53880819f91418125b44a6e503354c82c9dcd084bbb5c9fb18e8`）；JT/PT median ratio 为 1.31283、allocator live peak ratio 为 1.08066。它不是当前源码，也不含容量边界。新主包安装门发现实际 wheel 阻断，先处理后再以最终候选做匹配测量。

GPU2 readiness artifact `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-day45-readiness-current-r1-prepared/` 有连续 5 次 5 秒采样（0% 利用率、81038 MiB free、无 compute PID），日志 SHA256 `412d3d16da4afc38fd7eae09450dac92801cf930eedf8b9b927aefd94ef4a154`。当时没有启动计时实验；下次正式运行须在新鲜准入后执行。Day4–5 和 overall release 仍未完成。

## 2026-10-31 最新续接：主包 clean install selftest 暴露 wheel 缺模块；Day4–5 未验收

本轮在既有 `/home/cgq/jittor` 继续，主包包门绑定 HEAD `a067d2dedb44cf6d039bb54fed2a3f5af94b7864`、origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`（109/0）。构建前 103 项既有 dirty/untracked 的 status SHA256 `ad24d0fe82a758525cec03ee1e79373524c55422c274ac6a3b7e6c89a88b4402`；源码和既有实验文件未改。本轮仅有隔离 `/data6` 包 artifact：`/data6/cgq/jittor-lab/_state/release-day23/20261031-main-package-current-r1-prepared/`，adjudication SHA256 `b432ec86ea1bd6098789d393b5b02fb952d60327a5095f08d47f0fcf597096dc`。

主 Jittor wheel/sdist 构建成功，直接 wheel 与 sdist 派生 wheel 的 1,184 个成员逐项相同；但 manifest check 失败，core wheel 包进 `jittor/compat/...`，安装后 selftest 又因 wheel 漏 `jittor.backends.cuda.kernels.cublas.lt_linear_cuda` 而在 import 阶段失败。兼容层 wheel 单独的 CPU 20 步更新通过，但它借用源码树 runtime，不能证明主 wheel 或组合安装可用。全量日志、哈希和 checker 限制写在 Day2–5 结果顶部；不要覆盖主包失败原件或重跑已完成的构建/兼容训练 smoke。

后续先修包内容/manifest 边界，在隔离干净环境通过主包 selftest 和主包+compat 组合 smoke；继续关闭当前候选声明范围、DDP Trainer 保存恢复、held-out 任务指标、JT/PT checkpoint 显存和速度配对、JT portable-pickle 与 stock PyTorch 的互操作契约。Day4–5 需等稳定低负载且无无关 GPU PID，再按至少三轮 JT/PT 交替、每轮每侧至少十个热态样本记录 load/首步/热态 median/p95/离散度、allocator/NVML 峰值和容量/OOM 边界。当前共享 GPU 不满足正式窗口。Overall release **Not complete**。

## 2026-10-31 续验交接：当前 DDP owner CPU 门通过；双卡同步路径仍需扩展

在服务器 `/home/cgq/jittor` 现有工作区继续，HEAD `5fcc3f049f0c724c17db29acf77e10f5c1c777ff`，origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind `108/0`。工作区 103 项原有 dirty/untracked 与 CPU DDP 用例前后 status SHA256 `ad24d0fe82a758525cec03ee1e79373524c55422c274ac6a3b7e6c89a88b4402` 一致；测试 index 为空，CPU core SHA256 `51845b1462d8f3beaf5f9c2034bd96363a2925b90ff1dcec2bbcf02f9aeb5818`。`compat/tests/torch/test_ddp_owner_separation.py` 当前全模块 15/15 通过，产物 `/data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-ddp-owner-cpu-current-r1-prepared/`。

保留的双卡 NCCL no_sync 用例曾在历史 HEAD `fda5675c892fed75f206bf36d65c45628e4d55ed` 的 GPU2/GPU4 上两 rank 各 1 passed；能证明那一候选上的梯度累积收尾、归约、AdamW 更新和 barrier 窄路径，不能直接提升为当前 release candidate 的完整 DDP 通过。记录见 `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r20-current-source-r1/`。尚需冻结候选的多 rank 结构化参数/optimizer-state 对照、Trainer 训练及新进程 checkpoint 恢复、完整 DDP session 门；no sync 局部结果不能替代这些门。

当前 GPU2/4 利用率分别为 83%/78%，GPU1 空闲 19,385 MiB；没有合格 Day4–5 性能/显存窗口。任务进程与 cache holder 均已退出，根分区余 33 GiB、`/data6` 余约 5.5 TiB。后续继续按低负载和显存准入条件等待配对测量；只更新既有三份状态文档，不改动其余 dirty 文件，不 push。Overall release 仍 **Not complete**。

## 2026-10-31 继续工作交接：optimizer CUDA 门通过；性能显存仍等稳定窗口

本轮验证开始时工作区为服务器既有目录 /home/cgq/jittor，HEAD 91e11f8a3dfbcc18dd182a07327fc995c4c482c2，fetch 后 origin e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 107/0。本文档修改前 105 项 dirty/untracked 原样保留；六文件任务变更已作中文提交，提交后仍有 103 项既有 dirty/untracked，其中 optimizer 测试的混合修改继续留在工作树。未 stash/reset/clean/push。测试期间未改动其他 dirty 文件。新增的 CUDA marker 识别与 policy 回归通过；optimizer 全模块 27 passed、0 skipped。相关 source hunks 属于本任务，其中 optimizer 测试文件混有原有 dirty 修改，只能按 hunk 暂存，不能整文件暂存。

交接时十张 GPU 利用率均超过 10%，GPU2 空闲 72,387 MiB 但外部进程仍使用 8,651 MiB；这足以做过小型正确性验证，但不满足 Day4–5 正式窗口。没有本任务测试进程存活。根分区余 33 GiB，/data6 余 5.5 TiB。origin fetch 后无需要整合的远端提交；未 push。

新增可复用证据：
- optimizer module full CUDA: /data6/cgq/jittor-lab/_state/test-suite/torch-cuda/20261031-optim-placement-full-cuda-r1-prepared/，manifest SHA256 3048570b2cac1cb5488a112a3c5f2051de2a7fdfb09e4970e0e48c77c093763。
- checkpoint actual CUDA recompute: /data6/cgq/jittor-lab/_state/release-day23/20261030-main-qwen-checkpoint-recompute-practical-r1/evidence/cuda-checkpoint-llama-attempt04/，功能与 fallback 通过，但无 on/off memory 结论。
- seed821 held-out generation pair: /data6/cgq/jittor-lab/_state/release-day23/20261030-main-qwen-checkpoint-recompute-practical-r1/evidence/heldout-generation-r1/，仅 17 行探索分数，绑定历史 source fingerprint。
- 历史 optimizer CUDA gate r5 失败及 seed823 generation 超时均保留；不重复 optimizer full、R14 full/resume 或冷编译。

恢复后的优先顺序：先抓取满足低占用/显存阈值且无外部计算的 GPU 窗口，执行已规定的配对性能/显存和容量协议；窗口未到时继续做不占 GPU 的最终候选兼容矩阵、DDP 状态/训练路径及 clean install/package/import/selftest 核验。checkpoint recompute 功能有 CUDA Llama 证据，但必须补受控显存收益/速度对照。保持 stock PyTorch 对 JT portable optimizer 文件的读取失败与 Jittor 原生新进程 resume 通过分别记录。只更新三个既有状态文档，不改写历史失败，不 push；Overall release 保持 Not complete。

## 2026-10-31 接续：optimizer CUDA 定向结果与发布门状态

当前远端工作区 HEAD `e1a3cdb00109c1605f20d5347dbd1b6f1b7a7e4c`，origin ahead/behind `106/0`；103 项原有 dirty/untracked 与测试前状态 SHA256 `ad24d0fe82a758525cec03ee1e79373524c55422c274ac6a3b7e6c89a88b4402` 一致，index 为空。GPU2 的两个 optimizer 设备放置测试已在其现存 CUDA core（SHA256 `83d5c1a0ac0481afbdbd250e295807f5b7e9005504fae055a0c77bc5a37d8aff`）上执行：2 passed；通用 CUDA gate 因动态 CPU/CUDA 循环未被 accelerator 计数器识别而 rc=1，fallback 未采集，故仍是窄诊断，不关闭 `KI-TEST-009`。产物及启动失败记录见 `/data6/cgq/jittor-lab/_state/test-suite/torch-cuda/20261030-main-optim-placement-focused-cuda-r5-prepared/`，manifest SHA256 `57db0c231684fee57efb4fed7a24b44d3b41f93c25adaac6ae38f03ce6abda4e`。

正式 Day4–5 性能/显存尚未验收；仍需合格负载窗口、三轮交替 JT/PT 配对测量、每轮每侧至少十个热态样本、延迟/吞吐分布、allocator/NVML 峰值和容量/OOM 边界。更广 Day2–5 还需最终候选的兼容矩阵、真实 held-out 模型指标、activation recompute、恢复/序列化、DDP、干净安装/打包。不得标记 overall complete。

## 2026-10-31 增量：CPU BF16 AdamW 新进程续训轨迹与不中断训练一致

为评估上一轮单进程 BF16 AdamW exp_avg_sq 对照差异是否表现为训练轨迹退化，在当前 CPU core 上做一次确定性小型线性回归续训诊断。Jittor Torch shim identity probe 通过，CPU dispatch、use_cuda=0、Jittor core SHA256 51845b1462d8f3beaf5f9c2034bd96363a2925b90ff1dcec2bbcf02f9aeb5818。固定输入/目标 dataset SHA256 199f2caeeba4d29ecf89a419edcb1ec583d9463676334e3dee6f55ff7d2a9ef1；Linear(2,1)、BF16、AdamW lr=0.01、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01、40 次 full-batch 更新，无随机层、RNG 依赖或 scheduler。

三条独立 Python 进程运行结果完整保留在 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-bf16-resume-trajectory-cpu-r3-prepared/。baseline 进程完成 40 次更新，训练 loss 1.298828125→0.1537342072，固定验证 MSE 1.9765625→0.2781715393；seed 进程完成 1 次更新并保存 checkpoint；resume 是另一新进程，从 step 1 恢复后再做 39 次，结束于 step 40。39/39 个恢复后训练 loss 与不中断路径逐项完全相同，最终验证 MSE 完全相同，参数和 exp_avg_sq 的最终张量 max_abs 差都为 0；三进程梯度有限、实际更新 40 次合计、fallback delta 均为 0。checkpoint SHA256 9e55af6849cbf4a4a420e8398778f198d17e9067ae214f56a2b15bb1b61d9096，汇总 manifest SHA256 fc5e9d3ceb58d3f389f25496c75aff207db5c91231fccfe3f469df89651698e。

这说明之前单进程里参考与恢复两个 optimizer 交替更新产生的 exp_avg_sq 差异，在该小型 CPU BF16 任务的新进程续训轨迹中没有复现。证据只支持此线性模型、数据、配置和 CPU 路径；它不是 Qwen/CUDA 结果，不替代 held-out 真实任务质量门，也不能覆盖 AdamW 其他 dtype/device。上一轮单 node 严格状态比较的 FAIL 原件仍保留，未改写为 PASS。当前结论是功能与任务轨迹窄路径通过，严格的单进程 moment state 比较仍有差异，需要按声明范围继续审视。

实验期间未改动源码、测试文件或编译 core。代码目标文件 SHA 与实验前一致；工作树其他 103 项 status SHA256 仍是 2307d0e94eb9b3bc7fba2ede6944fa17e077934dd2beedfee42190737f7ea25c。runner 的初次 shell 语法失败、r2 因 worker PYTHONPATH 缺失而未执行，以及 r3 parent 聚合阶段对 dict 型 exp_avg_sq 的后处理错误都保留在各自 artifact；r3 三个训练进程本身成功，汇总由已保存 JSON 的 post-hoc adjudication 完成，没有重跑训练。

正式 Day4–5 设备条件仍未满足：最新快照 GPU1 利用率 0% 但空闲显存仅 19,385 MiB；GPU2 空闲 72,387 MiB但利用率 87%，有外部 PID 3160291 使用约 8,640 MiB。CPU cache lock 无 holder，/ 余 33 GiB、/data6 余 5.5 TiB；无本任务活动进程。origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 105/0。overall release 仍 Not complete。

## 2026-10-31 增量：CPU BF16 AdamW 续训差异定位到 exp_avg_sq

在 /home/cgq/jittor 当前工作区、HEAD 5a41561b17bbbb412725c6edb2697063df1d2506 上，对上一轮唯一失败 node 做一次字段诊断，没有重跑整套。CPU Jittor core SHA256 仍为 51845b1462d8f3beaf5f9c2034bd96363a2925b90ff1dcec2bbcf02f9aeb5818，JITTOR_NO_BUILD=1，CUDA_VISIBLE_DEVICES 为空。失败确认为 CPU BF16 AdamW 的 exp_avg_sq：恢复前 state 完全相同；第二次 step 后参数对照通过，但 exp_avg_sq 为 [0.0314781479537487, 0.07143609970808029]，对照为 [0.03147624433040619, 0.07136933505535126]，max_abs=6.676465272903442e-05、max_rel=9.354809299111366e-04。该诊断只执行一个既有 node，1 failed、1 executed、0 skipped；没有改动源码、测试文件或 core。证据位于 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-bf16-adamw-restore-field-r2-prepared/，manifest SHA256 40ddd1570a4d9db2f099bf942a6787e8dd2cd6e79f8bf9802e6ce5ca6efbd77c。attempt-01 因默认 python 缺少 numpy，未执行测试，保留在同名 field-r1-prepared 目录。

这是恢复后优化器二阶矩的数值续接差异；它可能影响更长续训，但当前只有微型 CPU 用例，不能外推为模型任务退化，也不能据此宣布完全无害。R14 的 Qwen seed821 CUDA 训练与 resume 仍是另一源码/配置上的实际模型证据。结合两者，后续应评估该二阶矩差异在声明支持的 BF16 训练续训上的实际轨迹影响；不为严格末位对齐无限重跑。

最新设备快照仍不允许 Day4–5 正式测量：GPU1 利用率 0%，但仅空闲 19,385 MiB 且被外部进程占用约 61,653 MiB；GPU2 有 72,387 MiB 空闲，但利用率 90%，外部 PID 3160291 使用约 8,640 MiB。其余 GPU 亦不同时满足利用率 ≤10%、空闲显存 ≥60,000 MiB、无无关 PID。CPU core 的 lock 文件没有 holder；/ 余 33 GiB、/data6 余 5.5 TiB。没有本任务测试进程存活。

origin/cgq_transformers 仍为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，分支 ahead/behind 104/0。其余 103 项 dirty/untracked 仍保持 status SHA256 2307d0e94eb9b3bc7fba2ede6944fa17e077934dd2beedfee42190737f7ea25c；overall release 继续 Not complete。

## 2026-10-31 续接：保存/续训 CPU 复核发现 BF16 AdamW 状态差异；正式性能窗口仍不合格

在 /home/cgq/jittor 当前工作区继续，验证时 HEAD 1a6afb04bfac79078d887dbe9c3ff9849d4a8cfb，fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 103/0。复核前后其他 103 项既有 dirty/untracked 状态的 SHA256 为 2307d0e94eb9b3bc7fba2ede6944fa17e077934dd2beedfee42190737f7ea25c；本轮未改源码或测试文件，新增实验记录只写入 /data6/cgq/jittor-lab/_state/。

CPU 保存/续训定向门在 JITTOR_NO_BUILD=1、use_cuda=0、实际 CPU Jittor core 51845b1462d8f3beaf5f9c2034bd96363a2925b90ff1dcec2bbcf02f9aeb5818 上收集 39 项，执行 25 项：23 passed、15 skipped（全为 accelerator 条件）、1 failed，用时 34.84 秒。test_serialization_api_owners.py 所执行用例通过，覆盖 Jittor portable round-trip、weights-only 安全拒绝、Torch zip storage 解析及 safetensors reader；这不表示 stock PyTorch 可以读取 JT portable-pickle 的 optimizer 文件，既有互操作失败仍单独保留。日志和 preflight 位于 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-serialization-restore-cpu-r2-prepared/，manifest SHA256 004869af2e46160d63fec0b9d83b772ee3b132226c3c73d066315a9b334be936。

唯一失败为 test_map_cpu_restore_to_per_parameter_device_and_next_step[cpu-adamw-dtype1]（CPU BF16 AdamW）。checkpoint load 后的参数组 lr、step=1、optimizer state 张量与原实例逐项相同；继续做第二步时参数对照通过，但一个 optimizer state 张量对照失败，观测值 [0.03147815, 0.0714361] 对 [0.03147624, 0.07136934]，max_abs 6.676465e-05、max_rel 9.3548e-04。这是续训状态数值差异，尚未由更长模型级训练证明会造成任务指标退化，也不能直接忽略为无影响；应保留为未关闭的 optimizer-state 行为问题，并结合后续实际训练证据决定优先级。该失败不推翻绑定特定 CUDA 源码/配置的 R14 Qwen seed821 full 与新进程 resume 结果。

初次 CPU attempt 因 nvcc_path 选中了不存在的 CUDA 配置且 JITTOR_NO_BUILD=1，在 pytest collection 前 fail-closed，未运行测试；旧原件保存在 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-serialization-restore-cpu-r1-prepared/，manifest SHA256 2245a83001a0374f30a74e478b437638009fc9241fde966d7f99d62837435e3f。对照既有 CPU runner 显式设 JT_BUILD_NVCC_PATH='' 后复用 cfg1d275ac1 core 才进入用例。另一个临时字段追踪 runner 自身语法错误，目标用例未执行，原件保留在 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-main-bf16-adamw-restore-diagnostic-r1-prepared/，不作为测试结果。两次不完整尝试和后续实测均未改动五个目标源码/测试文件；status SHA 与 core SHA 前后一致。

Day4–5 正式 GPU 测量仍不具备准入条件：快照中十张 A800 利用率为 22–100%，没有卡达到 ≤10%；GPU2（UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0）为 80%、空闲 72,387 MiB，有外部 PID 3160291 使用约 8,640 MiB。/ 仅余 33 GiB，/data6 余 5.5 TiB；因此实验临时产物写在 /data6，本轮不做性能/显存验收。正式门仍须等待低负载、无关 PID 清空的窗口，做至少三轮匹配配置 JT/PT 交替，每轮每侧至少十个热态样本，并报告 load/首步/热态、median/p95/离散度、allocator/NVML 峰值和容量/OOM 边界。overall release 继续 Not complete。

## 2026-10-31 dtype skip 更正：当前 dirty 主树原测试体全模块 34/34 通过

当前 HEAD fead5cd02be5fdb333ecc99e30e472e832e743e9、origin e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，工作树 103 项 dirty/untracked 与 status SHA256 2307d0e94eb9b3bc7fba2ede6944fa17e077934dd2beedfee42190737f7ea25c 保持不变。本轮没有修改源码或测试文件。

在真实 A800 GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 的 cuda:0 上运行完整 dtype module，临时只在 pytest 收集时解开两个陈旧 skip：34 passed、0 skipped、0 failed，fallback 0→0，core SHA256 83d5c1a0ac0481afbdbd250e295807f5b7e9005504fae055a0c77bc5a37d8aff。被跳过的 .long() int64 cast 与 16 组 Torch promotion 矩阵均通过 CPU/CUDA 分支。普通未修改 suite 仍记录 32/2，因为源文件里的 skip 标记没有改；内存 hook 结果不能代替正式磁盘测试更新。

证据 /data6/cgq/jittor-lab/_state/test-suite/torch-cuda/20261101-main-dtype-inmemory-unskip-cuda-r1/，adjudication SHA256 109ce722245f63f28631e4cff7177fd038c14db8078751c59d09a9e2010bdc20。Day4–5 正式性能/显存、activation recompute 实际收益、任务质量门、序列化互操作和干净安装/发布 runner 仍开放；目标继续 active，整体 release Not complete。

## 2026-10-31 接续：当前主树 dtype CUDA 窄门通过；seed823 generation 未形成配对结果

工作区继续为服务器 /home/cgq/jittor，HEAD 5d73e0fb9abdeccc46d9acdc21a93125ddb76f6d，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 101/0）。保留 103 项既有 dirty/untracked；功能验证前后 status SHA256 2307d0e94eb9b3bc7fba2ede6944fa17e077934dd2beedfee42190737f7ea25c 一致。本轮未改源码、未 push，仅更新三份状态入口。

新增当前主树真实 CUDA dtype module 证据：34 项收集，32 passed、2 skipped、0 failed，fallback 0→0；GPU2 UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0、逻辑 cuda:0，core SHA256 83d5c1a0ac0481afbdbd250e295807f5b7e9005504fae055a0c77bc5a37d8aff。两个未关闭差异为 .long() 输出 int32 与混合 dtype binary promotion 未遵循 Torch lattice。证据 /data6/cgq/jittor-lab/_state/test-suite/torch-cuda/20261101-main-dtype-cuda-current-head-r1/。

seed823 17 条固定 generation 补充只得到 PT 17/17（EM 0、token-F1 0.207141、ROUGE-L 0.150043）；对应 Jittor attempt 在 300 秒内未完成第一条，处于首次 JIT 算子编译，exit 124，不提供 fallback 或质量结论。seed823 Jittor adapter 来自旧冻结 candidate，与 seed821 当前源码证据分开。保留完整记录 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-qwen-seed823-heldout-generation-r1/；其中 status 计数误写的原 protocol 未改写，另有更正文件。

Day4–5 正式性能/显存仍需合格稳定窗口；本次目标卡有共享负载，不能作性能验收。下一步优先处理或明确限制 .long() 与混合 dtype promotion 支持范围，并继续关闭真实 activation recompute/显存、模型质量阈值、跨框架序列化、干净安装等 Day2–5 门；等 GPU 利用率 ≤10%、空闲 ≥60,000 MiB 且无无关计算 PID 后，再按三轮交替/每轮每侧十个热态样本启动正式性能验收。整体 release 仍 **Not complete**。

## 2026-10-31 接续：当前主树状态契约 CPU 与 CUDA alias 窄验证通过

继续使用 `/home/cgq/jittor`，HEAD `59c775df3b833ce9d1a008478949610ea4b8d78b`，origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，ahead/behind `100/0`。原有 103 项 dirty/untracked 保留，测试前后 status SHA256 `83cebe9ab499bf7eac33eaf704f20dd58441ca5b7ee5bc662b560e6cd827fa2f` 不变；未重新 clone 或清理工作区。

新增当前主树证据：CPU daily-state 契约 24/24（preflight HEAD=`59c775d`）通过，R9 失败路径 3/3 CPU 定向用例（preflight HEAD=`2e02c1e`；随后仅有文档提交，代码身份相同）通过；真实 CUDA readiness 后 alias/device 两项 2/2 通过，进程采样映射到 GPU2 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`。完整记录及失败尝试见 `/data6/cgq/jittor-lab/_state/test-suite/torch-cuda/20261031-main-tensor-alias-cuda-20261031T043048Z/`。

这是窄功能证据，不构成性能/显存结论。当前没有满足低负载、空闲显存和无外部进程条件的 GPU；Day4–5 及支持范围矩阵、发布安装/序列化契约等更广门继续开放，overall release **Not complete**。

## 2026-10-31 接续：R9 两个失败路径在当前主树的 CPU 定向复核通过

在服务器 /home/cgq/jittor 既有工作区验证，HEAD 2e02c1e17ce8ff99d515b297c882125fa8b1dbec，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（ahead/behind 99/0）。103 项既有 dirty/untracked 保留；测试前后源码与 status hash 不变。三个既有 CPU nodeid（fill/zero alias 与梯度、内部 FollowRuntime 写入、public Tensor(native) alias）3 passed、0 skipped、0 failed，使用 JITTOR_NO_BUILD=1 和已热 CPU core。证据目录 /data6/cgq/jittor-lab/_state/test-suite/torch/20261031-r9-focused-cpu-main-042414-r3/，preflight/raw/result SHA256 与文件身份见整改看板及 Day2–5 结果记录顶部。

该结果只支持当前主树的三个 CPU 分支；R9 隔离候选 7 passed、1 skipped、2 failed 的原始结果保持不变，不记为 R9 全组通过。下一步仍需围绕最终支持范围完成 API/device/dtype/layout/autograd/fallback 矩阵、模型训练/恢复和预设 held-out 质量门、真实模型 checkpoint 重算与显存收益、DDP 与干净安装/打包、明确 JT/PT 序列化互操作契约。

Day4–5 性能/显存需另等合格设备窗口：利用率 ≤10%、空闲显存 ≥60,000 MiB、无无关 PID；冻结匹配 workload，至少 3 轮 JT/PT 交替、每轮每侧 ≥10 个热态样本；报告 load/首步/热态 latency 与 throughput 的 median/p95/离散度、allocator allocated/reserved 与 NVML 峰值，并单独测容量/OOM 边界。当前共享 GPU 快照不合格，不能作为验收测量。Overall release 仍为 **Not complete**。

## 2026-10-31 当前接续：Torch storage alias CPU 门通过；CUDA 与 Day2–5 release 继续开放

继续使用服务器 `/home/cgq/jittor` 原工作区。当前 HEAD `ef21322dc9b38c89c2cbb07854ba65e8c442c025`，origin `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，分支 ahead/behind `98/0`。既有 103 项 dirty/untracked 保留；仅本任务源码增量与三份状态文档纳入此次整理，不 push。

已修复 `torch.Tensor(existing_var)` 未共享 storage 的问题，维护 Torch CPU core runner 230 collected/executed、227 passed、3 skipped、0 failed。默认 autograd 图关系仍有 KI-COMPAT-004 缺口。CUDA 定向尝试 attempt-01/02/03 均未到实际张量检查（首次 utility reload、第二次预检 cwd 错误、第三次 no-build core stamp 不匹配）；原始证据保存在 `/data6/cgq/jittor-lab/_state/test-suite/torch/tensor-alias-cuda-20261031/`，本轮未编译新 CUDA core。GPU 共享负载下不开展正式性能/显存测量。

接续优先项仍是实际功能门：在现有已构建且源码身份明确的 CUDA core 可用时补做 alias/device/fallback 检查；随后继续 Day2–5 支持范围 API 与训练/恢复/任务质量、checkpoint 模型重算和显存、默认 DDP、安装/打包、序列化互操作边界。Day4–5 性能/容量门必须使用匹配配置和合格稳定窗口，至少三轮交替、每轮至少十个热态样本，报告 load/first/hot、median/p95/离散度及 allocator/NVML 峰值和 OOM 边界。overall release 仍 **Not complete**。


# 2026-10-21 工作区快照与本机接续交接

## 2026-10-31 补充：Torch CPU core gate 有一项既有测试失败；正式性能门仍不具备条件

本轮继续在 /home/cgq/jittor 原工作区；HEAD 4c2909d45516db60aa45f6b1513dd3bf53a28c24，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81（97/0），103 项原 dirty/untracked 与空 index 保留。维护 CPU core runner 在隔离 cache 上完成 230/230，226 passed、3 skipped、1 failed；失败为既有 dirty test_independent_frontend.py 的 torch.Tensor(original) data_ptr alias 断言。完整日志 /data6/cgq/jittor-lab/_state/test-suite/torch/core-cpu-recovery-20261031.log，SHA256 b7f8820f1c28948cb5ca9398b5ff31b8c6e571ce95b3eb532a4bc22df3078fa2；本轮未记录 fallback 数。

测试后 GPU2/4 利用率 94%/96%、空闲显存 51,296/51,286 MiB，仍不满足 Day4–5 正式准入。后续先处理或限定 alias 失败对应的支持声明；性能门需等待 ≤10% 利用率、≥60,000 MiB 空闲及无无关计算进程，再冻结匹配 JT/PT 配置做至少三轮交替、每轮每侧十个热态样本，并给 median/p95/离散度、load/首步/热态和 allocator/NVML 峰值、容量/OOM 边界。完整剩余发布门见[Day2–5 结果](2026-10-26-torch-compat-release-day2-5.md)顶部更新；overall release 仍 Not complete。

## 2026-10-31 补充：R20 DDP 参数与 optimizer 状态报告通过

HEAD b45bc7eb9523f530dfc4973ce9b2eec55ce5d7b3（origin/cgq_transformers ahead/behind 96/0）上的双 rank CUDA/NCCL no_sync 用例 exit 0；GPU2/GPU4 进程均为逻辑 cuda:0，各 1 passed/0 skipped、sentinel 1/1。结构化报告确认首个 no_sync 梯度保持 rank-local，closing backward 后梯度、最终参数与 optimizer state 两 rank 匹配；15 个参数标量改变，max abs update=0.014990580268204212，optimizer step=[1,1]，梯度/参数有限。证据 structured-current-head-ddp-r1/ 中 audit SHA256 为 bb4310ee55dac696a22c2c2b22b93abdb4664f5d5d98aa8424864ecd54ee90e5。nox 未安装；只关闭此 DDP 窄路径，不代表完整 nox、FSDP 或其他 collective 通过。

预检 GPU2/GPU4 利用率 97%/94%、空闲 51,296/51,286 MiB 且有共享计算进程，不能做 Day4–5 正式测量；根分区余 33 GiB、/data6 余 5.5 TiB，无本任务进程或 cache holder。正式配对性能/显存与整体 Day2–5 release 仍未完成。


## 2026-10-31 最新接续：当前源码 Accelerate DDP 定向门通过；Day4–5 仍未开始正式验收

继续使用 /home/cgq/jittor 原工作区。验证 HEAD fda5675c892fed75f206bf36d65c45628e4d55ed；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 95/0。测试前 103 项 dirty/untracked 保留，本轮 noxfile.py 修正增加一项；没有 clone/reset/stash/clean/push。全部运行与失败原件在 /data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r20-current-source-r1/。

NCCL warmup 的 singleton 路径现显式调用 setup_nccl()。launcher 现将单卡屏蔽后的 backend_local_rank 用作标准 LOCAL_RANK，避免 Accelerate rank 1 把不可见 device id 1 传给 barrier。最终 attempt-03 双 rank 通过：GPU2/GPU4 各进程 cuda:0、local_rank=0；每 rank 1 passed、0 skipped，sentinel 1/1，launcher exit 0；梯度归约、optimizer.step、收尾 barrier 和 fallback 守卫通过。nox 不可用，pytest 实际为 7.3.1，本结果不是完整 nox session 通过；当前尝试未生成参数/optimizer-state JSON。

Day4–5 准入不合格：GPU2/4 利用率约 31%/29%、空闲 59,941/59,931 MiB，且有外部 VLLM PID。正式验收要等利用率 ≤10%、空闲 ≥60,000 MiB 且无无关计算进程，再冻结匹配 JT/PT workload，做至少三轮交替、每轮每侧十个以上热样本，报告 load/首步/热态分布、allocator allocated/reserved、NVML 峰值及独立容量/OOM 边界。

更广发布门包括声明范围 API/device/dtype/layout/edge/autograd/fallback，目标模型训练与 held-out 阈值，模型级 checkpoint 重算/显存对照，其他分布式范围，安装打包和跨框架序列化边界。Qwen 训练/恢复和 17 条 generation 对照、Llama 单层 recompute 各有窄证据，不能代替这些门。Overall release 仍为 Not complete。


## 2026-10-31 最新续接：当前主树 NCCL barrier 主机同步窄验证通过

当前工作区 /home/cgq/jittor，HEAD db4618b7f73e7dd77d1822a3d3d614366d44508a；评测前后 source fingerprint b5779f89d464541ecb464632e407d5fc5b9e361c7cff8dd6ddbd7b8e19bf3e97 不变。两个独立 rank 从本树导入 Jittor，在真实 GPU2/GPU4、NCCL、cuda:0 上验证显式与默认 barrier；rank 1 在 rank 0 延迟写入 host marker 后返回默认 barrier，marker 已可见、等待约 0.509 秒，fallback 两端均为 0。当前 barrier 源 SHA 3a374332a539cd862bea1ac9a34e0febe6bc4392ffb2aecf4683d823aea276e6，复用 nccl0 core SHA 3b646629cb5d3a3d5804ba0aefe066617c90364e4038438a936e87b301d6ffce。结果与限制见 /data6/cgq/jittor-lab/_state/release-day23/20261031-main-nccl-barrier-sync-probe-r1/，result.json SHA256 b5c9ec3d906e4c86d1927cfc2c381ca10173d3057d013e9d69d4ce7eaa9b7a06。

attempt-01 只因 Conda libstdc++ 缺少 GLIBCXX_3.4.30 在 Jittor import 前失败；attempt-02 预加载系统 libstdc++ 后 exit 0。R15 的临时 Var.sync 候选失败记录保持不变；本轮证明当前 marker.numpy() 路径，不证明完整 Accelerate DDP/no_sync。共享 GPU 下没有测性能或显存。下一步继续处理公开普通 DDP/no_sync 功能闭环；Day4–5 正式测量仍等稳定合格的设备窗口，overall release 继续 Not complete。


## 2026-10-31 最新续接：seed821 Qwen held-out generation JT/PT 对照完成

评测在 /home/cgq/jittor 原工作区完成，HEAD fcebf8748b1e7a8667f96313a0b42bd226cfd9ab，源码 fingerprint 9ebffa322d64e4bbad6a4eb6959c3756410a5cccfcaee6bf661571c1a074b27f；评测前后 source inventory/status hash 不变。fetch 后 origin 为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 93/0。评测前保留 103 项既有 dirty/untracked，index 为空；无 reset、stash、clean、clone 或 push。

Qwen3-0.6B LoRA seed821 JT/PT 最终 checkpoint 在共同 17 条 held-out fixture 上完成 CUDA greedy generation；训练数据/config 关键字段一致，base/训练/eval fixture SHA 一致，adapter config 只在 target_modules 数组顺序不同，语义归一后相同。两端 GPU2 cuda:0、FP32 inference、总 token 96。PT EM/F1/ROUGE-L 为 0 / 0.208718 / 0.149341；JT 为 0 / 0.204534 / 0.148379；原始输出 14/17 相同。Jittor fallback 0→0。仅支持当前 checkpoint 和 17 个样本的有界对照，不作为整体 generation gate PASS。完整证据：/data6/cgq/jittor-lab/_state/release-day23/20261030-main-qwen-checkpoint-recompute-practical-r1/evidence/heldout-generation-r1/paired-metrics.json。

attempt-01 因自动扫描构建外部 Motion324 扩展时缺少 omegaconf 在模型加载前失败；attempt-02 300 秒超时，8/17；attempt-03 使用 JITTOR_TORCH_SKIP_EXT_BUILD=1 后全量通过，前提是此 eager-attention 推理路径不依赖被跳过扩展。所有失败和 partial 产物保留。GPU 仍共享，评测后约 30% 利用率、59,941 MiB 空闲，未作任何正式性能或显存结论。

下一步仍先等待 Day4–5 正式设备窗口，再做匹配 workload 的三轮以上 JT/PT 交替、每侧每轮至少十个热态样本、延迟/吞吐分位数和 allocator/NVML 峰值及容量/OOM 边界。广泛 Day2–5 继续完成支持范围 API/autograd/fallback、默认 DDP 终审、模型级 checkpoint 显存收益、安装/打包和跨框架序列化边界；overall release 保持 Not complete。


## 2026-10-31 最新续接：CUDA LlamaDecoderLayer checkpoint 重算与梯度通过窄验收

在当前 Jittor CUDA 构建上，LlamaDecoderLayer 的 tiny config 单层测试将输入及 7 组 Linear 参数实际放在 GPU2/cuda:0（UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`），checkpoint 前向与 backward 重算调用 `[1,0]`，输入/q_proj.weight 梯度对照未 checkpoint 基线通过，fallback=0。attempt-04 为 1 passed/0 skipped；外置 wrapper 只放宽 exact 输出比较，实际 max_abs=0，梯度容差断言仍使用原值。attempt-03 原 strict 断言以 max_abs `1.1920929e-7` 失败并保留，attempt-01/02 也保留；不改写 strict FAIL。

实验在状态文档提交 HEAD `9dac3aaa02711cc9fe7b1af1c354158b2ee6e527` 运行。worker 启动行硬编码了前一快照身份；post-run inventory audit 显示与原训练快照相比仅三份状态文档变化，代码和测试一致。以 `identity-adjudication.json` 记录此事后身份审计，不把失真的启动行当作实时 HEAD 证据。详情、哈希和剩余门见 Day2–5 结果报告。

此项证明 CUDA 单层 checkpoint 重算和一阶梯度，不证明显存节省或完整模型训练。Day4–5 正式性能/显存窗口仍因共享负载不合格；整体 Day2–5 和 release 均未完成。



## 2026-10-31 最新接续：当前源码 Qwen 两 epoch 与 checkpoint-51 新进程恢复通过

继续使用 /home/cgq/jittor 原工作区。实验源码身份 HEAD 0f9fcab760199be9eb696806fb2fb88ea9951f6c，fingerprint de5a3c9a283bc4c4a0b4d4602dee409615a006d90bb97822474a929f9197d099；fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 91/0。开工时 103 项已有 dirty/untracked 保留、index 为空；本轮未改源码/测试，只在收尾更新指定三份状态文档；无 clone/reset/stash/clean/push。

隔离证据目录 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-qwen-checkpoint-recompute-practical-r1/`。当前源码私有 CUDA 构建绑定 GPU2 UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`、逻辑 `cuda:0`；Qwen3-0.6B LoRA seed821、BF16、2 epoch、102/102 次更新成功、0 skip、0 fallback，epoch 微批 loss 均值 1.205599→1.153534，固定 eval loss 2.469817→2.456431。新进程从 checkpoint-51 到 102 又完成 51 次更新；Python/NumPy/CPU/CUDA RNG audit 通过。连续 full 与 resume 用 atol=rtol=0 对照：930 个状态张量、样本顺序、更新协议和最终状态均匹配。reload/merge 生命周期 smoke 通过。此结果只支持固定源码和配置，不含 held-out generation 质量结论。

最新设备快照 GPU2 约 35%/59,941 MiB 空闲，GPU4 约 26%/59,931 MiB 空闲，存在共享计算；/ 可用约 33 GiB、/data6 约 5.5 TiB。无本任务 runner 或 cache holder。Day4–5 正式性能/显存验收未开始，仍待匹配冻结配置的 JT/PT 至少三轮交替、每轮每侧至少十个热态样本，报告 load/首步/热态统计、allocator/NVML 峰值与容量/OOM 边界。

更广发布门仍未完结：声明范围 API/device/dtype/layout/edge/autograd/fallback 矩阵、默认 DDP 完整 gate、共同 held-out JT/PT generation 判据、CUDA 模型级 activation recompute/显存对照、安装打包检查，以及 JT portable-pickle 与 stock PyTorch 的互操作边界。Overall release **Not complete**。细节与证据哈希见 Day2–5 结果记录顶部。


## 2026-10-31 最新接续：Torch factory leaf 训练路径通过，Day2–5 仍未闭合

继续使用服务器现有 /home/cgq/jittor。HEAD 15a6bc19832bd5af55d17104fe1ec2a6d78591a9；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 90/0。保留工作区 103 项 dirty/untracked；无 clone/reset/stash/clean/push。本轮前 index 为空，只更新本交接、整改看板和 Day2–5 结果记录。

当前 dirty factory/autograd 实现已通过 strict shim CPU 回归：autograd 语义文件加 JVP 高阶导用例 27 passed、2 accelerator skip、fallback=0；常量 factory leaf 收到预期梯度，并经 SGD 更新到 0.8/0.7。attempt-04 曾报告第一次读回仍为 1；attempt-08 无显式 sync 后立即读取通过，因此暂未复现为稳定功能故障。attempt-02/03 的命令与路径错误、attempt-04 断言失败及后续复核日志均保存在 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-torch-autograd-leaf-r1/；测试源码身份和文件哈希见 Day2–5 结果记录。

GPU2/4 利用率 30%/26%、空闲 59,941/59,931 MiB，存在共享 GPU 进程；正式 Day4–5 性能/显存窗口不合格，未做测量。后续发布门仍包括稳定窗口 JT/PT 配对性能和容量/OOM、声明范围 API/device/dtype/fallback、模型级训练与 held-out 质量、新进程 optimizer/scheduler/RNG/data-order 恢复、真实 checkpoint 模型重算/梯度/显存、默认 DDP 最终 sentinel、共同 generation 判据、跨框架序列化边界及安装打包检查。Overall release 保持 Not complete。


## 2026-10-31 最新接续：修复 checkpoint 反向重复路径，尚未完成显存与发布验收

继续使用服务器现有 /home/cgq/jittor。当前基线 HEAD bbb0e45f015a5bceb86886b26445e197709be659；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 是祖先（ahead/behind 89/0）。不得重新 clone、stash、reset、clean 或 push。既有 103 项 dirty/untracked 必须保留；本任务只改 checkpoint 实现、对应回归测试和本交接指定的三份状态文档。

最新功能增量为 compat/torch/installers/data.py 中将 checkpoint 上游 grad_output detach 后再构造重算目标，避免平方 loss 的上游梯度沿 checkpoint 原输出形成二阶回流。attempt-44 LlamaDecoderLayer 单层对照 1 passed；attempt-45 checkpoint 专项文件 6 passed、0 skipped，CPU strict shim、CUDA 隐藏、JITTOR_NO_BUILD=1。测试时源码 fingerprint d0528c457f3185f8b07c83ffba76b8b94013e281a8edeea69df30dbe29055166，Jittor core SHA256 前后均为 6a21f70f357d767f12c136a81f9449fb3420245db8b18c99990f31f5fbf1ce05。完整证据位于 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-checkpoint-recompute-r1/attempt-44/ 与 attempt-45/；attempt-39 至 43 的失败与诊断日志继续保留。该结果只证明一层重算和一阶梯度，没有完整 Llama 训练、CUDA RNG 或显存收益测量。

当前 GPU2/4 利用率约 32%/32%，空闲 59,941/59,931 MiB，未达到正式性能门槛；/ 剩余约 34 GiB，/data6 剩余约 5.5 TiB；测试结束后未发现本任务进程或 cache holder。正式 Day4–5 仍需同源码/权重/数据/配置、连续稳定 GPU 利用率 ≤10%、空闲 ≥60,000 MiB、无无关计算 PID、至少 3 轮交替 JT/PT 且每轮每侧 ≥10 个热样本，报告 load/首步/热态分布及 allocator/NVML 峰值，另测最大 batch/sequence 与 OOM 边界。

后续发布门依次补齐声明范围 API/device/dtype/layout/edge/autograd/fallback，代表模型训练、梯度/更新/loss 与 held-out 质量，新进程 optimizer/scheduler/RNG/data-order 恢复，真实模型 checkpoint 梯度和显存对照，默认 DDP 最终 sentinel，JT/PT 共同 generation 判据、portable-pickle 互操作范围，以及安装/打包检查。Overall release 保持 Not complete。


## 2026-10-31 checkpoint 重算续接与剩余验收门

当前服务器工作区仍为 /home/cgq/jittor，未重新 clone。HEAD f81f994266147c25c87135c7e00623dd2265a444；fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 88/0。原有 103 项 dirty/untracked 必须保留；本任务当前仅有 compat/torch/installers/data.py、compat/tests/torch/test_torch_checkpoint_recompute.py 两项实现/测试修改，状态文档仅允许更新 architecture board、Day2–5 results、workspace handoff。不要 stash/reset/clean 或 push。

当前 checkpoint 改动让 Transformers GradientCheckpointingLayer 的 CPU 小协议用例通过，attempt-30 全文件 5 passed / 0 skipped / 1.49s；真实 LlamaForCausalLM attempt-31 超时，DecoderLayer attempt-37 断言失败并报告 9 个 hold/live vars 增量。attempt-38 的 retain_graph=False 引发 native segmentation fault，该参数已经撤回。上述日志都留在 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-checkpoint-recompute-r1/，不要重跑 R14 full/preflight/resume 或已超时的 Llama 全模型探索。checkpoint 现在有实际函数级重算和梯度证据，但还没有真实模型通过，也没有节省激活显存的实测。

当前 GPU2/4 约 30%/33%，空闲显存 59,941/59,931 MiB，存在共享 VLLM 进程；/ 剩余 34 GiB、/data6 剩余 5.5 TiB。待 GPU 稳定达到 ≤10% 利用率、≥60,000 MiB 空闲且无无关计算 PID 后，才运行冻结配置的正式配对测量：3 轮交替 JT/PT，每轮每侧 ≥10 个热态样本，报告 load/first/hot 的 median、p95、离散度、allocator allocated/reserved 与 NVML 峰值，另测最大容量和 OOM 边界。更广 Day2–5 仍要完成支持矩阵、声明模型训练/held-out 质量、恢复状态、默认 DDP、真实 checkpoint 模型/显存证据、共同 generation 判据及跨框架序列化范围。整体 release 仍 Not complete。

## 2026-10-31 checkpoint 重算工作区接续

继续 /home/cgq/jittor 原工作区。提交前 HEAD 77b3ced2be7cae286f02f63839f822b704d56a0e；origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 87/0。所有既有 dirty/untracked 保留，未 push。任务新增 compat/torch/installers/data.py checkpoint 重算路径及四项定向测试；工作区整树 fingerprint 9b5fc3e218c41acd26ccf112597af96c761790f57c9c42c7ee757c811872a65，证据只绑定本次测试时源码身份，不等同于文档提交后的 HEAD。

attempt-29 在 CUDA_VISIBLE_DEVICES=""、CPU strict shim、JITTOR_NO_BUILD=1 下 4 passed / 0 skipped / 0.50s：覆盖闭包/模块参数梯度、嵌套输出、CPU RNG 重放与 caller RNG 恢复、no-grad 直通及 partial(super().__call__) 的 Jittor 模块形态。core 测试前后 SHA256 都是 6a21f70f357d767f12c136a81f9449fb3420245db8b18c99990f31f5fbf1ce05；cache 文件清单无增删，仅 jittor.lock 内容变化，详见 attempt-29/evidence/cache-delta.diff。证据目录 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-checkpoint-recompute-r1/。HF GradientCheckpointingLayer protocol 的失败日志见 attempt-25/27/28：没有 backward 重算，梯度为零；不得记为 HF/模型级通过。CUDA RNG 和峰值显存收益也未验证。

Day4–5 下一步等 GPU 连续稳定满足利用率 ≤10%、空闲 ≥60,000 MiB、无无关 PID；冻结匹配源码/权重/数据/精度/batch/sequence 后做至少三轮交替 JT/PT、每轮每侧至少十个热态样本，报告 load/首步/热态 median、p95、离散度、allocator allocated/reserved 与 NVML 峰值及独立容量/OOM 边界。最新快照 GPU 2/4 为 31%/29%、空闲 59,941/59,931 MiB，因此本轮不做性能/显存计时。继续发布门还包括修复 HF checkpoint 集成并测梯度/显存、声明支持范围的 API/device/dtype/autograd/fallback、默认 DDP 最终 sentinel、真实模型 held-out 质量、新进程恢复与跨框架互操作边界；R14 与 R19 既有状态不重跑、不改写。整体 release 仍为 Not complete。


## 2026-10-31 R22 当前状态接续

继续服务器 /home/cgq/jittor 原工作区；测试时 HEAD a5e5b99be0ce074df70e10670cf179ab71d81f34，fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead/behind 85/0。保留 103 项既有 dirty/untracked，index 为空；本轮实际功能验证通过后只会提交三份规定的状态文档，不 push。测试身份 c6722fd90cacb8265eee69f313a7f9d0486e3a7f1beab7fd8ff78b59ea94bf01 绑定文档提交前的代码树。

R9 native saved-artifact audit 为 PASS（原生测试 2 passed），但更广 terminal audit 因 cache drift/lock 条件为 FAIL；R9 旧 Torch focused 仍保留 2 个功能失败。当前源码 CPU autograd 全文件通过（26 passed、2 accelerator skip、fallback=0）；随后真实 CUDA FollowRuntime alias 定向用例在 GPU4/cuda:0 通过（1 passed、无 skip、fallback 未增加）。CUDA core/stamp/util 未变，借用 cache 的 10 个文件内容有变化；测试未改源码，也没有留下进程。详见 Day2–5 结果记录与 /data6/cgq/jittor-lab/_state/release-day23/20261101-main-follow-runtime-cuda-r1/。

优先继续完成实际支持范围的功能发布门：扩展 API/device/dtype/autograd/fallback 覆盖，明确声明模型的训练/任务质量和新进程 checkpoint 恢复，完成真实 activation recompute 与梯度/峰值显存对照，并闭合默认 DDP 最终 gate。当前 HF forward-equivalence 用例只比较 no_grad 输出，缺少 backward、recompute 计数与显存断言。Qwen 需共同 held-out generation 质量判据；JT 同框架恢复和 stock PyTorch 对 JT portable-pickle 的读取失败分开管理。

Day4–5 性能/显存验收另等稳定 GPU 窗口（利用率 ≤10%、空闲 ≥60,000 MiB、无无关计算进程），冻结匹配的源码和 workload，做至少三轮 JT/PT 交替配对、每轮每侧 ≥10 热样本，报告 load/首步/热态 median/p95/离散度、allocator allocated/reserved、NVML 峰值及独立容量/OOM 边界。当前 GPU 共享 VLLM 负载，只能用于功能测试；overall release 仍为 **Not complete**。


## 2026-10-31 当前主树 Qwen 实训与续训接续

继续使用服务器 /home/cgq/jittor 原工作区，未 clone。当前 HEAD 57189bde41a01d657b870e1a2c47ac955a175ec5；fetch 后 origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，ahead 84 / behind 0。开工时 103 项既有 dirty/untracked 保留，index 为空；本次只改本交接、整改看板和 Day2–5 结果文档，不 push。当前源码指纹 ae2cb7e24fcd6e58d555d829ecf0e33411c17ab475174ab7500a336d7b71fbfc，训练前后一致。

当前源码 Qwen3-0.6B LoRA seed821 实训在真实 GPU4/cuda:0 完成 2 epoch、102 updates、0 skip、0 fallback；epoch 加权微批 loss 1.205469→1.153617、eval loss 2.455552。新进程从 checkpoint-51 恢复到 102，执行 51 次更新，状态/RNG 核验和 atol=rtol=0 的 full-versus-resume 协议/样本顺序/更新比较均通过。完整证据位于 /data6/cgq/jittor-lab/_state/release-day23/20261031-main-qwen-practical-current-source-r1/，full 和 resume 产物未写入主仓库。R14 旧快照、当前主树训练与恢复分别记录；当前结果不外推其他配置。

接续优先做真正影响功能和显存的 checkpointing 路径：定位当前直接调用 shim 与 Hugging Face/模型检查点路径的关系，完成 activation recompute、前后向数值/梯度和峰值显存对照；然后继续补齐声明模型/API/device/dtype/autograd 和 generation/恢复矩阵。跨框架 portable-pickle 边界、R19 sentinel 外层失败保持原状态。Day4–5 等合格设备窗口后再做正式配对：GPU 利用率 ≤10%、空闲显存 ≥60,000 MiB、无无关进程，至少 3 轮交替 JT/PT、每轮每侧 ≥10 个热态样本，并报告 load/首步/热态 median/p95/离散度、allocator/NVML 峰值及独立容量/OOM 边界。当前 GPU4 约 29% 利用率、59,931 MiB 空闲，本次训练计时只能作诊断。Overall release 仍为 Not complete。


## 2026-10-31 最新接续：修正 DDP gate 的 CUDA 计数器

继续在服务器原工作区 /home/cgq/jittor。当前提交为 9f049b3e6472cc80ee246b4ac05932dfe1af1b73，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 为祖先（ahead 83 / behind 0）。保留所有既有 dirty/untracked；本任务只改计数器、对应结构断言和本交接要求的三份状态文档，不 push。独立结构断言通过：1 passed / 0.02s。它证明 R19 所用 Accelerate CUDA 集成 nodeid 会被 gate 计数器识别；R19 两 rank 测试体和旧外层退出码仍按各自原始记录陈述，没有为正常化 sentinel 而重跑。

一次误启动 shim pytest 在隔离 Jittor cache 开始冷编译后已被中断，cache /home/cgq/.cache/jittor/torch-shim/jittor-1a29170444cbf9f2（约 244 MiB）保留，没有清理或复用声明，也没有残留的 pytest/compiler/cache-holder 进程。正式性能快照仍不合格：GPU 2/4 利用率约 31%/31%，空闲显存 59,941/59,931 MiB，并有外部 VLLM 进程。根分区可用 33 GiB，/data6 可用 5.5 TiB。

完成 Day4–5 性能/显存门还需冻结并核对同一源码、模型权重和训练/推理配置；先满足 GPU 利用率 ≤10%、空闲显存 ≥60,000 MiB 且无无关计算 PID 的稳定窗口；在同一窗口完成至少三轮 JT/PT 交替配对，每轮每侧至少十个同步热态样本；每侧分别报告加载、首步和热态 latency/throughput 的 median、p95、离散度，以及 allocator allocated/reserved 和 NVML 峰值。独立给出最大可运行 batch/sequence 或相应 OOM 边界，不能把共享负载下的诊断数值作为验收结果。

更广 Day2–5 发布门还须按最终公开范围完成 Torch API、设备、dtype、edge case、fallback 和 autograd 对照；对声明支持的模型绑定真实设备/数据/权重/训练配置，给出梯度、参数更新、loss 轨迹、验证任务指标与 checkpoint；对需要 resume 的路径在新进程验证 optimizer、scheduler、Python/NumPy/CPU/CUDA RNG 和数据顺序；证明 gradient checkpointing 有实际 activation recompute，并比较梯度和显存；在共同 held-out prompts 上给 Qwen JT/PT generation 任务指标和预设判据；明确 JT 同框架恢复与 stock PyTorch 无法读取 JT portable-pickle optimizer 的跨框架互操作边界。当前只有 R14 单配置训练/resume 和 R19 窄 DDP 路径证据，不能替代上述发布范围，overall release 仍为 Not complete。

## 2026-10-31 最新交接：NCCL barrier 与 DDP no_sync 窄路径已验证

继续在 /home/cgq/jittor 原工作区；实验提交前 HEAD e17f90e1e5622c6f98d9c7f95534befc10a1967e，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 是祖先（ahead 82 / behind 0）。不 clone、不 reset/stash/clean、不 push；103 项原有 dirty/untracked 保留，本任务新增 distributed.py barrier 补丁单独记录。

R16/R17 将 NCCL barrier hang 收敛到 P2P transport 与 host materialization；在该主机需设置 NCCL_P2P_DISABLE=1。R18 barrier 改为 backend 对应 device marker，并用 host NumPy 物化等待。公开 barrier 测试在真实双 rank CUDA 上通过。R19 Accelerate no_sync→closing backward 目标测试体在两个 rank 均通过，首个 no_sync 梯度分离、closing 梯度一致、AdamW 更新一步，最终参数/optimizer 状态跨 rank 一致；设备、源码 SHA、逐 rank JSON 和外层 gate 计数失败均在下方 campaign 路径留有原件：
/data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r2/accelerate-ddp-no-sync-p2poff-r19/

attempt-01 pytest-timeout 参数错误与 attempt-02 缺少 nccl0 cache_name 都发生在测试收集前；attempt-03 的 pytest body 为 1 passed/rank，但 session sentinel 报 0 accelerator cases 并让总 launcher exit 1。按功能结果记录，不为改退出码重复该测试。core SHA 保持不变，R19 新增 JIT kernel 文件，运行后无 holder；未作 cache 清理。GPU 利用率过高，不能开始正式 perf/memory run。

后续优先顺序：
1. 等目标 GPU 连续满足利用率 ≤10%、空闲显存 ≥60,000 MiB 且无无关 PID，再为明确冻结源码/配置的 JT/PT 代表负载执行至少 3 轮交替配对；每轮两侧至少 10 个同步热样本，报告 load/first-step/steady 的 median、p95、离散度及 allocator allocated/reserved、NVML 峰值。容量与 OOM 边界另做，旧共享负载数字只作诊断。
2. 按目标支持范围逐项完成 Day2–5 模型和 API/device/dtype/autograd 矩阵，记录真实设备与 fallback、有限梯度/参数/loss、真实更新、验证任务指标，以及需要的新进程 checkpoint/optimizer/scheduler/RNG/data-order 恢复。现有 R14 只代表固定源码/seed/config。
3. 处理真实 activation checkpoint recompute/显存收益证据及 held-out Qwen generation 指标；将 Jittor 同框架续训和 stock PyTorch 读取 JT portable-pickle 失败分类记录，若跨框架读取属于发布声明再单独修复。
4. 保留严格数值对齐历史 FAIL，不把其改成 PASS，也不让无训练影响的末位差异替代实际训练目标。overall release 保持 Not complete。

## 2026-10-31 最新交接：NCCL barrier 双 rank host 同步未闭合

当前仍在 /home/cgq/jittor，HEAD e8dfc66b9d7e346d748f895e01e9c011ffe49966，origin e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 为祖先（ahead 81 / behind 0）。保留 103 项既有 dirty/untracked 路径；失败候选已撤回，未 push。

R15 真双 rank CUDA 2/4 探针显示：CPU marker 报 NCCL device 错误；CUDA marker 配合 device_sync=True 停住；普通 Var.sync() 虽使 barrier 返回，却让 rank 1 在 rank 0 写入主机 marker 前返回，故同步契约仍未满足。候选文件/补丁哈希、每次启动日志与脚本在 /data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r2/barrier-device-fix-r15/，manifest SHA256 e29ac2504de003b769bc8067d4258286394f2dfbcc30fa50a680498c6aafe5c7。生产文件已恢复到 HEAD，SHA256 8bfb068bf7f6e8c99b146deeac5b83163ad4607d63125168f081d52cbcc42f74。

下一步先用最小核心探针区分 NCCL all-reduce enqueue、双方参与、输出物化与 host wait，再决定 barrier/DDP 修复；之后才重跑 no_sync/closing backward/optimizer 更新与 rank 状态一致性。不要重复 R14 full/preflight/resume。Day4–5 等利用率 ≤10%、空闲显存 ≥60,000 MiB 的稳定窗口后，做至少三轮交替、每轮每侧至少十个热态样本，报告 load/first/steady 分布和 allocator/NVML 峰值，容量边界另测。held-out 质量、API/device 矩阵、真实 checkpoint recompute、更多模型与序列化范围仍未关闭；overall release 继续 **Not complete**。

## 2026-10-31 最新接续：optimizer 空恢复窄回归通过，普通 DDP 读回未完成

继续使用服务器已有 `/home/cgq/jittor`。提交前 HEAD `d28fd052dfa6219b20f62448e6226910bce9c801`，origin/cgq_transformers `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 为祖先（ahead 80 / behind 0）。全树已有 103 项 dirty/untracked 路径，均保留；本轮不 push。进程检查未见遗留 pytest/Jittor DDP 工作，GPU 2/4 仍约 33%/30% 利用率，不满足正式性能窗口。

Accelerate 包装器的 fresh empty optimizer restore 现在对 step-zero AdamW/SGD 复用既有零 slot，避免无状态初始化时多余的零张量分配与同步；对已 restore 后再加载空状态仍维持清零。相关 CPU/实际 CUDA 用例 10 passed、18 deselected、0 skipped、fallback=0；测试日志及 SHA256 见 Day2–5 结果记录顶部。

普通 Accelerate DDP no_sync 两 rank 测试仍未完成。R5 中 rank 0 在 all-gather 后的 `torch.allclose` 物化停住，rank 1 在初始参数 `_var_numpy` 读回停住；超出 70 秒预算后无测试终态或更新证据。原始尝试和栈采样在 `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r2/optimizer-restore-fix-r*/`，后续应针对两个 rank 的读回/图执行顺序做窄定位后才决定是否有依据再运行多卡测试。

Day4–5 仍要求匹配当前源码身份的私有 cold core/cache、连续稳定且无无关 PID 的 GPU 窗口、3 轮交替配对且每侧每轮至少 10 个热态样本、load/first/steady 的 median/p95/离散度和 allocator/NVML 峰值；显存容量/OOM 单独验收。广泛 Day2–5 仍需覆盖所声明模型的训练/恢复和任务指标、真实默认 DDP、activation checkpoint 实际重计算及梯度/显存证据、共同 held-out JT/PT 质量阈值、序列化互操作分类。整体 release 保持 **Not complete**。


## 2026-10-31 接续增量：普通 Accelerate DDP 首次同步路径未闭合

继续使用服务器原工作区 `/home/cgq/jittor`。代码身份 HEAD `4a03ff2eecb616b03e59b3a88c2a0533c4359da8`、fingerprint `674fc808ef91f0072b1d015fbfeee4f3a58e04940aaa43392bc894fcc27fb106`；文档提交前 HEAD `5ecd38739be63bb2738d989de0865640178dd1c0`，origin/cgq_transformers `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 为祖先（ahead 79 / behind 0）。本轮只产生外置实验记录，没有源码或测试改动。

普通 Accelerate DDP 的真实双 rank no_sync/closing-backward 用例未取得可验收终态。独立双 rank Accelerator 初始化探针已证明进程组包装和两个 CUDA rank 可启动；pytest `-s` 用例中 rank 0 到达第一微批次 backward 与 gather 完成，rank 1 未显示第一微批次开始，随后两边均未推进。600 秒 pytest timeout 没有终止 native 路径；进程约 10:49 后由本轮父进程 SIGTERM 停止。保留路径及哈希见 Day2–5 结果记录最上方新增项和 `/data6/cgq/jittor-lab/_state/release-day23/20261031-main-accelerate-ddp-nosync-r2/`。第一轮默认 pytest capture 造成“没有阶段输出”的解释已更正，不能把第一轮当作失败定位。

后续应从 gather 后阶段作最小诊断，给 rank-local 梯度比较、CPU 转换/记录以及进入第二微批次分别加有界标记或栈采样；先定位两 rank 的不同步点，再按真实 CUDA 设备、fallback、no_sync 梯度差异、closing backward 梯度一致、optimizer 更新及参数变化形成报告。不要从目前已执行的第一微批次推断 optimizer 更新成功。CPU MPI DDP 曾因构建无 MPI 而全跳过，仍无证据。

本轮测试期间目标 GPU 共享 VLLM 负载且利用率达到 100%，空闲显存低于正式门槛；Day4–5 性能/显存验收没有启动。待该门仍需匹配源码私有 cold build、连续稳定且无无关 PID 的设备窗口、至少三轮交替/每轮每侧十个热态样本、load/first/steady 统计与 allocator/NVML 峰值，容量另测。R14 的 Qwen 实用训练/resume 保持原有单配置证据；更广模型、activation recompute、共同 held-out JT/PT 质量和跨框架序列化范围继续未关闭。整体 release **Not complete**；唯一状态入口继续为看板。

## 2026-10-31 最新接续：R11g owner-aware alias 写回 CPU/CUDA 定向通过

继续在原服务器工作区 /home/cgq/jittor 推进，没有 clone/reset/stash/clean。主树 HEAD 4a03ff2eecb616b03e59b3a88c2a0533c4359da8，origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 为祖先（ahead 78 / behind 0）。最终源码 fingerprint 674fc808ef91f0072b1d015fbfeee4f3a58e04940aaa43392bc894fcc27fb106；2,815 项清单核验只有五条代码/测试路径变化，source manifest SHA256 c77f67cfcd3272fd4e94bbce63809cffb645db153916e3b3c395f87413544dda。

最终源码 CPU autograd 全文件 26 passed、2 个 accelerator 用例跳过、fallback=0；实际 CUDA FollowRuntime Tensor(native) fill_/zero_ alias 回归 1 passed、0 skipped、fallback=0，进程 PID 2942090 映射到 GPU UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0，逻辑 cuda:0，用量 428 MiB。owner 值、原 data_ptr、placement 和 device 检查通过。完整日志、worker result 和中间失败均保存在 /data6/cgq/jittor-lab/_state/release-day23/20261031-r11-current-main-root-verify-r1/attempt-18-r11-view-aware-write-fix/；详细 hash 和失败 attempt 说明见 2026-10-26 Day2–5 结果记录。

这些测试只关闭这条 constructor alias 功能问题，不覆盖完整 CUDA API，也不提供性能或显存结论。最新 GPU 快照中 GPU 2 为 33% / 59,941 MiB、GPU 4 为 28% / 59,931 MiB，没有卡达到正式测量要求。下一步要在最终 fingerprint 上单独建并审计私有 cold cache，等目标 GPU 连续稳定满足利用率 ≤10%、空闲显存 ≥60,000 MiB 且无无关 PID，再按三轮交替、每轮每侧至少十个同步热样本完成 load/first/steady timing、分布统计、allocator/NVML 峰值及独立容量测试。更广 Day2–5 的 T5/Qwen/Vision/BERT、默认 DDP、真实 activation recompute、共同 held-out JT/PT 质量阈值和序列化边界仍未关闭；整体 release 为 **Not complete**。

本任务的 R11 源码与测试修改保留在工作区 dirty 状态；本轮中文提交仅包含指定的三份状态文档，其他已有改动不暂存、不提交，也不 push。

## 2026-10-31 R11g 最新接续：隔离候选定向回归通过

本轮继续使用服务器现有 `/home/cgq/jittor`，没有 clone 或清理。当前 HEAD `dcc7f66ca412d3b2496478ddf27499dd21698bca`，远端 `origin/cgq_transformers` 最新提交 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 是祖先（ahead/behind 77/0）；80 个 tracked 文件和 21 个 untracked 文件仍 dirty，index 为空，全部保留。文档里较早的 `b2cf6bad...` / fingerprint `95eb54ba...` 是历史冻结身份，不与当前 HEAD 或 R11g 混用。

R9/R10 的 fill_/zero_ 后梯度缺失及 `torch.Tensor(native_var)` FollowRuntime 放置错误，在隔离 worktree `/data6/cgq/jittor-lab/worktrees/torch-compat-fill-zero-runtime-grad-r11-20261031` 推进到 R11g；HEAD `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，fingerprint `18500d23a6b25c2f5388aa7a8c3af4b057aff98ace9caa0be139ae9c02ab47d4`。attempt-14 三项 Torch focused、attempt-15 native JVP（shim 关闭）、attempt-16 单前端 CPU（`use_cuda=0`、隐藏 CUDA）共 5 passed、fallback=0。attempt-16 不是 CUDA 证据。manifest、日志、receipt、worker 哈希及失败迭代详见 Day2–5 结果记录。

R11g 尚未移植或合并到当前主树，接下来须先审查并移植到当前源码身份，再复测对应 CPU 语义及适用 CUDA 路径。性能/显存本轮未测：CPU cache 无 holder、无本任务残留进程、`/data6` 约 5.5 TiB 可用，但可见 GPU 利用率约 27%–100%。正式门须使用同源码私有 cold core/cache、至少三轮交替且每轮每侧至少十个热样本，报告加载/首步/热态 median、p95、离散度及 allocator/NVML 峰值，容量另测。其他模型、默认 DDP、真实 activation recompute、held-out JT/PT 质量与跨框架互操作仍未关闭；整体 release 保持 **Not complete**。




## 2026-10-31 历史接续记录：Day4–5 门槛与 R9/R10 功能失败

当前服务器仍使用 `/home/cgq/jittor` 原工作区。复核前 HEAD `d03df9e0a81fbcbc40d0ce68cafd73d0d0c88836`，origin 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 且是祖先，HEAD ahead/behind 为 76/0；原有 101 项 dirty、index 为空，没有清理、暂存或改写它们。三份状态文档此前均干净。源码性能候选身份继续绑定隔离冻结快照 HEAD `b2cf6bad4ce647294cb4c5663117145743bb02dc` / fingerprint `95eb54ba17b771ece4ed585c2036f19565075b66c858e599972fbee011f76558`，不要将 R10 隔离补丁或旧 R17/df45 产品混入此身份。

R10 对 R9 暴露的两个 CPU Torch 节点各复测一次，仍未修复：`fill_/zero_` 后线性输入梯度为 None；`torch.Tensor(native_var)` 的 FollowRuntime placement 仍从 -1 转成显式 CPU 0。pytest 是 2 failed/20.47s，证据和源指纹详见本文件上方发布门增量及看板；runner 后续写回路径错误，worker-result 缺失，但 raw log 与 receipt 已保留。这是 R10 终态与其 runner 写回错误的历史记录；R11g 后续结果及当前待办见本文档最上方最新接续状态。

正式性能/显存门暂不可入场：计划 GPU UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0` 当前 29% 利用、59,939 MiB 空闲，不满足利用 ≤10%、空闲 ≥60,000 MiB；其余 GPU 也未同时满足门槛。当前无本任务进程或该私有 cache lock holder；根分区约 33 GiB 且 100% 使用，`/data6` 约 5.5 TiB 可用。后续 Day4–5 要先为冻结候选建立并独立审计私有 cold cache，再在合格窗口完成三轮交替、每侧每轮 ≥10 个同步热样本，报告加载/首步/热态耗时 median/p95/离散度和 allocator/NVML 峰值；容量另测。更广 Day2–5 的 T5/Qwen/Vision/BERT、默认 DDP、真实 activation recompute/显存收益与共同 held-out JT/PT 质量门均仍未关闭；整体 release **Not complete**。



## 2026-10-31 当前源码候选冻结与后续执行边界

继续使用现有 /home/cgq/jittor，本轮未 clone、reset、stash、clean，也未改源码。主树 HEAD b2cf6bad4ce647294cb4c5663117145743bb02dc、origin e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 祖先关系通过；2,815 个非文档代码/测试文件的指纹为 95eb54ba17b771ece4ed585c2036f19565075b66c858e599972fbee011f76558。独立复核核对文件 inventory/bytes、符号链接、可变状态文档和工作区状态均通过。证据包 /data6/cgq/jittor-lab/_state/release-day23/20261031-main-current-source-r3/；源码清单 SHA256 2253605eceb1e301a6adaf476dd64200ae10b13f33c6537bc7e4a9009877c4ab，source verification SHA256 f0ec621b9a4b378282e59a792f86ce14849fc4046c948f74b4c5de94745eb27c，bundle SHA256 162cf21584d1662c7761f885501b551553b011e4c196757aca75be57b0ef2dd0。该 HEAD 是源码身份冻结基准；随后本任务只把这三份既有状态文档纳入中文提交，提交后 HEAD 是其文档专用子提交，代码指纹保持不变。首轮清单器因目录符号链接而失败，失败日志 SHA256 3cf35d96c62ff802566d85d1ca0a4081c4ae6314b9742865aaf7720ed4bb08ef 已保留；重做后两次扫描及独立复核通过。

当前 R17 BERT 性能包 df45…/e92681… 与当前 95eb…/b2cf… 身份不符，不能复用旧 cold core、也不能将旧测量作为本候选结果。其 source-guard 拒绝发生在 GPU/JIT 启动前，日志留在 evidence-map 包内（SHA256 511eb4b192cb08507533e36848411bcdf398b7487ed4bbd3b4201ef44631bc5e）。

资源快照显示目标 GPU UUID GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0 为 100% 利用、32,634 MiB 空闲；正式门要求 ≤10%、≥60,000 MiB、连续稳定采样并无其他 GPU PID。共享负载持续，故尚未做当前候选私有冷编或性能计时。根文件系统约 34 GiB 可用、已报 100%；/data6 约 5.5 TiB 可用。后续先用 /data6 新建当前指纹的私有产品并独立审核，再等候合格测量窗口；每侧三轮交替、每轮至少十个热态样本，同时报告加载/首步/热态和 allocator/NVML。容量验收独立设计。当前冻结也不补齐 Day2–5 的 activation recompute、默认四卡 P2P、JT/PT 生成质量等未完成范围；overall release 仍 Not complete。


## 2026-10-30 R14 Qwen held-out generation 补充状态

R14 Qwen LoRA 的两 epoch 训练与新进程续训仍按其冻结源码身份通过；本轮仅对已训练 seed821 的 JT/PT adapter 做了17条 prompt 的 greedy generation quality 补充。eval ID 与101条训练子集无交叉，但来源是 Alpaca train split。FP32、无 autocast、输入上限96、生成上限48。PyTorch 在真实 CUDA cuda:0 完成，token-F1 0.212716、ROUGE-L F1 0.164849、exact match 0/17；小样本和截断指标没有人工审评，不判质量通过。

Jittor 第三次 attempt 复用 R14 原 JITTOR_HOME、JITTOR_NO_BUILD=1，未重编 core；真实进程在目标 GPU UUID 上进入生成，300秒预算到达后未落盘完整结果及最终 fallback 计数。旧启动失败、timeout 和所有 cache 产物均保留。原 cache 仅新增75个 JIT 文件，无删除或既有文件改写；core/build stamp 哈希不变。完整证据 /data6/cgq/jittor-lab/_state/release-day23/20261030-r14-qwen-heldout-gen-r1-prepared/，终审 SHA256 7e0ed47258f310320ec9092517efcb1a8766dee5e1aaed207e9a164d15b3d69b。

此结果只补出 PT 侧生成指标，JT/PT 生成质量比较仍未完成；不影响原 R14 实用训练/续训结论、strict FAIL、性能/显存门或整体 release Not complete。继续时不要重跑 R14 full/preflight/resume 或 core 冷编；如要取得 JT 质量分数，应为模型载入与生成另设有界预算并按样本流式落盘，不能从这次未完成运行推断 JT 分数。


## 2026-10-30 R14 Qwen 实用训练与 resume 终态交接

当前主任务取得一组绑定 R14 冻结快照的实用训练证据：seed821 Qwen LoRA full102 与新进程 checkpoint51→102 resume 均在指定真实 CUDA UUID、`cuda:0` 上运行，fallback=0；full 102 次更新零跳步，epoch 微批次均值从 1.205469 降至 1.153617，固定验证 eval loss 2.455552，匹配 PyTorch 为 2.456644（约 −0.045%）。resume 恢复模型、optimizer、scheduler、四类 RNG 和第二 epoch 样本顺序，完成 51 次更新，112/112 LoRA 参数相对 checkpoint-51 改变，最终 eval loss 与 full run 相同。

R14 源快照 HEAD `e4a3894065953efa50ea23a0e52b45511b835f59`、fingerprint `89157afb357abbb882057230ecb0e86586deeef7fb010a7cc51b3577eea054a6`。

执行证据位于 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r14-qwen-practical-final-r1-prepared/`：full review SHA256 `f8521274bcfdf78a3728ca3457af5b52b99239b70eab0b739b463f950fc4d382`，首次 resume 的比较器阻断原件与终审 SHA256 `2e740cf0a3c3a7b1c3469545003e7e013e5533c672b853ecc3629a904b0421b7`，修正规则后的 attempt-02 功能 resume 终审 SHA256 `eebcda6a2f97dae4de8f6da7fb3f3864e0c9c15bdb89fe0518865737d0ae349c`。首次失败是设备映射审计把 112 个 scalar step 误列为 tensor placement；值仍逐项精确校验，新尝试使用独立目录，原始失败和收据保留。状态审计放宽的只是设备映射覆盖范围，step 计数、模型与 optimizer tensor 值/设备检查均保留。

保存恢复功能通过不等于跨框架序列化兼容：stock PyTorch 无法 `torch.load` Jittor portable-pickle optimizer 文件，原失败保留；同一 Jittor 新进程续训已经通过。没有 held-out generation 质量分数，gradient checkpointing shim 不执行激活重计算，正式性能/显存及更广泛 Day2–5 发布门仍待完成。后续按唯一看板继续性能/显存和剩余范围；保持 strict 数值 FAIL 历史、不从该短程结果推断通用长训或显存节省。

## 2026-10-30 跨卡 Tensor.to 修复与接续状态

跨设备 `Tensor.to` 的窄修复已在 `compat/torch/types.py` 中落地：对已经位于CUDA且目标ordinal不同的Var，改走可微分host staging；其他已有迁移路径不变。此前环境内跨卡读回NaN（API返回成功）是触发证据，不外推为所有设备/驱动的共同根因。

唯一有效的双卡用例 `test_explicit_copy_roundtrip_and_gradient_keep_source_placement` 通过，pytest 1 passed / 11.45s；实际检查 cuda:1→cuda:0→CPU 的 `[2,3]` 和反向梯度 `[4,6]`，两张绑定GPU均由进程使用，fallback=0。独立终审SHA256 `6ba61de0e06aa0962cf1a254ba01205f6d92cc42ac82e9df89c1af186b56597f`；执行包位于 `/data6/cgq/jittor-lab/_state/release-day23/20261030-torch-cross-cuda-stage-r7-test-hold/`。R1–R6启动/收集问题与R2未知结果均留档，没有覆盖或计为有效测试。

后续继续Day2–5剩余CPU/Torch/CUDA验收以及Day4–5独立性能/显存测试。此次不提供性能、graph capture、通用长训或其他API结论；既有strict FAIL继续保留。


## 2026-10-30 e7d171 Qwen checkpoint51→102 实用续训完成

当前e7源码下seed821 Qwen3-0.6B LoRA从既有checkpoint51续训完成51次更新、101微批次；step102固定17行验证集两次eager评估成功。112项CUDA0 FP32梯度每步finite且无缺失，零跳步/零fallback；首次更新112个LoRA参数均变化。JT per-step平均loss 1.1684248，PT full/resume 1.1688748；mean-abs差0.00219484、max 0.00686032。固定集loss JT 2.4555516、PT 2.4566441，训练曲线非单调。PT checkpoint不同，ID/shape相同不代表token字节相同。

checkpoint51功能恢复含模型、optimizer、scheduler和四类RNG通过；strict hash仍因两个额外initial_lr项失败。checkpoint102共18文件已核hash，保存global/scalar step=102；没有fresh reload或predict。当前身份为e7 Python + 原位df45 CUDA core/utils混合产品，不是e7冷编。两份独立终态审计SHA为6341d8ca21ab5d342afdb4f90036c329a0cd541af27131ed730c1540b8334a27和e60b5a26a411361b4229acd583f87b0e9d295db8a0dacfb96d6aa0955bac2e9c；保护cache未变，candidate cache零新增，unit自然rc0。/proc扫描1883次拒读，终态可见性有限。

任务仍未完成：下一步优先核验实际跨卡生产Tensor传输及其copy/capture失败；随后完成剩余Day2–5 CPU/Torch/CUDA受影响范围和Day4–5性能/显存正式验收。手工host staging是诊断，不是生产修复。性能、predict、checkpoint102 fresh reload和通用长周期收敛仍无结论，旧strict FAIL保持原样。

## 2026-10-30 BERT checkpoint100 新进程加载与评估

在上文 step50→100 的真实续训之后，独立新进程实际通过 Trainer 加载该 checkpoint100 的模型、优化器、调度器、TrainerState 与 RNG：12 项 checkpoint 文件身份核符，模型/优化器/调度器/scaler 状态签名与已保存值严格一致，201 个 optimizer scalar step 均为100，603 项参数及moment设备为CUDA0，四组RNG恢复一致。此为新进程加载和评估，未执行后续 optimizer 更新、训练或数据游标续接，不把此前50步续训误作此次加载后的训练。

固定评估向指标函数实际交付33条预测和标签，loss 0.4756673276、accuracy 0.7878787879；此前训练末 loss 0.4761481881、accuracy 0.7878787879，loss绝对差 0.0004808605，原因未知，不宣称位一致或将其归因于checkpoint错误。评估前后模型/优化器/调度器/scaler签名一致；TrainerState原102条日志前缀保留、仅追加一条step100评估日志，global_step不变，其他TrainerState字段未全部比较。四组RNG评估后相同；no_grad、禁止 optimizer.step 与 model.train(True) 是该runner的执行约束，并非全局backward探针。

两份独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-bert821-checkpoint100-load-eval-r4-cuda-replay-independent-terminal-r1/audit.json SHA256 764e6cb139669be45c90ede38c294dee4ddbc97db541a7ed449267553092f9fd；/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-checkpoint100-load-eval-r4-resume-next-terminal-r1/audit.json SHA256 bf8312ccb89a45943a8f5db65bd3149ff8523834b5874ee5b9ff1ffd4ebc8a33；原 receipt SHA256 4a5ef00d93ce7885a9fe4c5eead196ae6e91cc005f5a754d4289453676d8e37e。实际e7 Python与已审df45原位CUDA core为混合身份，非e7原生冷编；三份保护cache不变、功能cache未新增，五锁内核记录和可见FD holder为空，unit自然成功/cgroup空，但FD拒读和端点可见性限制保留。结果只证明该单次加载与固定评估，不覆盖新进程后续更新、逐样本ID验证、predict、长期训练或性能。


## 2026-10-30 恢复后 BERT checkpoint50→100 实用续训

任务已恢复 active。下方 peer-enabled host staging 直接 CUDA 双向同步/异步中转的四进程八阶段结果仍有效，旧 no-auto-peer 显式 peer 复制失败与未执行项继续保留。当前主树 Python 源指纹 e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90、HEAD 9724b453，复用原位已审 df45 CUDA core/utils（混合身份，并非 e7 原生冷编），从历史 df45 checkpoint50 单次完成 seed821 BERT 的 step51–100，显式将两组 AdamW 设为 nonfused/foreach=False。恢复模型/优化器/调度器签名严格一致，201 个 scalar step=50、603 个参数与 moment 的实际 CUDA placement、Python/NumPy/CPU/CUDA 四组 RNG 核验通过。

50 次续训更新均有 201 个有限 CUDA FP32 梯度，201 个参数每步实际变化；99 个 microbatch 的 sample ID/形状与历史同种子 PT 报告一致，未比较逐 token 字节。两条不同训练 microbatch 的 query projection 观察到 CUDA BF16 输出和 autocast，不代表全程或内核内部精度。续训 50 次更新的平均 loss 为 0.2081167603，历史 PT 为 0.1965655518；单步有正常非单调波动，例如 step51/52 为 0.28931/0.31714、step99/100 为 0.55750/0.08252，不能据此声称逐步单调下降。PT 使用不同 checkpoint 与优化器路径，比较仅为描述性单种子结果。

固定 heldout 的实际预测/标签各 33 条，loss 0.4761481881、accuracy 0.7878787879；PT 对照 loss 0.4836055934、accuracy 0.7878787879。checkpoint100 六个主要文件已保存并重新核 hash，终态模型为 step100、best metadata 为该 checkpoint；尚未新进程反序列化或验证再次恢复。无 predict、native EventQueue、性能或多种子结论。两份独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-e7d171-bert821-resume51to100-r2-cuda-replay-independent-terminal-r1/audit.json SHA256 0981a708aac6b497e5ecd15bc0812819da096807e80a109e046f2b0da820cd60；/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-e7-resume50to100-resume-next-terminal-r1/audit.json SHA256 39a3dde39b3d1128a541b12fc14ab427203ea85594152c5b6f8f381dad31a0ff。保护 cache 不变，功能 cache 新增4项且旧项内容不变；五锁内核记录和可见FD holder为空、unit自然成功，FD拒读与端点快照限制保留。下一步若要验证持续恢复，应独立设计 checkpoint100 的新进程加载与状态核验；跨卡生产修复仍需独立验证。


## 2026-10-30 任务恢复与 peer-enabled host staging 验证

任务已从下方历史暂停状态恢复。peer-enabled host staging r1 单次实际运行四个独立进程，覆盖两张 GPU 的双向传输及同步→异步、异步→同步两种次序，共八阶段；每次目标对源启用 peer 返回 0。实际 CUDA 指针和设备 ordinal、两张 GPU 的进程 UUID、portable pinned host、源流与事件及目标流顺序均已核验。源、host 和目标四个 FP32 值均为 [1.5, -2, 5, 7.5]，consumer 为 [2.5, -8, 13, 20.5]，原始字节匹配且设备读回两次一致。各进程分别编译 PTX 并执行 GPU driver JIT，driver/runtime 活体映射前后相同。

独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-host-async-resume-next-terminal-r1/audit.json SHA256 4c7f11e128f12ce6f812f686fe069530b1cf4a6a7c5de0145c448f56575ea97c，原 receipt SHA256 d7d7ebdb6c474fb34d71ab1b30b318fb3eccffb1f4bcc139a0ed0b4319d0d7e9。六份缓存的前后及终审快照一致，六锁无内核持有记录，unit 自然成功、cgroup 空及本任务进程退出；进程和 FD 可见性、端点快照之间的瞬态变化仍有限制。本实验是四个连续 FP32 元素的直接 CUDA 中转；记录和同步读回会干预执行，异步提交顺序不证明实际重叠，也不覆盖 Jittor Tensor.to、capture、反向、通用布局或性能。此前 no-auto-peer 八案例的显式 peer 复制 FAIL 和未运行项仍保留。


## 2026-10-30 按额度条件暂停

用户设定的 Codex 周额度暂停条件已触发（剩余 19%），任务暂停；现场可见范围内无正在运行的本任务 unit、主 PID 或 cgroup 进程，历史 active/exited 与 failed unit 保留。六份缓存锁无内核持有记录、可见 FD 无 holder；进程和 FD 有权限可见性限制，结论为当前时点快照。HEAD 为 7c078c39056c872316235421d53b0ac89e97babd，原 dirty 和历史产物保留。

peer-enabled 双向同步/异步 host staging 诊断包 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-enabled-host-async-sequence-runner-r1-hold 的 manifest SHA256 e7617bf62086d5021d2fc8eb3716db43ae738bc226dee9ff806faf36f4e99d77 已经两份静审，尚无现场 binding、未创建 attempt，也未运行该包 GPU 或 JIT；恢复后应重新核现场资源、源码及锁并生成新鲜绑定。该包是直接 CUDA 实验，不是 Jittor 修复或 capture 验收；八案例原 FAIL 和未运行项继续保留。暂停快照 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-staging-pause-handoff-r1/pause-snapshot.json SHA256 1bdcd3defb4de0c0148c1bdadfc2b02cd7e700439c2d66286f4699bda13d0597。


## 2026-10-30 no-auto-peer 隔离候选跨卡八案例：四项通过、第五项失败

在已审私有冷编候选上，五个独立进程实际调用 Torch Tensor.to。未由测试脚本显式启用 peer 的双向直接复制及 producer→copy→consumer 四项，四元素连续 FP32 源、目标及 pending 中间值按位匹配，实际 CUDA 指针 ordinal/dispatch 和两个物理 UUID 均核对，四项 fallback 0；此项不证明 peer 实际处于关闭状态。第五项在目标 cuda:1 对源 cuda:0 显式启用 peer 返回 0 后，0→1 直接复制的目标两次读回均为 16 字节 ff，源前后保持原值；记录数值不符并首错停止，后续三个显式启用案例未运行。失败案例未执行最终 fallback 和映射复核，原 FAIL 保留。

独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-no-auto-peer-eight-case-r3-resume-next-terminal-r1/audit.json SHA256 dca45d7f0525d5b4737a7f7fc76eb9470543f6ed70682417767d43275b479ef4，原 receipt SHA256 fd5cca369841a9779b426d29d963a320abd0f1563bc17e80b99687aaf87bc625。五份受保护 cache 前后与终审快照一致，候选 cache 新增 8 项且旧项未变；六锁内核记录为空，unit 以失败状态结束、cgroup 空，已记录进程退出。进程和 FD 的权限限制及快照之间的瞬态变化无法完全排除。每案为新进程，未显式启用不等于 peer 关闭；pending 构造期间无 Python 显式同步不证明内部未同步，读回及强同步会干预执行。此结果仅覆盖该隔离候选、两方向和四个连续 FP32 元素，不证明原 CUDA 700 根因，未覆盖 capture、反向或性能，不是主树修复验收。


## 2026-10-30 no-auto-peer 私有 CUDA 冷编及 CPU sentinel

有序跨 CUDA copy 路径移除主动 `accelerator_peer` 调用的单行隔离候选，在独立 JITTOR_HOME 实际完成 CUDA-enabled 原生冷编，unit 自然 rc0，约 381.3 秒；它基于当时主树工作树 `e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90`，不是主树生产补丁。日志记录 core 编译进度 `228/228`；stamp 列 231 个编译文件，独立审计区分 228 个 core 文件与 3 个 CUDA backend 文件，并核对 817 项 stamp 源输入。实际 private core／utils 的进程映射、路径、hash 与 stamp 核符，CPU sentinel 得到 `[3.0, 5.0]`、fallback 0。首次独立审计有一条将绝对路径误判为不合条件的审计断言失败，已保留；修正审计未重编。

两份独立终审分别为 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-cold-r2-independent-terminal-r1/audit.json` SHA256 `07e9ed58b4cdb4acf46c348ad149eb0d52286c3860b06e14b3a0d34ff124c6eb`，以及 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-private-cold-r2-independent-terminal-r1/audit.json` SHA256 `396b39d6cb0178afb87de5881f67bb75d3d72fbc574132c38b34a1b35bf7aaac`。五份受保护 cache 前后相同，含私有 cache 的六锁终审无内核记录，unit 成功退出、cgroup 空；快照和进程／FD 观察有可见性限制。

此次只运行 CPU sentinel，尚未执行 GPU tensor、跨卡复制或后续因果测试，不能证明复制正确、原故障根因或性能／capture 行为。移除该路径的主动启用不会撤销已启用的 peer，其他路径也可能启用；旧直接复制及 capture 失败和预算账目保留。


## 2026-10-30 接续状态与跨卡 peer 因果候选

以下为冷编前的历史准备记录；实际冷编状态见上文。

Checkpoint 控制参数任务代码已提交 `0d1030f8cc802e0ef1bbb27a8849aa39c23c2a7d`，三文档测试记录已提交 `7fd69c59bd2134787608486868f09f19d97939a4`：主树 CPU 代表五节点通过；隔离候选 13 节点实测通过后控制器节点过滤导致 unit 严格 FAIL、旧 smoke skip 和历史 PT 全套终态未知均保留。实现仍直接调用目标函数，不重计算 activation、不提供 checkpoint 内存节省，未实现的非默认控制显式拒绝。

此前实际完成的 Qwen checkpoint 51→102、BERT 非 fused 两步、双向手工 host 中转及其各自独立审计见下文：它们使用记录时的源码／checkpoint 身份和干预，不能自动外推到当前提交树、完整模型或其他配置。Qwen 恢复严格 hash 的 `initial_lr` 额外项、BERT 历史 checkpoint-100 严格差异、原跨卡直接复制与 capture 失败均保留；手工中转不构成生产修复。

针对 peer enable 前后同指针跨卡复制结果变化，正在独立准备一个有序跨 CUDA copy 的因果候选：只在隔离源码中移除该路径主动调用 `accelerator_peer` 的一行，保留原 copy／stream 流程；候选补丁 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-copy-private-cold-r1-hold/candidate.patch` SHA256 `cff08b726e5b4c1bb995318fcc84e8b591072acfdedae506a798a346b98eaa19`。此为尚在准备的假设检验，当前未构建或运行、未应用主树，也不证明原故障根因或复制正确。该改动不会撤销已经启用的 peer，也不保证其他路径不会启用 peer；跨设备性能、其他环境及 capture／replay 语义未知。后续需冻结源码与资源、独立冷编并按实际条件验证，旧实验和预算不重置。


## 2026-10-30 checkpoint 控制参数任务代码提交

任务代码已提交 `0d1030f8cc802e0ef1bbb27a8849aa39c23c2a7d`（`修复checkpoint控制参数转发`），提交仅修改 `compat/torch/installers/data.py` 的 checkpoint 控制参数处理：默认控制不会误传目标函数，非 reentrant 目标 kwargs 可按约定转发，不支持的非默认控制明确拒绝。它仍是直接调用近似，不提供 activation 重计算或内存节省；已记录的主树 CPU 五节点及隔离候选 13 节点实测各有各自源码／执行范围，不能外推完整 PyTorch、GPU、模型或内存验收。先前隔离运行 pytest 成功后 worker／unit 严格 FAIL、旧 smoke skip 和 PT 全套终态未知继续保留。


## 2026-10-30 主树 checkpoint 控制选项 CPU 代表测试

候选改动已应用到主树工作树，执行源码 fingerprint `e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90`；直接导入实际主树 `torch.utils.checkpoint.checkpoint`，五个代表节点的 15 个 setup／call／teardown 阶段均通过，测试前后主树函数身份、实际 CPU core／utils 映射与 fallback 0 均已记录。覆盖 `use_reentrant=False` 下默认 RNG 控制与函数 kwargs 转发、`preserve_rng_state=False` 和自定义 context 的显式拒绝及警告只发一次；这是五节点 CPU 小样本，不等于已实现完整 PyTorch checkpoint 语义。该实现仍直接调用目标函数、不重计算 activation、不节省相应内存，其他不支持的非默认控制按实现显式拒绝。

先前隔离候选运行实际 pytest 13 节点／39 阶段通过，但旧 smoke 实际 skip，worker 节点前缀误判导致原 unit 严格 FAIL；原失败及映射／fallback 未落盘限制均保留。此次直接主树五节点通过不能抹去该失败或把旧 smoke 标为执行通过；先前 PyTorch 全套 pytest 的终态仍未知。没有重跑 PT 全套、GPU、模型或 checkpoint 内存测试。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-checkpoint-controls-direct-cpu-independent-terminal-r1/audit.json` SHA256 `a989c48f0feeab9cf5dcb11dd162b958fac29ffc64cd35558045b6a5ed6ac0fc`。unit 自然成功退出；CPU cache 端点无增删且仅锁文件变化，四份保护 GPU cache 不变，五锁内核记录为空；缓存快照和进程／FD 观察存在可见性限制。


## 2026-10-30 checkpoint 选项隔离候选 CPU 实测与控制器失败

CPU 单次 pytest 对隔离 `data.py` 候选实际收集 13 个新节点，保存的 setup／call／teardown 共 39 阶段均为 passed，pytest 汇总 `13 passed, 1 skipped`、返回 0；覆盖控制参数消费、reentrant 目标 kwargs 拒绝、非 reentrant kwargs 转发、反向梯度、警告及不支持选项的显式拒绝。旧仓库 checkpoint smoke 节点在 setup 实际 skip，pytest 保存的理由为 `needs deployed torch-shim + nvcc`，因此旧节点未执行，不能算通过。另有测试插件对该外部测试文件显示 0 executed 的路径记账差异；上述 13 项范围由逐节点阶段记录独立重算。

pytest 返回后，worker 按 `test_checkpoint_options.py::` 前缀过滤，但实际新节点 ID 以 `::test_...` 开头，导致误判收集不匹配并以 rc1 退出，控制器和 unit 亦严格 FAIL；原 receipt／日志保留，不以独立重算改写为整包 PASS。按执行源码控制流，到达 pytest 前的 overlay 路径／hash、公开 checkpoint callable 和 CPU core／utils 活体映射检查未抛错，但失败路径没有持久化当时映射详情，也没有记录终态 fallback，不能声称独立取得它们。先前 PT 全套 pytest 的终态未知；该候选仍是直接调用近似，没有 activation 重计算或节省内存，未覆盖 GPU 或模型。

独立重算审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-checkpoint-controls-cpu-r2-independent-terminal-r1/audit.json` SHA256 `642969eb0efb7abf4ef5b94789f20547764a0421d75d6b1e12501a747867cf52`。CPU cache 端点新增 36 项、无删除，14 个原 shim 文件仅 mtime 变化且内容 hash／大小不变，另锁文件变化；四份保护 GPU cache 不变，五锁内核记录为空，unit 以 exit 1 失败结束且 cgroup 已空。端点快照和进程／FD 可见性有限。


## 2026-10-30 graph capture owner／close CPU fake 原型

独立 Python 状态模型及 12 项标准库测试通过，覆盖 unsupported backend、先分配失败、确定／未知 begin、空图／非空图关闭、设备切换及恢复、嵌套与跨线程 owner 拒绝、未知 end 时保留 hold 并拒绝再次 capture、select／restore 失败及首错保留、已关闭 graph 在恢复失败时保留，以及 fake caller 对 poison 的传播。测试中的两次 `checked_caller` 调用是同一个 fake 函数的重复，不是两个真实调用点的集成验证。原始 10 项测试记录和扩展至 12 项的结果都保留。

真实源码只读 AST 核对显示 `step_capture.py:597` 和 `graph_replay.py:757` 的 capture 调用路径捕获宽泛 `Exception` 后返回、不重新抛出；这说明若未来 native unknown-close 以普通异常传播，必须另外设计调用侧 fail-closed，当前没有运行真实调用路径验证。现有 BackendOps ABI 3 表要求版本和结构尺寸精确匹配，真实 end 的 null 不能区分失败和已关闭空图，fake 的明确状态并非现有接口；模型中的锁、owner token、poison 和 held list 也不能证明真实 allocator 全局状态、线程生命周期、CUDA custom context、外部 backend 或非空图 launch／release 的正确性。unknown 不代表进程仍可安全执行其他 GPU 操作。

这只是 CPU 行为原型与静态源码审阅，没有生产补丁、Jittor 集成、真实 backend 修复或 GPU／JIT 测试；历史 capture 906 与原两次干预观察保留。独立审阅 `/data6/cgq/jittor-lab/_state/release-day23/20261030-capture-fake-prototype-independent-review-r1/audit.json` SHA256 `ae72c4673e9f5f90fe7757ff124fd540b20a7a4cc18c53ce20c88be50026c679`；原型 manifest SHA256 `c27b6bcc746ee465c01f6da846848a39e76cfc9fb1c460b2f769e532c32b1fe4`。


## 2026-10-30 手工 host 中转双卡一阶梯度诊断

主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 使用实际 CUDA core，在两个独立进程分别测试 logical GPU 0→1 和 1→0：显式启用目标到源的 peer（两次返回 0，之前隐藏状态未知），对四个连续 FP32 元素手工执行 `x._copy_to_cpu().to_device(target)`，而非改变生产实现。两方向的源前后、host 中间值和目标值均为 `[0.25, -1.5, 2.0, 3.25]`，backward 得到源端梯度 `[2.0, -0.5, 1.0, 0.25]`、loss `4.0625`，与独立 CPU PyTorch 数学参考一致。host 的 public device／placement／location／dispatch 分别为 CPU／0／CPU／`cpu:-1`；源、目标和梯度的实际 DEVICE 指针与对应设备 ordinal、两个物理 GPU UUID、实际 core／utils／runtime／driver 映射及 fallback 0 已核。CPU PT 的原生库映射未采集。

另以独立 tensor 双向分别变更源和目标，并读回两端，观察到两次变更实际发生且另一端值保持，支持该特定样本的复制隔离。该实验含 peer 启用、显式同步和多次读回，范围仅限该四元素连续 FP32、一阶梯度及单线程；不覆盖直接跨卡 `.to()`、非连续布局、其他 dtype、非阻塞、capture／replay、性能或生产修复，也不能推翻历史直接跨卡错误及 CUDA 700／906 严格失败。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-manual-host-autograd-r2-independent-terminal-r1/audit.json` SHA256 `9ae248862e156f253a8a5c5915884c19a3fb6cb7342a3ad41f6f8f26c468e4c3`。unit 自然成功退出；主 GPU 功能 cache 端点新增 12 项、无删除且旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空；快照和进程／FD 观察存在可见性限制。r1 未运行，静审发现错误的 CPU dispatch 索引断言，r2 修正并独立审阅后执行，原审计保留。


## 2026-10-30 主树 4ed94 BERT 非 fused 两步恢复诊断

单 seed 821 的真实 BERT 在主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 和实际主树 CUDA core 上加载历史 df45 checkpoint-50，完整状态签名严格一致，201 个 scalar step 与 trainer／scheduler 均为 50，201 参数及 402 个 moment 的设备记录为 cuda:0，Python／NumPy／CPU／CUDA 四组 RNG 恢复检查通过。随后有意把两个 optimizer 组的 `foreach`／`fused` 从 `None` 设为 `False`，这是改变原配置的诊断干预；保留原 100 步 scheduler，实际只续训第 51、52 两步，0 skip、fallback 0。每步 201 个梯度均有限 FP32 CUDA，201 个参数均发生变化；每步真实 Python 逐参数 AdamW 更新与 optimizer scope 各 201 次、step 分别为 51／52，记录的 scope 进前／内部／退出 dispatch 均为 cuda:0。该 Python 观察不证明 kernel 内部执行或其他设备上下文行为。

四个 microbatch 的样本 ID／形状及两步 LR 与历史 PyTorch 报告匹配，未逐 token 比较；第 51／52 步 loss 分别为 `0.28930664`／`0.31713867`，历史 PT 分别为 `0.27783203`／`0.27587891`。PT 使用不同 checkpoint 且 optimizer 路径可能不同，差异仅为描述性对照，不证明同状态数值等价或收敛趋势。运行配置为 BF16，记录未建立全程及 kernel 内部精度；同步、逐参数检查和包装会干预执行。此次未做 eval、predict 或 lifecycle，也未生成新的 checkpoint-52；先前 full／resume checkpoint-100 严格状态差异与其他历史 FAIL 保留，不能用这两步取代。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-4ed94-bert-nonfused-r2-independent-terminal-r1/audit.json` SHA256 `249fc6695b642cc297a6549b6c4687a66381218ee2c7d478cb93933d2ae3ff9f`。unit 自然成功退出；主 GPU 功能 cache 端点新增 8 项、无删除且旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空；端点快照及进程／FD 可见性有限。r1 静审发现异常处理依赖缺失，r2 修复并完成模拟后才运行，原审计保留；本次不作性能验收。


## 2026-10-30 主树 4ed94 Qwen LoRA checkpoint 51→102 续训

单 seed 821 新用途在主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 和实际主树 CUDA core 上加载历史 r10 checkpoint-51，按原 102 步日程实际续训至 102：51 次更新、0 skip，每步记录 112 个有限 FP32 CUDA 梯度，首个续训更新中 112 个 adapter 参数均变化，fallback 0。恢复功能比较通过，112 个 optimizer scalar step 与 trainer／scheduler 均为 51，337 个已记录的 adapter／moment／buffer 设备为 cuda:0，Python／NumPy／CPU／CUDA 四组 RNG 恢复检查通过；337 不涵盖冻结底座全部参数。恢复严格 hash 比较仍为 FAIL：多出两个值为原始学习率的 `initial_lr` 字段，功能比较仅对该已限定差异作例外，不改写严格结果。历史 checkpoint 来自旧 r10，并非本工作树从头训练。

101 个续训 microbatch 的 sample ID／形状和 51 步 LR 与历史 PyTorch full／resume 报告匹配；未逐 token 比较。续训更新窗口等权平均 loss `1.16842480`，两个 PT 参考逐步 loss 平均／最大绝对差分别 `0.00219484`／`0.00686032`；固定 17 样本 eager eval loss `2.45555162`，PT 两个参考均 `2.45664406`。新 checkpoint-102 的 112 个 adapter 均与两个 PT checkpoint 不同，最大绝对差 `0.000506258`、整体差值 L2 `0.0237648`（PT 参考 L2 `8.6498742`）；这只是 adapter 数值诊断，没有 PT optimizer 状态一致性或长期收敛结论。checkpoint 文件和旧 checkpoint 原件身份均已核；另存 `final-state.npz` 的最佳模型与末态 optimizer 组合未证明为同一原子 checkpoint。

训练首个续训更新的两个 microbatch 各观察到首层 q_proj 输入／权重 FP32、输出 BF16、autocast 开启及 CUDA dispatch；不证明全程或内核内部精度。51 次更新均观察到 fused AdamW Python 分支调用，逐参数 `adam_update` 调用为 0，故本次不覆盖非 fused 参数 scope 路径；Python 计数不证明 kernel 内部执行。公开 `jt.fetch` 包装观察到两次调用并返回，但未直接观察 EventQueue。predict／lifecycle 明确未运行，旧严格失败及评估限制保留；同步、读回与包装均有干预，不作性能结论。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-4ed94-qwen-resume-r2-independent-terminal-r1/audit.json` SHA256 `1a1451fe83152e01f8702716961b6c68be4a1cf4503b28a0b82142361b91543f`。本次独立 3000 秒预算内 unit 自然成功并退出；主 GPU 功能 cache 端点增加 428 项、无删除，旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空。缓存为端点快照，现场进程／FD 可见性有限；历史预算和原失败不重置。


## 2026-10-30 Qwen checkpoint 51→102 新用途准备

以下为运行前计划的历史记录，实际运行与终审结果见上文。

正在准备在主树当前工作树上进行 Qwen LoRA checkpoint 51→102 的独立续训新用途，预算单列为 3000 秒；这是拟议运行，不是已完成的训练或通过结果。原有训练、续训和严格失败以及原预算账目保持不变，不因新用途而重置。runner 尚未冻结，实际源码与产品、PyTorch 对照、checkpoint／数据身份和现场资源仍须在可执行方案中绑定并审阅；尚未以本计划运行 GPU、JIT 或模型，不能据此推断新工作树的训练或恢复成功。

## 2026-10-30 graph capture backend 契约只读审阅

隔离 r4 的历史观察显示：在设备 0 的 per-thread stream 开始 capture 后，跨设备操作拒绝时当前设备变为 1；沿当前设备结束时 end 返回 401、后续读取返回 906，而另一独立进程显式恢复设备 0 后 end 返回 0、源字节匹配。GDB、同步及设备恢复均为干预，结果不能认定原 CUDA 700 的唯一根因，也不是通用 capture／replay 通过。

只读审阅发现现有 backend 表 ABI 为版本 3 并要求结构尺寸精确一致；新增字段会影响旧表兼容，原回调改签名同样改变二进制调用契约，外部 backend 与消费者未穷尽。当前 end 返回 null 既可能是结束失败，也可能是确认关闭后的空图或不可实例化图，不能据此证明 recording 已关闭；持有分配的 owner、起始设备与线程及非空图 launch／release 的设备归属都需要明确验证。发生关闭状态未知时，是否保留 held allocations、阻止继续 capture 并把致命条件穿过现有 Python 宽泛异常处理，仍是待设计和验证的问题；custom context、并发与错误恢复亦未解决，CoreX 和外部 backend 不可直接外推 NVIDIA 实验。审计仅提出受限方向与后续 CPU fake-backend／真实 CUDA 验证要求，没有补丁、冷编、GPU 实验或合入。审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-capture-backend-contract-readonly-handoff-r1/audit.json` SHA256 `e56b922245228e975e46347cb71981c8ce0008f657b029c3fc134ed4e2da75db`；原 906 严格失败和两次干预观察均保留。


## 2026-10-30 Adam 选项任务代码提交

任务补丁已提交 `672d1fcf47726c6b5a7b546f14aaa5c61b6c4244`（`支持Adam显式关闭批量与融合选项`），只修改 Adam 的前端选项、参数组选项保存和非 fused step 前校验：接受 `foreach`／`fused` 的 `False` 或 `None`，对 `True` 明确拒绝，其他错误类型拒绝；不声称实现 foreach 或 fused。此前主树工作树 fingerprint `4ed94ce1…` 的直接 CPU 七节点及三个新进程序列化恢复通过，隔离候选的 CPU／CUDA 固定梯度四步 PyTorch 对照也通过；这些各自带同步／读回的证据范围与工作树 dirty 身份保留，不等同于该提交树的完整模型、GPU 恢复或性能验收。原控制器入口错误、观察器错误、同进程状态共享及续步严格失败均不改写。


## 2026-10-30 主树 Adam 选项直接 CPU 恢复验证

使用当时工作树源码 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 与主树真实 CPU core、直接导入主树 Adam 模块（无 overlay），七个节点的 21 个 setup/call/teardown 均通过、无跳过。三个不同 `foreach` 组设置（缺省、`None`、`False`）各自保存 checkpoint 并由新进程加载，再进行一步更新；恢复前后两个参数的参数值、m、v、step 与同次父进程记录逐值一致，step 从 1 到 2，`None` 与 `False` 实测各自保留。另测 `foreach=True`／`fused=True` 显式拒绝、组修改和 closure 前置检查、加载后不支持组的拒绝及 AdamW 小样本回归。父子进程实际 core／utils 映射和模块身份已核，fallback 均为 0。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-options-direct-cpu-independent-terminal-r1/audit.json` SHA256 `aa5b6c34182fd8f168e9e38ff2ed79338b6f843c8b01d6e7356c6d16ce4c1ab8`；原 receipt SHA256 `ea5ffdbe1f0ab1ea8cdfa9c96b1c8f938e2985e80ce0858abaafda7a997684fc`。CPU 缓存端点仅锁文件变化，四个受保护 GPU 缓存端点不变；锁内核记录为空、unit 自然成功且终态无活动 cgroup，现场进程与文件描述符可见性有限。此前启动错误、观察器错误与同进程直接状态共享的记录均保留。本次为带同步／读回的小样本 CPU 恢复测试，没有独立 PyTorch 数值对照，也不代表 GPU、模型训练或性能验证。


## 2026-10-30 隔离 Adam 选项候选 CPU／CUDA PyTorch 对照

使用主树原生 CPU／CUDA core 和隔离的两份 Adam 选项候选 Python 模块，两端显式 `foreach=False, fused=False` 构造 Adam；各以四参数固定梯度运行四步，与已保存的对应 PyTorch 参考逐字段比较通过。两端参数和梯度最大绝对差均为 0；m 的最大绝对差 CPU `7.4506e-9`、CUDA `2.9803e-8`，v 两端均 `1.8626e-9`，参数 step 和 14 次逐参数更新已核。CUDA 参数实际在 logical cuda:1、ambient dispatch 为 cuda:0，端点 DEVICE ordinal 与两端 core／utils、CUDA runtime／driver 映射已核，fallback 0。

每端对 `foreach=True` 和 `fused=True` 的顶层与参数组构造以及 step 均观察到显式拒绝；step 拒绝前 closure 未调用，所观测参数与 step 未变。这不代表 True 路径已实现。测试使用同步和读回，范围是固定小样本而非完整模型、性能或无同步训练；PyTorch 原生库映射未采集，候选尚未合入主树。同进程直接状态共享和先前失败均保留。

CPU cache 与 CUDA cache 端点分别保持 1146 和 3033 项、无增删，保护 cache 端点一致，终态锁无内核记录；两个 unit 成功结束，记录 PID 在终审时不在现场。CUDA 运行期间三份按该协议可变的状态文档因补写 capture 记录发生变化，执行源码 fingerprint `f110…` 保持，不能称完整工作树前后字节不变；缓存快照和进程观察亦有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-oracle-cpu-cuda-r1-independent-terminal-r1/audit.json`，SHA-256 `10f736c562abca446a55d876b4e14554e7d0fed224a246d1fcf58eea27dabe3b`。

## 2026-10-30 隔离 r4 capture 906 的 GDB 双分支观察

两个独立进程均使用隔离 r4 core 与主树 optimizer overlay，先做 CUDA0→1 的 257 元素 warm copy，再观察 capture 拒绝。两分支实际 begin 均返回 0；拒绝后当前线程 runtime／driver device 从 0 变为 1，该线程的 PTDS capture status 为 0。A 沿当时设备结束：end 前 device 1、status 0，GDB 关联实际 end 返回 401，之后源 driver DtoH 返回 906，源后字节未能验证。B 在 end 前显式恢复 device 0：该线程 status 为 1，实际 end 返回 0，之后源双读与原始期望字节一致。两次都记录预期的跨设备 staging 在 graph recording 中拒绝信息。

进程 rc0 表示诊断记录完成，A 的读回仍失败，不能把两分支标作 capture 功能或 replay 通过。GDB 断点、当前线程查询、同步、读回和 B 的设备恢复均有干预；两个分支是不同进程，结果不能单独证明所有线程／流、生产安全或此前 CUDA700／906 的唯一根因。实际 begin/end 的线程、时间窗口、PTDS stream、返回码及 runtime/core 映射已核；原 remaining16 capture 严格失败保留。隔离 r4 cache 端点均为 982 项、无增删且仅 lock 文本变化，三保护 cache 端点一致，锁无内核记录；unit 成功结束、任务 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-capture906-gdb-compare-r2-independent-terminal-r1/audit.json`，SHA-256 `e90092e8f7a8216593ba5d74e9a9cc65710ac64a5f1c81df292ec1724ba56d3f`。

## 2026-10-30 Adam 选项候选 CPU 序列化新进程恢复

隔离两文件 Adam 选项候选的 CPU checkpoint r3 实际完成 8 个节点及 24 个 setup／call／teardown 阶段。新增组 `foreach` 缺省、显式 None 和 False 三种情况下，分别真实 `torch.save` checkpoint 并在三个独立新进程加载、恢复和续步；文件 SHA、进程身份、CPU core／utils 活体映射及 fallback 0 已核，恢复前后两参数的值、m/v 和 step 与原进程对应状态按 `atol=rtol=2e-6` 一致。其余节点覆盖不支持 True／类型的显式拒绝及 closure 前检查、加载不支持组选项的下一步拒绝，以及 AdamW／native Adam 单步回归。

这是 CPU 固定小样本、隔离候选的序列化测试，没有独立 PyTorch 数值、GPU 或模型训练对照；磁盘序列化新进程通过不证明同进程直接 state_dict 加载的状态独立。此前直接加载时 moment 对象共享观察、续步严格失败及观察器失败均保留。CPU cache 端点 1122 → 1146、新增 24 项无删且旧项仅 lock 文本变化，四份保护 cache 端点一致、终态锁检查通过；unit 成功结束，记录 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r3-independent-terminal-r1/audit.json`，SHA-256 `81f6dd171843ed4d230c2582c08b7d93cbe70d900e260e8939aa5650c485cb94`。

## 2026-10-30 Adam 直接状态共享观察与序列化观察器首错

隔离候选 CPU checkpoint r2 的首个节点实际观察到，同进程将 `state_dict()` 直接加载到另一优化器后，源与恢复优化器的 m 和 v 分别具有相同 Python 对象 id；源优化器单步后，两侧 m/v 数值同步改变，m 的 dispatch 均为 CPU。这支持该特定直接加载工况的状态共享，但不单独证明此前续步比较差异的唯一原因，也不代表序列化或新进程恢复行为。

第二节点已写出 `checkpoint-missing.pt`，随后观察器假定 step 支持 `.item()`，实际取到 float 而报 AttributeError；新进程未创建，checkpoint 未加载或续训，其后 7 个节点未执行。unit 按严格失败退出，原 r1 入口错误及此前续步失败保留；失败 worker 未记录终态映射或 fallback。CPU cache 端点 1110 → 1122、新增 12 项无删且旧项仅 lock 文本变化，四保护 cache 端点一致，终态锁检查通过，记录 PID 不在现场；快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r2-independent-terminal-r1/audit.json`，SHA-256 `0f5b69678978119a408a1cbc1e3bb390c14160c5f5385e3a42d580b64e9f1a8b`。

## 2026-10-30 Adam checkpoint 后继控制器入口失败

用于观察直接状态别名及序列化新进程恢复的 CPU checkpoint r1 控制器在入口调用 `predecessor.verify()` 时，因未导入 `predecessor` 报 NameError，unit 以 exit 1 失败。错误发生在 attempt 创建和 worker 启动之前，计划中的 9 个 pytest 节点与 checkpoint 子进程均未运行；不能将该结果称为框架验证或推断先前续步差异原因。原 journal 和绑定保留，不自动重试；此前静审未识别该未定义名称。

终审时 CPU、主 GPU 功能、隔离 r4、性能与封存 SDPA 五份 cache 快照与入场审阅相同；五锁无内核记录或可见 FD holder，控制器 PID 不在现场。1755 个 `/proc` FD 目录拒读，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r1-independent-terminal-r1/audit.json`，SHA-256 `22b2b5fc78c055ea268d489a57a7957a0755827af948ddc5aeafc308deb2c878`。

## 2026-10-30 隔离 Adam 选项 CPU framework 恢复续步首错

隔离两文件 Adam 选项候选的 CPU framework r2 收集 12 节点：四种顶层 None／False 构造和参数组保存组合通过；第 5 项在新增组缺省选项的 state_dict 保存和加载选项断言通过后，同进程依次对原优化器和恢复优化器续步，参数比较失败：原为 `[0.9812166, 2.0200136]`，恢复为 `[0.9801547, 2.0216746]`。按首错停止，余下 7 节点未执行；这不是候选整体通过，也没有独立 PyTorch 或 GPU 对照。原 r1 控制器入口错误及此次失败均保留。

源码显示 state_dict 直接引用 moment Var，恢复时 `_to(copy=False)` 对已在 CPU 的 Var 可能原样返回并安装，因而存在两个优化器共享 moment 的可能；本次没有保存对象身份或各阶段 moment，不能确定它就是失败原因。失败 worker 未保存终态映射或 fallback；CPU cache 端点 1098 → 1110、新增 12 项无删且旧项仅 lock 文本变化，四份保护 cache 端点一致，终态锁检查通过；unit 以 exit 1 失败、记录 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-framework-r2-independent-terminal-r1/audit.json`，SHA-256 `19b0f157598fbe1cdb1d8df5ef69d4e4e3b4422a316f1f0caef5b7844c107c11`。

## 2026-10-30 Adam 选项候选 CPU 控制器入口失败

隔离 Adam `foreach`／`fused` 选项候选的 CPU framework r1 控制器在入口引用不存在的 `source_guard.COLD` 时抛出 AttributeError；实际模块导出的是 `GUARD_COLD`。unit 以 exit 1 失败结束，错误发生在 attempt 目录创建和 worker 启动前，12 个 pytest 节点均未执行；这是控制器准备错误，不能判定候选框架行为或将其标为测试通过。原 journal 和绑定保留，不自动重试。此前静审未识别此属性不匹配。

终审时 CPU、主功能 GPU、隔离 r4、性能及封存 SDPA 五份 cache 快照与入场审阅一致，五锁无内核记录或可见 FD holder，控制器 PID 不在现场；1751 个 `/proc` FD 目录拒读，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-framework-r1-independent-terminal-r1/audit.json`，SHA-256 `d750b4f3abaf50872de641d2d88dadfcca78a866945451c2ad237e8b98785d38`。

## 2026-10-30 隔离 r4 剩余跨卡复制与 capture 首错

基于隔离 r4 host-staging 原生 core／Python 源及显式加载主树 optimizer 模块的一次有界后继中，原 suite 未执行的前 14 个复制节点完成：两方向各四种 dtype 的 17 元素，以及 FP32 两方向在 8 MiB 前、恰好和后各一个元素的边界。每项的源前、目标、源后三份原始字节和 driver 双读均与独立重建的确定性期望一致，实际两卡 DEVICE ordinal、UUID、映射和 fallback 0 已核。同步、指针物化和读回均为干预，不能外推到所有大小、模型或原主树跨卡实现。

第 15 项 CUDA0→1 capture 的 warm copy 字节符合期望，并记录跨设备 host staging 在 Jittor graph recording 期间不支持的拒绝信息；随后 source-after 的 driver DtoH 报 CUDA 906，进程 rc1，未完成源后数值验证、终态 fallback 与映射检查。因此该节点严格失败，不能称 capture 通过或 replay 已验证；第 16 项反方向 capture 未执行。原 CUDA700、隔离候选异常／资源生命周期及 context 等限制仍保留，不能由这些结果确定原故障根因。

隔离 r4 cache 端点 970 → 982、增加 12 项且无删除，旧项仅 lock 文本变化；三份保护 cache 端点一致，终态锁无内核记录。unit 以 exit 1 失败结束、已记录任务 PID 在终审时不在现场；缓存快照和进程观察有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-devicecopy-remaining16-r2-independent-terminal-r1/audit.json`，SHA-256 `b033b8be360c91828331f2a361014a14c5fd95c013a4338c38977380d1cd6ce0`。

## 2026-10-30 Adam 更新作用域代码提交

任务代码已提交 `c1569515961a5c4b9739c8b4ab2916bb531d9f22`（`修复优化器更新的参数设备作用域`），该提交仅包含 `compat/torch/optimizer_api.py` 的 Adam／AdamW 非 fused 逐参数更新 scope 改动。真实主树工作树 f110 身份下的 CPU 与 CUDA 固定梯度四步对照已完成，独立审计 SHA-256 `e9c2ea3011d87abc7cda5a5bb7b9db951b0bb172706bb6083be80b625c38fa51`；完整工作树还包含既有 dirty 内容，不能把该工作树身份当成提交树身份。测试包含同步和读回，不证明完整模型、无同步运行或跨设备复制；原 Adam `foreach` 构造失败、历史 CUDA 700 及候选限制继续保留。

## 2026-10-30 主树 f110 Adam／AdamW CPU 与 CUDA 固定梯度对照

主树 f110 Python 源直接加载已集成 scope 的 optimizer，复用已审主树 df45 的原生 CPU／CUDA cold core：CPU 两个 Jittor 阶段使用已保存的 PyTorch Adam／AdamW 参考，CUDA 新生成两个 PyTorch 参考后运行两个对应 Jittor 阶段，六阶段均自然 rc0。每种算法各以四个 FP32 参数、固定梯度运行四步，逐字段的参数、梯度、m/v 和 step 与对应 PT 参考通过预注册容差；每个 Jittor 阶段记录 14 次逐参数更新，fallback 0。CUDA 参数在 logical cuda:1（UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`），ambient 为 logical cuda:0；最终 CUDA 参数及状态的 driver DEVICE 指针和 ordinal 已核。

主树 Adam 调用省略不支持的 `foreach`／`fused` 参数，AdamW 两侧显式设为 False；先前 Adam 对 `foreach=False` 的实际构造失败和相关 API 缺口仍保留，不能由此通过结果覆盖。每步均显式同步和读回，观测会干预调度；这是固定梯度小样本，不是完整模型训练、收敛、性能、无同步或跨设备 copy 验证。实际主树模块及 core／utils 和 CUDA runtime／driver 映射记录已核；PT 的原生库映射未采集，CPU 物理设备仅按公开分类。

CPU cache 端点 1090 → 1098、CUDA cache 3029 → 3033，分别新增 8 和 4 项、无删除，已有项仅 lock 文本变化；各自保护 cache 端点一致，终态锁无内核记录。两个 unit 成功结束，记录的任务 PID 在终审时不在现场；端点快照和进程观察有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-post-r4-independent-terminal-r1/audit.json`，SHA-256 `e9c2ea3011d87abc7cda5a5bb7b9db951b0bb172706bb6083be80b625c38fa51`。

## 2026-10-30 主树 Adam scope 集成后 CPU 首错

主树 f110 源码的直接 CPU 小样本对照中，独立 PyTorch Adam 与 AdamW 固定梯度参考各完成四步、进程 rc0；随后的主树 Jittor Adam 接受 `foreach=False, fused=False` 的构造调用在进入 `step` 前报 `TypeError: Adam.__init__() got an unexpected keyword argument 'foreach'`。堆栈显示 Torch frontend 将参数传至不接受 `foreach` 的 native Adam 构造器，这是该调用实测的 API 失败；尚未运行 Adam scope 更新，不能据此推断 scope 的更新正确性或失败原因。按首错停止，Jittor AdamW 和后续 CUDA 对照均未运行；保留原失败，不自动重试。

失败进程已记录实际主树 optimizer 路径和 CPU core／utils 运行中映射，因构造失败没有产品映射后记录或 fallback 结果；PyTorch 新参考未采原生库映射。CPU cache 端点均为 1090 项、无增删且仅 lock 文本变化，四保护 cache 端点一致、终态锁无内核记录；unit 以 exit 1 失败结束、记录任务 PID 终审时不在现场，缓存快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-post-cpu-r2-independent-terminal-r1/audit.json`，SHA-256 `a7850f486d736b5a514611b2a46602f811f0de9a326832d0dabad8d0472a4e41`。

## 2026-10-30 隔离 Adam scope CPU 异常恢复单例

在主树 df45 CPU core 和预导入的隔离 Python scope 候选下，通过真实 AdamW.step 在 `adam_update` 调用内注入异常。进入时观察到 Python placement 与 native TLS 均为 CPU；异常后恢复到事先设置的合成 CUDA1 外层标记，退出外层后恢复测试前初值。这个 CUDA1 只是 TLS 标记，没有真实 GPU 操作；step 在异常前已增至 1，测试不证明优化器状态事务回滚。frontend result-type token 和 runtime device／flags 没有单独验证，异步 GPU 错误和部分 moment 更新亦不在该单例范围。

候选模块与主树 helper 的加载身份、CPU core／utils 活体映射、stamp 和 fallback 0 已核。CPU cache 端点均为 1090 项、无增删且仅 lock 文本变化，GPU 功能及另外三个保护 cache 端点一致；unit 成功结束，记录任务进程在终审时不在现场，终态锁检查通过，进程和端点快照存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-scope-exception-cpu-r3-independent-terminal-r1/audit.json`，SHA-256 `4bbb0d9df5c43df6d085c88bbccdbef893ae1add396b99512017c4918d84d7cf`。

## 2026-10-30 主树 core 与隔离 Adam scope Python 候选定向对照

在 df45 主树 Python 和原主树 CUDA core 上，预导入加载仅 `optimizer_api.py` 的隔离 scope 候选，一次串行 suite 的 13 个独立阶段自然 rc0：1 个新 PyTorch 双卡分组固定梯度参考与 12 个 Jittor 候选阶段。CPU+CUDA、双 CUDA 同组、双 CUDA 分组各四步及其 checkpoint 第 2 步新进程恢复至第 4 步均通过预注册逐字段对照；恢复初态和后续结果也与各自 Jittor full 第 2 步和尾段对照。两设备的进程 UUID 以及 CUDA 端点参数、梯度、m/v 的 driver DEVICE 指针和 ordinal 有记录；CPU 端点的设备为公开分类，不是 driver 内存证明。实际 overlay、core、utils 和 CUDA runtime 映射前后身份及 fallback 0 已核。

固定梯度场景还覆盖四步中间无显式逐步读回、手工梯度 AMP、FP16、BF16、capture 拒绝及 fused：fused 真实选择和调用各 4 次，逐参数 fallback 0；capture 拒绝不表示 replay 已验证。无逐步显式同步不证明底层无内部同步，其他阶段的观察和端点指针检查会同步并物化。各项按预注册的 dtype 对应容差比较；新双卡 PyTorch 参考未采原生库实际映射，历史 PT 参考按保存工件绑定，恢复并非 PT checkpoint 续训对照。这是四参数小样本功能实验，不覆盖真实模型训练、收敛、性能、完整 AMP／GradScaler 或所有设备路径，Python 候选尚未合入主树；历史 CUDA 700 和先前严格失败保留。

主功能 cache 端点 2985 → 3029 项，新增 44、无删，旧项仅 lock 文本变化；隔离 r4、性能及封存 SDPA 三份保护 cache 的端点快照相同。unit 成功结束，记录的任务进程在终审时不在现场，终态锁无内核记录；缓存端点和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-overlay-functional-r2-independent-terminal-r1/audit.json`，SHA-256 `958186907811568d5909d95fda64c8a653dafc6ac0eb1895c03372cb6fa6051c`。

## 2026-10-30 主树 df45 core：临时 scope 的 Adam 单步对照

在真实主树 df45 Python 源及其原主树 CUDA cold core 上，以独立进程对 ambient CUDA 0、参数位于 CUDA 1 的场景执行一次 Adam 更新。测试在调用期间临时包装 `adam_update`、`optimizer_state_scope` 和 `_optimizer_arithmetic` 并观察实际分派；p0、p1、p3 三个有梯度参数参与更新，p2 无梯度。与保存的 PyTorch 参考相比，梯度精确一致，参数和 m/v 在 `atol=rtol=2e-6` 下通过，step 已检查，fallback 为 0。实际 core 和 CUDA 库的映射身份以及目标参数的 DEVICE 指针、ordinal 1 均有记录。

逐参数观测中，scope 前、内、后的 driver context 和 runtime device ordinal 均为 0，而 dispatch 为 `cuda:0 → cuda:1 → cuda:0`；实际分支记录为 `native_or_other`。测试未证明 driver context 切换到 1，也不能确定先前 CUDA 700 的根因。观察器的临时包装、指针物化及 scope 外强同步改变了执行条件；这只是受干预的单步对照，不是无干预多步训练或永久产品修复。此前隔离候选 suite 的 CUDA 700 原始失败继续保留。

主功能 cache 的端点记录为 2981 → 2985 项、增加 4 项且无删除，既有项中仅锁文件文本变化；隔离 r4、性能和封存 SDPA 三个保护 cache 的端点快照一致。systemd unit 自然成功结束，已记录的任务进程退出；锁与进程审阅受可见性限制。独立终审：`/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-adam-control1-scoped-r2-independent-terminal-r1/audit.json`，SHA-256 `4dfee9f053d4d13ddfb80cb686acff683cccdd694b8bfb754cadce11513a5d58`。

## 2026-10-30 隔离 r4 Adam 环境与 placement scope 对照

使用隔离 r4 冷编产品，两个独立进程分别执行 CUDA1 参数的单步 Adam 对照：ambient CUDA1，以及 ambient CUDA0 且仅在 Adam 调用中进入参数 placement scope。两项均完成 p0、p1、p3 三个有梯度参数的更新；梯度与保留的 PyTorch 参考精确一致，参数和 m／v 在预定 `atol=rtol=2e-6` 内一致，fallback 0。

在 ambient0 scoped case 的三个参数上，依次采样的 driver context ordinal 和 runtime current device 均为 0，scope 前／内／后 ambient dispatch 为 CUDA0→CUDA1→CUDA0；记录的算术分支为 `native_or_other`。这些读数并不表示实际 CUDA context 已切到 CUDA1，也不证明唯一算术路径或先前 CUDA 700 的根因。scope、指针物化和强同步改变调度；这只是隔离候选上的单步小型对照，并非原主树或完整模型验证。此前 suite 的 CUDA1 Adam CUDA 700 和其他未执行项保留，候选异常恢复、非 primary context 和 capture 并发限制仍未解决。

候选 cache 端点新增 4 项、无删且既有项仅锁文本变化，三份既有保护 cache 端点一致；unit 成功结束、任务 PID 不在现场，阶段间任务 cgroup 核验仅 controller；端点快照和 `/proc` 可见性受限。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-adam-control1-scoped-r2-independent-terminal-r1/audit.json`，SHA-256 `67c7a44f28cb9b4971cc74ce4b34650cb44e82a7b54b158eee228fbb6551d853`。

## 2026-10-30 隔离 r4 staging 候选定向功能首错

基于 r4 私有冷编产物的一次定向 suite 在首错停止前完成五项：FP32 scalar 的 CUDA0→1 和 CUDA1→0 原始字节复制，两方向各 257 元素的 producer→copy→consumer 链输出逐字节符合独立期望，以及 CUDA0 上四参数、三个有梯度更新的 Adam 单步与保留 PyTorch 参考在预定容差内一致。pending 链没有显式 Python 同步，但 staging 后端本身会同步，观测也会物化和同步；这五项实际 CUDA 设备和 core／utils／runtime／driver 运行时映射前后记录核符，fallback 0。

第六项 CUDA1 Adam control 的首个参数更新结果 `sync(True, False)` 期间，设备查询 `cudaMemGetInfo` 报 CUDA 700／illegal address，worker 以 rc70 停止，首更新未完成；launch ring 仅提供候选调用，不能证明肇事 kernel 或先前 CUDA 700 的根因。该失败 worker 仅有映射前记录、没有终态设备 PID／UUID 记录。按首错停止，后续 16 项未执行，包括 8 MiB 边界、多 dtype 和 capture；不能将前五项外推为完整跨卡、模型或错误恢复通过。

候选仍仅在隔离源和私有 cache，未知完成状态下的设备分配生命周期、非 primary context、capture 并发及异常恢复限制未解决，不能直接合入。候选 cache 端点增加 20 项、无删且既有项仅锁文本变化，三份既有保护 cache 端点一致；unit 按失败退出、任务 PID 不在现场，终态锁记录核验通过；端点快照和 `/proc` 可见性存在限制。原失败均保留，不自动重试。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-staged-copy-candidate-functional-r2-independent-terminal-r2/audit.json`，SHA-256 `ff90f05d2cf42b31c9fec2b66a78e518dbe2e94d8e0cd1174175ac23217efbe9`。

## 2026-10-30 隔离 r4 跨设备 host staging 候选 CUDA 冷编

在独立源副本及专用 functional cache 中，ordered 跨不同 CUDA 设备的 pageable host staging r4 候选完成一次真实 CUDA 冷编：日志记录 core 228/228 与 CUDA backend 3/3，stamp 列出 231 个编译单元，controller 自然 rc0、约 393.38 秒。源副本与 df45 执行清单的区别仅为候选 driver，38 个忽略 pyc 按已审资源绑定；实际私有 core／utils／stamp 的磁盘路径、内容与身份已核。逻辑 CUDA0 对应 GPU UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`，单卡基础算术 sentinel 得到 `[3, 5]`、fallback 0；没有单独执行 CPU sentinel。

该冷编和单卡算术不验证跨卡 staging 功能、双卡模型或错误恢复，也不能作为可合入结论。未知完成状态下源／目标设备分配的生命周期、非 primary context 身份与恢复、capture 并发、失败资源累积及设备范围同步的影响仍待处理或验证；原 CUDA 700 和所有失败证据保留。构建进程运行时的实际共享库 maps 未采样，已记录 readelf 与当前库磁盘身份，不能据此冒称实测映射身份。三份既有缓存端点一致，候选及既有锁现场无记录且可见 holder 空，unit 成功结束、任务 PID 不在现场；1792 个 `/proc` FD 目录拒读，端点快照和采样存在可见性限制。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cuda-staging-r4-private-cold-independent-terminal-r1/audit.json`，SHA-256 `35a01578d7e70853a741521b2a9c2e45f6a92701b2e060b2ea1d91e20181ffea`。

## 2026-10-30 启用 peer 后 driver 与 host 中转对照

在单个独立进程中，显式 target 1→source 0 peer enable 返回 0 后，针对逻辑 CUDA0→CUDA1 的同一源／目标指针对固定依次执行 C→H，传输一个四字节 FP32 scalar。C 使用目标 context 下 driver `cuMemcpyPeer`，目标双读 `ffffffff`（NaN，MISMATCH）；H 实际从源做两次 driver DtoH，得到 `e286013d` 的 host buffer，再在目标 context 同步 HtoD，目标双读 `e286013d`（MATCH）。每步源前后为 `e286013d`，目标事先重置并双读 `a5a5a5a5`，设备、context、线程、地址、返回码及次序记录已核；进程 rc0 表示记录完成，不能将 C 的错误结果标为通过。

这是先 C 后 H 的单次、单方向、四字节观察；host 中转、重新设置 sentinel、同步和双读均会改变执行，不能证明 H 对其他大小、方向或长期运行可用，也不是通用 fallback 或性能结论，不能单独确定此前 CUDA 700 的根因。实际 core／runtime 映射、两卡 PID／UUID 和 fallback 0 已记录；功能缓存无增删且仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场、终态三锁无内核锁记录；端点快照与 `/proc` 可见性受限，不能排除瞬态活动。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-peer-enabled-host-stage-r1-independent-terminal-r1/audit.json`，SHA-256 `8e698bd03ce26eb08cd50f1f48f907eafe67142983cbb75c2eb6dd269b565a5d`。

## 2026-10-30 异步复制显式 peer enable 前后对照

在独立单进程、逻辑 CUDA0→CUDA1 的同一源／目标指针对上，使用同一个目标 nonblocking stream 和同一对 events 执行两次四字节 `cudaMemcpyAsync`：D0 目标双读为 `e286013d`（MATCH），随后显式 target 1→source 0 peer enable 返回 0；D1 目标双读为 `ffffffff`（NaN，MISMATCH）。两次复制及两设备同步均记录返回 0，每步源前后双读 `e286013d`、目标预写 sentinel 双读 `a5a5a5a5`；同一线程的指针、context、event／stream 次序与结束后资源清理已核验。进程 rc0 表示观测完成，不能当作复制功能通过。

D0 前源码路径没有显式 peer enable，但资源创建或框架可能改变隐含状态，且固定先后顺序、sentinel 重置、事件、同步及读回均会干预；该单 scalar、单方向观察不能单独确定唯一原因或此前 CUDA 700 的根因。实际 core／runtime 映射、两卡 PID／UUID 与 fallback 0 已记录；功能缓存 2981 项无增删、仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场，终态三锁无记录；终态前现场检查的可见 holder 为空、约 1787 个 `/proc` FD 目录拒读，端点快照和采样仍有可见性限制。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-async-peer-transition-r1-independent-terminal-r1/audit.json`，SHA-256 `ab2344668889cfd31c9a4a4f96b5bf30b50bd2e987b225637166366067c3cb0c`。

## 2026-10-30 同指针对显式 peer enable 前后对照

在单个独立进程中，逻辑 CUDA0→CUDA1 的同一保活源／目标指针对，第一次在目标 context 执行 driver `cuMemcpyPeer` 后目标双读为 `e286013d`（MATCH）；接着显式执行目标 1 到源 0 的 `cudaDeviceEnablePeerAccess`，返回 0 且清除状态返回 0；再以同一指针对执行第二次 `cuMemcpyPeer`，目标双读 `ffffffff`（NaN，MISMATCH）。两次复制 API 均返回 0，每步源前后双读 `e286013d`、目标 sentinel 事先双读 `a5a5a5a5`，两次复制的指针、设备、context 和线程顺序已核；进程 rc0 表示观测完成，不表示该复制行为通过。

第一次复制之前，源码可见路径没有显式 peer enable，但原有隐含 peer 状态及第一次复制对状态的影响未知；sentinel 重置、同步、读回及固定先后顺序都会干预，不能单凭这次前后观察确定唯一原因或此前 CUDA 700 的根因，也不能推断其他方向、大小和配置的结果。实际 core／runtime 映射、两卡 PID／UUID 和 fallback 0 已记录；功能缓存无增删且仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场，三锁与可见 holder 空；约 1784 个 `/proc` FD 目录拒读，端点快照与采样仍有可见性限制。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-peer-enable-transition-r1-independent-terminal-r1/audit.json`，SHA-256 `417b63b0e1f4e93b6690f5a83acba4ca36caa4bfdef92308a470a74b0fa0f073`。

## 2026-10-30 同一指针对三种跨设备复制顺序对照

两个独立进程分别按 C→D→P 和 P→D→C 执行逻辑 CUDA0→CUDA1 单 scalar 控制，其中 C 为目标 context 下 driver `cuMemcpyPeer`，D 为 runtime `cudaMemcpyAsync` 默认方向，P 为 runtime `cudaMemcpyPeerAsync`。每进程内三步使用同一已核验的源／目标指针对，且每步源双读 `e286013d`、目标预写 sentinel 双读 `a5a5a5a5`；三种操作之后目标均双读 `ffffffff`（NaN），源仍双读 `e286013d`，六个观察均为 MISMATCH。两个进程 rc0 表示记录完成，不能作为复制功能通过；两进程的实际分配地址彼此不同。

两进程各自先启用目标到源 peer 访问，后续 D/P 均重新执行 runtime prepare／peer enable，记录返回依次为 0、704、704；D 两次均处于第二步。目标 sentinel 写入、同步、事件链和 driver 读回均会干预执行，顺序或已有状态可能影响结果，不能由这两种顺序推断完整排列或此前 CUDA 700 的根因，也不能将该结果泛化为所有复制。实际设备、context、线程、指针和 core／runtime 映射及 fallback 0 已记录；功能缓存无增删、仅锁文本变化，两个性能保护缓存端点一致，unit 成功退出且任务 PID 不在现场，三锁及可见 holder 为空；约 1775 个 `/proc` FD 目录拒读，端点快照和离散采样仍有可见性限制。此前失败与对照全部保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-same-pair-sequence-r1-independent-terminal-r1/audit.json`，SHA-256 `b1b0c6bab4f4141bd6b9d38dbb2a061cd45cdcb54c735388d8ee797d011d9ab7`。

## 2026-10-30 CUDA runtime 目标指针先读对照

在独立单次 DS 控制中，目标 sentinel 事先双读为 `a5a5a5a5`，从逻辑 CUDA0 向 CUDA1 的四字节 `cudaMemcpyAsync`、事件链及两设备同步均记录返回 0。完成后，源码可见路径在显式目标 Var sync／地址 getter 之前，保活原目标并以已核验的 device/context/地址范围直接做两次 driver 读回，均为 `ffffffff`；随后执行目标 Var sync 和 getter，地址仍为 `0x78819f425600`，核验后再次双读仍为 `ffffffff`，源前后双读均为 `e286013d`，分类为 `BOTH_MISMATCH`。rc0 表示观测采集完成，不能当作复制正确。

这排除了该次记录中“显式目标 Var sync／getter 之后才第一次出现错误值”的狭义解释；同步 DtoH、地址属性查询、事件和设备同步本身均可能干预，地址相同也不能证明没有同址写入，框架或 CUDA 内部隐含行为仍未知。该单 scalar／方向结果不能确定此前 CUDA 700 根因或代表所有复制。实际 core／runtime 映射、两卡 PID／UUID、fallback 0 已记录；功能缓存无增删、仅锁文本变化，两个性能保护缓存端点一致，任务 unit 成功退出，三锁及可见 holder 为空；约 1793 个 `/proc` FD 目录拒读，端点与采样存在可见性限制。此前六 case、driver、GDB 与原失败保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-pointer-first-r1-independent-terminal-r1/audit.json`，SHA-256 `2927aff26f2e548897337c6e34828b1fe3e7366c83e877098ca29ccb035a4ae8`。

## 2026-10-30 CUDA runtime 跨设备 scalar 六控制结果

六个独立进程分别对逻辑 CUDA0→CUDA1 的同一已知 FP32 scalar 执行四字节 CUDA runtime 控制：`cudaMemcpyAsync` 默认方向推断与 `cudaMemcpyPeerAsync` 各测无事件链、显式事件链以及事件链加目标 sentinel 的变体。六个 case 的原始记录均为源迁移前后各两次 driver 读回 `e286013d`、目标迁移后两次 `ffffffff`（NaN）；两个 sentinel 变体在复制前已双读确认目标为 `a5a5a5a5`。每个 case 的复制调用和两设备同步返回 0，记录的指针、设备、context、单线程与事件／stream 顺序通过对应校验；六次进程 rc0 仅表示采集结束，六项观察均为 MISMATCH，并非复制功能通过。

这些是同一方向、同一数值和特定构造／初始化条件下的有限实测；显式物化、同步、读回、事件链及 sentinel 都可能改变执行时序，runtime 与 driver 对照仍不能单独确定此前 CUDA 700 的根因，也不能推断所有跨设备复制的行为。六个独立 worker 的实际两卡 PID／UUID、core 与 CUDA runtime 映射和 fallback 0 已记录；功能缓存前后均为 2981 项、无增删且仅操作锁文本变化，两份性能保护缓存端点一致，任务 unit 成功结束、任务 PID 不在现场，三锁无记录且可见 holder 为空。约 1789 个 `/proc` FD 目录拒读，端点快照与离散采样不能排除不可见或瞬态活动。原 mixed、分段、driver 两 case 与 GDB 观察及失败均保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-cross-device-control-r3-independent-terminal-r1/audit.json`，SHA-256 `1b41b0b4481e8af0e81d05e4150009d0f531887359f9af40fdf605b7ae09fb77`。

## 2026-10-30 跨设备 scalar 控制与 GDB 调用记录

在两个独立进程、同一已知 FP32 scalar、逻辑 CUDA0→CUDA1 的单次诊断中，driver host-staged 和 driver peer 控制各自记录源迁移前、目标迁移后、源迁移后均为 `e286013d`；另一个进程的 Jittor `.to` 记录源值前后仍为 `e286013d`，目标两次 driver 读回均为 `ffffffff`（NaN）。两个 case 进程 rc0 表示记录和分类完成，不代表 Jittor 跨设备迁移功能通过；不同方法使用各自构造的 scalar，不能称所有复制均有同样问题。

另一个受控 GDB 单例中，实际同线程 `DeviceCopyOp` 与 `backend_copy` 记录 4 bytes、源 device 0 到目标 device 1、ordered；`cudaMemcpyAsync` 记录相同源／目标指针、4 bytes、kind 4 及目标 stream，入口对应返回 0。事件记录／stream wait 和目标设备同步的返回也记录为 0，但目标仍读为 NaN、源值保持 `e286013d`。异步 API 返回 0 不证明目标数据正确，也不能单独确定此前 CUDA 700 的机制；GDB 断点、强同步、设备初始化和读回均改变时序，该单 scalar、单方向结果不构成一般硬件或完整 optimizer 结论。

两项诊断的源码、core 与保护缓存端点核符，功能缓存无增删、操作锁文本变化，任务 unit 自然完成且 cgroup 清空，三锁与可见 holder 空；`/proc` 权限拒读、离散采样和端点快照限制仍在。两 case 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cross-device-scalar-two-case-r1-independent-terminal-audit-r1/audit.json`，SHA-256 `2f10a6b93d0afa2ef21b0b9bb6f6c2cd2a6c7f7ebde5e6d06cfbc9cb835f0afc`；GDB 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-device-copy-gdb-production-r2-independent-terminal-audit-r1/audit.json`，SHA-256 `5428b142a46c6459f8ff9c3f4595d543765d3ebc6cebdddf99ba0a046311998d`。


## 2026-10-30 跨 GPU scalar driver 读回校准失败

在显式同步、driver context push/pop 和两次 DtoH 读回的单次正向校准中，逻辑 CUDA0 的 native scalar 在 clone 前后均为 FP32 位模式 `e286013d`，与 CPU PyTorch 参考一致；显式 `.to(cuda:1)` 后返回值经强同步，所观测指针为 CUDA DEVICE memory、ordinal 1，两次 driver 读取均为 `ffffffff`（NaN），未匹配参考。该次复制前后指针与属性、context 恢复端点检查通过，两张物理 GPU 的本进程归属均有记录。

这是干预下的一次迁移后异常，不能单独证明 backend 哪一环出错或此前 CUDA 700 的根因；迁移后源 scalar 尚未复读，反向设备顺序及后续操作未执行。此前 mixed、分段及本次读回的严格失败原件均保留，没有自动重试。功能缓存无增删、锁文本变化，源码和 core 产品及两处保护缓存端点一致，unit 失败退出且 cgroup 已清空，三锁与可见 holder 空；`/proc` 权限拒读、离散观察及端点快照不能排除瞬态变化。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-gpu-driver-readback-calibration-r1-independent-failure-audit-r1/audit.json`，SHA-256 `fb89b51dba181a90987f1fbc14935754c24f55b10f3a22582c0bdf0aca73b99f`。


## 2026-10-30 双 CUDA 基线错误的分段定位

在显式同步和指针物化的单次分段诊断中，control0 的 3 个有梯度参数完成首步，与已保存 PyTorch 首步对照最大绝对差为 0；21 个所观察指针均为 CUDA DEVICE memory、ordinal 0。control1 将全部参数置于逻辑 `cuda:1`、ambient dispatch 为 `cuda:0`；首个参数的参数、梯度及两个 moment 指针均为 DEVICE memory、ordinal 1，而采样的当前 CUDA context ordinal 为 0。随后首个 Adam 更新结果的 `sync(True, False)` 报 CUDA 700 `cudaErrorIllegalAddress`。

这缩小了有同步干预的观察范围，但不能证明 context 差异导致错误，也不能根据有限近期 launch 记录认定肇事内核；显式同步与指针物化改变原始异步调度。失败 worker 未到终态 PID→UUID 查询，逻辑到物理映射仅有 `CUDA_VISIBLE_DEVICES` 与指针 ordinal 支持；mixed、初始快照和设备顺序反转的后续三个阶段未执行，未自动重试。此前 r4 因启动命令绑定路径占位符未替换而在 worker 前退出的操作者错误及原件同样保留。

功能缓存新增 4 项、无删除，锁文本变化；源码、core 产品和两处保护缓存端点一致，unit 失败退出且 cgroup 已清空，三锁与可见 holder 空；`/proc` 权限拒读、离散观察与端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-two-gpu-segment-r6-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `a6767e10196dfe5caf3eb594494d80ea31b5be41addee7607439279d6c329904`。


## 2026-10-30 隔离 AdamW 候选 mixed-device 与双卡基线失败

CPU 与单 CUDA 同组固定梯度 full 及各自从第 2 步 checkpoint 在新进程恢复的 PyTorch、Jittor 基线和隔离候选共六阶段通过；候选 full helper 为 14 次调用、4 次命中，resume 为 8 次调用、2 次命中。该覆盖限于小型 optimizer fixture 与实际设备状态，不代表完整模型训练或性能。

PyTorch 双 CUDA 同组 full 阶段通过；Jittor 基线在双 CUDA 同组首个更新期间，于逻辑 device 1 同步时检测到 CUDA 700 `cudaErrorIllegalAddress`。前三个参数的更新调用有记录，但无完整步骤；双 CUDA 候选及后续阶段未运行。日志近期 launch 候选不能证明故障内核或原因，也不能归因于尚未运行的候选；严格失败原件保留，未自动重试。功能缓存新增 8 项、无删除，锁文本变化；源码、core 产品和两处保护缓存端点一致，unit 失败退出且 cgroup 清空，三锁及可见 holder 空，`/proc` 权限拒读和离散观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-fallback-mixed-r2-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `55f9a20bb993d6d87a2a52ea730eebd91fe4fa3287eeeda83e748f585e5fca4d`。


## 2026-10-30 隔离 AdamW 候选单 GPU fallback 验证

单 GPU 固定梯度实验以 PyTorch、Jittor 基线和隔离候选分别运行 AMP optimizer 上下文、FP16、BF16 和 capture 拒绝四类 fixture，各 4 步，共 12 阶段通过。AMP、FP16、BF16 候选 helper 均为 14 次调用、0 次缓存命中；capture 阶段活跃捕获的 3 次 helper 调用无缓存命中且发生预期拒绝，replay 为 0，非活跃调用另有 6 次命中。基线与候选的对应值、dtype 和设备精确对照；PyTorch 低精度 moment 使用参数 dtype，Jittor moment 为 FP32，跨框架按预先登记的对应 dtype 容差比较，不能称为位等。

这是固定梯度下的 optimizer 局部检查，不覆盖完整 AMP 训练、GradScaler、capture replay、mixed device 或性能；候选尚未合入主树。功能缓存新增 44 项、无删除，锁文本变化；两个保护性能缓存端点一致，systemd unit 自然成功、cgroup 清空，三锁与可见任务进程/holder 为空。`/proc` 权限拒读和离散观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-fallback-single-r1-cuda-replay-terminal-r1/audit.json`，SHA-256 `02e047a7f874dda7c2279305edcfdb3aa4112949be20f0133f300e91c78bff61`。


## 2026-10-30 隔离 AdamW 候选 BERT 48 预热 A/B 严格失败

基于此前 30 步池轨迹，新用途 A/B r2 在双方采用固定 48 次预热并保持原 1 MiB 热态稳定门槛。baseline0 完成；candidate0 预热末稳定，但第 6 个热样本 reserved 增长 90 MiB，超出门槛，allocated 恒定且热态 JIT 快照不变。基线在第 4 和第 30 次预热各扩张 90 MiB，候选在第 26 次预热扩张 90 MiB；其具体分配机制仍未知，不能认为延长预热可以保证下一个窗口稳定。

严格 FAIL 原件保留，其余四阶段未运行、没有自动重试。单个不稳定候选／基线热样本中位比约 `0.9803` 仅供探索，不能作为正式配对速度比或收益结论；candidate 未生成最终 result，因此没有该阶段完整终态 fallback 和 source-after 证据。不再以单纯延长预热作为后继方案。unit 以失败退出且 cgroup 已清空，源码与 core 产品、两处保护缓存端点一致，私有性能缓存无增删，锁与可见 holder 空；`/proc` 权限拒读、离散采样和端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-warm48-r2-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `2a3add79ae0670e22f2a948329aa26322ffb18b6645c6689c99e3502111b3db2`。


## 2026-10-30 隔离 AdamW 候选 BERT 双侧 allocator 轨迹诊断

在已审真实 CUDA core、相同 BERT base B8/S256 FP32 配置下，隔离基线与候选各执行首步、16 次预热和 30 次带同步及观测的更新。基线预热后 reserved 为 `4266655744` bytes，第 14 个诊断更新升至 `4361027584`；候选预热后为 `4172283904`，第 10 个诊断更新升至 `4266655744`，各增长 90 MiB，之后到第 30 步未再增长。两侧 allocated 恒为 `2184966656` bytes，32 份 JIT 快照在各自进程内一致；候选首步 helper 调用 197 次、命中 196 次。

这是单次有限的池聚合轨迹；逐步同步、扫描和记录会扰动执行，不能确认具体分配位置、长期稳定或性能收益。此前 A/B r3 的严格 FAIL 仍然保留，不能由此改判。unit 自然成功并清空 cgroup，源码与 core 产品、功能及封存 SDPA 缓存端点一致；独立性能缓存 1227 项前后无增删，两个探针文件仅 mtime 变化、锁文本变化，三锁及可见 holder 空。`/proc` 权限拒读、离散采样和端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-allocator-trajectory-r3-cuda-replay-terminal-r1/audit.json`，SHA-256 `cd443b4bfc8116fb0b316356138dc62ae80636df22615ff6481ccf4039cb8a63`。


## 2026-10-30 隔离 AdamW 候选 BERT A/B 热态诊断失败

新用途诊断 r3 保持每侧首步、固定 16 次预热及 10 次热更新的原严格稳定门槛，并在计时结束后逐样本保存 allocator 与 JIT 快照；新增扫描和写盘可能扰动后续样本。baseline0 完成，candidate0 首步 helper 197 次调用、196 次命中，预热末两次稳定、10 个热更新 loss 均有限，但热态 reserved allocator 前九次为 `4172283904` bytes、第十次为 `4266655744` bytes，范围 90 MiB，超出原 1 MiB 门槛；allocated 恒为 `2184966656` bytes，12 份热态 JIT 快照一致。baseline 的同一 90 MiB 扩张发生在预热第 4 次后并随后稳定；具体分配机制及长期行为未知。

该次严格 FAIL 保留，未产生正式配对速度比，也不能反推此前 r1 合并断言失败的原因。unit 失败退出且 cgroup 已清空；三处缓存锁无内核持锁记录，源码、产品和两处保护缓存端点一致，私有性能缓存无增删，仅探针 mtime 与锁文本变化。FD 权限拒读及离散采样限制现场结论，端点快照不排除瞬态活动；未自动重试。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-r3-independent-failure-audit-r1/audit.json`，SHA-256 `1673299b3f176a114d31e33248f7dcf20bb3b511f87b21ebcc36cff01061e9e4`。


## 2026-10-30 隔离 AdamW 候选 BERT 性能 A/B 单次失败

在当前 df45 源码和已审独立性能 CUDA core 上，按 AB/BA/AB 计划运行 BERT base B8/S256 FP32、eager、nonfused 的基线与隔离候选性能对照。baseline0 完成首步、16 次预热、10 个热样本及各 3 次独立 memory/attribution 更新；candidate0 首步 helper 197 次调用、196 次命中，16 次预热末两次达到稳定门槛，10 次热更新的 loss 有限，但随后因合并断言 `hot JIT or allocator unstable` 失败。热样本时序、热态 JIT 变化及 allocator 明细未在异常前落盘，不能确定是哪一条件触发；candidate 结果与配对速度比均未产生，后续四阶段未运行，失败原件保留且没有自动重试。

该实验不能证明候选性能收益或正式性能门槛通过。systemd unit 失败退出且 cgroup 已清空；私有性能缓存新增 4 个同一 JIT 算子相关文件、无删除，两个数学探针文件仅 mtime 变化、锁文件文本变化；源码及 core 产品和两处保护缓存端点保持一致，三锁无内核持锁记录。进程/FD 权限拒读和离散采样仍限制现场观察，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-r1-independent-failure-terminal-r1/terminal-audit.json`，SHA-256 `a970ea0741db18b1c2d48793959a1fde389f98711030118cc4f0f5d5de28a8fa`。


## 2026-10-30 隔离 AdamW 候选：固定 BERT 梯度单步对照

在真实 CUDA0 上，将此前 BERT base B8/S256 FP32、nonfused 且 foreach=false 的 r3 基线保存的同一组 197 个梯度分别注入隔离基线和候选 overlay。逐名核实 199 个参数槽及 197 个梯度对象在 `adam_update` 入口被实际消费，另外两个 pooler 参数没有梯度；两端各执行一次 optimizer 更新。独立逐数组审计确认 197 个梯度、199 个更新后参数、197 组一阶／二阶 moment 的 FP32 位模式精确一致且有限，step 均为 1；197 个参数实际变化。候选 helper 调用 197 次、复用命中 196 次，实测 CUDA dispatch，fallback 为 0。

此对照固定使用已有梯度，未执行 forward/backward、PyTorch optimizer 状态对照或性能测量，不能证明训练连续性、长期数值一致或收益；候选尚未合入主树，mixed device、AMP 等未覆盖范围不因本结果消除。此前 r3 单步严格 FAIL 原件保留，该实验不覆盖或改写其结论。独立审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-injected-r2-independent-numpy-audit-r1/audit.json`，SHA-256 `1c1108aea2ca6591305a9859ded98d0231106b852c1ba75051aecb45050edc66`。


## 2026-10-30 隔离 AdamW correction 候选 BERT 单步与严格差异

隔离的双模块 Python overlay 在真实 CUDA 上分别执行 BERT base B8/S256 FP32、eager、nonfused 且 foreach=false 的基线和候选各一次更新；候选 helper 实际调用 197 次、命中 196 次，命中记录核对了 CUDA0 dispatch、FP32 frontend 及 AMP=0。两侧保存的 197 个梯度、199 个更新后参数和 197 组 moment 均有限；197 个参数实际变化，199 个更新后参数的 FP32 位模式全部相同。

严格 comparator 仍为 FAIL：197 个梯度中 196 个及 197 组 moment 中 196 组位模式一致，唯一差异来自 `embeddings.token_type_embeddings.weight` 梯度及对应 m/v。梯度在 optimizer.step 之前已经不同：711/1536 个位模式不同，最大绝对差 `8.881784197001252e-16`、相对 L2 约 `3.8524e-7`；m 最大绝对差约 `9.0206e-17`，v 约 `8.3304e-28`。这是单步且更新后参数完全相同的直接证据；差异的具体机制与长期训练影响仍未知，不能因这一次而宣称长期一致或性能收益。该实验未执行新的 PyTorch 梯度/optimizer 状态对照，包含大量同步和读回，也不构成计时结果。独立逐数组审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-scalar-r3-independent-numpy-audit-r1/audit.json`，SHA-256 `0ae999b3e8c480f896dc65fa95fc6c131821644aeac41b101ee8044a17cf3ca2`。

此前 r2 在权重加载、forward 及 step 前因配置观察器直接比较内存字典与历史 JSON 字典而 FAIL；失败时保存的 JSON 配置与历史参考相同，本地源码及模拟表明 JSON 的整数键转字符串足以解释误判，但原进程内存类型未保存，不能认定当次唯一原因。r2 和 r3 两个原严格失败均保留，r3 使用全字段严格规范化及类型记录；候选尚未合入主树，未证明默认 fused/foreach 命中、完整训练或性能改善。

## 2026-10-30 隔离 AdamW correction 标量复用候选 CUDA 小型验证

尚未合入主树的两个 Python 模块 overlay 在 df45 真实 CUDA core 上，使用单 GPU、五参数、FP32、nonfused 且 foreach=false 的合成 fixture；PyTorch、Jittor 基线和候选分别完成 full、checkpoint 新进程 resume 及无逐步显式同步的 async，共 9 个阶段。基线与候选按 FP32 位模式精确对照通过，与 PyTorch 的对照在 `atol=2e-6, rtol=2e-6` 下通过；候选 helper 在 full/resume/async 分别记录 18/10/18 次调用与 8/4/8 次实际复用命中。核验了真实目标 GPU UUID、参数和 moment 的 CUDA dispatch，Jittor fallback 为 0；恢复 step 核对的是数值，不能据此宣称 PyTorch step 标量本身位于 GPU。

async 脚本没有逐步显式同步或读回，在最终一次显式同步后检查四步 loss；底层是否发生内部同步及其完整异步生命周期未证明。该小型实验不覆盖 mixed device、AMP/BF16、图捕获、BERT 实际命中或性能收益。终态功能缓存新增 68 项、无删除、既有项仅操作锁文本变化，独立性能缓存与封存 SDPA 缓存端点不变，锁与可见本任务进程/缓存 FD 为空；权限拒读与端点观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-r2-independent-terminal-r1/terminal-audit.json`，SHA-256 `6c832ca93c7639c584371fc1e28f447b16db0cc68802df56a6e1d0b524ec640f`。

## 2026-10-30 隔离 AdamW bias correction 标量复用候选 CPU 证据

隔离候选在当前 df45 工作树上以两个 Python 模块 overlay 验证，尚未合入主树。r4 小型 CPU fixture 分别运行真实 PyTorch、Jittor 基线和候选各 4 步，并从第 2 步 checkpoint 恢复后再执行 2 步；候选标量 correction 记录 28 次调用、12 次实际复用命中。独立复核中候选与基线的 FP32 位模式一致；与 PyTorch 的对照在 `atol=2e-6, rtol=2e-6` 下通过，观测最大绝对差为 `6.103515625e-5`，差值不能单独代替相对/绝对组合阈值判断。r4 的 comparison 进程自然 rc0 且结果文件 PASS，但控制器未在活体扫描中观测到短命进程身份，原整组终态仍为 FAIL 并保留，不能回写为通过。

独立后继 r5 仅执行原本未开始的三个 CPU focused 节点：连续与 checkpoint 恢复对照、低精度不缓存、native 与 AMP 不缓存；三个节点的 setup/call/teardown 共 9 阶段全部通过、0 skip，systemd 单元自然完成。r4 已有轨迹未重跑。审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-r4-independent-partial-terminal-r1/audit.json`，SHA-256 `f174f60be5d79f1b31caf7b3d0eb05709ccd6c552b5f3b6a4c622fef049c3dab`；r5 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-focused-r5-independent-terminal-r1/audit.json`，SHA-256 `910b7ec1ff9e71c9abe721e97db3c8a11e23fb77810a54e12879ee96619adb8c`。这是隔离小样本 CPU 证据；CUDA、多设备、异步图生命周期、真实 BERT 是否命中以及性能收益均未验证。

## 2026-10-30 BERT B224/S512 近容量单步校准保护停止

在实际 81920 MiB GPU 上，BERT base FP32、B224/S512、合成输入及 AdamW 的一次有界校准首先运行 PyTorch；单步期间 NVML 采样最低可用显存为 2326.75 MiB，目标进程采样峰值为 78700 MiB，低于自设的 6144 MiB 全局余量后控制器仅向本任务 worker 发送 SIGTERM，进程 rc -15，Jittor 阶段未启动。200 ms 离散采样及停止延迟不能保证余量始终不低于阈值，也不能将进程采样峰值当作 allocator live 高水位。原运行没有 result 或 allocator sidecar，不能确认更新完成、实际 allocator 峰值，也未记录 OOM 异常；不能称为 OOM 或宣称该尺寸的双端完整能力。

自设 80% allocator live 与 6 GiB 余量属于本次单步校准，不是原正式门槛；完整 near-capacity 三轮性能和显存验收仍未完成。失败原件保留，不自动换尺寸或重试。systemd 单元 failed/exit-code 1，cgroup 消失；源码、原生产品、私有性能缓存与两受保护缓存端点快照不变，相关锁和可见任务进程/缓存 FD 为空，权限拒读进程的状态仍未知。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-capacity-r4-independent-terminal-r1/terminal-audit.json`，SHA-256 `ef52b815d966addf6b30734a654ff287a5c928ceffe8fb710abc3bebd0ad1732`；另一独立审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-capacity-r4-independent-failure-audit-r1/audit.json`，SHA-256 `81b8f566cb0a1199c13abf40f6136a97865f47082398ba03555d7e76b955f143`。

## 2026-10-30 df45 BERT 代表格三轮性能证据

基于 df45 工作树及独立 CUDA 性能缓存，r18 完成 BERT base、B8/S256、FP32、固定合成 token/mask、实际 AdamW 的三轮交替配对；每端每轮固定 16 次 warmup、10 个同步热样本，另做独立的 3 次 memory 与 3 次 CUDA attribution 更新。六阶段均有 33 次更新后的有限 loss 记录；加载时逐项核对 199 个参数和 2 个 buffer，197 个梯度有限且位于实际 CUDA，fallback 为 0。该配置不是容量边界或真实下游任务质量实验。

三轮共 30 个热样本的 pooled 中位耗时：PyTorch 118.458 ms，Jittor 155.516 ms，JT/PT 为 1.31283，超过原 1.10 的时延诊断阈值；逐轮中位数比为 1.4556、1.2137、1.2680。pooled p95 分别约 130.296/166.738 ms，标准差约 13.350/19.875 ms。分离测量的 allocator live 峰值为 PT 3,343,963,136 字节、JT 3,613,674,496 字节，比值 1.08066，低于 1.10 的该代表格诊断阈值；其为各框架所报告的 allocator 指标，不能当作总设备显存或整个发布验收。采样的进程 NVML 峰值分别为 3996/4544 MiB。

这项新协议明确使用 eager、固定 math 设置和 16 次 warmup，memory/attribution 与热计时分离，与历史固定 4 次预热等协议并非同口径；加载和首步含观测开销，热样本间写记录可能影响采样。五次稳定入场与阶段 NVML 为离散观察，无法排除全部瞬态负载或权限不可见进程；未证明厂商 kernel 内部算术精度。原 Day4–5 全范围性能与 near-capacity 显存验收仍未完成。旧 r7、r11、r16 失败及其原因不确定性均保留，不回写成通过。

六阶段及 controller 自然 rc0；独立 systemd 单元终态成功且 cgroup 消失。执行前后功能缓存与已封存 SDPA 缓存快照一致；私有性能缓存既有 math probe 两个文件仅 mtime 变化且内容、身份与大小经核验，操作锁文本变化，终态锁和可见任务进程、缓存 FD 均空；端点快照不证明期间没有瞬时写入。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r18-independent-terminal-r1/terminal-audit.json`，SHA-256 `08d7958485ff53880819f91418125b44a6e503354c82c9dcd084bbb5c9fb18e8`。

## 2026-10-30：df45 BERT 性能 r16 单对诊断及失败终态

r16 的固定 B8/S256 FP32 配置仅完成一对 PyTorch/Jittor：10 个热样本中位数分别为 96.51/154.35 ms，JT/PT 为 1.599；独立三步 memory 阶段分别记录 PT allocator peak 3,343,963,136 B 和 JT pool peak 3,613,674,496 B，比值 1.081，两端分配器指标口径存在差异。两阶段各自 rc0，配对条件检查通过；该结果仅作单对诊断，未完成正式三轮验收，也不是近容量结果。逐步有限性、设备、fallback、采样和数学条件的具体证据及限制见独立审计。

控制器终态为失败：性能缓存前后快照无新增或删除，两个既有探针 tmp 源文件内容哈希与大小不变、mtime 变化，既有锁文件内容与 mtime 变化，保护缓存端点快照不变；端点快照不能排除瞬态写入。控制器在 finally 中用缓存元数据失败覆盖先发异常，故未继续到第二对的直接原因未知，不能推断为特定负载或编译问题。锁及进程的陈述以各次实际审计为准，权限或采样不可见范围不外推。旧 r11 失败和本次失败均保留，未自动重试。独立审计：`/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r16-pair-independent-audit-r1/terminal-audit.json`，SHA-256 `ed6b0f377c9c9de05b62c518f3447fdc9a91dac003200d42a14b4b55bf0fb4db`。


## 2026-10-30 df45 BERT B8/S256 配对性能 r11 失败保留

在工作树源码指纹 df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3、执行时 HEAD e92681d291272db1f1b9d0e3fe18cf35775a4de2 下，r11 的 PT0 阶段完成并保存 10 个热态样本；JT0 的首次更新及 4 次 warmup 按控制流返回后，合并的 JIT 缓存或 allocator 稳定性断言失败。该次未保存 warmup 各次缓存差异与 allocator 值，无法判断是哪个条件触发，也不能声称已测出具体不稳定原因；JT 热态样本数为 0，没有完成 PT/JT 配对性能比值。失败保留，不自动重试。

独立终审确认独立性能缓存端点新增 277 项、没有删除，既有项仅锁文件文本变化；受保护缓存和绑定产品前后未变。锁终态无内核记录及可见持有者，但进程/FD 权限拒读与离散采样仍是边界，缓存端点对比也不能证明期间没有瞬时改写或确定各 warmup 的 JIT 活动。该结果不覆盖近容量格、完整性能验收或其他版本。独立终审：/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r11-independent-failure-terminal-r1/terminal-audit.json，SHA-256 1f1541c3c850494b213acd9891c966e449fff823846547b87ff839e914284cba。

## 2026-10-30 df45 独立性能冷编与 BERT 代表格首阶段失败

为隔离旧功能缓存和已封存的 SDPA 性能缓存，基于 df45 工作树执行源码指纹 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3` 在新私有缓存真实冷编 CUDA：231 个编译单元，约 400.13 秒，自然退出成功；CUDA sentinel 实测数值 `[3, 5]`，实际 dispatch 为 CUDA 设备 0，fallback 前后为 0。旧功能缓存与 SDPA 封存缓存的冷编前后快照一致，不能由端点快照推断期间不存在瞬时写入。独立冷编终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-cold-independent-terminal-r1/terminal-audit.json`，SHA-256 `6fd06492e5eb57e7f07e48661c1e0c9f16d1ee8bf8041fda8bb78760d577224b`。该产物来自工作树，不等同于纯提交树产品。

BERT B8/S256 FP32 合成输入代表格 r7 的五次稳定采样 READY 后，首个 PyTorch 阶段在热态及 CUDA attribution 后因映射观察器遇到 deleted 映射而以 rc1 结束；未记录触发的映射路径，原失败原因的具体映射类型未知。代码独立查证显示观察器在筛选目标库之前对任何 deleted 路径报错，无关的 memfd 映射也可能触发；不能据此认定本次实际触发项属于该类。Jittor 阶段未启动，未产生可用双端配对比值，不构成 Day4–5 性能或显存验收通过；单次失败原件保留，不自动重试，现准备限定该观察缺陷的独立后继。失败终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r7-independent-failure-terminal-r1/terminal-audit.json`，SHA-256 `6ebcd8885b8ca30079b88def29c10a4162e3fecef5a235d866b78f2a77bacb7e`。NVML 是离散采样，受权限和采样间隙限制。

## 2026-10-30 Qwen 历史 checkpoint51 在 df45 工作树的单步续训核验

历史 Qwen checkpoint 51 加载到 df45 工作树（执行源码指纹 df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3），仅执行一次更新至 step 52，不能称为完整训练或纯提交树运行。112 个 adapter 梯度在 CUDA 上有限、112 个 adapter 参数发生更新；两组学习率由 1.59375e-5 变为 1.5625e-5，fallback 前后均为 0。设备核验的 337 项仅包括 112 个 adapter 参数、224 个 Adam moment 和 1 个 buffer，不涵盖冻结的 base 模型。

两个真实训练 microbatch 的首层 q_proj 输入及权重为 FP32、输出为 BF16，autocast 开启且记录实际 CUDA dispatch；未证明 kernel 内部算术精度。恢复时对照 Python、NumPy、Torch CPU 和 CUDA 四种 RNG，以及样本 ID、输入和标签形状；没有逐字节比较 token。严格恢复比较仍有两个新增 optimizer group initial_lr 字段，不能称为严格相等。两个 microbatch 的平均 loss 为 1.3973119258880615，与历史 Jittor 对照相同，历史 PyTorch 为 1.3938316106796265；该单步没有新固定评估、predict、完整训练或性能结果，观测同步及物化也不支持性能推断。历史原 strict 失败与 Qwen predict 限制继续保留。

缓存端点快照记录新增 304 项、无删除、旧非锁项未变；操作性锁文件内容变化，锁 inode 保持且终态释放。端点快照不能证明期间没有瞬时写入，权限拒读的进程与 FD 仍属未知。第二独立终审纠正了第一版将操作性锁文本变化误算作普通缓存变动的 false 结论，原审计保留，未重跑。证据：独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-r1/terminal-audit.json（SHA-256 83600fd6dddb9b5693e0411bf20a00d65886510ac2beeb4afcfd41b7d5989e80）；第二独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-audit-r2/terminal-audit.json（SHA-256 b393b5448bb0dbe2d1319ac015cff824543a833a12a41ec6c2d56855f7142f15）；其前一版 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-audit-r1/terminal-audit.json（SHA-256 b08906628c4caf9e8567f7ae5ee9db54f0bbc674dd58059d2e6614fb6461e42c）；包内终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-preparation-r1-hold/attempt-01/terminal-audit.json（SHA-256 f1c0a1ed23db4dd715681ed74e8258433fc30080b04bfe7be7009b1b3aecc6e2）。

## 2026-10-30 df45 源码范围与四卡 NCCL 默认传输复核

只读内容检查确认：当前工作树的 df45 执行清单 2815 项均与原清单内容一致；当时 HEAD 05630e82 对应有 79 个已跟踪路径不同、24 个清单路径未跟踪。此检查未重新盘点完整工作树、未重新构建或验证当前运行，因此工作树训练证据不能称作纯提交树同源。审计 /data6/cgq/jittor-lab/_state/release-day23/20261030-postcommit-worktree-source-recheck-r1/audit.json，SHA-256 72650d4a2c3e0f1bfb3772748d41254474353c67eae72ac788ef75042dcedffc。

历史默认四卡 NCCL Trainer 在 PyTorch 阶段超时，Jittor 阶段未启动；独立 PyTorch 原生四 rank all_reduce 默认配置同样在 60 秒内未返回，另一次禁用 P2P 后四 rank 通过并观察到 SHM 通道。证据支持有界传输配置诊断，未定位根因或唯一故障链路；不能据此断言默认 P2P 可用或归因 Jittor。历史四 rank tiny BERT LoRA full/恢复通过绑定 NCCL_P2P_DISABLE=1 的特定配置，不能外推默认配置；历史源码身份及其他边界不变。审计 /data6/cgq/jittor-lab/_state/release-day23/20261030-four-rank-nccl-default-readonly-audit-r1/review.json，SHA-256 06060640e972e1bdc4c9cad3c7442a11513ded34e66156d9c6334524a6bb3736。


## 2026-10-30 历史T5 BF16续训功能证据复核

历史候选两seed checkpoint50的model/optimizer/scheduler/scaler共2088项加载签名在新进程on_train_begin比对通过；续训50→100各50更新、0 skip、99 microbatch样本ID/shape与各自full后半一致、50组LR一致，梯度missing/nonfinite均0、参数cuda:0/fallback0。第二epoch记录loss均值 full/resume：821 `1.48445840/1.48436376`，823 `1.49720019/1.49706295`；固定eval loss full→resume分别 `3.06211638→3.06272078`、`3.05675101→3.05602074`，exact match均0。仅支持历史该配置的有界功能续训和验证loss接近，不能断言任务生成质量或当前main适用。

两seed原strict各771/2088 FAIL，均为257模型+257一阶+257二阶moment；最终257 FP32模型tensor均可读有限，full/resume模型全局相对L2分别 `8.8914e-7/8.8376e-7`、max abs `0.00032434/0.00022142`，moment数值差幅未知。加载签名未包含RNG，原有RNG文件存在不等于四类实际恢复匹配，moment加载后的真实设备也未逐项核验。两seed的JT/PT full checkpoint50所需模型/optimizer/scheduler/RNG/签名文件此处实际存在，resume目录引用full的checkpoint不是缺失；历史其他22文件缺口本次未复核。只读终审 `_state/release-day23/20261030-t5-historical-resume-readonly-audit-r1/audit.json` SHA `2b46c2674051ee643897fd20000fba6ddeac12d0959d04f1792958428375164c`；原strict与预算 `51.29068389162421s` 均保留。


## 2026-10-30 Vision r25 五格历史实用训练与恢复证据审阅

五格历史r25实际训练均完成100更新、两epoch样本加权训练loss下降：CNN FP32 seed821 JT `2.312916→2.217066`、seed823 `2.322480→2.221158`，CNN BF16 821 `2.313635→2.218671`，ViT FP32 821 `2.299955→2.261349`，ViT BF16 821 `2.300069→2.261586`。固定65样本eval JT/PT loss与accuracy依次：CNN FP32-821 `2.190925/2.190382`, `9/65 vs 11/65`；CNN FP32-823 `2.184424/2.183609`, `7/65 vs 10/65`；CNN BF16-821 `2.189611/2.191783`, `7/65 vs 10/65`；ViT FP32 `2.211886/2.211886`, `21/65 vs 21/65`；ViT BF16 `2.211897/2.211983`, `21/65 vs 21/65`。小验证集CNN accuracy差异如实保留，不能由loss接近推断无差异或外推收敛。

五格resume 62–100样本ID/输入摘要/LR与各自full均39/39匹配，PT/JT full样本/输入100/100一致、fallback0；代码加载optimizer/scheduler与四类RNG，但未有加载前后optimizer状态及RNG独立对拍，恢复功能范围因此有限。两格CNN FP32同运行时strict exact通过、PT严格仍分别FAIL160/177；ViT FP32/CNN BF16/ViT BF16同运行时exact分别FAIL100/245/143，后续跨框架strict未运行，不能称全部门禁通过。只读原报告/NPZ及hash审计 `_state/release-day23/20261030-vision-r25-five-cell-offline-audit-r1/five-cell-summary.json` SHA `16c3ec54fafae6b0e896c7743aad4220edd6ccfce7cd0c115c1f0ce9dc6813f3`；绑定历史r25 source `1be46e54…`，非当前工作树或提交树，原310.095694191秒诊断预算未动。


## 2026-10-30 任务范围中文提交及保留的工作树

已提交 `cf5be28b618b6d99c89af8856d67dab618a053f3`（修复 Torch 兼容训练与恢复），parent `3dedae001372b6be106ae6700bcf10ebc7faebac`，tree `657d6e1da328fd064aba6735be29a0e8f2eb76d0`；仅由独立审阅的59路径任务patch暂存，cached diff --check通过，未push。提交前备份全部144个dirty路径与原空index；提交后逐项验证除3文档有审定的可逆摘要插入外，其余文件内容/模式不变，index为空，保留的dirty相对新HEAD仍存在；后审 `_state/release-day23/20261030-task-only-commit-postcheck-r1/postcheck.json`。候选代码与此前r12仅两处空白、types.py两空行和文档变更，结构单节点已验证前一同代码tree；工作树与候选22条路径不同，旧dirty相关依赖重合已披露，不能将working tree的df45源码身份自动转成提交树训练结果。更改HEAD后旧执行fingerprint包含旧HEAD，未来执行必须新绑定，不能把原缓存/历史证据重新贴标签。性能测量未取得稳定窗口、BERT续训strict差异原因及影响仍未完全解释，原两seed strict FAIL、T5/Vision预算与历史失败均保留；发布验收未完成。


## 2026-10-30 任务证据更新（历史记录按原身份保留）

HEAD-only r12 两卡实际训练与恢复使用 **r10 原生 core 和 Python 根目录，加 r12 两个 scheduler Python 模块 overlay**，并非完整 r12 冷构建。FP32 tiny BERT LoRA Trainer 两 rank 完成 full 4 次更新、checkpoint 2 新进程续训至 4；两卡真实 CUDA、fallback 0，有限梯度及参数更新被检查。JT/PT 两侧各 rank 平均 microbatch loss 分别为 0.34503534/0.34503534、0.34377843/0.34377844；固定 16 样本短程证据不能推及长程质量、AMP 或多于两卡。独立终审 fc2fc52657c9e6a31ef5ae9d8c2be617833018ffdd4ee24e3579ca599436fc17。

主树 df45 是另一源码身份：BERT seed821 配置 BF16 从头完成 full100，记录的参数和优化器状态为 FP32；训练前 initial evaluation 显式 autocast 下的 CUDA logits 为 BF16。projection dtype 和 autocast 状态按跨阶段集合记录，未标记训练时段，因此训练期间 BF16 覆盖未独立确认；历史 PT 报告同样未分阶段。训练结束后生命周期保存/重载对照使用 keep_fp32_wrapper=False 移除 forward 包装；两 epoch 平均 loss JT 0.354515→0.208091、独立历史 PT 0.354172→0.196566，固定验证 loss/accuracy JT 0.477709/0.787879、PT 0.483606/0.787879。新进程从同源 checkpoint 50 恢复至 100，恢复时 603 命名 CUDA 张量、RNG 和状态通过检查，后续 50 次更新有限、fallback 0；验证 0.478523/0.787879。最终 201 参数和各 201 一阶/二阶 moment 的严格哈希共 603 项不同，201 step 相同；相对全局 L2 分别约 4.46e-5、0.0505、0.00580，原因未定。原 strict 失败和原两 seed 15.425953905% 对 15% 门槛的失败均保留。

补充的独立训练阶段诊断采用同源 df45 与历史 PT 结果：两侧各两个 training_step microbatch 的首层 attention query projection 均记录 CUDA autocast 开启、FP32 输入及权重、BF16 输出；JT 同时验证 native CUDA dispatch 和 fallback 0。两侧各 201 梯度有限且参数更新；首个 optimizer step 前的零学习率人工改为 6e-6 后执行一次更新。该诊断未覆盖原 full100 的每一步或其他 projection，未追踪内核内部算术与累积精度，也不能作为性能基准。终审 SHA-256 c5cc3bb5a5c84d555753e0ee22d9f1e564f526274a29cf0d35bc6b59ddb9d044，路径 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-bf16-training-phase-jt-shim-successor-r1-hold/independent-terminal-audit.json。原 full100 跨阶段集合的观察边界不变。

同源 checkpoint100 同批单步诊断比较联合状态差异：两侧都真实 CUDA、201 梯度有限及参数更新、fallback 0；重置 seed、将 checkpoint 的零学习率改为 3e-5 且不推进 scheduler，更新向量相对 L2 差约 0.0861、cosine 0.9963。参数、moments 与梯度起点均不同，不能归因单一 optimizer 或视作自然续训；其对长期训练影响未知。

隔离性能 cache 的 SDPA 合成同形输入 A/B 正确性通过（输出及梯度对照、11 个边界节点），并非整模型性能。唯一 60 秒配对 readiness 全 13 次采样未达到空闲显存、利用率及无外来进程门槛，未启动计时；无加速或性能验收结论。Day4–5 性能/显存仍未完成。原性能预算与历史失败保留。

本候选结构节点在 3068da63… tree 上实际 1 项通过、无跳过：原测试方法读取完整候选源码，使用桩处理无关框架导入；不是完整结构套件。后续候选只修改三份文档，代码 blob 相同，训练及测试仍按各自原混合或主树身份解释。旧 HEAD 802/800 结构失败、旧执行错误和用户已有 dirty 均保留。


## 2026-10-30 df45 BERT真实训练阶段BF16补充验证

针对完整full100历史observer跨阶段混合set的问题，另做相同配置、真实Trainer训练阶段标记的两microbatch单步：PT/JT首层query projection两次输入及weight均FP32、输出BF16，CUDA autocast=true；JT三类张量native dispatch均cuda:0，指定物理GPU、fallback0。两侧样本及输入/标签hash相同，201梯度有限及201参数实际变化；两侧均将原首步零LR人工设 `6e-6`，观察sync与RNG历史差异明确保留。PT scaled losses `[0.4248046875,0.3701171875]`、JT `[0.427276611328125,0.34375]`，跨框架随机流未证明一致，不能声称loss逐步对齐。独立终审 `_state/release-day23/20261030-main-df45-bert-bf16-training-phase-jt-shim-successor-r1-hold/independent-terminal-audit.json` SHA `c5cc3bb5a5c84d555753e0ee22d9f1e564f526274a29cf0d35bc6b59ddb9d044`；首个observer读取shim不存在的 `torch.__file__` 造成输出未保存的失败原件保留。只证明两个训练microbatch首层投影的输出精度，不能外推原full100每步/每层或kernel内部计算及累积精度；以下对full100跨阶段observer的范围更正继续有效。


## 2026-10-30 df45 BERT训练精度记录范围更正

上文df45从头full100与同源续训均使用 `mixed_precision=bf16` 配置，真实CUDA、FP32 master参数/optimizer状态和有限梯度及记录loss/验证指标证据成立；但投影dtype/autocast观察器在训练前initial eval注册，只保存贯穿训练、评估与lifecycle的集合而无阶段标记。initial eval明确在accelerator.autocast内、CUDA logits为BF16；该汇总不足以证明训练阶段BF16覆盖，亦不能断言训练全程FP32。Accelerate源码有native_amp forward包装路径，但该运行未按训练阶段独立证实。`keep_fp32_wrapper=False` 移除forward包装仅发生训练结束后的lifecycle保存/重载对照，不应写成训练期已移除。历史PT报告的投影汇总亦不区分阶段，因此先前数值比较为同训练配置下的描述性对照，不能凭现有集合声称训练期两端混合精度覆盖相同。已准备补充阶段标记的有界单步验证；原产物、loss、严格失败和验收边界均不改写。


## 2026-10-29 SDPA隔离A/B正确性通过，稳定窗口未取得

本次新用途使用df45源码真实独立performance JITTOR_HOME冷编core，与功能cache隔离；CPU／CUDA原冷编路径及初始两次controller/驱动入场失败均保留，成功cold `_state/release-day23/20261029-sdpa-r5-independent-cold-terminal-audit-r1/terminal-audit.json` SHA `522b1bb2502073af1fae872d44cd0972dae655f25e3926ebbc8eae81a32862e0`。原A/B尝试因导入后修改sealed cache_path而失败，改为共享同一独立performance JIT cache后真实验证A原模块/B独立shuffle模块：物理seq-major合成输入，float64 output/dQ/dK/dV参考及三次重复通过，A/B max差 `0/1.19e-7/0/0`，B侧11节点无skip，实际源码/so/key不同且marker相符，双方CUDA/fallback0；终审 `_state/release-day23/20261029-sdpa-r6-shared-cache-correctness-terminal-audit-r1/terminal-audit.json` SHA `eecbd81183e3487871ee7e769813f03d8e4de46d7100b4cc0940f021ed40b5e1`。driver cache环境由controller设置但worker未独立记录，不能称实际路径已观测。

60秒13采样单次readiness因目标GPU利用率92–100%、每次有3个外来PID而未达稳定窗口，计时未启动、无速度结论、不再自动排队；终审 `_state/release-day23/20261029-sdpa-r6-readiness-timeout-terminal-audit-r1/terminal-audit.json` SHA `2c64f1686c65d14dd72f2002b1d83da07a03ab4c4e7c85b30447fb0cff23ad3b`。历史BERT性能ratio1.735仍是既有测量及其共享负载边界，不能由本合成正确性测试证明优化收益或根因；旧诊断预算未重置。


## 2026-10-29 df45 full与resume最终状态同批单步实际诊断

针对checkpoint100 603项strict hash差异，离线静态bias-corrected m/(sqrt(v)+eps)方向续训相对full整体L2差10.096%、cos0.994923；保存lr实际为0，因此该方向不是下一步真实更新。一阶moment相对L2差5.052%、二阶0.5799%及模型4.4625e-5继续保留。

独立同批压力诊断分别在新进程加载两份同源checkpoint100，固定原训练两个microbatch `[14,136]`/`[171,66]`、BF16/GA2/clip1，人工将两组lr设为3e-5并重设seed821，分别真实AdamW step101；这是额外干预，非原自然schedule。两侧checkpoint恢复签名/603 CUDA状态、201有限FP32 CUDA梯度及201实际变化参数核验、fallback0；两microbatch scaled loss均 `0.076171875/0.150146484375`，裁剪前梯度相对L2差 `1.943%`，裁剪后 `1.838%`，实际参数delta相对L2差 `8.6095%`、cos `0.996302`；更新后模型相对norm差由 `4.4625e-5` 增至 `5.0940e-5`，一阶moment相对差由 `5.052%` 至 `4.461%`。两侧起点模型、moment和梯度都不同，此为联合效应，不能单归因或认定长期无影响。独立终审 `_state/release-day23/20261029-main-df45-bert-checkpoint100-matched-batch-update-plan-r1-hold/independent-terminal-audit.json` SHA `ed9df0163d615db6d084329cb1dc1033af9ff1dc7aa5a8918d6b83a29495a456`。原strict 603项差异、原两次离线解析失败及双seed strict FAIL均保留；实际full/resume固定验证均26/33且loss接近，分别作为独立功能/任务表现证据。


## 2026-10-29 统一main df45 BERT BF16同源checkpoint续训

主树 `df45a565…` 从本源码full50的真实checkpoint新进程续训至100：checkpoint50加载签名、603 CUDA模型/moment状态、201 step及Python/NumPy/CPU/CUDA四类RNG一致；50续训更新0 skip、每步201 FP32 CUDA梯度有限、首个续训更新201参数变化、fallback0。99个microbatch样本/形状及50步LR与同源full和历史PT一致；第二epoch更新窗口等权均值resume `0.20797455`、full `0.20809052`、PT `0.19656555`，固定验证loss依次 `0.47852302`、`0.47770923`、`0.48360559`，准确率均26/33；功能恢复在此配置通过。独立终审 `_state/release-day23/20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/independent-terminal-audit.json` SHA `f3edc6ba4b820d0d163d70b7f99b37552188211fafaac64c6961e1cc32a2729f`。

最终checkpoint100严格状态603项hash不同（201模型、201一阶、201二阶moment，无缺失/新增，201 step相同）；离线可信本地产物数值分析确认dtype/shape相同且有限，其整体相对L2分别 `4.46e-5`、`5.052%`、`0.5799%`，full/resume单步loss最大差 `0.00354004`，差异原因与实际影响未定，不能以验证接近自动称无害或最终状态严格一致。数值后继 `_state/release-day23/20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/checkpoint100-numeric-comparison-successor-r2.json` SHA `f14388795d5fe399d1880306709190225ea54922775d466eaccb8ce9f92fbd5a`；前两次离线比较因weights_only opcode和portable包装误解析失败均保留。此为单seed、sync/物化诊断干预，原双seed strict FAIL、非性能结论及EventQueue未直接观测保留。


## 2026-10-29 统一main df45 BERT BF16从头两epoch学习趋势

`df45a565…` 在真实CUDA单卡以固定seed821、197训练/33验证、B2/GA2从头完成两epoch/100次更新、0 skip；每步201个FP32梯度有限且实际CUDA，step2非零LR后201参数均变化，1/50/100步603模型与moment状态CUDA，fallback0。198 microbatch样本ID/形状及100步LR与已审历史独立PT一致；按每epoch 50更新窗口等权的mean_microbatch_loss口径，JT `0.35451462→0.20809052`、PT `0.35417236→0.19656555`；最终固定验证JT loss `0.47770923`、PT `0.48360559`，双方accuracy `26/33`，JT选checkpoint100。另一按99 microbatch数加权的epoch1记录均值约0.347694，两种口径不得混用。LambdaLR helper last_epoch=-1实际调用1次、step101次，serialization调用 jt.fetch 4次并返回；native EventQueue未直接观测。checkpoint50/100模型/optimizer/scheduler/RNG文件与签名已核，但未独立解码或新进程续训；梯度/参数检查有sync与物化干预。独立终审 `_state/release-day23/20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/independent-terminal-audit.json` SHA `96f990e7895121b37db7e5b15b728c97839a88883517203e0f129bb1f4821265`。单seed实际学习趋势与任务表现支持此配置的实用训练，但不覆盖新源码optimizer-only重建分支的模型验证、新进程续训、其他网络或原双seed 15.42595%>15% strict FAIL，后者保留；共享负载及诊断干预下不作性能结论。


## 2026-10-29 统一main df45 BERT BF16两epoch实训运行中

固定seed821、197训练/33验证、B2/GA2、两epoch 100更新的全新单卡BERT BF16训练用于验证统一main真实学习趋势及新scheduler/fetch路径；对应历史PT权重/数据/超参及23项引用已核。controller PID 2031669、child PID/PGID 2031672/start_ticks 189609483，单次cap3000秒，运行目录 `_state/release-day23/20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/`，绑定 SHA `7225df80…`。逐步梯度设备/有限性、参数签名和关键步optimizer状态会同步/物化，属于明确诊断干预；Python wrapper记录LambdaLR/helper/jt.fetch调用，native EventQueue未直接观测。启动时GPU UUID `GPU-2be51…` 空闲约51292 MiB，外来负载/利用率100%，仅正确性/学习趋势，不作性能结论。尚无当前训练或验证结果，接续须核PID/start/receipt、锁和实际状态，不重复；历史双seed strict FAIL保留。

HEAD-only r12从r11仅加入持久CUDA启动回归测试，完整HEAD patch SHA `250c43dffca948b5e027ea20f65b0e4141ecffc54b039f410398f6f6b69718c5`，fingerprint `2394be191ac3fbe00a8dc304c623f65c97c25339cadf2f66785c3d58c50c8349`，隔离index对2796文件/链接blob/mode核对，主index不变；静态审计 `_state/release-day23/20261029-headonly-r12-startup-regression-candidate-r1-hold/candidate-audit.json` SHA `8f7453bac4006ff97e3db73346b4c0a076bf9b174f3e0dbcdefbec976f3b8594`。原会话前相关dirty片段重合按r9审计披露；r12未单独实测，候选未暂存或提交。


## 2026-10-29 r11候选CPU调度器及tiny BERT训练/恢复

HEAD-only r11相对r10仅LambdaLR实现/helper/测试三文件，r10真实CPU core + 两个r11 scheduler overlay及r10其余任务模块：八个PT oracle场景、16个调度器/Accelerate节点通过、fallback0，独立终审 `_state/release-day23/20261029-headonly-r11-lambda-cpu-focused-terminal-audit-r1/terminal-audit.json` SHA `efea8cd356d3aa8d73a25bbd0a8fb34d1769d68cd6f30b9d07dc4a2cf72f3f2e`。其后合成tiny BERT CPU、dropout0、八步：25参数每步实际变化、梯度有限、参数/50优化器状态CPU、fallback0；step4保存并新进程恢复到step8，连续与恢复最终25参数最大差0，PT/JT训练loss最大差 `5.96e-8`。固定验证集两端趋势同向变差：JT loss `0.696845→0.767809`、accuracy `0.25→0.125`，不能称收敛或任务表现改善。终审 `_state/release-day23/20261029-headonly-r11-tiny-bert-cpu-model-terminal-audit-r1/terminal-audit.json` SHA `9118537ee24255fe75c3756e4f2070446f129a38cf843c633c6f66decb890239`。仅短程CPU混合身份，未证明GPU、长训或RNG恢复。统一main df45 100步BERT BF16从头学习曲线实验已独立准备，正在入场审阅，未运行；r11候选与main不可混同。


## 2026-10-29 统一main df45 CUDA冷编及功能专项通过

主树 `df45a565…` 同源CUDA真实冷编231 TU、38既有资源、全部产品/锁终审 `_state/release-day23/20261029-main-unified-fetch-lambda-cuda-r1-hold/independent-terminal-audit.json` SHA `783aeffb80ca67ea63e2534e4653a1c1dd94638b7005412f2172871b7e756add`。main core ELF init fetch 88/device 198，故本次pre-import通过不能独立证明device-first；该危险顺序在r10候选已实测。真实CUDA功能六阶段均通过、无skip：preimport `use_cuda=1/use_nccl=1`；持久startup、原fetch/GIL/EventQueue 4/4；32回调/cleanup/异常退出；placement 61/61；CUDA restore 11/11；LambdaLR GPU探针含两次实际参数更新、moment CUDA/有限、optimizer-only重建和缺字段。全阶段CUDA dispatch/fallback 0→0，终审 `_state/release-day23/20261029-main-unified-cuda-focused-r1-hold/independent-terminal-audit.json` SHA `3585f23ab5330b3a39960dfe5590ecf7f02e55dc863218abaa1e79217b77276f`。GPU Lambda中间数字仅断言未单独保存；本次不含NCCL collective/多GPU/超过20同时排队。d900的两rank模型与checkpoint证据及旧CPU BF16 strict失败均保留各自身份，不能转写为df45模型通过；任务候选r11本身与main不等同，待独立回归及模型学习曲线。


## 2026-10-29 统一main df45 CPU构建、调度器及fetch通过

统一主树 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3` 独立CPU cold自然rc0，227 TU、38既有pyc资源、源码/产品/stamp与锁终审 `_state/release-day23/20261029-main-unified-fetch-lambda-cpu-r1-hold/independent-terminal-audit.json` SHA `99244535aa8a4b46785dd63c1f1becbfc50395d812e7083fba1a8c200d2ba860`。随后当前main真实LambdaLR新增6和原scheduler/Accelerate 10节点共16/16、optimizer-only重建8案例与已审PT oracle一致、fallback0；终审 `_state/release-day23/20261029-main-df45-lambda-cpu-focused-plan-r1-hold/independent-terminal-audit.json` SHA `fd32291e46de58ea99b1c0a06db29498ed443c113090405479ce68400d944470`。原d900数值缺陷、CPU BF16 strict失败未覆盖；重建是内存state_dict、非新进程模型恢复。

同源码CPU fetch后继3/3无skip：原fetch、32提交回调值、正常cleanup及预期异常退出子进程，CPU dispatch/fallback0；终审 `_state/release-day23/20261029-main-unified-cpu-fetch-successor-r1-hold/independent-terminal-audit.json` SHA `845fbf58e45b45f42639b8c05cf0decb2e02fe8de438367bb62284843bd598b6`。32提交不证明排队长度>20。正在以三份同源CPU审计为前置准备统一main独立CUDA cold及startup/EventQueue/fetch、optimizer placement/restore、LambdaLR GPU测试；没有当前main CUDA结论。


## 2026-10-29 LambdaLR与fetch队列修复已移入主树，统一验证待运行

经先前隔离CPU/PT八案例和原10回归通过后，LambdaLR修复已最小移入main：保留/设置各组initial_lr并在last_epoch恢复构造时按字段建base、缺字段按PT报错，其他scheduler不改；新增覆盖多组、tensor、optimizer重建和文件checkpoint的测试尚未运行。整合前目标lr_scheduler干净、旧dirty和index均已保存，`_state/release-day23/20261029-main-lambda-integration-r1/integration-review.json` SHA `8259d6fced9bc17730e3b4b1ea2a6e2f23b22d52f186d111a180af0783be5d30`。隔离overlay终审 SHA `6d4e2f342e6b610fc3a947b5618b1549683ce1de928db3e9a64d2a635aa657d3`，原d900学习率错误未覆盖。

随后已审r10 fetch惰性队列四文件最小补丁移入main：目标与候选字节一致，原 `var_holder.cc` 82行既有dirty及主index保留；整合审计 `_state/release-day23/20261029-main-r10-fetch-queue-integration-r1/integration-review.json` SHA `b15cfde8c384c7523ac45596a14a6db270d104cd4d72de3ac90d6fa2c3079842`。仓库新增持久子进程回归 `tests/runtime/test_flag_env_and_setter.py::TestUseCudaAtProcessStartup::test_cuda_enabled_before_jittor_import`，明确导入前设置CUDA、实测dispatch与fallback；CPU-only依据capability skip，实际测试尚未执行。新统一执行源码 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3`，manifest SHA `46243b6f25ecf6e897a4e5e05a306cc5d6f71b207fcd249bf8660eebb6cf443d`。候选r10 CPU/GPU功能与LambdaLR overlay证据不可自动转成统一main通过，正在准备新身份原生产物及定向验证；原r10 CPU strict SKIP失败、初始化signal11和BF16 strict差异均保留。


## 2026-10-29 fetch初始化崩溃隔离候选GPU专项通过

r10 HEAD-only独立候选以惰性fetch队列修复原pre-import `use_cuda=1` signal11；真实CUDA cold 231 TU，实际ELF仍维持原危险顺序device静态初始化索引100、fetch索引145。原环境 `use_cuda=1/use_nccl=1` 进程启动、真实CUDA core加载和 `sync_all` 成功，dispatch CUDA/fallback0；原CUDA fetch、callback GIL和CPU跳过的原EventQueue节点均实际通过无skip，32回调、正常cleanup与异常退出子进程通过。GPU独立终审 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cuda-cold-hold/gpu-functional-independent-terminal-audit.json` SHA `4b30b1babe55b273519b6d9f1f7e172e303788a8397a502db185627cd9f6d590`；CUDA cold后继终审 SHA `318adf9184b61dd2ee4fc5df47e2d7efbccb0562ea901fcd52f871ceba9a6cbb`，先前审计对CPU/CUDA符号可见性相同的错误及原signal11均保留。NCCL环境已启用但未执行collective；32提交未证明队列并发长度曾>20，外部ABI及析构边界仍如实记录。修复尚未移入主树，正在准备与dirty基线兼容的最小移植；调度器独立候选先运行。


## 2026-10-29 LambdaLR重建实测缺陷与fetch崩溃候选构建

主树 `d90073…` 的独立PT/JT CPU实测确认：只恢复optimizer、按 `last_epoch=2` 重建 `LambdaLR`，PT base/LR为 `0.003/0.00075`，JT为 `0.0015/0.000375`；自定义 `initial_lr=0.012` 时PT为 `0.012/0.003`，JT仍为 `0.0015/0.000375`；缺 `initial_lr` 时PT报KeyError、JT成功。完整optimizer+scheduler恢复两端step3均为0.00075，因此不能误称已验证的Trainer恢复失败。内存state_dict独立终审 `_state/release-day23/20261029-main-d900-cpu-scheduler-reconstruction-jt-successor-r1/independent-terminal-audit.json` SHA `66a08bca04c6dd8a53254c6b167bc2a595cbaeff7988d6558f8ad397cf52ff53`，fallback0；初次JT遗漏已审隔离stamp导致导入失败的原件保留。其他scheduler部分使用链式公式，正在针对LambdaLR隔离设计修复，不外推所有类。

r10独立候选将fetch两个队列改为惰性构造以处理 `use_cuda=1` 静态初始化崩溃，主树未改；独立CPU cold 227 TU真实通过 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cpu-cold-hold/independent-terminal-audit.json` SHA `90f6da11c0575ca518c7c7aafb2cbcd137a9b08717a0ff4ca37a1693c3f59930`。CPU fetch严格阶段3 PASS/1 CUDA-only SKIP、原门槛FAIL保留，派生审计仅判三个适用项通过、GPU必须执行原跳过项；`cpu-applicable-successor-audit.json` SHA `5af589cfaf7a023276a087a4e7b2c11eff9e850fa32509cfe0b24a3ac3bbc96d`。旧Linux core队列变量为本地符号、275缓存扩展无未定义引用，未知外部/Windows仍有ABI及无cleanup时队列不析构的生命周期边界。r10独立CUDA cold controller PID 1987842/start_ticks 189307494、worker 1987844/start_ticks 189307558、cap900，目录 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cuda-cold-hold/attempt-01-cuda`；正在编译，恢复先核实际PID/receipt/锁，不重复。原pre-import signal11及GPU功能尚未验证修复。


## 2026-10-29 候选r9两rank模型与恢复后继通过

HEAD-only候选的真实运行身份为r2 native/root + 覆盖9个r9生产Python差异：两rank、16样本tiny BERT LoRA FP32，full4/stop2/新进程resume4以及已审PT对照均rc0；六个rank进程不同、真实GPU/core/NCCL匹配、fallback前后0，每rank六个参数实际更新，候选scope/Torch AdamW step/CUDA fused更新真实调用已记录。每rank step2参数/optimizer/scheduler共40字段恢复最大差0，四类RNG与样本衔接；full/resume最终参数差诊断0。PT共同数值字段allclose（梯度max `1.49e-8`、参数 `1.041e-7`、optimizer `3.73e-9`），resume的四个仅PT状态字段保留；8/9 overlay在训练中实际加载，distribution未用，不能声称完整r9根目录或冷编。sync与Python profile是诊断干预。独立终审 `_state/release-day23/20261029-headonly-r9-bert-wrapper-successor-r1/independent-terminal-audit.json` SHA `791be9292968706fb1e7974c96df289218feeacf962c651b28c77bca0beeb585`。

原full4在首步后观察器把Accelerate包装器误判为候选类型而失败，日志未保存当次包装器类型；分析还发现fused路径不会调用per-param adam_update，不能以此断言未更新。该失败 `_state/release-day23/20261029-headonly-r9-bert-model-successor-r1/independent-failure-terminal-audit.json` SHA `7b431099b185723493dddded63b4822043ce39343d99fa5d7f2cb0a76ffda75c` 保留；后继实际观察包装链及真实fused更新。原use_cuda=1 core初始化signal11仍为未修复问题：只读静态分析显示候选ELF device静态初始化先于fetch_op全局list，崩溃指令读该list且fault地址0x10，与未构造容器相符；需进一步验证、审查另一fetcher_to_free的生命周期，不能将相关假设当成修复通过。


## 2026-10-29 候选r9模型后继运行中

候选safe-init成功导入后原finder绝对第0 guard失败（未到NCCL），原件 `_state/release-day23/20261029-headonly-r9-nccl-safe-init-diagnostic-r1/independent-terminal-audit.json` SHA `b9a6dcff223dccff57f0d8d49cce7c22711ed3632096cebc6b5460d870e9f03f`。独立新诊断验证finder唯一、前置11钩子对9候选模块共99次不抢占，8/9实际加载并核hash（另1未观察）；单进程NCCL实际warm成功/fallback0，审计 `_state/release-day23/20261029-headonly-r9-nccl-finder-diagnostic-r1/independent-terminal-audit.json` SHA `ebba681d9ededee2c4d31b42efb65380a7295ecbea4774a9bf5868695f264fed`。首次core signal11与finder失败都保留，不能以成功后继抹去。

混合r2 native/root + r9九生产Python overlay的独立两rank BERT LoRA full4→stop2→fresh resume4模型后继已启动，controller PID 1955525、full launcher 1955530、rank 1955532/1955533，目录 `_state/release-day23/20261029-headonly-r9-bert-model-successor-r1`；full launcher cap1200秒，失败即停不自动重试。实际两rank已在真实launcher导入时初始化NCCL，首步前各6参数/12状态槽检查完成；当前算子编译、尚无更新和训练结论。检查有显式sync与Python调用profile干预；接续先核PID/start、receipt、实际JIT/锁/资源和source/product，勿从本状态外推完整r9冷编。性能A/B保持HOLD、不得与任务JIT并发。


## 2026-10-29 提交候选r9 NCCL warm在core初始化崩溃

任务候选需独立模型与恢复证据，因其与d900主树有45个生产文件差异；候选模型真实身份为r2原生/根目录加覆盖全部9个生产差异的r9 Python overlay，不能称完整r9根目录冷编。首个单进程NCCL warm在r2 CUDA core动态加载、NCCL setup前signal11，模型未运行，原失败独立审计 `_state/release-day23/20261029-headonly-r9-bert-runtime-r1-hold/warm-failure-terminal-audit.json` SHA `6f0f552673a51ea2ec18ffa3edf34d5e62eeed1bbfdd1e1fc48f06b5474f5b3a`；core/utils/stamp及双源仍匹配，无可见任务进程或锁holder。失败前环境导入前强设 `use_cuda=1/use_nccl=1`，已通过151节点的focused按 `use_cuda=0/use_nccl=0` 导入，栈符号指向 `init_use_cuda → setter_use_cuda → sync_all`；环境仍有其他差异，不能认定唯一根因，signal11实际失败不能抹去。独立safe-init warm诊断正在准备；成功也不能自动证明两rank launcher可用。

SDPA backward独立A/B方案仅静态准备、未运行：`_state/release-day23/20261029-sdpa-f32-backward-shuffle-ab-r1-hold/` manifest SHA `784f6f85f23776f462663268d660868275fb00eca55784f09525caafdd951443`，不重置旧性能预算。候选训练与诊断优先、JIT串行，性能收益与缓存隔离均未实证。


## 2026-10-29 d900两rank BERT LoRA保存及新进程恢复通过

主树 `d90073…` 真实CUDA/NCCL与固定16样本tiny BERT LoRA FP32，已有full4基线后新做stop2与不同PID的fresh resume4；两阶段两rank fallback均前后0、6个可训练参数实际更新。每rank checkpoint step2共40个参数/optimizer/scheduler字段恢复最大差0、Python/NumPy/CPU/CUDA四类RNG匹配，样本游标衔接；同源码full与resume最终参数最大差0。共同梯度/参数/优化器/调度数值字段与此前审定PT对照allclose；resume轨迹PT另有两个 `param_groups.initial_lr`、`scheduler._get_lr_called_within_step`、`scheduler.verbose` 四字段，不能声称state_dict字段完全一致。首步前与stop首步后state显式sync属于诊断干预。独立终审 `_state/release-day23/20261029-main-d900-bert-resume-successor-r1-hold/independent-terminal-audit.json` SHA `9154144a522dfd39914f59e2a8cc089d1f7787cfa6657b8e630c1f2d97abbab1`。旧CPU BF16 strict FAIL与首次full4超时保留；不外推17样本尾批、AMP、四rank或性能。


## 2026-10-29 d900 CPU优化器专项

主树 `d90073…` 绑定已审CPU core、按源码真实mtime隔离派生的stamp并显式 `use_mkl=0`：scope/异常2/2通过，restore 10/11，唯一失败为原BF16 AdamW恢复后下一步moment strict比对，最大绝对差 `6.676465e-05`、最大相对差 `0.00093548`；旧阈值与FAIL未修改。两个阶段fallback0，实际锁及可见holder空；审计 `_state/release-day23/20261029-main-d900-cpu-optimizer-focused-r1-hold/independent-terminal-audit.json` SHA `e9b7f0394c0b41eb17e9f97a24bff4e11b7154e7365c8dda815e23bec82024f9`。该数值差异长期训练影响仍未知，不能仅因数值小认定无害；旧GPU恢复11/11和新两rank full4功能证据单列。


## 2026-10-29 主树新fresh optimizer真实两rank训练通过

主树 `d90073…`、真实新CUDA core/NCCL、两rank tiny BERT LoRA FP32、16样本的full4后继自然rc0，每rank 4更新/8 microbatch，全部6个可训练参数实际变化；两rank UUID、样本序列、有限梯度及fallback前后0核实。每rank首步前及首步后的12个AdamW状态经显式sync物化为对应CUDA上的FP32 Torch Tensor，scope恢复一致；与已审旧PyTorch full4固定对照的梯度/参数/优化器状态最大绝差分别 `1.49e-8`/`1.041e-7`/`3.73e-9`，均在诊断容差内，loss最大差 `2.98e-8`。独立终审 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-successor-r1/independent-terminal-audit.json` SHA `615b1453353082158e74fd6e23741031e9c65391b69ef7c5f68fd6a83f79fefb`。原首个full4冷算子JIT 270秒超时已单独保留，新用途独立预算后继利用已产生缓存，约48.6秒完成；本次状态sync是诊断干预，四步短程不能证明epoch学习曲线、d900续训/尾批/四rank/AMP/性能。旧源码BERT/Qwen/T5历史曲线和对应保存恢复证据保留其身份，不能重标。


## 2026-10-29 主树fresh AdamW真实BERT首步前检查通过，训练超时

主树 `d90073…` 新冷编CUDA缓存先单进程NCCL warm rc0；后继两rank 16样本tiny BERT LoRA FP32 full4运行，按270秒launcher上限在首步算子JIT阶段rc124，首步、梯度、参数更新及PT对照均未完成，不能据此判断训练可用或失败。两rank均实际初始化NCCL，各自6可训练参数与12状态槽在首步前经显式sync检查为Torch Tensor/FP32和正确CUDA设备、构造前后scope一致；fallback仅训练前0，sync为诊断干预。相对较早双卡后继前快照新增78个jit .so，其mtime晚于本模型启动（该旧基线亦早于后续warm，归因按时间界定）。进程组结束、锁与可见holder空，部分外部FD不可读。超时独立终审 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-plan-r1-hold/terminal-timeout-audit.json` SHA `13447e7a50e6d3c09ac2e7ba8f82dd1f3133b08d3371c0c9d45bfdd6181b2847`；正在评估以明确新用途和预算完成实际更新，原超时不覆盖。


## 2026-10-29 主树optimizer设备专项通过，模型验证待执行

主树 `d90073…` 绑定同源码真实CUDA冷编产品，单卡placement 61/61、原18例AdamW/SGD混合设备冲突矩阵state无错放、原FP16/BF16失败节点2/2、主树restore 11/11；双卡后继2/2，driver核对cuda:0 `GPU-2be51…`、cuda:1 `GPU-b6f598…`，功能阶段均fallback0。首次双卡被控制器残留单卡数量断言在收集前阻断，原失败保留；后继仅运行尚未收集的双卡节点。独立终审 `_state/release-day23/20261029-main-d900-cuda-placement-dual-successor-r1-hold/independent-terminal-audit.json` SHA `b2affb58a8d89a65a925fbd7f67a658688090c9f7e88277ea5eab35a9d4171ec`；初始四阶段审计 SHA `9f23352e9240079bc19115740e80d9eb3ddb31e4af8c2014874e9939d081e6c7`。旧d08 CPU 2/2不能标作新源码CPU通过；旧BF16 strict差异保留。显式参数状态创建的前端类型改变可能切换算术路径，实际模型训练及更新效果尚待验证。

已准备两rank tiny BERT LoRA FP32、fresh AdamW实际状态位置和训练更新的有界方案 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-plan-r1-hold/plan.json` SHA `b5b75f81c9e5f738a0ceecd451e632bb7e376825b137e2f4107a363b4579cf84`；逐参数显式同步状态属于诊断干预，不能冒称无干预曲线。现准备新产品/NCCL绑定和必要预热，尚无该身份训练结果。


## 2026-10-29 主树新CUDA冷编与候选打包通过

主树 `d90073…` 在新独立缓存单次真实CUDA cold自然rc0；独立审计确认231 TU（194源码/37生成）、817源码签名、core/utils/stamp与libcuda_extern实际产品及新缓存RUNPATH、导入后 `use_cuda=1`，锁与可见holder终态空；无张量功能测试。审计 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/independent-terminal-audit.json` SHA `7f9081857286dd547819a29f2dcd4cda6d8ed054ebe1a4992bd690cc16c157d7`。后续须绑定该实际产品运行设备矩阵、双卡及恢复，不能以构建通过替代。

候选r9隔离源码副本的root/compat两套wheel及sdist均实际构建rc0；新增state placement helper、storage header、optimizer restore和DDP hooks逐字节在包内，wheel全部RECORD路径、hash、大小通过。终审 `_state/release-day23/20261029-headonly-r9-packaging-isolated-r1/terminal-audit.json` SHA `a67ae633c3c22673c7b450739fdbb968f26236118486040c7c161ca70f5cd371`；未测安装、模型及从sdist重建wheel。任务候选中与暂停前相关dirty重合的三处stop_grad及九行iadd逻辑已单独记录，原始字节不齐的路径不能声称已穷尽证明没有旧dirty；审阅 `_state/release-day23/20261029-headonly-r9-commit-readonly-audit-r1/review.json` SHA `b5b3dbf0ee88d4aea22011917bd6237e466ce8d0ba8f310e29ba0342df614a97`，尚未提交或暂存。


## 2026-10-29 候选r9定向通过、主树真实CUDA冷编运行中

HEAD-only候选r9真实身份为r2 CUDA core + r9九个Python生产模块/测试overlay；单次九阶段151/151通过：placement单卡61、双卡2、optimizer restore14、alias22、write1、cat14、RMS2、pow1、factory34。逐阶段真实UUID/dispatch/fallback0、源码产品、锁和cache已审，factory绑定原PyTorch oracle；终审 `_state/release-day23/20261029-headonly-r9-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `c65204378ab7d8f18dbbd71da754f38ccc0de74bdb7fea184ab9e7e69cd494d3`。修正测试辅助代码后的CPU frontend恢复2/2亦通过，原r8 3pass/2fail及历史BF16 strict FAIL保留；混合overlay不是完整r9冷编或真实模型训练。候选helper与主树 `d90073…` 逐字节相同，其他身份不同，不能直接转移主树结论。

主树独立真实CUDA cold已单次启动，controller PID 1916153、child PID/PGID 1916169/start_ticks 188769085，cap900、无自动重试；新缓存及产物路径 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/attempt-01-cuda/`。入场审阅 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/release-admission-review.json` SHA `2c46cc5b93652beeca2ea7e0f40921adcc55a1db45a183faaf6191042fa1c4e6`。接续时先核PID/start、receipt、锁、产品与源码；尚无主树新构建或设备回归通过结论，其他用户不可读FD不作绝对断言。


## 2026-10-29 optimizer add_group实际错设备及后继修复待验

HEAD-only r7 的 r2 CUDA core + r7九Python模块实际回归：原optimizer restore 14/14通过（含此前FP16/BF16两失败）；后续55节点24通过/31失败，双卡及后续6阶段未启动。独立终审 `_state/release-day23/20261029-headonly-r7-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `c75ee23fff9874ee2a0fa7c7c6a423a2b6a1a4df165b46c8368e38b2c2727467`。其中26项源于测试工厂只设置Python ContextVar、原生参数实际为FollowRuntime；独立探针在真实CUDA下确认其余五种Torch optimizer的新增CPU参数组，其CPU参数实际CPU、而新增state实际CUDA（native Var backend=-1、dispatch CUDA、sync后location=device），非仅显示差异。探针 `_state/release-day23/20261029-r7-cuda-add-group-location-probe-r1/` receipt SHA `45e14b2b91f22217cf6832d2700b02fdc3e221346a27631e23378f422255aad3`；两层placement token嵌套异常恢复经探针验证。显式CPU native Var的 `.device` 可能显示runtime CUDA，需按backend/dispatch/实际location判断，Torch state另测公开device与类型。

主树新源码 `d900732c0398909488f9525be5a11d29371d538c7af8f9bed3dd9fe8dd72857f` 将显式参数的Python placement、native TLS、frontend result type同时覆盖并逆序恢复，FollowRuntime保留原路径，修正原生测试工厂并增加类型/公开state/异常测试。相应HEAD-only r8 candidate fingerprint `4067dead1617683710defaf44b8e9901e5859f4548a5f98d76bbc0a36cc99f9d`，helper与主树逐字节一致；两者实际回归尚未运行，frontend类型变化的数值影响待验证。主树独立CUDA冷编包 `_state/release-day23/20261029-optimizer-state-both-placement-genuine-cuda-cold-r4-hold/` 仍HOLD，待r8终态和串行资源释放后放行；旧stamp、旧错误和历史构建全部保留。


## 2026-10-29 optimizer placement CPU通过、CUDA扩展stamp阻断

主树 `d08f9e…` 使用独立派生core stamp、`use_mkl=0` 完成CPU scope/异常恢复2/2；派生仅更新两个内容已按SHA核实不变的C++文件的真实mtime，核心及完整ingredients运行时检查通过，非新冷编。pytest部署14个shim文件被保守缓存检查拦停，逐项比对均与受审源码一致，无JIT源码或二进制变化；CPU终审 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r3-hold/independent-terminal-audit.json` SHA `27af5f3e263392eb9abfc33343d3bbd8fc56d87e6e0a734d2cf53b3be6cd6a89`。后续GPU尚未收集测试，独立 `libcuda_extern` stamp含相同mtime差异、无core那样的隔离路径覆盖，原产品和stamp与冷编快照哈希一致但当前完整运行时ingredients未验证；终审 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r4-gpu-hold/independent-terminal-audit.json` SHA `a2e0564a4599e3acabfed57252313a497975410fe734636600372645e47f0db5`。原签名和各次失败均保留，正在评估可审计的独立缓存，不可将GPU回归写成通过。

FollowRuntime混用原生参数和torch optimizer存在ambient placement冲突的代码路径，但目前无实际故障复现；直接固定设备会改变动态语义，不能以此假设修复。只读审计 `_state/release-day23/20261029-d08-followruntime-optimizer-readonly-review-r1/audit.json` SHA `663a7daff5a5b7c4351089ee24374a2d9abcf1936472e406251d8537ca6a00af`。


## 2026-10-29 optimizer placement回归受stamp检查阻断

主树 `d08f9e…` 单次串行回归在CPU测试收集前退出，CUDA/18例矩阵/双卡均未启动，不能记为通过。`JITTOR_NO_BUILD=1` 检测到core stamp过期；只读比对显示CPU和CUDA原stamp均仅有恢复后的 `py_tensor_frontend.h/.cc` 两个源码mtime差异，其大小和SHA与原冷编及当前清单一致。原生文件和已审产品未改，失败进程已终止、缓存锁与可见holder为空，其他用户不可读FD不作全机断言。独立失败审计 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r1-hold/independent-terminal-audit.json` SHA `a03d25bbc2050eb55087748c36cae2b97c5e3a7d4e386c43d0a4d1ebf0e75b2a`。后续仅在核实完整编译输入、生成器、工具链及产物等价后，评估独立派生stamp；原stamp及失败记录不覆盖，尚无测试通过结论。


## 2026-10-29 optimizer显式参数state placement修复待验

矩阵证实constructor ambient放置错误后，主树引入optimizer私有scope，针对已有显式placement的每参数临时覆盖上下文并finally恢复，覆盖zeros/adaptive FP32 cast及SGD contiguous；FollowRuntime参数直接走原路径，ambient冲突仍未解决。新指纹 `d08f9ec25b8669fc9d171c1e41274d4061f605e9e79c8a354090d412b64f80ff`，相对旧版仅4 Python/测试路径，曾尝试的native clear API已撤去且两个C++文件与事前备份逐字节一致；现有native产品能否复用需重审。静态review `_state/release-day23/20261029-optimizer-per-parameter-placement-fix-r1/static-review.json` SHA `03fa7cb604c82dc7202a7cae5c99728ea435039d52c363e218d2c39a554589c4`；新增CPU/CUDA混组、显式冲突、异常嵌套和公共factory语义测试，尚未运行，原30个state错设备证据保留。

## 2026-10-29 当前主树optimizer初始state实际设备缺陷

主树 `cbfa19…` 真实CUDA测试 FP16/BF16 AdamW新建moment公开device为CPU（原测试在native dispatch前失败），独立机制探针进一步证实同步后真实CPU location；手工zero/cast在CUDA，helper trace定位constructor期间 `tensor_frontend` 默认CPU placement覆盖了CUDA参数的 `zeros_like`，add_param_group上下文不同。18例AdamW/SGD×FP32/FP16/BF16×默认/显式CPU/显式CUDA，含CPU/CUDA混组，共30个初始参数state与参数设备不一致；新增组均匹配、数值有限、fallback0，异常路径未测。矩阵独立终审 `_state/release-day23/20261029-main-cbfa-optimizer-placement-matrix-r1-hold/independent-terminal-audit.json` SHA `1283b8d70c9baceac2f2df47e495f810c7b0094207d397604d7f4f6ac88e2259`；辅助helper trace SHA `09cd32f8a55e0169a999edb9a7c8f6e32ae106679531f8644c0c4d639fbaf953`。这是功能设备缺陷，非仅显示错误或末位数值差；正在设计每参数内部状态独立scope，同时保留公开工厂显式device与FollowRuntime语义，尚未修复。

## 2026-10-29 候选CUDA发现moment设备显示差异

HEAD-only候选实际 r2 CUDA core + 6个r6生产Python模块/测试：replay 6/6通过，optimizer 12/14，新增FP16与BF16 moment初始状态公开 `device=cpu`、测试期望cuda:0；断言发生在native dispatch检查前，不能断言物理存储在CPU。独立终审 `_state/release-day23/20261029-headonly-r6-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `fd98594a1d069e3565cb4e180db9a17d2b6e7d038afc0dbb37729cbaff732f7b`，fallback0，余alias/write/cat/RMS/pow/factory未执行。原CPU BF16严格moment数值失败保留。当前main有相似半精度状态构造，正准备独立真实CUDA专项同时测公开device、dispatch/placement，不能预判是否受影响。

## 2026-10-29 HEAD-only候选CUDA cold通过，定向回归运行中

候选r2独立CUDA cold自然rc0、真实UUID及231编译单元核验，194源码/8生成器与r6逐项相同；独立审计 `_state/release-day23/20261029-headonly-r2-cuda-cold-independent-audit-r1/terminal-audit.json` SHA `010575095d68c3ec8638655c0856bc554ece59c98c0f716f3694fe10a4ff0579`。后继实际为r2 CUDA core+六个r6 Python模块/测试，单次八阶段replay/optimizer/alias/write/cat/RMS/pow/factory；controller PID/PGID 1868906/start_ticks 188472844、首worker PID/PGID 1868907/start_ticks 188472935，路径 `_state/release-day23/20261029-headonly-r6-cuda-targeted-overlay-r1-hold/`；锁/缓存/资源已核，GPU有外部负载，非性能测量。恢复先核PID/start/receipt/锁，失败停止不重试；目前无GPU测试/训练通过结论，原CPU BF16严格失败及结构基线问题保留。

## 2026-10-29 HEAD-only候选独立CUDA cold运行中

候选r6需要真实CUDA而非沿用main产品；以已审r2 native源码与CPU cold前置、r6 native 190编译源/777文件/8生成器逐项等价，建立独立CUDA cache，r6仅后续Python/测试overlay。单次cap900，controller PID/PGID 1863596/start_ticks 188427923，worker PID/PGID 1863659/start_ticks 188427981，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，入场约76827MiB空闲且外部利用率59%，只作正确性；目录 `_state/release-day23/20261029-headonly-r2-cuda-cold-r6-overlay-r1-hold/`。恢复先核PID/start/receipt/锁，不重复；cold与后续定向测试尚无结论，不能把main GPU训练转移到候选。

## 2026-10-29 HEAD-only r6事务与DDP CPU overlay通过

任务独立提交候选r6完整patch SHA `6e7be1ab21e98642fd19e9a0b0f096169787ec77ded4afe019c96afd8be2c6a7`，将主树hook事务修复最小移植，HEAD transaction/context/fidelity依赖字节一致；r5及其故障原件保留。单次CPU测试真实身份为r2 framework/native core + r6六个Python生产模块/测试、use_mkl=0，事务8/8、hook16/16、owner20/20、原隔离故障注入2/2通过，CPU dispatch/fallback0；独立审计 `_state/release-day23/20261029-headonly-r6-hook-transaction-cpu-overlay-terminal-audit-r1/terminal-audit.json` SHA `16de455dd8df575f7fa76566b0b618c47a759c5a11e26323f426d4e0c8703fbc`。此为混合身份、不证明r6 native cold或完整shim安装故障，r4 BF16 moment严格数值失败、原HEAD types.py 802行结构失败、候选CUDA/多rank/训练未验证均保留；真实index未改，候选未提交。

## 2026-10-29 BERT热步间NVML间隔诊断（共享负载）

新主树 `cbfa19…` 的单次PT/JT各12步QNNQ交错诊断、每端6 query前置/6无query，模型/权重/config一致、实际197非空梯度和199参数有限、fallback0、loss最大差2.38e-7。PT两组更新median 87.33/85.33ms，JT 235.82/209.18ms；JT组内差+23.30/+44.73/-4.70ms波动显著，查询本身约105–110ms不计入更新。两阶段目标卡持续有外来PID 1849785、只有5/7离散负载采样且无等长sleep对照，不能区分查询、闲置间隔或共享负载的影响，不作为新source正式性能。独立终审 `_state/release-day23/20261029-main-bert-b8s256-nvml-gap-cbfa-r1/terminal-independent-audit.json` SHA `33aa1b06961b2d764d7ec773cc5ebfaf604483e42435d53fc4a85c041ac803fd`；原 `31857f…` 正式B8/S256 1.735倍结论及原负载监测限制继续保留。

## 2026-10-29 最新主树两rank Trainer full4通过

事务修复后 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7` 的单次两rank HF Trainer tiny BERT LoRA FP32 full4自然rc0，两个指定GPU真实source/core/NCCL、fallback0，均8 microbatch/4更新、16样本，副本参数一致；复用旧独立PT full，gradient/parameter/optimizer最大差1.49e-8/1.04e-7/3.73e-9，loss最大差2.98e-8（诊断）。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-full-transaction-r2/terminal-independent-audit.json` SHA `82d04f818d1283ea5a9bf6a7635bb2207fd158096d87a9d79c69fac28e1ed616`。首次启动因旧helper hash在任何attempt/模型前被拦截，原失败保留；此次不含新source的resume或四rank，旧 `31857f…` 两/四rank续训结果按其身份保留。

## 2026-10-29 hook事务修复CPU回归通过

新主树 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7` 的五阶段单次回归：新增rollback故障注入8/8、hook16/16、Torch owner20/20、既有事务/边界27/27、结构24/24均自然rc0，CPU dispatch和fallback0；源码/产品与四份冻结文档一致、进程组及可见锁/holder空。独立终审 `_state/release-day23/20261029-main-ddp-comm-hook-transaction-cpu-focused-r1-hold/independent-terminal-audit.json` SHA `c3e238a4d1b35fbc6a852460894de5f9e15f659dd458e389b05ce50a9f90408c`。覆盖新建/预存namespace三个故障点及重复安装；原r5隔离事务2失败保留，CPU mock不等于真实多rank，之前2/4rank Trainer与性能结果继续按 `31857f…` 身份报告。

## 2026-10-29 hook事务故障注入失败与主树修复

HEAD-only r5 混合CPU测试 hook16/16、owner20/20通过；原HEAD `types.py` 802行超过800的结构失败与两项受控故障注入失败均保留。注入在已有hook模块上证明InstallTransaction.rollback后属性未恢复，限独立registry而非完整shim安装；独立终审 `_state/release-day23/20261029-headonly-r5-hook-cpu-overlay-terminal-audit-r1/terminal-audit.json` SHA `a335a7965fabbae23fa985874b56a4240c634a1d25b81674c32fe9071f10790b`。主树同一helper已改用事务属性和模块发布账本，新增新建/预存及多故障点回归，旧文件备份 `_state/release-day23/20261029-ddp-comm-hook-transaction-fix-r1/`；新指纹 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7`，相对 `31857f…` 仅helper及新测试变动、native源码不变。尚未运行真实回归，旧训练/性能结果保留各自身份；不要预报修复通过。

## 2026-10-29 BERT B8/S256性能归因诊断

在相同main `31857f…`、权重、B8/S256默认attention/optimizer配置及独立性能cache上，PT/JT各3步profiler自然rc0、JT fallback0；两端实际均为BertSdpaSelfAttention/sdpa。PT memory-efficient attention前/反各36次事件总3.90/8.01ms，JT mea前/反各36次总9.44/21.05ms，约6.20ms/步事件差提示attention值得查，但不足以证明无profiler正式64.12ms/步差的根因。JT三步host wall310.10ms/device busy240.71ms/device wait176.83ms、2043 kernel；PT kernel区间并集238.15ms/2994 kernel，两个profiler捕获口径不同，不得直接相加或作benchmark。JT trace未截断，PT无独立截断指标；各侧只有5个离散负载样本、无可见外来PID。独立审计 `_state/release-day23/20261029-main-bert-b8s256-attribution-r1/terminal-independent-audit-v2.json` SHA `eaccfb0460905b5ca4e9bcea0f0aa2864ce4a54515ca5bb2b3868b24f4f34529`，原正式1.735倍结果及采样限制保留。

## 2026-10-29 主树BERT B8/S256 FP32性能三轮完成

当前主树 `31857f…` 的独立性能cache、60秒readiness 13/13合格后，PT/JT交替三轮各10次同步热更新自然rc0；PT轮median 87.36/86.98/87.47 ms，JT 152.45/154.24/146.11 ms，合并median 87.30 vs 151.42 ms、JT/PT=1.735，热loss最大差2.98e-7、JT fallback0。独立3步内存测量allocator peak allocated PT/JT约3.041/3.033 GB、reserved约3.282/3.329 GB，逐热步NVML采样峰3638/3650 MiB；不构成近容量证据。独立终审 `_state/release-day23/20261029-main-bert-b8s256-performance-r1-hold/terminal-independent-audit.json` SHA `9ac32552295e89f445465fe37f10341d48f547376071054a7872028e932ad940`。cold编译期间253次负载采样一次nvidia-smi超时，其余有记录采样未见目标卡外来PID，离散采样不能排除间隙干扰；该单格稳定偏慢是需处理的性能差距，完整Day4–5及near-capacity仍未验收，历史r3诊断与readiness失败保留。

## 2026-10-29 主树BERT B8/S256配对性能实验运行中

目标GPU空闲状态相对前次60秒失败明显变化后，新用途一次60秒readiness的13/13点连续合格，单次放行当前主树 `31857f…` 的独立性能缓存实验：FP32 BERT B8/S256，PT/JT交替三轮，每轮实际首更新、4次预热、10次同步热样本及独立allocator/NVML采样，非近容量验收。controller PID/PGID 1831102，独立JT cache `_state/release-day23/20261029-main-bert-b8s256-performance-r1-hold/performance-cohort/runtime/jittor/jittor-home`，日志在同目录。首次JIT串行，阶段失败即停不重试；恢复先核PID/start/receipt及锁。当前无性能结论，历史r3共享负载诊断与前次readiness失败保留。

## 2026-10-29 四rank HF Trainer禁P2P恢复通过

当前主树 `31857f…`，tiny BERT+LoRA FP32、16样本、4rank、batch2/G2，在PT与JT均明确 `NCCL_P2P_DISABLE=1` 下，full4/stop2/fresh-resume4六阶段及PT连续性/最终分析自然rc0；24个rank执行设备/实际产品与fallback0已核，六阶段NCCL各14条channel均SHM、0 P2P。checkpoint2位于epoch2，四rank RNG齐全，各端各rank自身40项checkpoint状态恢复max_abs0、full/resume最终参数max_abs0。跨框架共有字段最大差梯度2.98e-8、参数1.0966e-7、optimizer 2.794e-9、loss 2.98e-8仅作诊断；resume轨迹PT多两项initial_lr与scheduler `_get_lr_called_within_step`/`verbose`，不声称跨框架全部状态一致。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-no-p2p-r1/terminal-independent-audit.json` SHA `40d432deb41cacc0178f8243c5f828c6dea382b9f7ab5cc9ae4cde3df6678b88`，进程/可见锁/FD终态清楚、拒读PID未知；终审两次只读脚本因字段差异/字符串finite中止已记录，原运行未重做。此限禁P2P/SHM，默认P2P PT full和微诊断超时、17样本尾批恢复、AMP/长期质量/性能尚未通过。

## 2026-10-29 四rank禁P2P Trainer后继运行中

同四卡微型NCCL禁P2P诊断通过后，新用途对PT与JT均明确 `NCCL_P2P_DISABLE=1`，其余16样本/epoch边界checkpoint协议不变；PT full/stop/fresh-resume及连续性先过才进JT。controller PID/PGID 1823071，binding SHA `2fafe9325667d2f6a81fc2f40c0528349dac48de1ce96bd9f27d91b7cb3fea73`，目录 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-no-p2p-r1/`；单次有界、失败停止不自动重试，恢复先核PID/start/receipt/锁。该适配配置不能称默认P2P通过；原默认PT full超时及17样本尾批恢复失败均保留，目前无四rank训练结论。

## 2026-10-29 四rank NCCL默认与禁P2P环境诊断

同四卡、同PT NCCL 2.21.5库，默认配置的单次简单all_reduce 60秒超时，四rank均记录进入collective前、无数值结果，日志混合P2P/CUMEM与SHM；独立 `NCCL_P2P_DISABLE=1` 单次自然rc0，四rank均得10.0，日志通道均为SHM。独立终审 `_state/release-day23/20261029-pt-4rank-nccl-diagnostic-r1/terminal-audit.json` SHA `fa9a71c2ce3033c3134722189bb4cf9ed47ab6f8241b5cc232dc30f41e6d18f2`。此支持传输路径相关环境问题，但不确定具体根因/故障链路；不能将禁P2P诊断称默认P2P或Trainer通过。正在准备独立、明确禁P2P的四rank PT/JT Trainer后继，原默认full超时保留。

## 2026-10-29 四rank HF实验PT full超时

四rank新用途的初始权重与NCCL四rank初始化完成，PT full单次300秒超时停止；日志显示部分通道P2P/CUMEM、部分SHM/direct/direct，卡住根因未知，不能归因Jittor。PT stop/resume及JT所有阶段均未启动，无重试；controller/launcher/rank进程组退出，锁及可见FD空，拒读PID未知。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-r1-hold/terminal-failure-audit.json` SHA `297674242dc9ae2ba6b25eaababce195d09b21205dc37098c54326f9e299143d`；控制器TWO_RANK标签瑕疵保留，实际证据四rank。四rank公开Trainer与性能/显存仍未验收，两rank16样本续训通过及17样本尾批恢复失败分别保留。

## 2026-10-29 四rank HF Trainer epoch边界实验运行中

当前主树 `31857f…` 新用途：tiny BERT+LoRA FP32、16样本、四rank/batch2/累积2，每epoch1次更新，checkpoint2要求epoch2.0；PT full/stop/fresh-resume及连续性先过才运行JT对应三阶段，每阶段cap300、整体cap1800，失败即停不自动重试。四UUID依序 `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`、`GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`、`GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`、`GPU-b6f59824-d1a3-cd6f-8100-9149728fc8eb`；余量约81/72/73/76 GiB，外来负载与不可读PID保留。controller PID/PGID 1817682，binding SHA `de3bf82b539cf4ec70e24c312e72d39a235c558a28fd2133336dcc1376e4cf19`，路径 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-r1-hold/`。控制器成功状态字符串沿用TWO_RANK属于标签瑕疵，必须以实际四进程与设备证据为准；当前尚无四rank结果，恢复先核PID/start/receipt/锁，不重复。旧17样本尾批恢复仍未通过。

## 2026-10-29 noneditable compat wheel混合布局入口通过

旧 `2dd2b3…` 身份compat wheel安装到隔离target后，184 RECORD记录及optimizer_api/optimizer_restore实际路径/哈希核对通过，wheel入口spec明确，动态torch namespace无 `__file__` 记null；使用当前 `31857f…` 主树CPU core及显式扩展的 `jittor.__path__`，公开Optimizer.load_state_dict的零参数零状态调用实际进入wheel恢复实现，将lr 0.1恢复为0.2，fallback相关张量未涉及。独立终审 `_state/release-day23/20261029-compat-wheel-noneditable-import-check-r3/attempt-01/independent-terminal-audit.json` SHA `5ad909581b409c6adaf2a0dfa38943d4f2a05f9a14da31401b27a9a3cb1bbe23`。早前RECORD相对脚本路径和namespace `__file__` 两次观察器失败均保留，lock内容变化单列；此混合布局不证明最新wheel或完整core+compat干净安装，也不证明张量状态恢复。

## 2026-10-29 HEAD-only候选 leaf/factory/numerics 后继通过

候选r2→r3仅更正旧 `torch.ones` 非leaf断言：独立PT 2.6实测为leaf、grad_fn None，运算输出非leaf；旧失败/误选CUDA skip保留。后继实际身份为r2生产源码与CPU core + r3测试文件、显式use_mkl=0，leaf 1/1、factory17/17、numerics18/18通过、fallback0；独立审计 `_state/release-day23/20261029-headonly-r3-cpu-test-overlay-terminal-audit-r1/terminal-audit.json` SHA `ec8a7c874d59052cfb3c6bf393d02de1b36707ba90237eb94557a61db4998407`。先前候选BF16 AdamW moment dtype失败、默认oneDNN缺产品与CUDA/multirank未验证继续保留；不能仅据这些CPU结果提交或转移主树训练证据。

## 2026-10-29 主树两rank HF Trainer epoch边界续训通过

新源码 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 的tiny BERT+LoRA FP32、16样本、两rank、G=2：Jittor full4/stop2/fresh-resume4与复用的独立PT对照均完成，checkpoint2 epoch1的40项模型/optimizer/scheduler状态与Python/NumPy/CPU/CUDA RNG匹配，16样本ID及顺序符合full控制；两rank实际CUDA UUID/参数/梯度ordinal正确、fallback0、6可训练tensor实际更新。JT full/resume最终参数max_abs0且副本一致；对已核PT最终参数max_abs 1.04075e-7、梯度max_abs 1.49e-8、scaled microbatch loss最大差2.98e-8（诊断）。NCCL实际SHM/direct/direct，进程组/可见锁及FD空，拒读PID未知。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-jt-hook-r1-hold/terminal-independent-audit.json` SHA `78e0bef8c0931f33a88dc58e174579fcaf12fff8006ff52063bda4206c645ed2`。这不覆盖旧17样本尾批恢复、4rank/AMP、性能或长期任务质量；原缺hook失败及17样本PT恢复/观测失败均保留。

## 2026-10-29 HF Trainer 16样本修复后JT两rank运行中

新主树 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 经hook/native/Torch/structure CPU四阶段89项通过；同一16样本PT full/stop/resume及41项权重/报告/checkpoint资产已重新只读核实，不重跑PT。JT-only单次full/stop/fresh-resume后继已启动：controller PID/PGID 1808060、full launcher PID/PGID 1808065、rank0 PID/PGID 1808067、rank1 PID/PGID 1808068，两rank框架/锁/NCCL身份前置通过；binding SHA `850d2b7cee25ec1fbf822c9c21994969fc5e67051d1c57b584005b1c1086fb23`，目录以 `_state/release-day23/20261029-hf-trainer-ddp-epoch16` 后继记录为准。恢复先核PID/start/receipt/锁，失败停止不自动重试；原JT缺模块与17样本失败保留，当前无JT训练结果。

## 2026-10-29 默认DDP comm-hook导入修复CPU回归通过

针对HF两rank默认NO hook导入失败，新源码 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 提供Accelerate无条件读取的hook模块和符号，实际自定义/压缩/PowerSGD hook或wrapper执行明确抛NotImplementedError并标注UNIMPLEMENTED，默认NO不注册；Accelerate对NO hook+非NO wrapper的既有静默忽略不宣称支持。CPU单次四阶段 hook16/16、native DDP29/29、Torch owner20/20、完整结构24/24通过，fallback0，冻结文档一致；独立审计 `_state/release-day23/20261029-main-ddp-comm-hook-cpu-focused-r1-hold/independent-terminal-audit.json` SHA `df976f0ee009ab3966615d07d0d2c31bc8d11b396b646a52ddbbcf0ee009b73e`。旧16样本Jittor导入失败保留；CPU mock不代表真实多rank。新模块和既有dirty备份 `_state/release-day23/20261029-ddp-comm-hook-surface-fix-r1/`。

## 2026-10-29 HF两rank 16样本对照通过、Jittor缺comm-hook导入

16样本epoch边界协议PT两rank full4/stop2/fresh-resume4及连续性均通过：两rank 40项checkpoint状态、四类RNG、样本顺序、最终参数匹配。PT resume rank1实际有额外GPU context，但当前ordinal/参数与目标UUID一致；不能反推原17样本失败的未记录列表。Jittor两rank在HF Trainer→Accelerate.prepare_model的默认DDP kwargs导入 `torch.distributed.algorithms.ddp_comm_hooks` 时均ModuleNotFoundError，未开始更新；fallback仅前值0，异常后未记录。独立审计 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-r1-hold/terminal-independent-audit.json` SHA `c116512c2b88c9703c2b959d64bd016eb5235d521decc33d757a51e300d49547`；旧17样本失败不覆盖。准备默认NO hook的导入兼容，实际压缩或PowerSGD hook必须显式不支持；16样本并不覆盖尾批恢复。

## 2026-10-29 HF两rank 16样本epoch边界新实验运行中

另建16样本、G=2、checkpoint2位于epoch边界的独立协议；PT/JT各full4、stop2、fresh-resume4，先核PT自身样本/状态/RNG连续性才放行JT。原17样本尾批fresh-resume失败保留，16样本不声称尾批恢复覆盖。main `a2a524d5…`、既有CUDA/NCCL产品及锁/资源已核；release SHA `cdcd1a436156ac790035e3d23bfcb52e858b1df745f3bbffc14690e502857b91`、binding SHA `0484a11c1d6891dfd8508ed345b0230af14fa390cc62e51db79d149791658793`。controller PID/PGID 1799310，PT full launcher 1799317自然rc0；当前PT stop launcher PID 1799402、rank 1799410/1799411，目录 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-r1-hold/`。恢复先核PID/start/receipt/锁，不重复；当前无PT完整恢复或JT通过结论。

## 2026-10-29 两rank HF Trainer原17样本实验失败终审

独立PT full4/stop2完成；fresh resume 的rank1因设备观察器仅允许单个NVML UUID而退出，异常未记录完整UUID列表和恢复后参数ordinal，NCCL cudaDev1/nvmlDev2/busId53000与目标PCI一致但不能复原现场或断言原因。rank0 checkpoint step2模型/optimizer/scheduler 40/40及四组RNG匹配；同时full只有7 microbatch，stop4+resume5为9，样本与sync序列不一致，原因未知。原独立终审 `_state/release-day23/20261029-hf-trainer-ddp-resume-r1-hold/original-failure-terminal-audit.json` SHA `16208f0c09037f7197b5cf70cca02938622a62fa31f789443ceac561461992cd`；所有进程组和可见锁/FD空，拒读PID未知。Jittor阶段未启动，不能声称尾批恢复通过；准备单独16样本整除累积、epoch边界checkpoint的新协议，保留原失败且明确不覆盖尾批恢复。

## 2026-10-29 主树两rank HF Trainer新进程恢复实验运行中

以新主树 `a2a524d5…` 为Jittor身份，使用独立两卡及匹配PyTorch、固定tiny BERT+LoRA、G=2的单次新用途：两端各full4、stop2、fresh-resume4，逐rank报告/真实设备/恢复与同步，单阶段cap300、总cap1800；按实际失败停止，不自动重试。release review SHA `d7693adf334c45a0b1c025c4a0bfb8b0256dec03628f66ae197d0dede01efe0c`、binding SHA `c10e4eca0e0e2f3bd2603ca1db4fb69c59c1a8d3482529386b7a5dc961598796`；controller PID/PGID 1793439，PT初始化权重PID 1793445 rc0（weights SHA `4e595646d39667958e2e88efb0bf6950e9baf2d82b8ece57b31e709431391cf3`），PT full launcher PID/PGID 1793464、rank0 1793584、rank1 1793585。目录 `_state/release-day23/20261029-hf-trainer-ddp-resume-r1-hold/`；恢复先核PID/start/receipt/锁，不重复；当前任何训练、恢复或对照尚未终态。

## 2026-10-29 HEAD-only候选CPU阶段结果

独立HEAD候选首次focused的native3 3/3、native DDP 17/17通过；Torch DDP因未预装oneDNN导致11节点失败，原rc1与审计 `dd0bb57238eaf912443429f04894ac51aa75d7263ffcd52b3d4e6d954b627059` 保留。明确 `use_mkl=0` 的新后继中 Torch DDP 20/20、replay 6/6通过，optimizer 10/11；BF16参数的exp_avg实际BF16而非测试要求FP32，属于候选状态dtype差异，与主树先前FP32 moment下一步数值末位差异并非同一失败。独立终审 `_state/release-day23/20261029-headonly-r2-cpu-no-mkl-focused-terminal-audit-r1/terminal-audit.json` SHA `3f63936d2eddfa21575365b071adcd87612363eaa27e555a245f1a5d6c71a255`；alias/factory/numerics因失败即停尚未执行。目标候选未达到提交验证，不把主树训练或默认oneDNN结果转移到候选。

## 2026-10-29 HEAD-only提交候选CPU cold通过，定向回归运行中

候选独立CPU cold自然rc0，227单元、native/core/utils/stamp及全快照校验通过，独立审计 `_state/release-day23/20261029-headonly-r2-cpu-cold-independent-audit-r1/terminal-audit.json` SHA `d4fe42c9446f7bf4746ac468f5a026fabba9763b6713d2cc2f8886281b00a036`；仅编译和导入身份，无张量或训练结论。候选CPU focused已单次启动，controller PID/PGID 1788240/start_ticks 187872989，首worker PID/PGID 1788242/start_ticks 187873025；首worker identity.command启动时为空，receipt单独存实际命令，不补造。路径 `_state/release-day23/20261029-headonly-r2-cpu-cold-focused-r1-hold/focused-attempt-01/`；失败即停不自动重试，恢复先核PID/receipt/锁。候选与main的既有dirty依赖不同，main实训不能转移；用户dirty与真实index未改。

## 2026-10-29 任务独立提交候选CPU cold运行中

为不夹带既有dirty，在非Git HEAD快照上构造任务补丁 `_state/release-day23/20261029-task-only-head-candidate-ddp-r2/candidate.patch` SHA `b536de4a831202a0ce45b135c7ce7210cf8a3e45ca960660afe800bdce3cf7bb`；主进程用隔离临时index将补丁应用HEAD并逐一核对2789文件/符号链接与快照一致，真实index未改，代码适配与既有dirty依赖仍须测试。候选完整清单指纹 `d55d62807b579f420260c457ecef71552054a009c6a1640f5b994b8c20609a9b`，CPU cold单次cap600：controller PID/PGID 1781640/start_ticks 187834786、worker PID/PGID 1781642/start_ticks 187834826，目录 `_state/release-day23/20261029-headonly-r2-cpu-cold-focused-r1-hold/`。恢复先核PID/start/receipt与锁，不重跑；cold仅构建身份，候选测试/训练尚无结论，不能转移主树证据或提前提交。

## 2026-10-29 当前主树性能窗口单次检查未就绪

`a2a524d5…` 主树完成真实两rank终审后，对GPU `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 新用途单次60秒离散readiness检查：13点全部不合格，free恒59939 MiB（原准入≥60000）、利用率28–31%（原准入≤10%），外来PID 1461480全程可见，最终连续合格0；自然 `WAIT_TIMEOUT_NO_RELEASE_WINDOW`，无benchmark/JIT或自动重试。独立终审 `_state/release-day23/20261029-main-day45-readiness-a2a-r1/terminal-audit.json` SHA `210310c0c4a40e0e19198b6eff3be99f86b69e89ba8e509bcb590f04694419e2`；采样离散且GPU/PID查询非原子，不能推断间隙状态。这是无正式窗口而非benchmark失败；历史r3倍率仅诊断，near-capacity未证明，Day4–5性能/显存仍待验收，不重置旧T5/Vision预算。

## 2026-10-29 主树真实两rank Accelerate DDP尾批通过

新源码 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465` 的两rank tiny FP32 Accelerate DDP后继自然rc0：两物理GPU UUID实际匹配，NCCL日志记录SHM/direct/direct，fallback0，17个样本ID全覆盖，累积sync标记false/true/true，参数实际更新且两rank最终参数max_abs=0；各rank对历史PT的254数值项、39精确项通过，最大绝对差1.4901161193847656e-8。独立终审 `_state/release-day23/20261029-ddp-owner-successor-r1-hold/rank-terminal-independent-audit.json` SHA `200495b3dbc2d6d2ffb3a001fdd995caa8dce25c88452ecf012973cd8317fcbc`，进程组和可见锁/FD已退出，拒读PID未知；旧主树两rank TypeError及CPU首轮失败保留。历史PT报告未记录fixture脚本hash及确切版本，匹配依据和限制保留；此结果不覆盖4rank、HF Trainer、DDP checkpoint新进程恢复、AMP或性能。

## 2026-10-29 DDP修复后两rank后继运行中

新源码 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465` 已有native 29/29、Torch 20/20 CPU终审；经旧NCCL产品、当前源码、共同锁及两卡资源复核，单次真实两rank后继已启动：controller PID 1770885、launcher PID/PGID 1770886、rank0 PID/PGID 1770888、rank1 PID/PGID 1770889；最终binding SHA `7334c1e6d7c2bc3bf753506e82b24d6d1a2e4742ea09df089fa8ec31433fd2c5`，目录 `_state/release-day23/20261029-ddp-owner-successor-r1-hold/`。原launcher LOCAL_RANK保持，旧两rank TypeError和CPU首轮失败保留；恢复先核PID/start/receipt/锁，不重复启动。新测试结果尚未终态。

## 2026-10-29 DDP owner及dtype修复CPU定向通过

native DDP计算梯度桶大小改为从native Var读取dtype，保留现有分组；新增native/Torch多dtype与桶边界测试。新源码指纹 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465`；CPU独立终审 native 29/29、Torch 20/20，覆盖先前失败的双owner用例和新增dtype节点，fallback0；审计 `_state/release-day23/20261029-main-ddp-owner-bucket-cpu-terminal-audit-r1/terminal-audit.json` SHA `7a9007d7dc95425305bde32abdb35a59d046e831ca96ea90c77960202ebdc1bb`。原先28/28与18/19失败、原两rank TypeError均保留；CPU mock collective不等于真实多卡，下一步重绑后单次测试真实两rank。

## 2026-10-29 DDP修复首轮CPU回归与打包检查

DDP源码 `5466a63e…` 下 native原测试28/28通过；Torch owner相关18/19通过，唯一失败是在构造native DDP包装Torch Linear时 `parameter.dtype` 为torch.dtype，无 `dsize()`；该失败发生在预期双owner拒绝前。两阶段fallback0，独立审计 `_state/release-day23/20261029-main-ddp-owner-cpu-focused-terminal-audit-r1/terminal-audit.json` SHA `427d53c55c13fa443c277df7a337682e3c02e80ca812c08fa2360da7e4bcc5bb`，原失败保留；准备使用native dtype描述符修复并复测。真实两rank仍未通过。

新optimizer_restore模块的隔离sdist/wheel已实际构建rc0，两归档模块字节、wheel RECORD及显式required-member审计通过，审计 `_state/release-day23/20261029-main-compat-optimizer-restore-package-audit-r1/terminal-audit.json` SHA `f9bc8a67727c5bb55fccf3b89e226793a6be99039e1ec970492b41f51903d4ae`；未安装wheel或证明非editable运行，新文件仍为Git未跟踪，主树旧egg-info尚未刷新。

## 2026-10-29 DDP owner隔离修复及CPU回归运行中

针对原两rank SimpleNamespace冲突，Torch参数使用独立 `_torch_ddp_state`，native保留自身marker，双owner包装在通信前拒绝，native loss驱动路径对Torch参数/输入列表提前拒绝并保持状态；标准Torch backward/step、no_sync和closure需回归验证。原dirty备份 `_state/release-day23/20261029-torch-native-ddp-owner-fix-r1/`，新执行指纹 `5466a63e9f2e83d585fcf6deaffa977cb191f32aa6c237ba74b847e669577a7d`，native编译输入不变。CPU native DDP全文件与Torch owner/closure两个阶段串行执行：首worker PID/PGID 1765387、start_ticks 187738264，目录以实际controller记录为准；恢复前核进程/receipt/锁，不重复。旧两rank失败保留，CPU与新真实两rank结果尚无结论。

## 2026-10-29 主树两rank Accelerate DDP真实功能失败

新源码 `2dd2b3b5…` 两rank单次测试均完成preimport guard、真实不同GPU UUID及NCCL初始化，fallback各0→0；目标测试call均在optimizer step失败，native `_assert_ddp_step_ready` 把public Torch DDP 的不可hash `SimpleNamespace` 当字典键。该state也缺native后续所需属性，不能只改hash；无更新/对照通过证据。独立终审 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/rank-terminal-failure-independent-audit.json` SHA `2e98c575f56a2b5511af592aef56f4a6ea4759019227aa450fafe67d686ed5a9`，全部进程组和可见锁/FD已释放，拒读PID未知。原失败保留，正在审查隔离Torch与native DDP状态和避免重复allreduce的最小修复；新源码需重绑，不能复用旧binding或自动重试。

## 2026-10-29 主树两rank Accelerate DDP 尾批测试运行中

新源码 `2dd2b3b5…` 的单进程NCCL custom-op预热自然rc0、真实库/NCCL初始化和fallback0，独立终审 SHA `422a2969e01914da46141219ac96d7c17b19575c49ec632c2b7bd9f36d160a6f`。受限Jittor路径共同锁审阅 SHA `b80d307b71ac33c2b52c7660aa451b7087c46485bbf47ab25ffb94bbbc4f9106`，第三方/vendor未观测行为作为边界。实际两rank已单次启动：controller PID 1757090、launcher PID/PGID 1757092、rank0 PID/PGID 1757094、rank1 PID/PGID 1757095；绑定 SHA `7a14dc0f26194d493da642b3604b629ae42911fac7f08d215b2aba2596ece712`，目录 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/`。GPU UUID为 `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 与 `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`。保持原launcher LOCAL_RANK，恢复先核PID/start/receipt及锁，不重复启动，运行结果未终态。

## 2026-10-29 主树两卡DDP的单进程 NCCL 预热运行中

新主树指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 的BERT单步及CUDA optimizer已终审，受限调用链及可见进程/共同锁核验后，单次NCCL预热worker PID 1755063、启动epoch 1793267445.751、cap300，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`；目录 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/warm-attempt-v2-01/`。另候选第二卡 `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`，只要求足够显存，外来负载和拒读PID未知保持记录。恢复前先核PID/receipt/锁，不重复；当前预热及两rank均未通过，NCCL产品须实际绑定。

## 2026-10-29 新 optimizer拆分身份BERT单步实证

新指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 下BERT seed821 BF16由旧checkpoint50真实恢复并单步更新至51：新restore模块实际载入201组状态并返回，603模型/moment CUDA、四组RNG一致、201梯度FP32 CUDA finite、201参数变化，样本/shape/LR与旧full及PT一致，fallback0；模型child自然rc0。原包审计器错误只检空状态call000而忽视checkpoint call001，控制器审计rc1保留；独立审计 `_state/release-day23/20261029-bert821-main-restore-split-step51-independent-audit-r1/terminal-audit.json` SHA `8be210b7258def3e6aec6d6e14c5d95009f7f1a6d09f505c2fe4a409af73a741`。旧与新step51前态201签名全同、后态200/201相同，token_type_embeddings后态只有hash不同比较，值差未知；本步loss JT 0.3322448730、PT 0.2778320313，逐步值仅观察。此单步不是完整曲线/固定eval，拆分前主树BERT/Qwen续训结果继续按旧指纹报告。

## 2026-10-29 新 optimizer结构拆分身份的BERT单步运行中

新执行指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 基于旧BERT checkpoint50实际恢复并仅执行第51步，验证拆分后optimizer恢复调用、CUDA状态、RNG、梯度、参数变化及相同样本/LR；不声称完整训练或eval。单次cap1200，worker PID/PGID 1750026、start_ticks 187639585；binding SHA `3aacccfb90e4c116e9a9f8c9a5118a48bdba4c561020ca89a4801f012b33d0a1`，目录 `_state/release-day23/20261029-bert821-main-restore-split-step51-r1-hold/`。恢复先核PID/start/receipt及锁，不重复启动；拆分前BERT/Qwen实训继续按其旧身份记录。

## 2026-10-29 optimizer结构拆分后定向复测

主树执行指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 相对Qwen/BERT实训身份只有optimizer_api.py和新增optimizer_restore.py两项Python变化，native编译输入未变；旧入口保留，恢复函数体AST相同。原结构失败节点现通过（705行，原800上限未改）；CPU optimizer 10/11，既有BF16 next-step moment差异依原阈值仍失败、fallback0；CUDA optimizer 11/11、fallback0。独立终审 `_state/release-day23/20261029-main-optimizer-structure-split-focused-terminal-audit-r1/terminal-audit.json` SHA `ce9c37344587d8022c0551baeb832b873ba24dd9c3b7e193711acb9c93c30e63`；原十文件总体失败、这次三阶段总体rc1均保留，不能称十文件全套复跑通过。旧身份两网络训练与新身份定向回归分别记录。

## 2026-10-29 主树 Torch CPU 十文件核心集结构门槛失败

230项实际执行，226通过、3原规则skip、1失败：`optimizer_api.py` 为852行，超过原800行结构上限。独立终审 `_state/release-day23/20261029-main-torch-cpu-core-ten-terminal-audit-r1/terminal-audit.json` SHA `35fb18dbf425dcaa4e138c6a66a75c400919ad1ee48e7609eb169db5d4fc2778`；四份文档测试窗口冻结一致，终态fallback未记录，进程组和可见锁/FD空，拒读PID未知。原结果保留，不调整阈值；准备拆出optimizer恢复实现并绑定新源码身份、按受影响范围复测。之前Qwen/BERT训练均属于拆分前源码 `6c3decf3…`，不会自动转移为新身份。

## 2026-10-29 主树 oneDNN 显式 provider 及三个CPU节点通过

默认启用、显式指定已核库的oneDNN sentinel真实CPU matmul及数值通过；新后继复用该custom-op和主树CPU core，原先受阻三个节点的setup/call/teardown全部通过，父进程fallback 0→0。两个frontend子进程均rc0且日志分别记录1、6条真实oneDNN matmul，子进程fallback未独立测量。锁及可见FD无holder，拒读PID未知。独立审计 `_state/release-day23/20261029-main-r12-onednn-cpu-successor-terminal-audit-r1/terminal-audit.json` SHA `96fa0652da767dc61c4ccf7cbc712b200cc56e0d18cc86f733053e7fc7c09d22`。原recorder参数错误rc1、原no-build oneDNN缺失失败、CPU BF16 optimizer差异及CPU cold rc1均保留；此结果不证明默认库自动发现或完整CPU套件。

## 2026-10-29 oneDNN 实际 sentinel 通过，补测脚本失败保留

主树默认启用并显式指定现有oneDNN库的单次sentinel自然rc0，实际加载所指定libdnnl、选中 `_mkl_matmul` 且数值检查通过。后续pytest recorder把hook参数写作 `r` 而非hookspec要求的 `report`，三个节点收集之前失败，控制器rc1；原结果完整保留在 `_state/release-day23/20261029-main-r12-onednn-cpu-supplement-r1-hold/attempt-01/`，不得写为三项通过。正在独立修正测试记录器并准备只对未执行的三个节点作新后继，复用已经构建的custom-op，不重复冷编；自动发现及frontend子进程fallback仍未证明。

## 2026-10-29 主树 oneDNN 显式库 CPU补测运行中

Qwen/BERT续训独立终审后，实际源码、CPU产品、provider库及头文件和可见进程/锁已核对；仅可见范围无FD holder，拒读PID未知。oneDNN单次补测controller PID 1723686、sentinel PID/PGID 1723687、start_ticks 187515046，sentinel cap420；目录 `_state/release-day23/20261029-main-r12-onednn-cpu-supplement-r1-hold/`。先核实际载库和matmul，成功后再跑三个旧受阻节点；恢复先核PID/receipt/锁，不重复启动。此为默认启用但显式指定库的配置，不预报默认自动发现或frontend子进程fallback通过。

## 2026-10-29 主树 BERT 50→100 续训独立终审通过

主树seed821 BF16从旧候选 checkpoint50 续训50次、0 skip，50×201梯度实际CUDA0/FP32且有限，首步201参数均变化；603模型及moment真实CUDA、fallback0。99个microbatch的样本/shape及50个学习率与旧Jittor full及PT一致；固定33样本主树eval loss 0.4661458433、accuracy 26/33，历史独立PT full loss 0.4836055934、accuracy 26/33。实际checkpoint100模型、moments、step与签名相符，scheduler epoch100；原checkpoint50严格恢复及Python/NumPy/CPU/CUDA RNG检查通过。进程自然rc0、未超时、可见终态锁和FD无持有者，拒读PID未知；独立审计 `_state/release-day23/20261029-main-bert-resume-terminal-independent-r1/terminal-audit.json` SHA `7dae1c2561ec5b7eeeb4fb3055e9d0744e2cbb5d828b1696c844c23cb90403d8`。前50步属于旧候选，PT为独立full任务参照，不声称主树从头训练或抹去旧双seed strict失败。

## 2026-10-29 主树 BERT 50→100 自然完成，待独立终审

主树seed821 BF16 BERT续训单次自然rc0、未超时；包内终审14/14报告50次更新、0 skip，逐次201个FP32 CUDA梯度有限，99 microbatch样本游标及50学习率与旧full一致；step51 201参数变化，step100 201模型/402 moment真实CUDA、fallback0，checkpoint100核心文件存在。固定33样本eval loss 0.4661458433、accuracy 26/33；同协议历史独立PT full为0.4836055934、26/33，属于任务级参照而非同一续训来源。receipt SHA `36c649d749d139ac4f0a2974d4a7d3d18bfb5e89e6584c6339395f17eb83e0c3`、包审计 SHA `3601fe446a3d570e82ca79ebb6049b201b7198a46b1bba91d257bdaa2120e4cc`，独立终审待完成。前50步为旧候选，不称主树从头训练；旧BERT两seed strict FAIL保留。

## 2026-10-29 主树 BERT 首次续训更新通过

主树BERT seed821 BF16第51步非零学习率实际更新，201/201模型参数发生变化，更新后201模型与402 Adam moment真实CUDA0、fallback0；已记录step51–59每步201梯度设备，进程 PID 1710979/start_ticks 187428056 继续至100。checkpoint50的逐名状态及四组RNG此前已实际恢复，前50步来自旧候选；训练全程有限值、固定验证和终态仍待报告，不预报结果。

## 2026-10-29 主树 BERT checkpoint50 加载通过，续训进行中

实际主树seed821 BF16 BERT checkpoint50 的model/optimizer/scheduler/scaler具名签名strict与functional均通过；201模型参数及402 Adam moment实际public/native cuda:0，201 optimizer step、trainer及scheduler为50，Trainer真实加载边界Python/NumPy/CPU/CUDA四组RNG一致。worker PID/PGID 1710979、start_ticks 187428056仍运行，进入更新；报告路径 `_state/release-day23/20261029-bert821-main-resume-executable-r1-hold/runs/bert-821-resume/`。当前只是加载通过，不预报训练与固定验证终态，前50步来源旧候选。

## 2026-10-29 主树 BERT checkpoint50 续训运行中

seed821 BF16单次主树BERT 50→100续训已启动，worker PID/PGID 1710979、start_ticks 187428056，cap1800；release review SHA `7a1a74e902ea377a611a092d9a051b792e30d4a66e269d57932a7769703bf8ae`，binding SHA `b428f7e81d13c865b83342952c1951c7fa543208e1859e44e22dd0a21f989da7`，目录 `_state/release-day23/20261029-bert821-main-resume-executable-r1-hold/`。旧checkpoint50已离线逐名核对201模型tensor、201 optimizer state及scheduler，RNG文件记录各组但旧raw hash不可得；实际主树设备、恢复、更新、数据及验证仍需实测。前50步来源旧候选而非主树从头训练。恢复前核PID/start、receipt、锁，不重复启动。oneDNN与其他JIT保持等待。

## 2026-10-29 主树 Qwen 实用续训及 lifecycle 终态通过

主树seed821 BF16从旧r10 checkpoint51续训至102，实际51次更新、0跳步；每步112梯度均FP32、真实CUDA且有限，101个样本ID与旧full及PT后半程一致，112 LoRA参数均变化并与旧full最终参数一致。固定验证loss主树2.4555516243、PT 2.4566440582；新checkpoint原始portable内容和sidecar严格一致。predict返回loss 2.4555516243，adapter重载/生成及merge完成，merge logits最大绝对差0.000289917在原0.001预算内，fallback0；进程自然rc0、未超时。独立终审 `_state/release-day23/20261029-main-qwen-resume-terminal-independent-r1/terminal-audit.json` SHA `2960455d2342bb592ed7a1e83756fb430984c4d3eac738131429d103e94308d8`。此为主树后51步续训，前半程与完整full来自旧r10；eager evaluation为限定workaround，当前checkpoint直接由原生PyTorch torch.load optimizer仍因portable格式失败，旧strict/CPU BF16和oneDNN失败保留。GC实际重计算及正式性能/显存不在通过范围。

## 2026-10-29 主树 Qwen 续训已达102步，lifecycle运行中

主树seed821 BF16从旧r10 checkpoint51续训报告写出：51→102共51次optimizer更新、0跳步；51组梯度无缺失/非有限，dtype FP32；101个续训microbatch的step与sample ID同旧full后半程逐项一致。checkpoint 337实际CUDA张量、112 step、scheduler和四组RNG恢复已核验，冻结基座不变、fallback0；两次eager evaluate返回，固定验证loss 2.4555516243。报告和记录位于 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/runs/qwen-821-resume/output/`。前51步与对照full为旧r10源码，主树不是从头训练。进程PID 1674876/start_ticks 187236743仍运行，strict lifecycle/predict尚未终态，不预报最终通过；恢复前核PID、receipt与锁。

## 2026-10-29 主树 Qwen checkpoint 加载通过，续训进行中

主树Qwen训练进程 PID 1674876/start_ticks 187236743 仍运行；checkpoint51 同进程状态比较strict hash及functional通过：337个张量（112模型参数、224 optimizer张量、1 buffer）实际public及native cuda:0，112个scalar step均为51，trainer step及scheduler epoch均51，Python/NumPy/CPU/CUDA RNG恢复检查通过。报告在 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/runs/qwen-821-resume/output/`。仅证明加载，不预报后续更新、样本、验证或终态；恢复先核实际PID/receipt及锁，不重复启动。

## 2026-10-29 16:39 +0800 主树 Qwen 实际续训运行中

主树checkpoint51续训已单次启动，controller PID 1674865，训练进程PID/PGID 1674876、start_ticks 187236743，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`；同进程身份文件确认导入主树及CUDA core SHA `bbd4a209b3f0e780bc19cac53dde4d1b7f7ea98d305fae55d70c012800084839`。路径 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/`；恢复先核PID/start、receipt、锁与实际记录，不重复启动。尚无训练/续训终态结果。oneDNN补测及多卡补测仍HOLD，不与首次JIT并行。

## 2026-10-29 主树 CUDA 定向回归通过；Qwen 续训放行

主树CUDA 8阶段91/91节点、各节点setup/call/teardown通过，replay 6、optimizer 11、alias 22、writes 1、cat 14、RMSNorm 2、pow 1、factory 34，全部自然rc0、fallback0；writes与pow含CPU子分支，代表性alias/pow探针确认CUDA，不能推断每节点独立dispatch遥测。独立终审 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/terminal-audit.json` SHA `2203487021fbfe15d226a72a39bf9cea36d4afb3f20669bac9437df2689f0c98`；目标锁和进程组空、终态cache 1576项。factory 使用与旧PyTorch独立对照完全相同的测试文件，本次不是重新运行PyTorch。

主树Qwen从旧r10 checkpoint51继续至102，固定seed821、BF16、eager evaluation、无Python overlay；审核实际产品、CUDA阶段、数据/checkpoint和GPU后放行，release review SHA `c79c7396686ca97befea50aba544dbb1378dc5a79e149ccae03550f1260067b2`。前51步及完整两epoch训练曲线来源仍是旧r10，主树只验证后半续训，不预报结果；CPU BF16数值失败、旧oneDNN失败及旧Qwen失败均保留。性能独立未验收。

## 2026-10-29 主树 CUDA 定向回归进展

实际主树源码及CUDA产品下，replay 6/6、optimizer 11/11、storage alias 22/22、显式写入1/1（含CPU/CUDA子分支）、autocast cat 14/14、mixed RMSNorm 2/2、pow 1/1（含CPU/CUDA子分支）依次自然rc0、fallback0；代表性alias/pow设备探针为真实CUDA，不推断每个节点均有独立dispatch遥测。factory 第八阶段正在运行，PID/PGID 1671783、start_ticks 187221014；恢复先检查进程与 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/serial-receipt.json`，不要重复。结果未终审，不能推断训练通过。主树Qwen续训待所有阶段终态和审查。

## 2026-10-29 主树 CUDA 定向回归已启动

串行8阶段计划基于已核验主树CUDA cold产品与指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2`，首先运行 replay；worker PID 1656851、PGID 1656851、start_ticks 187179714，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，入场可用显存58054 MiB。记录位于 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/serial-receipt.json`；各阶段单次cap900，失败即停，不自动重试。恢复时先核进程和receipt，不预报测试结果；完成后才释放主树Qwen checkpoint51实际续训。

## 2026-10-29 主树 CUDA 冷构建身份核验通过

主树执行指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2` 的 CUDA cold 自然 rc0，耗时393.37秒、231个编译单元；实际设备 GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，Jittor及Torch导入均为 has_cuda/use_cuda=1。receipt SHA `0ae76fd99a6f28033d7a76a3fdae80df8e1ade9ef454353fa3c62ae096cb2aec`；独立终审 SHA `ec4847e591ce955d0e89f68bae0d56b7dd3112610a9428704b8a081386051332` 核对源码、38个补充pyc、core/utils/stamp、编译输入及缓存终态，可见锁与holder为空，拒读PID仍未知。仅证明冷构建和shim身份，不证明tensor数值、训练或性能。原CPU cold rc1及限定native审计继续保留。主树CUDA定向回归准备串行执行，主树实际Qwen checkpoint续训尚待；历史mixed身份结果不直接转写为主树通过。

## 2026-10-29 16:19 +0800 主树 CPU 补测通过，CUDA cold 运行中

主树CPU补测：修正后的 FollowRuntime 内部分支在默认配置1/1通过，3个原oneDNN缺失节点在明确 `use_mkl=0` 的CPU配置3/3通过，两个测试worker sentinel native CPU、fallback0；独立终审 SHA `67f92e338d0c2fa81a143cb713b70000877032b958e4754b89a22094654ed18b`。原四项失败保留，`use_mkl=0` 不证明默认oneDNN路径；frontend自建子进程的fallback未单独记录。原主树optimizer BF16严格moment差异10/11保留；别名、factory、replay及library已分别按前节记录。

测试断言修正后主树执行指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2`；与已审CPU native core对应清单只有该测试文件变化，生产和编译源码未变。主树独立CUDA cold 单次cap900已启动：目录 `_state/release-day23/20261029-main-r12-cuda-cold-resources-testfix-r2-hold/`，worker PID 1648007/start_ticks 187120314，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，根审阅 SHA `364fe9f057b84dea7666d8396214ea2970374bd3415c191f74dea8da99c03800`。38个gitignored pyc及对应源码单独绑定，不删除或重写；恢复时先核 PID/start/receipt/lock，禁止重复。CUDA构建与后续main设备、optimizer、replay测试尚无结论，性能验收独立。


## 2026-10-29 16:16 +0800 主树 CPU 定向结果与补测

独立审阅采用实际主树 native core 后，native JVP/pow三项全部通过、fallback0，终审 SHA `43ba3c2e9e2e1fdf4a1feb11f12a92b0939c6dc1400137467495d7590736ec6a`；最初 pytest hook 参数错误的0执行失败与 cold rc1均保留。主树 Torch CPU：replay 6/6，optimizer 10/11（已知BF16下一步moment差异），storage alias 22/22、factory 17/17、library/owner 17/17通过，fallback0。autograd semantics 23 passed/2 failed/1 CUDA-only skip，independent frontend 0/2 failed：其中 CPU fill/zero 和两个frontend均被默认启用oneDNN但 `JITTOR_NO_BUILD=1` 且无本缓存匹配oneDNN产物拦住，不是测试通过；将单独在明确 `use_mkl=0` 的配置补测，并保留默认oneDNN路径未验证。

另一失败为内部 FollowRuntime 测试错误地要求 native `detach()` alias 的 requires_grad=False；历史实际诊断显示该 native alias 创建时为True且写入前后不变，frontend target为False。主树仅修该测试：记录并检查 native alias原值不变，保留 frontend标志、共享地址、数值和dispatch/fallback断言；修改前备份 `_state/release-day23/20261029-main-followruntime-test-fix-r1/`，原失败保留。生产源码/已产main CPU core未改；新测试身份需重绑，不能将后继补测回写为原通过。


## 2026-10-29 16:03 +0800 主树 CPU cold 产物已生成，原终审因 ignored pyc 失败

独立限定审计 SHA `a244fb856a7f467ac2d2b2fa05c17b8e5481c108cde5aafe3d39f9a30d5050de` 确认实际主树CPU core、utils、stamp、227单元及8个生成器可作为 native 后继验证产品；38项pyc都不属于实际编译单元或生成器，header与对应冻结源码及递归编译语义一致，静态对比不能确定哪些曾被加载。原 child rc1、Torch shim 未导入的边界保留，后继须实际 warm import 和运行，不重跑冷编。cache-after SHA `504f145f17e7c15503c8cc709498a2a52d7d07ec6d479898e61e44c883565c97`，可见锁/holder空、拒读未知。

主树独立 CPU cold 实际编译227个单元（190仓库、37生成），core/utils/stamp已生成；child 自然 rc1、非超时，原 receipt SHA `1a832710782011869fc03a488c6796bb4788cd052ed70426a41c9c1c152dadc5` 保持失败。fresh-build 检查38项报错均为 `backends/cuda/**/__pycache__/*.pyc` 不在 tracked+非ignored source清单；尚不能将产物直接写成已通过。已核实38个pyc均是既有被忽略文件、对应冻结 `.py` SHA一致、Python magic/时间戳/大小匹配，按同解释器递归比较编译语义码38/38一致；是否可在限定身份内采用产品仍在独立终审，不重复冷编、不删除字节码。终态组/可见锁空；实际 native/Torch测试仍待运行。


## 2026-10-29 15:57 +0800 Qwen 实用续训通过；主树 CPU 冷构建运行中

r10 native/base + r12 三个 Python overlay 的 seed821 BF16 实用续训已独立终审通过：checkpoint51 实际加载337 tensor均真实 CUDA0、状态值严格一致，112 scalar step与scheduler一致、Python/NumPy/CPU/CUDA RNG恢复；后续51次更新及101个样本 ID/shape与full后半程一致，51×112 gradients finite/FP32/真实CUDA、fallback0，112个LoRA tensor实际变化，checkpoint102全部928数组与连续full相同；验证与predict loss均2.4555516243。此前崩溃的 predict 此次返回，adapter保存/重载、生成及merge在原协议内完成；进程自然rc0、未超时。独立终审 SHA `19d01e4bf02a2595a4522587727c2356d1224d62d32ba352a50800b2da5e9f70`，旧 r10 full/predict 和 resume rc1均保留。该结论只覆盖本配置、mixed源码身份、eager evaluation；CPU BF16 optimizer 10/11差异和通用图内写回未解决，性能/显存未验收。

主树代码已整合相应补丁，自己的执行源码指纹 `647717bf832a74e555e6d3b12f8a9b77f1c530a49ff04322391af99e440ab0d6`；独立 CPU cold 单次cap600正在运行，目录 `_state/release-day23/20261029-main-r12-cpu-cold-docmutable-r1-prepared/`，controller PID 1630556、worker PID 1630567/start_ticks 186981816，根源码审阅 SHA `e5e21fa657fb1861ba004f9df08cba88e5dcf6614827d463594aa6a15c675501`。恢复先查PID/start与receipt，严禁重复；该构建不是测试通过，主树 native/Torch/真实CUDA回归与模型验证待后续单独执行。状态文档例外只用于此构建/指定native，不用于读取看板的结构门禁。


## 2026-10-29 15:41 +0800 r12 混合 CUDA optimizer 回归通过，Qwen 续训已启动

独立实用续训审计 SHA `4d4d6529efe3be362d2a92f452a1ed97812008b31292ddf0a0b149309b9da7e9`：continuous full 与 resume 的 checkpoint102 sidecar各928键、无缺失或数值差，112个 LoRA tensor 自51→102均变化。程序已越过原 `trainer.predict` 崩溃点并创建后续 lifecycle 保存目录；预测数值及 lifecycle/进程终态仍待实际报告。该审计范围仅本次 r10 native + r12 Python mixed、seed821 BF16、真实 CUDA/eager评估，不能推及其他任务。

真实 mixed Qwen 新进程续训报告已保存：从51到102共51次 optimizer更新、0跳步，后半程101个 sample ID/shape与原 full 逐项一致，记录 loss 均值均1.153616764167748；每步112个postclip梯度均存在、finite、FP32、native cuda:0，fallback0，冻结基座未变，固定验证loss与full同为2.4555516242980957。checkpoint模型/optimizer/scheduler数值、337 tensor CUDA设备、112 scalar step和四组RNG此前已实测恢复；practical-report SHA `fbdd09311192df7dd56795d5afd801c83771fc5b1d2c258e825b11ac3216332a`。该直接证据支持此次 mixed 身份的实用续训，预测/lifecycle 与进程仍未终态，不把整个流程预报为成功。

新进程 checkpoint51 实际加载通过：strict hash与functional比较均true，928项值无失败，337个模型/optimizer tensor 均真实 cuda:0/native CUDA0，112个 live scalar step 均51且与 checkpoint/trainer/scheduler 一致；Python、NumPy、CPU、CUDA 四项 RNG 恢复检查通过。后续样本游标、51次更新及终态验证仍在运行，不能提前认定续训通过。

主树已按完整前置字节核对带入 r12 optimizer 设备恢复修复与测试，移植前后哈希及原文件备份位于 `_state/release-day23/20261029-main-optimizer-restore-integration-r1/`。r11 replay 缓解主树备份在 `20261029-main-replay-integration-r1/`。main-specific 构建和运行验证均待执行；CPU BF16 10/11、ACL未验等范围仍保留，不能直接转为主树通过。

r10 native/base 加冻结 r12 三个 Python overlay 的真实 CUDA 定向回归11项、33阶段通过，覆盖 CPU+CUDA 两组 AdamW FP32、BF16参数/FP32 moment、SGD momentum、CPU map_location后迁回每个参数的真实 device、下一步更新与异常原子性；fallback0，独立终审 SHA `f14b9e3616f5f8350c2f8b76cca887b4950aa1558b0a43fc30ef2346e78ec677`。范围为 mixed 产品，不是 r12 native冷编或其他NPU/多GPU的证明。CPU 同源11项为10 passed/1 BF16下一步 moment比较FAIL，原 receipt `3c90b5b760fd42e7f729e82117a0e9f81c95cf3c165c34cda8789091e5ff9767` 保留。独立诊断四场景记录到两侧第二步前参数、梯度、FP32 moment一致，第二步参数一致，moment最大差6.676465e-5/约0.093548%；与一次BF16中间舍入相符为数值推断，尚非已验证根因或长期无影响证明。

基于同源已审核的 r10 full checkpoint51，r10 native + r12 Python mixed 新进程 resume821 已启动，单次 cap1800s，worker PID 1619262/start_ticks 186889301，根审阅 SHA `b8ab39769dc547649b454edd513f8313da0560d2c1b05595207e7107b7fb7fb2`，新状态目录 `/data6/cgq/jittor-lab/_state/release-day23/20261029-r12-mixed-qwen-resume821-eager-r1-hold/`。其目标是实测模型和 optimizer state 的设备/值、step/scheduler、RNG、样本游标及后51次更新；启动不等于恢复通过。旧 r10 resume224个moment落CPU失败以及full后predict失败保持。需要恢复时先检查 PID/start、receipt、锁，勿重复运行。

主树已按单独备份移入 r11 已观测 alias写入时预先拒绝捕获的缓解及测试；main源码身份尚未独立构建/测试，不能转移 mixed 结果。optimizer r12 已按本节增量带入 main，主树验证待执行。


## 2026-10-29 15:36 +0800 r11 真实 CUDA 六项通过，r12 CPU 选择器待修

r11 混合验证使用已审 r10 native 产物及 r11 两个实际加载的 Python 模块，真实指定 GPU、cuda:0 native dispatch、fallback0，六个目标测试与18个 setup/call/teardown 阶段全通过，pytest/worker rc0，实际 item.path 均为冻结 r11 测试；独立终审 SHA `af8f53df14a0feb0c4c496db1f4584141efca0b5deff68a2bc1e92a6a52936ab`。包括已观察 alias 写入时拒绝捕获、别名与旧值、多次不同输入、异常恢复、无写模型仍回放和仅捕获阶段写入的 guard；不是 r11 原路径冷编，也不能宣称通用图内原位写回。原前置 use_cuda 断言失败及文件级 accounting 警告保留。

r12 optimizer CPU 首次混合尝试在收集结束前失败、0测试实际调用：`-k cpu` 同时命中了函数名并选入 CUDA 参数，recorder 因22项非预期中断；原 receipt SHA `9e4388c7c775e701c742285b91e974aed757690dfef436dbd7f8cb9d8de9a0cd` 保留。尚无 r12 optimizer CPU/CUDA 结果，正在按真实参数 device 精确筛选后继续。真实 Qwen checkpoint resume 与 predict 问题仍待解决，性能验收仍独立未完成。


## 2026-10-29 15:32 +0800 恢复：r11 CUDA 启动断言失败，r12 待功能验证

已重新检查现场并 fetch `origin/cgq_transformers`：SHA `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，仍为 main/r10/r11/r12 HEAD 的祖先，不需整合；所有树的既有 dirty 状态保留。上次 r11 混合 CUDA 尝试在收集前 rc1、0测试：worker 在 `import torch` 后错误断言 `jt.flags.use_cuda==0`，而同源真实 preflight 已记录导入后 use_cuda=1；该测试基础设施失败原 receipt SHA `b76b9e01426199dcecf5bd01cf91720284e0bc4abbf0e09c6a94ab7c400872d4` 保留，未证明 CUDA 功能失败，也未证明通过。当前可见范围无本任务活跃训练或构建；不可见进程/锁持有者仍按权限范围核查。

r12 已冻结为 `torch-compat-replay-optimizer-restore-20261028-r12`，fingerprint `3b98d6922e55309fc10342220d1cf5cf7fd7964fc6621fa2132ca8fc306b0206`：r11 基础上按参数 native dispatch 迁移 optimizer moments并保留 dtype，新增 optimizer 恢复测试；尚未实际运行。当前推进修正 CUDA 启动断言的 r11 定向后继及 r12 CPU/CUDA 状态恢复测试，保持首次 JIT 串行。原 Qwen 新进程 resume 的224个CPU moments是真缺陷、112个 scalar step设备检查为检查器误判；完整训练/eager eval 已有证据但恢复仍未通过。


## 2026-10-28 17:07 +0800 r11 replay 写入缓解：混合 CPU 六项实际通过

r10 native/core + r11 两个 byte-verified Python 模块及 r11 测试的明确混合 CPU 运行，6节点、18阶段实际通过、pytest/worker rc0、fallback0，覆盖已观察写入时 alias、旧reader、自动/显式replay拒绝、异常恢复、无写入正常replay，以及捕获时才发生写入的 guard 反例。controller 原 receipt `3812b428bf9983900064d81a5e95c563b841b7bf76a44cedfbab9df64c458743` 保持 `FINAL_GUARD_FAILURE_RETAINED`：14个现有 shim 文件只有mtime改变、SHA/size与canonical source均相同，另新增24个 CPU JIT 文件、无删除；完整保存证据独立终审 SHA `4714dbf888fd3317b1119b2b76623025d2b9073ec145c83a4a5f5932f6ad8a3d`。pytest六个call确有记录但 nodeid文件部分为空，JITTOR 文件级统计报0 executed，保留此统计界限。此结果不是 r11 native冷构建、CUDA验证或任意图内写入重放支持。

optimizer moment CUDA恢复缺陷正在 r12 新候选修复；新进程恢复审阅器单独修正 Python step 标量误判，15项无GPU假对象检查通过，仍未绑定实际新源码或证明恢复。


## 2026-10-28 17:04 +0800 r10 checkpoint51 恢复被真实 optimizer 设备缺陷阻断

seed821 新进程 resume 自然 rc1、非超时，严格模型/optimizer/scheduler 状态 hash 全一致，928项数值检查通过；模型参数/缓冲 113 项仍为 cuda:0，但 112组 Adam 的 `exp_avg`/`exp_avg_sq` 共224个保存时为 CUDA、加载后实测为 CPU。单卡 Trainer 用 `map_location=cpu` 加载 checkpoint，`compat/torch/optimizer_api.py::_load_state_dict_torch` 直接引用这些 tensor 而未迁到对应参数 device，属于实际设备恢复缺陷，首个续训更新、RNG及样本游标尚未执行，不能判恢复通过。另有112个 optimizer step 是 live Python标量，检查器错误地要求它们具备设备元数据，须与真实 moment 缺陷分开修正。

原失败及对照证据保留；独立终审 SHA `07d935cac14d5b1bd29b6ad9de65925f81711b0378a541049c6d4fda4bd1f4a2`。下一步在新候选修复参数对应 device 的状态加载及检查器标量处理，先做CPU与真实CUDA聚焦测试，再从原完整 checkpoint 新进程验证；不抹去已完成的 full/eager eval，也不重置旧T5/Vision预算。r11 graph-replay 缓解测试与 optimizer 恢复为独立功能项，首次 JIT 仍串行。


## 2026-10-28 17:02 +0800 r10 Qwen checkpoint51 新进程续训进行中

基于新 full 实际 checkpoint51，不采用旧缺失文件或 best模型/末优化器拼装状态，seed821 resume 新进程已启动，单次 cap1800s，续训51→102，评估仍显式 eager；worker PID 1214963/start_ticks 178729759、argv 已核对，目录 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-resume821-eager-final-r1-hold/`。根启动审阅 SHA `875a368721158b8b33d794b6871e100b40bf0b6cdf156506ec64a64b81502ba8`，preflight→full 缓存新增1220、无改删的转换审阅 SHA `35c076a5fefc9fdeb3d2e7383547f96ba39c2c1eadcea3cd0eed11fdc58d47b8`；启动前 GPU 空闲39214 MiB，可读锁/holder空，1658个拒读PID未知。恢复时先核 PID/start、原 cache/receipt，禁止重复。

checkpoint portable CPU 文件与112组状态已验证，但真实 GPU 加载、RNG、数据推进与续训训练行为尚无结果。原 full 的102更新/验证证据有效，随后 predict rc1 仍保留。r11 针对自动捕获写入的缓解与主树回归仍待在所有首次 JIT 串行条件下实测。


## 2026-10-28 16:59 +0800 r10 Qwen eager full 已完成训练，predict 仍失败

独立终审 SHA `7a74ecef613a8a2056198b65017e6467b58e137e4ad8578be16afdd9013473f5`：202条 microbatch 的样本 ID/shape 与匹配PT一致，epoch1 JT/PT 1.20546931/1.20568009，epoch2 1.15361676/1.15406398；每步112个 postclip 梯度均 finite/FP32/真实 CUDA0，checkpoint51→102的112个可训练 tensor 均变化，固定验证 loss JT/PT 2.45555162/2.45664406。新进程续训尚未运行；predict 的 rc1 仍是未修功能缺陷。

新 seed821 BF16 run 真实 CUDA 完成 102/102 更新、202 microbatch、0跳步，fallback0、冻结基座未变化；按每 epoch 101 条 microbatch 同口径计算训练 loss 为 1.205469→1.153617，匹配 PyTorch seed821 约1.20568→1.15406。固定17样本验证 loss JT 2.4555516243、PT 2.4566440582，单seed观察接近；三个 Trainer eager evaluation 均返回，step51/102 checkpoint 已保存。`practical-report.json` 在后续阶段之前保存，当前训练+eager验证已有直接证据，仍待独立终审和新进程续训；模型/optimizer的各文件必须审阅，best模型与训练末optimizer不是同一原子快照。

完整进程自然 rc1、非超时：训练和验证后 `trainer.predict` 未处于 eager 作用域，再次触发 r10 graph replay / alias-preserving write 捕获错误；不能写成整个工作流或预测通过。原严格结果、失败 traceback、两次 checkpoint 和报告均保留。r11 只针对 warmup观察到的写入，尚未运行；checkpoint51 portable CPU 对照通过不等于新进程恢复。正式性能/显存验收仍未完成，不能从训练耗时推断性能。


## 2026-10-28 16:50 +0800 Qwen eager 评估新 full 已启动；r11 定向缓解待测

checkpoint51 的9个文件已确认稳定并用独立 PyTorch CPU 与可信 portable pickle 显式映射核验：112组 optimizer state 的 step 均51、scheduler step51、四项 RNG 均存在，926条 sidecar 数值严格一致；保存 LR 与 scheduler/6步warmup线性计划吻合。CPU audit SHA `7480abd38074fd7ac74ab05a3ae6c815c84ffbefa64022b98f10d78cfa6fe232`。原 `torch.load` 直接读取 optimizer.pt 因该兼容层使用 portable pickle 格式失败，属于跨框架读取限制，保留原失败，不称文件损坏；一次日志/保存LR时序误判也保留并以实际计划复核。此为CPU文件审计，不证明Jittor新进程设备、数据游标或续训行为。

新 full 已走过原崩溃点：首次 Trainer eager evaluation 自然返回，inside `auto_graph_replay=0`、`keep_graph=0`，423个参数/缓冲 native cuda:0、fallback0，退出后 replay 恢复1；epoch1 eval loss 2.468442678451538，terminal SHA `7b781e179925ec8f91dcf96be5b0a48f8be897e4b47c576dec1f2ad15a367724`。checkpoint51 的 adapter、optimizer、scheduler、RNG、trainer_state和 sidecar 已出现，保存和完整训练仍进行；文件存在不是已核验恢复，不声称训练/恢复或 graph replay 通过。首两步实际CUDA梯度/更新门禁也通过。

针对 r10 自动 graph capture 验证崩溃，新的 seed821 BF16 full 以相同权重、数据、102更新预算单次3600s启动，仅在整个 `Trainer.evaluate` 作用域设 `auto_graph_replay=0` 并记录真实设备、keep_graph、fallback与标志恢复；这是实用训练的 eager 验证路径，不是 graph replay 修复。状态目录 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-full821-eager-eval-final-r1-hold/`；worker PID 1200873/start_ticks 178667264、argv 已核对，恢复时先查原进程身份和 receipt，禁止重复。根启动审阅 SHA `3c5efd89b500c1c8e74038b6fe2c6eb5762d6c321b4db09badb0b97163cf43d1`；原失败前后 cache 无旧文件改删，新增1200项，与当前 baseline 逐项核对，转换审阅 SHA `82e1b4e13accfb63c17c2a2c3142907dfca368c39612823cef83eb5d8ce57390`。启动前 GPU 空闲33290 MiB，1624个拒读PID未知；实验/正确性 cache 串行，不用于性能。尚无新训练、评估、checkpoint或续训结论。

新候选 `torch-compat-graph-replay-inplace-20261028-r11` 另行实现只针对 eager warmup 实际观察到的 alias-preserving 写入：完成该次结果后拒绝 capture 并返回结果，后续走 eager；保留 native capture guard，不能认定任意状态变化、非连续路径或图内原位写均得到支持。CPU/CUDA定向回归尚未执行；r10原崩溃和日志原样保留。


## 2026-10-28 16:20 +0800 r10 Qwen 首 epoch 后验证崩溃，修复中

seed821 full 自然 rc1、非超时；前51次更新日志 loss/grad_norm/LR 均有限，同口径51日志平均 loss JT 2.387310、PT 2.387724，首 epoch 局部变化受样本顺序影响，不能据此断言两 epoch 下降。验证阶段 graph_replay 捕获执行 Qwen causal mask 的 `kv_arange += kv_offset`，进入 `_copy_into_preserving_aliases` 时因 `state_capture || keep_graph` 被显式拒绝而崩溃；这是当前统一候选真实功能缺陷。Trainer 在保存前先验证，故无 checkpoint51、无固定验证结果和续训证据，不可从该运行续训或声称实用训练通过。首两步真实 CUDA、finite 梯度和实际更新证据仍有效。

原 receipt SHA `c53039bfdc0845d0bbabad842c8c2ec39491d60d4e20e6a8615143f3908867d6`，worker.log SHA `6346ee30ef7cc212e7bd5ebbcf2db1f251b238ea56ffc90413955c8289a6e5fe`，终态 cache SHA `76cee5443414f3a41cb9604c223c0637bdbedfcc6da5a5638de2e5ce8f42a3d3`；可见所属进程/锁清理，1601个拒读PID仍为未知。已开始检查保留别名语义的 capture 拒绝/安全降级或图内写入机制；未经具体测试不能简单改回 assign 或宣称修复。原失败不覆盖，旧T5/Vision预算不动。主树已整合的 r10 代码受同一缺陷影响，CPU cold HOLD 只适合验证该旧身份，修复后须形成新身份。


## 2026-10-28 15:53 +0800 r10 Qwen seed821 full 实际运行中

更新：第2步即首个非零 optimizer LR（约5e-6）完成，112个参数的梯度及 Adam moments 真实 CUDA、FP32且 finite，fallback0，56个 LoRA B 全部实际更新，全部112参数数值变化；56个 A 此刻梯度为零，其变化可能来自 weight decay，不能把变化当成 A 的非零梯度证据。`first-nonzero-update-evidence.json` SHA `efb7194c2b27a48138d6870e57e8806994b86b0a9fd7a5972891a1e326066033`。证明实际前后向与一次非零更新；完整训练、固定验证和恢复仍无结论。

更新：真实 Qwen 第1个 optimizer step 已通过有限值和设备门禁，两个 batch1 CUDA 输入、112个梯度及两组 Adam moment 逐参数均 finite/FP32/真实 CUDA，optimizer 覆盖无遗漏重复，fallback0。warmup 实际 optimizer 前 LR 为0，112参数保持不变符合预期；两条累积口径 loss 0.913073/1.469827。首步证据 `first-update-evidence.json` SHA `d1a106e68094e3fcb27800ce2a1722b07a44d74b2eb2124e09604c63026ffa96`，仅证明首步前反向和零LR状态；首个非零更新、学习曲线与恢复仍待实际结果。

新增 `test_library_saved_alias_version.py` 检查注册的 library backward 保存输入后经 detached alias 修改必须拒绝反向；同字节测试副本在独立 PyTorch CPU 实际1 passed，Jittor 主树尚未执行。FollowRuntime 新测试补充写入前后 stopped-gradient 断言；这些均未替代真实主树构建与回归。

主树进一步按依赖整组带入 shared storage version、反向实际读取校验、no-grad 保别名 copy、fill/zero 目标 placement 及存储回归；既有 dirty 内容前后副本与哈希保存在 `_state/release-day23/20261028-main-storage-integration-r1/`。FollowRuntime 内部回归改用已独立诊断的 native holder swap 构造，其旧错误前提的 r10 FAIL 原件仍保留。15个受影响 Python 文件已做 AST 语法检查，主树冷编译、真实运行、CPU/CUDA 和新回归均待执行，实用训练尚无主树结论。

主树另外逐项带入 SiLU 低精度中间值、autocast cat、梯度 dtype、混合 dtype RMSNorm 和 pow 舍入修复及对应回归，修改前副本、前后 SHA 保存在 `_state/release-day23/20261028-main-numerics-integration-r1/`；autograd 只移植梯度 dtype，存储版本 hunks 尚未混入。主树重新编译和相关 CPU/CUDA 验证均待执行，不能把候选测试当作主树通过。

主树已单独带入常量工厂叶节点修复和17项回归测试：只在 `compat/torch/installers/factories.py` 插入六个常量工厂的 stop_grad，保留原有 dirty 修改，插入前文件与哈希保存在 `_state/release-day23/20261028-main-factory-leaf-integration-r1/`；测试为 `compat/tests/torch/test_factory_leaf_training.py`。此为新的主树源码身份，尚未在主树运行回归，不把 r10 通过结论转移给主树；正在运行的 r10 训练源码未改。其余存储/alias、SiLU、cat、gradient dtype、RMSNorm、pow、fill/zero 改动仍需依赖审阅后整合。

新用途为同源 r10 的 Qwen LoRA BF16 实用训练，单次上限3600秒、102步、2 epoch，checkpoint51 留待新进程恢复；不动旧 T5/Vision 诊断预算。实际 final 在 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-full821-final-r1-prepared/`，source fingerprint `2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22`，launch review SHA `70b4b906596dd40e6583d4a70db7cca2c04c59e25e38853e04b07fd899a650e8`。controller PID 1144682，worker PID 1144693 / start_ticks 178318133，argv 指向真实模型、BF16和 CUDA；进程现正在执行，恢复时先核查 PID/start、receipt 和 cache，禁止盲目重复。启动前GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 空闲33108 MiB，正确性不以利用率为条件；可见cache holder和锁为空，1625个拒读PID仍是未知范围。

首步 warmup 学习率为0，记录逐参数梯度和 Adam 状态，下一非零学习率步骤检查实际更新；个别非零梯度参数不变保留数组待审，不自动中止。尚无完整学习曲线、固定验证指标或真实恢复，不能提前判定通过。完整 logits 原严格诊断与 CPU focused 原23/24失败仍保留，性能和显存验收独立。


## 2026-10-28 15:48 +0800 Qwen r10 no-training CUDA preflight 终审通过

preflight-only final 20261028-r10-qwen-practical-preflight-final-r2-prepared 单次自然 rc0；同一 PID/start 在物理 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 上实际导入 r10 source与 CUDA core，完整231 compile order和ingredients与已审计产品一致。小张量输入/输出 cuda:0、native dispatch cuda/device0、值[3,5]、fallback初值/增量均0。receipt 24fca00c9d88fdcaefa0fa892d5a7d51ec975d7e82b99fb6714eece711efe97e，result c2b9b9a1e5ad04e2faf532ead4f15424b08fc58a219fa883190d55d43e7a29fd，root terminal audit 3440e3b2b965e387f9317ace71ff430ef0a8846bb4f9cb1efc783dc883852ad7。实际cache常规文件1137→1141仅新增4项JIT、无删改，after SHA 0b3ea7f0062e117bb0a3917d24928234d9d6617ec5c118a70759ebe3dce6bf9d；锁及可见holder空，1624个拒读PID的FD仍为未知，不扩展为全机无holder。此仅小张量身份前置，未加载Qwen或训练。

训练final仍待冻结审阅：首step实际学习率为0，必须记录optimizer之前的LR，零LR时参数应保持且检查梯度/Adam状态，首个非零LR步骤再检查实际变化；batch积累、逐梯度设备/dtype、optimizer组和moment逐项验证，失败先留证。新模型full/resume与任务指标、CUDA dtype补充和Day4–5性能均未执行，CPU focused原23/24 FAIL保持。


## 2026-10-28 15:42 +0800 Vision 原始结果已逐格复核

旧 r25 Vision 五格真实训练、验证及恢复的实际指标、严格比较和样本限制见 [Day2–5 报告](2026-10-26-torch-compat-release-day2-5.md)，不要将其转移到r10；唯一状态源为 [看板](../architecture/refactor-board.md)。


## 2026-10-28 15:37 +0800 r10 真实 CUDA 35 项定向通过

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22、真实 A800 UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 的 Jittor focused 35/35 pass、0 skip/xfail、fallback0，105 个 setup/call/teardown 全通过，322.43s / 600s 自然 rc0。相同 34 项 PyTorch 同 GPU oracle 先通过；Jittor 多测原 CUDA fill/zero 对显式 CPU/CUDA目标的 alias 和 storage 语义。覆盖 FP32/BF16 六工厂叶节点、前后构造梯度、_like图/存储独立、输入和参数梯度、SGD及二阶，sentinel 实测 cuda:0/backend1/device0/dispatch(cuda,0)。receipt d4d483fcf7e67655472c255f90b9d669dd35ef9dbfc9a02e1cca3812dfab03d3、result 2a66f1fc21ae4e3eb44ff675ff970489b1e0e8d88c04fcce98747ed04751c31f、root terminal audit 6a9a4b56d034ec8e2b3f5bbcc04d7cb21575cf33740617e6d99915259f454530；源码/实际产品/完整编译顺序与环境一致，cache 942→1138仅新增196、无删改，终态 SHA 16257e96394bb034a016392f28028644c841e973f44d655367b604328fd3feaa，组和锁空。CPU focused 23/24 原FAIL仍保留，内部 FollowRuntime补充独立结果见下节。

这只证明有界算子与更新功能，不是 Qwen/BERT/T5 的 r10 实用训练通过，也不证明长期训练或性能。Qwen 新 r7 按此真实产品与缓存准备无训练 preflight，训练与日志另置 state、同功能 cache 串行、性能 cache 独立；preflight、模型加载、LoRA full/resume 及任务指标均尚未执行。真实checkpoint映射、scheduler/RNG/数据游标仍须新进程验证，旧严格结果不改。


## 2026-10-28 15:30 +0800 r10 CUDA 35项定向运行中

独立 PyTorch CUDA 34项对照与终审见下节。r10 Jittor 同源 CUDA focused final 20261028-r10-cuda-factory-focused-final-r3-prepared manifest 3c4101d313d6840cd5c3880bedbdd19acb1f0d67efae54a59c5e6033da84297c 已绑定实际 CUDA cold/core/942缓存和 oracle，root launch review ea2c43f47bf7c6d5d86581da33c47d0c359bcf29e707d6c4fefc0f0cf32e4d5d。单次 cap600、controller PID1120719、worker1120721、同UUID已启动，实际进程身份已记录，正在编译算子；恢复时先查 receipt/PID/start/锁，禁止重复。34项工厂/输入梯度/优化器/二阶/断图和原 CUDA fill-zero alias共35节点，尚无终态，不预填计数，不声称训练/性能通过。资源部署只允许14个精确路径和内容哈希，外部测试包不自动加载仓库 conftest，实际环境漂移仍记录失败。


## 2026-10-28 15:25 +0800 独立 PyTorch CUDA 对照 34 项通过

原 oracle 首次因 pytest recorder hook 参数名错误在收集前失败，0项执行；receipt 0e38282b48d942669bdb86812e1ab90a55990839fb4595b86f70586b57c31b71 和独立失败审计 4e8ffeaca525de3e2fe02b92d7cb2266036a01a1f9eebf68ada38cf319a2a626 保留。新 hookfix successor 使用完全相同测试字节 d2b99208885b39070a20ac986f6e1f09f40fb91e45919b241172d5719e9c3175，在独立 PyTorch 2.6.0+cu124 和指定物理 A800 UUID 上 34 pass/0 skip、102阶段完整，前后 CUDA 同步，28.22s自然 rc0；receipt 38b6ee43cd60dd34b0f1ae115261f899038e8b8c68a5fd2579b02600237284fb、result 0d7ed1c362c8187e66ef5c24df8cb8fdb16aec0a13d346b614ad5d1dcf0d4406、root terminal audit 75a03b0b1f3e17413f3531b6c4c5d8e244459061b03d07870a70262552e07d59。此仅是独立 oracle 通过，不是 Jittor CUDA、实训或性能通过。

下一步把该 oracle 终审和真实 r10 CUDA cold 产品/942缓存绑定到新 Jittor 35项定向包，核对真实 CUDA dispatch、梯度、SGD、二阶、alias及 fallback0，再逐项审计；尚未启动。Qwen r5 仍 HOLD，SIGTERM、恢复有界功能对比和身份均只有无 GPU dummy 验证，真实 checkpoint、新进程数据游标、训练产品与模型仍待实测。


## 2026-10-28 15:19 +0800 r10 CUDA cold 终审通过，数值待验

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 在物理 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 上的新 CUDA cold 自然 rc0、524.12s / 900s，231编译单元（194仓内、37生成），Jittor 与 Torch 导入后均 has_cuda=true/use_cuda=1；未创建 tensor，不能当成真实 CUDA 数值或训练通过。receipt 49bc5fa678b9e5716fc8cd50a89ea3927990d721a28c933aba8af34197559f97，child a1767d8d3a08e83aff3aabfa3a4b3e04af0495dbaad4eefd23cbfe6a5bd77c17，实际 942 cache c58b6866cb55ce273d487f84306fcb656a18c756a7a80699e46f1b2409f10d21；root terminal audit a3d5c1c1c9c862bf232495a4af014d78a53363e320b8a384c9863e20ba8b7340。初始 child 命令采样为空，存活期独立身份证据 2d8ae797e4a894ffe645b93ded82fe4c139fb7e044b6d7488076ff614cfb742b 核对相同 PID/start/组/实际命令，原收据不回填。

下一步在同物理 GPU 对 CUDA 常量工厂/输入梯度/SGD/二阶/断图的34项执行独立 PyTorch oracle，再绑定实际 CUDA core与终态执行 Jittor 34项+原 CUDA fill/zero alias节点。实训先决条件和性能验收均未完成。CPU core 的历史 dtype skip 不直接表示当前失败：同源 Torch promotion 测试实际覆盖旧16个 dtype 对中15对和两顺序，int8+int32仍需直接补测；.long()小值/负数在 CPU 正向测试通过，大数和真实 CUDA仍待验。


## 2026-10-28 15:11 +0800 r10 CUDA 冷构建进行中

实际 CPU core 终审已通过（见下节）；r10 同源 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 的新独立 CUDA cold final manifest 9a8fe2efc96f9787fae831c77dcb0fc4a762ecc1792e700a7c6ba1cc4c7dfd36 已单次启动，cap900；root launch review SHA 60a23df1d39eb6bbf3b34a7ba880b6f100813a628e9ac45b850ec76d15fbaca5，入场可用显存54024MiB，未以利用率设门槛。controller PID1096115、child1096119，原 receipt 初采 child command为空，存活期独立 identity 20261028-r10-factory-leaf-cuda-bootstrap-final-r1-prepared/root-cuda-cold-live-identity.json SHA 2d8ae797e4a894ffe645b93ded82fe4c139fb7e044b6d7488076ff614cfb742b 已核对 PID/start/命令/私有组。当前正在编译；不得重启，恢复先查 receipt、PID/start、子进程及锁。冷构建成功只证明产品身份，真实 CUDA 34项工厂/梯度/二阶及原 alias 定向、实训均未执行。

Qwen runner r4 的无 GPU dummy 覆盖 normal/timeout/异常/SIGINT/另 session后代和训练证据先保存，但独立审查仍发现 SIGTERM 清理和 exact signature 误阻断实用续训问题，正在新 r5 准备，不能运行训练或以 dummy 代替实训。


## 2026-10-28 15:08 +0800 r10 十文件 CPU core 终审通过，准备 CUDA

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 的原维护 Torch core 已一次自然完成：230 collected/called、227 pass、3 skip、fallback0，450.22s / 600s；十文件实际全部执行，新工厂和 autograd_semantics 定向方法不在此计数。三 skip 分别为 CPU 无 CUDA、历史 .long() dtype 和混合类型提升标记，后两项是跳过声明，不自动当作 r10 已复现。receipt e5b751e11bdee16dfaae0c446ceaebafa249f61eb8571311d8cf892195ab0163，result dd98aa4a4875c4a92e09816d59895180ee94428361bdcfc684473c7f9063dd4f，root 独立终审 a5e2b3f1b71906e9e79c3c02b404cac9ae0a6f5d813de1556ae3107c7d6e0941；源码2807文件、core/utils/stamp、环境、进程身份与实际7314 cache核对，6590→7314新增724、无删除或内容改变，14个shim部署文件仅mtime变化。原 focused 23/24 FAIL和独立 FollowRuntime补充诊断分别保留。

恢复后再次 fetch origin/cgq_transformers 为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，main、验收/修复树、r9、r10均包含该提交，无需整合，已有 dirty 保留。共享 GPU 有足够显存但负载繁忙，正确性可在显存足够时进行；不能作为正式性能窗口。下一项审阅绑定同源 CPU 证据的新 CUDA cold final，单次900s并串行启动；CUDA前反向、factory/别名/梯度和实训尚未通过。Qwen 新 runner 和控制器仍 HOLD，dummy 验证中的元数据问题要保留并先修，严格生命周期结果单列，实训数据先保存。


## 2026-10-28 14:10 +0800 FollowRuntime 内部 CPU 补充诊断通过

r10 的原 23/24 Torch focused 收据仍为 FAIL；唯一失败在写入前的对象构造断言，不回填。后续单次 120s 新诊断以内部 swap 构造真实 Torch frontend、placement=-1 的停梯度 holder，且与 native alias 同址；_api_fill(2)/_api_zero() 均保持返回自身、placement、共享地址与 alias 值，真实 CPU dispatch、fallback0。常规 frontend array/empty/ones 均实际发布 CPU placement0。该补充仅证明内部 swap 路径的 CPU 停梯度写入，不证明公开构造、CUDA 或模型训练。receipt 19ebd9cdc92d0f4500311f551fe0993558a1a6370d8b8aeebb08749f53d79b4b、result 5b4a4008a509ba762ae01e616bd9c830643b373f156076a20d3d3b4dfce9d6f3，独立审计 dd32272e6257e81f313d63da2f14ef02f675369c51ff547cdb7a2af0eb38117c；cache 6590 无增删改、终态 SHA 17752785201408dd9b8706831cd87c1eda77b9f7ef25142c498fd5df8424034a。下一步依据 23 项实测与此补充审计绑定原十文件 Torch core；原失败与不在 CORE_FILES 的新增测试单列。CPU core、真实 CUDA、实训和性能仍未验收。


## 2026-10-28 14:07 +0800 r10 Torch CPU 23/24，内部 FollowRuntime 前提失败

修正 selector 的单次 Torch focused 自然完成 331.76s：23 pass、1 fail、0 skip、fallback0。17 项独立常量工厂测试（构造时/后启用梯度、_like 图与存储独立、SGD、二阶梯度）及原 CPU fill/zero 的别名/值/输入和参数梯度回归通过，旧 leaf 回归、独立前端和相关构造器通过。唯一失败是内部 FollowRuntime 测试在调用写入之前，构造的前端张量 placement_backend 为 0 而测试要求 -1；该结果不证明写入失败，也不证明该内部路径通过。原 receipt b0284aa30c8f83e6692cbc360e2152fb2258bc2d27bc9db827e15cd9aa77fab3 保持 CPU_FOCUSED_FAILURE_RETAINED，result 5c290baaff40c21243aaec3c91ac493cd096393e2dafc06cba86759c9a3b30f3。独立保存证据审计 90498970762652b007f87766bcabcb1d2d1a015bb9d0c43ba78f46bd37770bb4，仅采用原真实 23 项与失败边界；actual cache 6590、SHA cf16c05b253a3c632371d8a453863b61b4210e127ff3c3c40fb83679459f4fe9、锁及身份有效。

下项只准备有界真实 holder 构造诊断，区分不可构造与内部写入缺陷；不改原 FAIL 或反复整组。r10 未采用 main，CPU 原十文件 core、真实 CUDA、实训和正式性能仍未完成。Qwen 后继保持 HOLD，需先持久化实际训练/验证证据，再将严格生命周期检查单列，并补异常退出进程清理与 checkpoint51 状态审计。


## 2026-10-28 14:01 +0800 r10 Torch focused：修正 selector 后运行中

首次 Torch final 因旧内部测试名称在收集阶段失败：pytest rc4、0 collected/0 executed，原 receipt dfc5812f94416a45ea05b60c2f56665df07ea310738ddde15ec681769eee667d 保留，不能解释为源码通过或失败。独立审计 2afb9dae163acd7a58305338ab5644b688891c89927462af4542a0eebf7889a1 确认实际 778 cache 及唯一失配名称；源码/产品未变。新 selector successor manifest ffb6e2e8d995949b29851687eeec6a211efbf9a47eb8b55dc89a8b7850e1e272 只修正测试名并绑定该真实终态，单次 cap600 已启动。当前 17 项新工厂测试全部通过，后续 fill/zero 回归在编译 oneDNN，整组仍无终态；不要重启或提前声称通过。进程 controller 1023968、worker 1023974，恢复时仍需核对身份/receipt 和实际锁。CUDA、模型实训与正式性能未验收。


## 2026-10-28 13:58 +0800 r10 原生 JVP 通过，Torch CPU 定向进行中

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 的 CPU cold 385.86s 自然 rc0，227TU、732 cache；receipt 656c802ff699fb0b3224962d417964544a04735e6d11ef492ba212cdc11bb32d，独立 terminal audit 990f9c3f59c856dfcf0367cfc04797fc5cde1aba206631262a8fd15e6f3a3b3e。仅构建身份。继而原 native 多输入 JVP 旧反例 1 pass、0 skip、fallback0，13.71s；receipt 2462b27c692b1714c736c92a8ad27a58cde23906a177d8f3a7e10de2338b87f5，独立审计 6e0c4f7f6b535444a58169cccc0ea6455a90807ef34f61f33ea61503c57e3962，终态 cache 764。此结果仅 native CPU，不外推 Torch 或 CUDA。

已绑定真实 native 终态启动一次 Torch CPU focused（cap600，final manifest b4ad82cc3b617ebc3b2d35212d34d1cb8c4fbc76a5fb7dfaa0d8a728f5395ceb），涵盖 17 新工厂测试和 7 项别名/leaf/前端/构造回归，仍在执行；先核验真实进程与 receipt，勿重启。Qwen 训练后继 20261028-r10-qwen-practical-training-r2-hold 有控制器静态实现但仍 HOLD，尚无 CUDA 产品/功能绑定及动态审计。实用训练和性能显存仍未验收。


## 2026-10-28 13:54 +0800 r10 CPU 冷构建进行中

r10 仅常量工厂叶节点修复、相关回归和内部 FollowRuntime 检查，独立源码指纹 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22；source manifest 20261028-r10-factory-leaf-source-r4/candidate-source-manifest.json SHA 9e059ac7691e39121d432c7fb68e51851d9243c59d32ca53226c84fb1b02514a，root 静态审计 SHA 7c05f2f9607bd442e2fe696d04e10b9005d299dd5d344270557c3e04cbc3aeab。新增 17 项测试在独立 PyTorch 2.6 全通过（oracle-r2.log SHA 41c3aee667fd6fd4d97f0094c8388df337b2aa54bc6c58df49c00c10289a7a65）；Jittor 尚未验证。CPU 单次冷构建包 20261028-r10-factory-leaf-cpu-bootstrap-r4-prepared manifest d45f0c3c4c19c78264aa3622bc7370f11958858d8adf299ec309cce117cd3d10，cap600，controller PID 1009630、child 1009635，仍在实际编译；勿重启，先核验进程身份和 receipt 终态。完成后先独立审计再绑定 native JVP 与 Torch focused，CUDA 和训练未启动。后继 Qwen 草案 20261028-r10-qwen-practical-training-r1-draft 为 HOLD，不具可运行训练控制器；原 strict 失败及 r9 不改。


## 2026-10-28 13:42 +0800 r9 工厂叶节点梯度缺失已复现

新 21 个独立 CPU 进程诊断自然完成（113.36s / 600s，fallback0），receipt SHA 0eb5633396f1683e392dbad25321718997c076152a59826ad593c892d8db44ec，未版本化目录 20261028-r9-fill-zero-grad-diagnostic-r1-prepared/attempt-01/。18 个 torch.ones(..., requires_grad=True) 案例包含无写入/fill/zero 与六种 alias/物化状态，全部 grad=None；无写入时创建即 is_backward_leaf=false、未登记。fill/zero 后可变为 leaf，但仍未登记。3 个 Linear Parameter/SGD 对照的参数梯度全 1、lr0.1 参数逐项减 0.1；相同案例中的 ones 输入梯度均缺失。这是 KI-COMPAT-004 的实际功能缺陷，不能继续描述为仅信息查询；也不能据此推断所有参数漏更新。已有结果仅 CPU，不能外推 CUDA 或模型训练。

源、产品、进程身份、零 fallback、6594 文件终态及锁已独立只读核查；cache-after SHA f1364c252582bdc80f5737717a373ca1427547a17480936cb90b597405856bc7，6586→6594 仅新增 8。新 r10 隔离工作树正在从 r9 构造，拟仅修 Torch 常量工厂输出叶节点并验证真假 leaf、grad、SGD、原 JVP 和真实 CUDA；尚无修复结果，不修改 r9 或覆盖原失败。


## 2026-10-28 13:35 +0800 恢复交接

用户恢复授权，优先真实设备训练、有限梯度和更新、学习曲线、验证指标及功能续训；原 strict FAIL 和预算保留，性能显存独立验收。fresh fetch e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 为各相关 HEAD 祖先，dirty 保留。r9 source 8afd3ed550f5cf113c67c0b4f7c159334781047bfda7e11232c9ab3184e7089d 保存 native 两节点独立终审 2 pass、0 skip、fallback0；2806 源文件、实际产品、740 cache 和精确锁核对。审计未版本化路径 20261028-root-r9-resume-audit-r1/root-native-terminal-audit.json，SHA fb26aee6e3cb6498d57962c6b564f06d6315939ce4318ab15bc1814783a90950。Torch focused final manifest 440072a5b064cbd3b3f5b55fb4114be8b19708a8d369e728068f5f639ecfce21 单次运行约 313.7s 终态：2 fail、7 pass、1 CPU capability skip、fallback0。普通可训练 tensor 写入后 target.grad 为 None，需无写入/写入及别名对照；FollowRuntime 测试在写入前因 torch.Tensor(native) 被构造为 CPU placement 而失败，其测试前提须核对。独立 frontend、RMSNorm4 和 public pow2 通过。原 receipt 5908db0e91b8d8e44c5f01dfbe0b7e0461d3069ab999d9fba0f421206aa60d8a 保持失败；11 条 MKL .o.d 目标为实际观察到的独立 frontend 子进程 PID 982482，合同预期 worker 982201，原 metadata 错误保留。源码/产品及 740→6586 cache 和锁的独立证据审计为 20261028-root-r9-torch-focused-audit-r1/root-torch-focused-evidence-audit.json SHA ad786152422b2c5465380fbccbeeee4dfd9979667953a36d30ad8c401fb4e5a4；仅允许将实际终态作为后续诊断基线，不将测试或原 metadata 改为通过。r9 实用训练和 Day4–5 性能尚未验收，发布 Not complete。

历史实训见 [Day2–5 报告](2026-10-26-torch-compat-release-day2-5.md)，唯一状态源为 [看板](../architecture/refactor-board.md)。


## 2026-10-28 13:10 +0800 用户要求暂停：恢复先审保存native终态

用户明确要求暂时暂停并汇报，线程goal已设 **paused**，发布仍 **Not complete**。未发现可见同UID本任务进程或JIT锁FD，r9精确锁inode无内核记录；4个拒读PID不作全机无进程证明。无活跃子agent，没有测试/构建/模型/性能队列需要续追，也未启动自动后继。原工作树、checkpoint、cache、日志及失败均保留。暂停快照与三文档更新前备份位于未版本化 `20261028-user-pause-1310/`；PAUSE_SNAPSHOT.json SHA `b564c8439e623e300f5e3e9d37def655593bae1b5eb7b37d8a00b0905ae0312b`。

最新候选仍为 `torch-compat-storage-silu-cat-grad-rmsnorm-pow-fill-zero-20261028-r9`，HEAD765f1791/source8afd3ed5，未采用main。CPU cold已通过且不重跑；07:19之后native consumer-first pow两节点也已实际 **2 passed、0 skipped、fallback0**、单次120s内自然rc0，正式receipt `20261028-qwen-fill-zero-r9-focused-cpu-native-r1-prepared/attempt-01-native-focused/receipt.json` SHA `da8ece3048fad10c74056b3bd79e0cc6db558b9d9936a02ea710fcb94f1d1b55`，result b3612b3e、cache-after13ee1d9a（740项，+8无改删）。独立root终态审计未写，Torch前置仍HOLD；不要重跑native、改写执行包或把它当fill_/zero_已通过。

恢复顺序：按原规则重核进程/源码/真实缓存与远端→对上述保存native证据完成独立root审计→复制到新的Torch final，绑定真实740终态及审计→Torch focused600s→原10文件Torch core600s→同源r9 CUDA冷构建及真实定向回归→原完整无hook gate。新增三方法不在core白名单，不能预填233。尚未决定的MKL元数据审阅合同不可默认为通过。完整Qwen仍8,690,528/29,171,712 FAIL，LoRA不启动；局部SiLU/cat/RMSNorm等历史PASS只属于各自source。

T5双seed strict各FAIL771/2088、singlemicro超时且原家族仅余51.29068389162421s；Vision五格FAIL、无合法最终continuous，最新余额310.095694191s，旧1289.681100597838s只是早期快照；BERT任务gap15.425953905%>15%。跨三任务11个原checkpoint目录存在但22个原model/adapter与optimizer文件仍缺。Day4–5没有正式稳定三轮/真实容量边界/峰值验收；r4/r5及后继readiness已超时，不能从旧RUNNING/WAITING文件重启。暂停后的完整目标与所有原门槛不变，详细成果和剩余表见[本日结果](2026-10-26-torch-compat-release-day2-5.md)，状态只维护[唯一看板](../architecture/refactor-board.md)。

## 2026-10-28 07:19 +0800 接续入口：r9 CPUcold已终态，下一项定向回归

r8 fill/zero四失败已复现并保留；新r9 explicit-target-only scope修复与consumer-first pow回归顺序已冻结，HEAD765f1791/source8afd3ed5、manifest67e71aa4，main/r8不改。CPU cold600 `20261028-qwen-fill-zero-r9-cpu-bootstrap-r1-prepared/` 在337.6935s自然rc0，227TU/core1562721b、732cache终态，root13审计a1d1645e全真、组锁空；exec43630已收回，不重跑cold。下一项绑定真实产品的native pow两节点120s→Torch focused10节点600s→维护原10文件Torch core600s；focused原独立前端用例不改，新增CUDA方法在CPU明确skip，不能预填core233或以focused代core。尚无r9数值/梯度或CUDA通过。原r8 core226pass3skip1fail与缓存事件保留，原完整Qwen/LoRA阻断、其他FAIL/预算/缺件与Day4–5门槛不变，发布 **Not complete**。精确路径、SHA与边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 07:02 +0800 接续入口：Torch core失败已确认，修复临时值placement

r8 Torch core终态226pass/3skip/1fail，唯一独立前端Linear weight fill_失败；不得再按旧“下一项Torch core”重复执行。原失败审计、7919缓存终态及MKL四文件重写证据保留。新 `20261028-r8-fill-zero-placement-cpu-r1/` 四项公开CPU fill_/zero_全部复现：目标raw backend0，独立RHS mirror backend-1，device_id均-1，公开cpu/dispatch相同；fallback0、source/core/env有效、缓存无变、组锁空，receipt bb10a303、保存审计b6c84dff。新r9隔离候选正在离线准备显式target-only scope修复与consumer-first测试修订，尚未执行，main/r8不改。下一项审阅r9真实源码身份后串行冷编译和定向CPU别名/梯度/原独立子进程回归，再按同源前置准备CUDA；旧CUDA HOLD包不启动。发布 **Not complete**，原完整Qwen/LoRA阻断、其他FAIL与预算、Day4–5门槛不变。详细路径与边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 06:47 +0800 接续入口：pow old-fail/new-pass已完成，先运行Torch core

当前无本任务框架/JIT进程。old-r7/new-r8 consumer-first控制已自然rc0，62/62/25→0，原native测试顺序的盲点已直接证实；root15×2＋paired4通过。r8最新7155基线为 `20261028-qwen-pow-r7-r8-consumer-first-cpu-r1-prepared/attempt-01-r8/cache-after.json` SHA3ca6a79f，组锁空，不重跑原阶段。native core103pass/45skip/1xfail与focused6经独立保存审计采用，原metadata rc1均保留；consumer-first测试patch仅proposal、未改r8。下一项独立Torch core600s final正在按7155绑定，之后才CUDA900与原完整no-hook；发布 **Not complete**，原FAIL/预算/LoRA阻断保持，具体收据/审计入口见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 06:13 +0800 接续入口：r8 CPUcold已终审，先运行受影响CPU回归

当前无本任务框架/JIT进程；r8 CPUcold的exec15130已自然退出，不能重跑cold。新隔离树为 `/data6/cgq/jittor-lab/worktrees/torch-compat-storage-silu-cat-grad-rmsnorm-pow-20261028-r8`，HEAD765f1791、source `f6751970c2da991ee09e23b7b08f50d8d777b9e916c21a76b2b895daea6879ef`；manifest位于 `20261028-qwen-pow-r8-source-r1/candidate-source-manifest.json`，SHA `7a08b93941b4466eea82267bce1dc1d045fdcc37f23a51ec9da6ab995edd2dd1`。相对r7只改CPU pow返回dtype cast和native/public两条回归，未改CUDA helper或主源。旧控制中CPU原链对物化分支25项差、同PT power公共输入两runtime逐位exact的证据已确认；完整模型因果与发布结论仍未闭环。

CPUcold包为 `20261028-qwen-pow-r8-cpu-bootstrap-r1-prepared/`，manifest979785d3、正式receipt6f9e4639，实际227TU/core4944108e；独立root17终审 `20261028-root-pow-r8-continuation-r1/root-r8-cpu-cold-terminal-audit.json` SHA `fefa46c12d45bf81d4b5428b2f56a7fa95a5e5b220d8f789e58a3f0d06c6fa1c` 全真、组锁空。下一项先收取qwen_meta_repair的实际CPU core执行final并静态审阅，在同core/cache上分别单次native core与Torch core+原mixed四项+新旧pow，新增各600s cap；尚未执行。native必须SHIM0且移除PROJECT_ROOT/RUNTIME_ROOT等shim激活入口，按明确native合同记录环境，不能照搬Torch post-import。CUDA900仅离线HOLD，等真实同源CPU回归再绑定final，首次JIT由root串行放行。旧r7不变，CPUcache816/CUDA1358是r7保存终态，不能混作r8 cache。原r7完整no-hook仍FAIL8,690,528/29,171,712，LoRA不启动，发布 **Not complete**；其他原FAIL、T5/Vision余额及性能门槛见[本日结果](2026-10-26-torch-compat-release-day2-5.md)，不要重置或重跑旧阶段。

## 2026-10-28 05:46 +0800 r7公共pow四侧控制完成，CPU初始化链可复现保存频率

新最小包 `20261028-qwen-r7-public-pow-micro-r1-prepared/` manifest `388c709b978aea0ff81610f2c86832b6099c4bfdc897c5c68e9b7a4fcc9d7d41` 已由root依次单次执行CPU→真实CUDA；两device PT/JT均自然rc0，CPU各2.391/8.577s、CUDA各3.395/36.933s，均在各自新120/180s微预算内，未重置旧模型/T5/Vision预算。CPU receipt `d978831aa55b80ef7b6643d74bb87f35c4b44d8bfd29400145cdf9ab878e824d`，CUDA `2865c203c8ebd2eaa6d0e187c3f05202d3c10bd1e27e920dd2ef437eb11bb220`；双方真实cmdline/PID/start/私有组和worker自身身份均有交叉核对，实际device/fallback0、source/core/env/组锁有效。最新CPUcache808=780+28，CUDA1350=1326+24，旧文件无改删。

root两侧独立22项保存数组审计全部通过：`20261028-root-rmsnorm-r7-continuation-r1/root-r7-pow-micro-cpu-terminal-audit.json` SHA `c032331d84ad17ae72ab3e206e5f14cffe7719ed5526e525d2caf5e070a443a6`，CUDA `root-r7-pow-micro-cuda0-terminal-audit.json` SHA `2d6ec391b19d858559b13dc498c5a8afe8fa32223e997ff800102133d0b00aee`。64项中独立PT CPU inv_freq逐位exact保存PT，JT CPU逐位exact保存JT；两CPU的power仅1项不同，但inv_freq及reciprocal均24项不同。新PT CUDA相对保存PT不同2项、JT CUDA相对保存PT不同1项；同device CUDA两侧power不同4项、inv_freq不同3项。故不能把保存PT频率当作CUDA初始化结果，或直接修改CUDA double-only `_signed_pow`来解释本次保存频率。四值既有reflected-pow数值控制及六项正负/整数/非整数NaN语义均通过；NaN payload位不同不等于有限数值差。

新CPU生成源码可见整数theta与float32 exponent经 `std::pow(int,float)` 产生double的auto局部，后续融合double除法再castfloat32，虽然pow张量声明float32；这是待最小干预验证的中间精度假设。下一项仅离线准备int/float/f32 tensor基底和保存power重建tensor的断融合控制，尚未执行，未修改r7或创建r8。独立审计首次把CPU launch环境直接等同进程环境，因已知CPython增加LC_CTYPE而rc1；失败记录保留，改用冻结expected_process_start精确合同后只重跑NumPy审计，没有重跑四侧数值。fresh fetch仍e5cffe1a为五树祖先，原dirty字节保持；原完整no-hook仍FAIL8,690,528/29,171,712，LoRA禁跑、候选未采用main，其他FAIL/原预算与Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 05:38 +0800 r7首层attention同输入控制有效

新微回放包 `20261028-qwen-r7-first-attention-replay-r1-prepared/` manifest `e739f1bade43ee3a7d6d814b3d98dbcb1476077fcb81dd1a2244352bb9fe849d` 已由root逐阶段单次执行。独立PT/JT均native原概率逐字exact、manual/public概率及pre-o exact、显式BF16 matmul与AMP exact；各在新300s微预算内自然rc0（10.672/33.255s），不是性能结果。正式PT receipt `33d2fd50b529812d14823a056f8954ae86fec32ad9f980a234d77aae3d6caa2a`、JT `ba9f058573a2cd2845fad5533318fcad3acaa2c0a6d1e14830b6a040f7c4f001`；独立saved比较receipt `1fb8dbcc82c8e2ec8b06074c4dba2e9d7ca1dd3580fbfc16954bcff38804c068`、result `d4c86356ba678e5adc619a493766a082cf5be13e39b354b867054af74794de6c` 已在原120s内自然完成。三个exec30264/59089/26655均收回。

root独立26项saved终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-first-attention-replay-terminal-audit.json` SHA `85290495d6cc4c3c0009062462af2d4c00aa7e27daf5ea19053fbeeee3df6b3c` 全真：固定同一cos/sin时，PT/JT全部32数组exact，包括mask、BF16 QK、scale、masked logits、概率和pre-o；只替换PT/JT cos/sin，两侧均产生40个Q/58个repeatedK BF16变化、61个masked-logit变化，并恰重现原1302个概率差（max0.004319220781326294）及466个pre-o差。它证明本首层固定QKV控制的cos/sin影响，不证明完整模型根因或发布通过，不能继续以旧cohort的RoPE无影响结论代替本次结果。

实际CUDA/fallback0、完整JT环境/core231/order/ingredients/source和组锁通过；CUDAcache1326=1306+20、旧文件无改删，终态 SHA `1d7ea7b9d2100ff836cba12de4ca82bf4ff90b97510e05c2b3e7c074a9318c31`。两原launch identity的cmdline因Popen瞬时采样为空，parent Popen argv/PID/start_ticks/private组保留；JT由root存活期另采实际命令（`root-r7-attention-jt-live-identity.json`），PT退出前未补采，该身份观察界限已明列，未回填原件。下一项仅离线准备公共pow→reciprocal的CPU/真实CUDA四侧最小控制，对双方保存inv_freq逐值核对，不假定PT初始化device；静态CUDA double-only `_signed_pow`仅是待证假设，未修改源码或创建r8。原完整no-hook仍FAIL8,690,528/29,171,712；LoRA不启动、候选未采用main，其他FAIL/预算和Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 05:21 +0800 r7独立root38通过，原dtype差清零

root单次独立复算117数组/113有序边界及五probe全部有效，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-layer-capture-terminal-audit.json` SHA `9416389296587a6c103ba8669c22e6967495be834e4716188d9ad4978e73b8de`，38项全真、exec84321自然rc0。实际CUDA/fallback0、完整环境/source/core231/order/ingredients、1306终态cache与组锁有效；这仅允许采用诊断边界，full/nohook/causal/release仍全false。

本次113边界原tensor dtype差异为0（旧r6为六处q/k norm），首层q_norm/k_norm输出、输入、weight及q/k/v投影均对PTexact；不能继续把旧mixed-dtype差写成r7剩余失败。首层attention概率仍不同1302/294912、max_abs0.004319220781326294，o_proj不同1515/196608、max0.001953125；更早RoPE order1仍有1284项微差/max7.539987564086914e-6，首层input_layernorm有FP32微差。新首差不是根因证明。下一项仅离线准备saved r6/r7逐边界对照和首层attention同输入算子回放，区分PT/JT cos-sin单因子及后续计算；两侧本地回放须先exact各自原概率，尚未执行，不先修改新源。旧r56/r58对其原cohort的无影响结论保留，不能未经验证推广至改变dtype后的r7。原完整logits仍FAIL8,690,528/29,171,712，LoRA不启动、候选未采用main，Day2–5仍 **Not complete**；其他FAIL、预算、Day4–5门槛保持。

saved r6/r7/PT逐边界对照也已单次完成，未运行任何框架/模型；包 `20261028-qwen-r6-r7-pt-capture-diff-r2-prepared/` manifest `cb348b2ddf4aca9e7d4526856ce49acb41162778ba29403369bf07a529e45190`，正式receipt `b99d7ebcfabc577c7c730e4e1691462e4643ff538a3cd8bc9673d96f5a4588c2`、comparison `1edd164b6d955b33891f9b5e7b6c0d6c757eef99627bcd44aa4fb394e2ffc5bc`。113边界中六处原dtype改为PT匹配、零新增dtype差；两数组由不exact变全exact（首层q/k norm），零数组级exact退化，但元素级仍有2,594,086项旧exact→新不同，不能称单调改善。第二/三层q/k norm输入已与PT不同，不能以这些不同输入的输出误差孤立归因RMSNorm。sum113含重复底层激活，仅为保存边界exact诊断，不是完整logits容差统计。

## 2026-10-28 05:16 +0800 r7 capture已终态，独立边界复算待验

新r7原113边界capture的worker/comparator均自然rc0（64.618/7.088s），exec89591已收回；正式receipt `cdb04b21cf898066c72c9c81187a5b01e93fb4a049a813f749bbe69e7f980362` 为 `DIAGNOSTIC_CAPTURE_COMPLETE_GATE_UNCHANGED_FAIL`，comparison `8b25c1c290f6c44d140b97a0eb224c857b30dda46f7d12de7c4ab54a980a146b`。原capture/comparator确认五probe逐字复现本次r7 no-hook；尚待独立root38，暂不采纳新首差异解释。

最新CUDAcache1306=1274+32、无改删或非法新增，cache-after SHA `e22439219f48fa12839a9624b764adce0d5d02c17d22cffd805ad8d97cc2f652`；自有组与精确锁均空，未重跑PT或修改候选。下一项只离线绑定真实终态的独立审计final，再由root单次复算117数组/113顺序及五probe、原dtype与float32存储区分。完整no-hook仍FAIL8,690,528/29,171,712，所有发布/LoRA前置门槛与其他FAIL/预算保持，Day2–5仍 **Not complete**。

## 2026-10-28 05:15 +0800 r7原113边界capture单次运行中

root19项启动审查全部通过，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-capture-launch-review.json` SHA `38dd87687ca96c31e094f902f2a6977ebc1b833a57c9426f06a272cd944dcb52`。新包 `20261028-qwen-r7-layer-capture-r1-prepared/` manifest `c9689d9022ee8cdaa4fbb25aa1a3209b666b239a5e0f2b02aeb14020480578d2` 已单次启动（exec89591），原worker1800/comparator120、113边界/117数组/五probe exact合同保持；capture、comparator及controller逐字复用原root38模板，环境/source/products/support绑定实际r7，最新1274缓存exact、精确锁unheld。入场free51296MiB，真实CUDA导入及原worker正常，当前尚无终态，不重复启动。

fresh fetch远端仍 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为五树HEAD祖先，无需整合；main519d0c的99项逐文件dirty字节全部保持。完整no-hook仍FAIL8,690,528/29,171,712，LoRA不启动、候选未采用main。完成后绑定真实终态执行独立root38，只能依据有效剩余边界继续定位；Day2–5仍 **Not complete**，其他FAIL、原预算与Day4–5门槛保持。

## 2026-10-28 05:07 +0800 接续入口：r7原完整no-hook仍FAIL，下一项原边界capture

exec32616已自然退出，禁止重复启动。r7原worker/comparator自然rc0、257.699/3.173s，正式receipt e32c3233为 `FULL_NOHOOK_NUMERICAL_FAILURE_RETAINED`；8,690,528/29,171,712 logits超限、max3.9296875，active3,714,487/max.4375、padding4,976,041；原输入exact/lossPASS。root28独立终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-nohook-terminal-audit.json` SHA `f8ae8cb66c4f9e5c7bbbf245d8ee2d742c0c527f0d7608869278b810e6b1e8e2` 全真，但numerical_gate_passed=false。真实CUDA/fallback0/source/env/products/清理有效，最新cache1274仅增256，组锁空。CPU4/CUDA2/单列unit1的局部PASS保持，不重跑。

纯保存数组对比SHA23517f62确认mixed guard使25,811,571 logits改变，修复旧超限3,643,481同时新增超限2,214,035，净少1,429,446但完整门槛仍FAIL。下一项r7原113边界capture和root38后继正离线准备，要求五probe exact复现本次no-hook、PT113复用、dtype与存盘float32分列；尚未执行。主源和旧失败保持，不采用main、不启动LoRA；发布 **Not complete**，其余FAIL、原预算和正式性能门槛均不变，详见本日结果。

## 2026-10-28 05:01 +0800 接续入口：r7原完整no-hook正在运行，禁止重复启动

原CPU4/CUDA2及单列mock unit1已通过并独立终审。root18项launch审阅 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-nohook-launch-review.json` SHA `a26fa337643e1d26415ed3dbb0635b1a2d605786ff68c6ddf3c4fce6384efda9` 全真后，单次启动新final `20261028-qwen-rmsnorm-r7-nohook-final-r1-prepared/` manifest e46ac379；exec32616，`attempt-01/receipt.json` 当前RUNNING，实际CUDA及原worker已正常开始。不要重复启动或由旧PID推断状态；先查这次receipt和实际进程身份。原29,171,712含padding/.005/.03/1800+120、七数值文件/原validator suffix、实际CUDA2环境仅support前缀、最新1018 cache及组锁已核验，尚无新完整模型数值终态。

完成后沿既有generic28独立复算并终审，再按真实结果决定下一项；不能把局部回归PASS替代本次完整门槛。不采用main、不启动LoRA，发布 **Not complete**；其余FAIL、原预算和Day4–5性能/容量门槛保持，详情见本日结果。

## 2026-10-28 04:53 +0800 接续入口：双cold、原CPU4/CUDA2及unit1通过

新r7 CUDAcold已自然rc0，正式receipt `1d2aa20b…`，231TU=194+37、新core9b4d1a11及完整环境/实际双phase CUDA flags有效；root17独立终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-cuda-cold-terminal-audit.json` SHA `866ecddc21d5fc7140ea118a26d5601c1f4e4dfa269cb777652385979bbd50da` 全真，进程、私有组及精确锁均空。最终包 `20261028-qwen-rmsnorm-r7-cuda-bootstrap-final-r1-prepared/` manifest b7a10651，cap900，仅构建身份，不是数值通过；禁止重复cold/CPU4/PT。

实际产品绑定的新CUDA2包 `20261028-qwen-rmsnorm-r7-original-cuda2-r1-prepared/` manifest75ef462c已单次原pytest两例全过，58.59s、自然rc0、fallback0；receipt974dba80/resultd86bc1fd。原FP16/BF16输出/梯度dtype与module/direct-forward exact均通过，root独立31项审计 `root-r7-cuda2-terminal-audit.json` SHA `4f56a3795bfc91d3b1967a16db30924f31353a4a384bbd8ffa7d78992137eb7a` 全真；16张量saved PT decoded exact只作诊断。单列同dtype原unit1也已单次通过（21.96s、自然rc0、fallback0），unit1包manifest353ec2ad、receiptb1d46c1e/resultefded387；root29项终审 `root-r7-unit1-terminal-audit.json` SHA `aecee948ab7af7c3eb752ead4c82fbf54dc823fa6c73bb91baa8f3972f86a659` 全真，明确mock routing only。最新CUDAcache1018只增20，无改删/holder；不重跑cold/CPU4/CUDA2/PT/unit1。下一项仅原完整no-hook新final，实际前置和最新baseline已交离线绑定，尚未执行。发布 **Not complete**，不采用main、不启动LoRA。CPU证据文档本地提交765e0368，无push；其他原dirty、失败、预算和性能门槛均保留，详情见本日结果。

## 2026-10-28 04:34 +0800 接续入口：r7 CPU四例已通过，下一项CUDA冷构建

r7 CPU cold及root17已实际终态PASS（receipt `df8a4750…`，audit `159c707c…`，227TU），随后实际产品绑定的CPU4包 `20261028-qwen-rmsnorm-r7-original-cpu4-r1-prepared/` manifest `e1584946…` 单次原pytest数值2+mock2全通过，13.17s，fallback0，自然rc0；不要重复cold或CPU4。正式receipt `31e5f5ef…`、result `d5666a8f…`，root独立31项终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-cpu4-terminal-audit.json` SHA `215b435e0f740426382176eab95ebcadf7cd8f333d6eb4e3a38adb747be3758a` 全真。原输出/梯度dtype、route exact和完整身份/环境/source/cache/组锁有效；新CPU cache780只增48，独立PT16张量decoded exact仅诊断。进程已退出，CUDA尚未执行。

下一项仅审阅离线准备的新CUDA-only900 final，绑定上述实际CPUcold/CPU4/root审计，单次冷构建后root17，再原CUDA numeric2（不重复CPU/mock/PT）与单列同dtype邻近unit，最后按原完整no-hook门槛验证。原HOLD包不原地修改，不采用main、不启动LoRA。fresh远端仍e5cffe1a，为五树祖先；main ec12e67a及原99逐文件dirty（聚合视图96）保留。发布 **Not complete**，T5/Vision原余额、BERT gap及Day4–5性能门槛不变；完整绑定见本日结果。

## 2026-10-28 接续入口：旧mixed CUDA失败有效，r7源已隔离修复

原8维旧控制已完成并由root24r2终审实际PASS，`20261028-root-cat-r6-continuation-r3/root-r6-rmsnorm-old-terminal-audit-r2.json` SHA `f74d35c18b902d1d9f93d8ad5320dae06be318d2eb177ba01da8f26c14500112`。独立PT两device/两dtype通过，旧JT CPU通过；旧JT CUDA两dtype只在eval module输出dtype/对应array_equal失败，training及两类grad断言和比较都通过。mock2 ACL失败单列。PT CUDA原LAZY元数据rc1、root终审r1错误预期training失败等原件均保留，不重跑旧控制。最新CPU984/CUDA1430，组/精确锁空。

root仅在新 `torch-compat-storage-silu-cat-grad-rmsnorm-20261028-r7` 相对r6加入module_methods混合dtype守卫和原mixed测试，source15cafdfa、manifest `5e2ce491c5b3742dbf5efff970ea19356803d69258e7e302e2f1b5184fd79346`；完整路径与review见本日结果。04:22 +0800 root14项launch审阅8573a351通过后，CPU600 cold包 `20261028-qwen-rmsnorm-r7-cpu-bootstrap-r1-prepared/` manifest8b38981b已单次启动，当前仍编译，禁止重复；尚无构建终态或数值PASS。先读本次终态并root17审计，再原CPU4；CUDA900与完整no-hook只作离线HOLD，不采用main、不启动LoRA。原完整模型仍FAIL，发布 **Not complete**，其余门槛/原预算保持。

## 2026-10-28 04:11 +0800 mixed旧控制接续：只缺首次JT CUDA和单列unit

发布仍 **Not complete**。原8维mixed控制包manifest88d1b625已单次执行，PT CPU/JT CPU原两dtype×两route均通过；PT CUDA完成36/36公共断言和全部输出/梯度，但最终自动新增CUDA_MODULE_LOADING=LAZY触发完整环境守卫，原自然rc1/FAIL receipt c4327f07不改。root31独立终审 `20261028-root-cat-r6-continuation-r3/root-r6-rmsnorm-old-prior-terminal-audit.json` SHA `ff30a5636cf1c62679247982d9dd9100467b15cad54cffb69f2bb19ca35962e2` 已实际PASS，核验48数组/原fixture/断言、source/core/完整环境唯一区别、cache/组锁，仅接受该独立PT CUDA数值，不给JT增加环境例外。最新CPU964/CUDA1354，下一项metadata后继只首次JT CUDA→分列mock2 unit，不重跑前三侧、不改旧包或源码、不先创建r7。独立capture38通过但原完整模型仍FAIL；其余门槛/预算保持。

## 2026-10-28 03:57 +0800 接续入口：capture 已终审，mixed-dtype 旧控制待执行

发布仍 **Not complete**。root 单次独立capture终审38/38通过，文件 `20261028-root-cat-r6-continuation-r3/root-r6-layer-capture-terminal-audit.json` SHA `9ef576c67018306a1123b6b1b3918ce4e0497b206cd6ac3421fcd6674de8f013`；117数组/113顺序记录及五probe原始bytes有效，cache终态1354且source/env/core/组锁通过。numeric-first RoPE与dtype-first首层q_norm分列；六处q/k norm输出PTfloat32/JT BF16只是可信诊断，不能宣称完整模型因果闭环。r6原no-hook仍10,119,974/29,171,712失败，禁止重复capture/no-hook及启动LoRA。

fresh目标远端仍e5cffe1a，为本次五树HEAD祖先，无需整合；main48dcbd3dd、99项dirty/空index及其他工作全保留。下一项仅审阅原8维MixedRMSNorm的独立PT/旧r6 JT CPU→真实CUDA包，eval/train/module/direct-forward与两条mock unit分别记录，包尚在离线准备；root放行前不执行，不先创建r7。真实旧控制确认后才考虑仅mixed-dtype guard的隔离后继。其余FAIL、原预算与Day4–5准入HOLD保持，细节见本日结果和唯一看板。

## 2026-10-28 03:33 +0800 r6 原完整 no-hook 数值仍 FAIL

Day2–5仍 **Not complete**。r3 元数据修正后，原完整 Qwen BF16 worker/comparator 均自然 rc0（225.725/3.116s），正式receipt `1644a95cb3da74682d98712b61c874857a19a6891fec9e9b1d47f3c58abfb89d` 为 `FULL_NOHOOK_NUMERICAL_FAILURE_RETAINED`。原全部29,171,712含padding logits仍超限10,119,974项，max_abs4.587890625；有效位置4,375,563/22,030,720超限，max_abs0.4375。输入exact、loss满足原门槛，不改变0.005/0.03。root独立28项终审 `356cfd07ce83a178857aef66fc1c7f4c6d6e17fc2f713f7107869b80e9256415`确认实际CUDA/fallback0、source/products/完整环境/cache/清理有效并独立复算一致，但 **numerical_gate_passed=false**；证据有效不等于模型通过。

候选未采用main，LoRA不启动，不重复本次cohort。只读分块比较已确认r4/r6完整NPZ及全部五数组bitwise exact，29,171,712个logits全部相同、修复项与新失败项均0；analysis SHA `014567ebec376f2c01b021e587584c9eb3d25d343f22d7854b549faf77f2e76a`，仓外 `20261028-qwen-r4-r6-nohook-forward-diff-r1-analysis/analysis.json`。因此cat/grad局部修复在本次cohort无可观测前向作用，不能据此关闭模型门槛。r6逐层capture包 `20261028-qwen-r6-layer-capture-r1-prepared/`（manifest `ccbbd038…`）已单次完成，worker/comparator自然rc0、50.757/6.943s；正式receipt `aa3e373e23af600dbfb50173013bcd2d3a8211b7e879817377d36d3b9dd91968` 为 `DIAGNOSTIC_CAPTURE_COMPLETE_GATE_UNCHANGED_FAIL`，仍非模型通过。原r54数值主体不变，PT113边界复用、五probe数组对本次r3逐字exact，cache只加16至1354；独立终审尚待执行，禁止重复启动。观察到embedding及首层q/k/v投影、q/k norm输入和weight exact，但q/k norm输出原dtype为PTfloat32/JT BF16，q_norm374342项不同/max0.1484375、k_norm191884/max2.25；RoPE更早也有差异，不能据边界直接归因模型根因。下一项只准备原 `test_rmsnorm_mixed_dtype.py` 数值主体的独立PT/旧JT CPU→真实CUDA最小控制，先复现混合dtype标准RMSNorm dispatch；没有修改r6或提前采用历史mixed-dtype guard。T5/Vision原余额与禁止重试边界、BERT原gap、跨模型原checkpoint缺口及Day4–5正式稳定三轮/容量/峰值门槛均保持。

精确收据和核验范围见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

原件和容量准入窄复核 `20261028-original11-stat-and-capacity-admission-review-r1/`（manifest `1722585fbabfc01fa021aa585b0da2519f4fcc0dd023265940703b8dc369870d`）按旧audit精确逐项stat：11个checkpoint目录存在，分布BERT5/Qwen LoRA3/T5 3，但22个model或adapter safetensors与optimizer.pt原文件仍全缺，零新增/零stat错误；这不是BERT独有的11份原件。一次GPU只读采样显示10张A800均未达71680MiB free，最大66466MiB、差5214MiB；单次采样不证明稳定性能窗口。未启动waiter/JIT/模型，source/adoption/Qwen-success HOLD独立保留。

## 2026-10-28 r6原cat CPU/CUDA闭环后的续接

原完整no-hook r2已自然rc1终态：receipt a47f1674/entry39848a23在模型前的worker.py line31缺`_hf_trainer_e2e_runner`；没有原probe/NPZ/comparator，不能声称logits结果。root15项失败审计ce429068确认source/products/记录的导入环境、cache1090全delta0和组锁有效，失败后child环境未记录。根因是新包漏原r4唯一PYTHONPATH support前缀，PathFinder反事实已定位原字节模块；下一项仅新r3 metadata后继恢复该键及完整合同并绑定r2正式FAIL/entry/terminal-cache，不重冷编/原14、不改七数值/原validator或门槛。原失败不改写，LoRA保持阻断。

Day2–5仍 **Not complete**。r6/source5f4107c5已原CPU14、维护cat_stack1/grad5及CUDAcold231TU/root17通过；CUDA14新final `20261028-qwen-cat-grad-r6-cuda14-r2-prepared/` manifestcab69124现已149.042s自然rc0、原14/14 PASS/零skip，cat/rotate/梯度/二阶原断言完整执行，实际CUDA0/fallback0。receipt c2dcd738、root18终审cbd67896通过，source/products/env exact，CUDA只新增148允许文件至1090、CPU924不变、组锁空。旧失败和PT原件保持，候选未采用main。下一项只审新原完整no-hook final，必须绑定本次CUDA14真实终态cache/环境及全部分阶段root审计；原全29,171,712含padding/.005/.03/1800+120不变，当前尚无模型结果，不启动LoRA。T5/Vision原预算、BERT原gap/缺checkpoint及Day4–5正式三轮/容量/峰值保持；精确绑定见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 02:57 +0800 r6原CPU14及维护通过后的续接

CUDA cold终态补充：单次CUDA-only900已约423.131s自然rc0，receipt fd36f0bd、root17审计4ee9af94全真，231TU/新core036023be/stampe7d42d42/942文件cache/完整环境有效、组锁空。不要重启构建；下一项仅新产品绑定CUDA14，当前仍无CUDA数值或原完整no-hook结果，LoRA保持阻断。精确root审计为`20261028-root-cat-r6-continuation-r3/root-r6-cuda-cold-terminal-audit.json`。

03:05 +0800现场增量：r6 CUDA-only fresh cold包 `20261028-qwen-cat-grad-r6-cuda-bootstrap-r1-prepared/` manifest9518cb60已root单次串行启动，900s上限、无自动后继，仍在编译。先核验该真实终态和root17，不重复启动；CUDA14/原no-hook模板仍严格HOLD。合法launch审阅为root目录`root-cuda-cold-launch-review-r2.json` SHA34e36129，r1错误要求CPU cap字段的FAIL保留。

Day2–5仍 **Not complete**。r6/source `5f4107c5…` CPU cold227TU自然rc0（receipt451435ee/root17审计f774a8b9），真实产品绑定CPU14新包 `20261028-qwen-cat-grad-r6-cpu14-r2-prepared/` manifest c5bd04b0已14/14 PASS、零skip；原cat/rotate值、dtype、梯度与二阶导数均完整执行。receipt8355e773/root16终审2b795145确认CPU/native dispatch/fallback0、source/products/env/terminal-cache与组锁有效。继承原终态的维护包 `20261028-qwen-cat-grad-r6-cpu-maint-r3-prepared/` manifest33ec687c已原cat_stack1与grad语义5分次通过，receipt e6a8799d/2664d86a、root15/16项审计60c21c3d/f28c771a全真，当前924文件terminal-cache有效。原r5梯度dtype失败保持，r6未采用main。下一项只审新r6 CUDA900 cold→原14，之后才可考虑原完整no-hook。LoRA及其他模型不启动，T5/Vision原预算、BERT原gap与checkpoint缺口、Day4–5正式三轮/容量/峰值门槛保持。fresh远端仍e5cffe1a为六树祖先，原main96dirty/空index保留；精确绑定见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 02:34 +0800 r5 CPU14保留4项梯度dtype失败

Day2–5仍 **Not complete**。r5原CPU14已终态10pass/4fail/0skip，receipt2420d5c9…；mixed training在line70、二阶在line112：FP16/BF16 leaf收到FP32 gradient/curvature。cat前向断言先于这些失败通过，不代表被短路的后续rotate/梯度数值均完成。root14项终审630fa856…确认真实CPU/fallback0、source/build/env/current-cache/组/锁有效；维护CPU1、CUDA cold与原no-hook均未执行，禁止绕过失败进入模型或LoRA。下一项仅新r6移植公开autograd.grad的返回dtype转换独立9行，保留当前storage/version和create_graph；已在隔离r6仅autograd.py应用9行（source5f4107c5/manifest131c2edc），仅静态身份；CPU600 cold已单次启动（包bd67c774），尚无终态/数值证据，CUDA保持HOLD。main/r5冻结修改及原失败保持；其余FAIL/预算/Day4–5门槛不变。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 02:19 +0800 cat旧端终态与r5续接

旧cat独立PT CPU/真实CUDA原14各通过，旧JT两device各10fail/4pass，root终审r2 b5331c68的59检查全真；不能把失败case中的未执行rotate/梯度写为已验。PT CUDA原工具元数据FAIL通过独立审计接受原数值，不改原件、不重跑。root新隔离cat-only r5 source25708e7d/manifest c6e4d2fa已逐文件核验，只改dense concat的autocast amp_reg范围并加入原14测试，main/r4保持。r5 CPU cold227TU已rc0，root17项终审a8bb58fa通过；仅构建身份，无修复数值证据。先核对新CPU600s bootstrap包，CPU原14成功后再分别审CUDA900s及原14；未通过则不运行原完整no-hook，不启动LoRA、不自动retry。全部旧FAIL、T5/Vision原预算与Day4–5正式三轮/容量/峰值保持，详细收据及fresh远端/六树快照见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:57 +0800 原Qwen no-hook已终态FAIL，禁止重复启动

r4原full no-hook receipt `544f7f9d…`、comparison `2fc9e50c…`仍FAIL：10,119,974/29,171,712超原阈值、max_abs4.587890625，有效位置亦FAIL；worker与comparison均自然rc0。root独立终审`a0da4374…`28项全真，source/env/core/cache合格、组/精确锁空，原数值失败有效。新CUDA终态cache只能取该attempt/cache-after；局部SiLU PASS不能替代模型。下一项cat autocast单因子旧端14case×独立PT/JT×CPU/CUDA仅离线准备，root审阅后才逐阶段执行；当前未修cat、未启动LoRA、未采用main、无重试。Day2–5未完成，其他门槛与原预算保持。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:46 +0800 SiLU局部CPU/CUDA通过，原full no-hook待验

Day2–5仍 **Not complete**。r4原SiLU四个23点unit/native-AMP行在CPU及真实CUDA的前向/输入梯度均对独立PT和量化数学exact，nonunit同样exact，FP32无退化；CPU原维护2+邻近1通过，CUDA原维护2+activation1+groupnorm2均通过，零skip/xfail、fallback0，source/products/environment/compile-order exact。CUDA numeric receipt5c0200ff、root39项终审870851c0；CPU终审a1fcde78。旧FAIL不改写、候选未采用main。下一项仅新source/core/terminal-cache绑定的原完整no-hook，当前尚未执行；LoRA不启动，其余FAIL/预算保持。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:40 +0800 CUDA构建已终态，接数值门禁

CUDA cold-stage已自然rc0，约401.89s、receipt a45491fa为PASS_COLD_BUILD_IDENTITY_ONLY；新core b1e8991a/utils24b09003/stamp2e157b0b、231TU/完整环境及source通过，root终审6b87523f、组空/精确锁空。不要重复cold-stage/CPU数值，下一项绑定本次产物和NO_BUILD的CUDA numeric、维护2/activation1/groupnorm2后继；目前仅构建身份通过，不能视为CUDA数值或full no-hook通过。Day2–5仍未完成，详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:33 +0800 CPU数值已通过，接CUDA终态

CPU metadata9c5c57a5、SiLU原4unit精确对拍5620f81d、维护2pass/邻近1pass已完成，root终审a1fcde78见20261028-root-silu-r4-continuation-r1/root-silu-cpu-terminal-audit.json。旧环境误判FAIL保持；CPU不必重跑。root唯一CUDA cold-stage现为20261028-qwen-silu-r4-cuda-bootstrap-successor-r1-prepared/attempt-01-cuda，manifest40a9c54d，900s单次，新cache且禁止自动后继。先核实其终态与产品，再绑定CUDA numeric/维护/邻近后继；当前无CUDA数值或原full no-hook证据，禁止据CPU PASS启动LoRA。Day2–5及其他FAIL/预算保持，详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:24 +0800 CPU构建终态续接

CPU227TU及两次fresh-build审计通过，约344s后工具将公开shim的HOME/TMPDIR重定向误判，receipt485d77df/child00500cf1保留FAIL/rc1；未超时、未执行张量或测试。root14项终审20261028-root-silu-r4-continuation-r1/root-cpu-cold-terminal-audit.json SHA da27fee8核对新core487a8fd8/utils24b09003/stamp2078dac1未漂移，自有组空、精确锁无kernel holder。先以新metadata后继严格验证完整环境转换，NO_BUILD采用该产物，禁止重冷编和改旧FAIL；CPU后继成功才单独审CUDA900s。修复数值/维护/full no-hook尚未执行。fresh远端e5cffe1a为六树祖先，main298bb8422原96dirty/空index保持；其余FAIL、预算及HOLD不变。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 01:10 +0800 SiLU真实旧失败与隔离r4入口

storage r3的lifecycle通过后，先按原审阅完成SiLU-only旧路径独立CPU/真实CUDA。新包`20261028-qwen-silu-current-r3-r1-prepared/` manifest `c3fdde08…`、receipt `6f895f7b…`、root终审 `8aaa4665…`：每device四个原23点unit低精度/native-AMP行PT/math exact，JT forward/input-grad均差；BF16样例4.53125对4.5625。source/core/stamp不变、fallback0/cache无改删/组空，旧失败有效，不是修复PASS。

root新建`torch-compat-storage-silu-20261028-r4`，完整r3复制仅activation.py8行SiLU精度修复和原维护回归新增；source `b3c6c32a…`、`20261028-qwen-silu-r4-source-r1/candidate-source-manifest.json` SHA `ea42f5e7…`。main/r3保持，原96项dirty与空index未变。新bootstrap包`20261028-qwen-silu-r4-bootstrap-r1-prepared/` manifest `c52ba629…`已由root唯一释放CPU600s冷构建，CUDA900s尚未释放；两cap继承原cold stage、无自动重试/后继，源和工具冻结。不得把构建身份PASS当数值支持。下一项先核验CPU终态后单独审CUDA，再采用新products/NO_BUILD完成独立PT继承对拍及维护/邻近测试，只有真正PASS才考虑一次原全部logits no-hook；旧HOLD包不原地修改。Day4–5新审阅确认无正式性能行，T5/Vision预算与BERT失败不变。完整证据见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 00:45 +0800 lifecycle通过后的续接入口

Day2–5仍未完成。storage r3/source`2dad903f…`在未版本化`20261028-qwen-storage-r3-lifecycle-r1/`完成独立PT/JT CPU及真实CUDA有界生命周期验证，正式receipt SHA `5319668de6bce14cfd9e45b02bf4c38295e06813d29d16c0930476945035915f`、manifest `a8ea56ca…`、root终审 `3e1fb752…`。每runtime/device独立/共享exp producer各重复三次，非求导可变常量、无中间holder、物化loss后写、先读梯度均通过原1e-12绝对/相对对拍；JT每device另通过kept旁观不重放及相关reader拒绝/目标不变，八次清理均回到live_vars=0。实际CPU/CUDA dispatch与fallback0，source/core/utils/stamp不变，CPU/CUDA各新增16个允许JIT产物、无改删，四自有组/两个精确锁内核记录空。未重复48/22，未重冷编；main原96项dirty哈希与空index保持。

下一步绑定同source、原完整no-hook worker/独立PT参考/原comparator及已验证CUDA core后，再审阅有界原门禁；不重跑已通过lifecycle，不改padding/阈值，不先跑LoRA。候选仍未采用main，原main完整logits失败保留。T5禁止重试且原预算余51.29068389162421s，Vision无合法最终continuous且余310.095694191s，BERT gap/缺11份PT checkpoint与Day4–5正式稳定三轮/容量/峰值门槛均不变。main本段起点`ef09b6233`，目标远端fresh `e5cffe1a…`为相关HEAD祖先。详细证据见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-28 00:31 +0800 维护22项CPU/CUDA通过后的续接入口

Day2–5仍未完成。隔离storage r3 source`2dad903f…`的原48 CPU与维护22 CPU有效，现同维护文件真实CUDA22/侧也已通过；新包`20261028-qwen-storage-version-r3-cuda22-lock-metadata-successor-r3-prepared/` manifest `a045fc77…`、正式receipt `9f0ca4514d13a30a79af3f9eae0fdb2c47dbf4f5ddf1c406a245f1cf6f3f4f0a`。新JT22继承已通过PT22，同nodeids、0skip/error/failure、66份native CUDA0张量证据、fallback0，不重跑任何已通过部分。采用本次CUDAcore`16a21ef1…`/utils`4eeb093b…`/stamp`93a33c63…`并禁重冷编；真实import快照has_cuda=true/use_cuda1、环境0，source/runtime compile_order/build通过，新增64个JIT产物且既有编译产物无改删，自有组和精确锁内核记录空。原flag工具失败、正常owner记录被误判的metadata-r1、执行前哈希不匹配r2全部保留；最终只区分绑定的单一锁运行元数据，inode/属主/权限仍严格校验。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)及root终审`ccc38102…`。

下一项是两组最小lifecycle验证：多个无holder saved-root、stop-grad常量及重复图释放；compat eager-write下kept旁观图隔离、相关reader拒绝且目标不变。先独立PT/JT CPU，再真实CUDA；Jittor特有kept边界单列，不重跑48/22，不自动扩大矩阵。r3未采用main，完整no-hook logits与LoRA门槛仍未关闭；T5、Vision原预算/禁止重试边界、BERT及Day4–5正式性能门槛不变。main此前三文档提交`4a6988f40`之后runtime源码未变，原96项dirty/index继续保留，不将隔离candidate身份或旧性能source改绑到main新HEAD。

## 2026-10-28 CUDA22导入守卫失败后的续接入口

Day2–5未完成。隔离storage r3 CUDA22新包`70d45275…`已由root唯一串行执行：PT真实CUDA22/22 PASS；JT完成新core/扩展后在安装shim后的flag0守卫失败，零collection/维护测试，receipt `8540056b7993d6be89d78efd35c6048fede386fecf3d0499b3206d5e7dd73c5c`保持。旧child未记录实际has_cuda/use_cuda，不能回填；公开shim安装自动启用CUDA与该工具假设冲突，core/utils合法CUDA父子目录也被旧工具误判。新core`16a21ef1…`的has_cuda1/779源签名/231单元已事后核验，边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)，不替代JT数值证据。组/精确cache lock记录空，无重试或重冷编。

下一步只审离线metadata-only后继：显式采用本次新CUDA core/cache、继承已通过PT22，记录两次import后的真实flag，再单独释放JT维护22。通过后仍须先补多个无holder saved-root/stop-grad/释放及kept隔离的最小lifecycle证据，不能自动采用main或启动完整Qwen/LoRA。CPU原48与维护22 PASS有效；T5单micro超时后不重试/余51.29068389162421s，Vision、BERT和正式性能门槛不变。现场仍以唯一看板及终态收据为准。

## 2026-10-27 Qwen原48与维护22项CPU通过后的续接入口

发布仍未完成。fresh目标远端`e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`为五相关树祖先，无需整合；main起点`8a2e9b74e8c8d8c8cb7091eb829c500c53efe085`、96项原dirty与空index保留。Qwen隔离storage r3 source`2dad903f…`已完成独立PT/JT原48项CPU逐字段对拍，pair receipt `83c8acf15651993820717e5949c9dff84da75821772e3c7634905a31d3b6b927`，48/侧、differences=[]；随后维护22在pytest hook注册失败，零测试，旧overall失败原件保留。只修hook的maintained22-only新后继已真实PASS，manifest`59fc1742…`、receipt `17522991df431ab8003046c6ec74b95408ace1aef8d448bed0ecea06621088e5`：PT/JT各22pass/22CUDA deselected/0skip，同nodeids，CPU/fallback0，source/core/utils/stamp exact，cache仅+20无改删、自有组空。原48不重跑，维护22/device不与原48相加成新矩阵。

下一项是审查已准备的CUDA22包：须精确绑定本次CPU成功receipt、独立CUDA cache/新core、足够显存和root串行JIT释放；只准备不等于执行。采用has_cuda=1/use_cuda=0导入后公开CUDA scope，不叠加T5初始化patch。r3未采用main，lifecycle、原完整no-hook logits及LoRA门槛均OPEN；T5唯一单micro超时后未重试，余额51.29068389162421s，事后独立审计不改正式失败receipt。Vision、BERT与Day4–5正式性能FAIL/OPEN不变；精确绑定与审计路径见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

## 2026-10-27 23:29 +0800 T5终态与Qwen双例CPU修复通过

fresh origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81仍为四相关树祖先；main HEAD c9d9633277ba0465bc9fa06f6ade798e16f34255、99项dirty/空index保留。T5唯一20261027-t5-step51-singlemicro-ordered3-r1已于210s工作deadline超时，receipt a6b3aef207994256dc44a5b977f7ae6820f556fcdbff42e390a6fb6f932901dd；仅preforward签名，无完整NPZ/trace/比较，report集合序列化缺陷未动态触发。收费212.9110761990305s，原1800s累计1748.7093161083758/余51.29068389162421，无重试；root验证冻结bindings exact、自有组与可见holder空，不声称全机holder绝对空。清理后唯一启动Qwen隔离r3的600s/runtime冷CPU双例，PT已PASS；JT冷编完成但数值前build守卫误分类仓内codegen源码，receipt b5293d93ece12a837c73468e14bfd1ef9f281a8820d3e18b40b4d84780106ae4，零JT语义结果；原r2静默错梯度FAIL、原48/CUDA/no-hook/LoRA及所有发布FAIL不变。详细source/manifest和边界见本日结果，不按历史PID发信号。


Qwen新metadata successor只修compile_order分类并显式采用本次r3 core，已PT/JT两例PASS_BOUNDED_CPU_ONLY，receipt da045adbf1cad6979e7cc99605d386c34052fe082d3c74a785d1408da8c0d84b；无holder与保留holder均返回原exp([2,3])梯度，source写后[3,4]、旧loss保持。真实CPU/fallback0，source/core/utils/stamp守卫通过；cache仅新增12、changed/removed0，自有组空。仅局部CPU修复证据，未采用main；原48＋维护22CPU正在离线准备，CUDA/no-hook/LoRA及发布FAIL不变。

## 2026-10-27 22:00 +0800 当前续接入口

Day2–5仍未完成。main765f1791与全部dirty/index保留，fresh目标远端e5cffe1为祖先。T5已定位并在隔离source715081d6修复fetch全局初始化次序与ordered代码生成的KernelIR容器问题；CPU44及真实CUDA44、全部VJP/provider负例、两个32次归约各1hash、fresh41fetch/twice-cleanup全PASS，CUDAreceipt0fb79f7b…、coreb63e552b…、fallback0且组/holder空。这只关闭底层前置，旧模型771/2088FAIL不变。public norm32fresh flag3已rc0/PASS，96份y/dx/dgamma各1hash，receipt8c296ff7…；仍为synthetic upstream的确定性诊断，不是原模型因果/恢复通过。同源flag0控制已完成（receipt524b348c…），y1/dx32/dgamma32hash，flag3各1hash，192份数组和provider实物独立审计通过，公开norm局部干预成立；模型恢复仍未验证。原1800s家族累计1535.7982399093453/余264.2017600906547。Qwen storage r2无holder exp安全探针PT两casePASS，JT已冷编但工具读取torch.__file__在数值前失败，receipt757337b6…保留；元数据r1工具失败保留；r2精确pre/post环境后完成原两例，receipt8d88976f…为SEMANTIC_FAILURE：无holder反向返回写后的[e^3,e^4]，保留holder对照及PT通过，真实CPU/fallback0/core/source未变，r2不得采用；新隔离r3只准备，原alias/no-hook仍FAIL。Vision无重试/余额310.095694191，BERT15.425953905%>15%，Day4–5 r2 NVML工具bootstrap失败保留；修正加载登记的新r3绑定c9d963327/完整08446013…/计算46a8，60s readiness终态超时（12样本/0合格，receipt0c320249…），源码工具exact，未启模型或performance cache；正式三轮/容量/稳定窗口仍未完成。详见[本日结果](2026-10-26-torch-compat-release-day2-5.md)最新增量及原收据。

## 2026-10-27 20:50 +0800 当前续接入口

Day2–5发布仍未完成。主树 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe` 与dirty/index原样保留；fresh fetch的 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`是祖先，无需整合。主树source-files fingerprint `46a8fbb7f15786462a530bb9a2a855a983ced8e796afeae3fe6b80362ebe2a74`仅指1223纳入源清单的文件内容。

T5 ordered-sum候选source `930732a5be98e897ff1dfbe49b184c97aa5130b110346e72b714ad44c9bc5e92`完成隔离真实CPU44例（前一有效source/build绑定）数值、VJP、provider和负例通过，receipt `11ff8c2c2b6bbc1c59cdadc8c36c438d0018a924f34b51ef8e93f693103e5b3a`；CUDA首编初始化在`CUDA enabled`后、调用`sync_all`的`init_use_cuda/setter_use_cuda`路径发生SIGSEGV，receipt `994d36b3d1c6b62cc9d90871399ef63046844dc3dded0086ea5af2783438ebee`，禁止自动重试。含之前CPU工具失败累计原1800s家族532.228476549s、余额1267.771523451s。下一项先离线符号/源码定位该初始化崩溃，再另行审查任何新cohort。

Qwen assign storage alias仍FAIL：r2四类别名全错，r3为7pass/1fail，direct Parameter/detach r4仍未共享存储。assign/meta补丁已在main且未提交，失败保持；新的shared-version隔离patch尚未应用main。新native shared-version与lazy-reader物化方案仍未编译，先审完整覆盖和不支持范围。原完整Qwen no-hook gate在旧source上FAIL，不启动LoRA。

Vision恢复runner timeout且改写一个在先阶段新生成的cublas JIT `.so/.d/.key`；trigger不可由保留原件判定。其continuous阶段无最终report/NPZ，余额310.0956941910554秒，不续跑。Vision BF16离线数组只能指出首个已归档norm标量差，非根因。Day4–5只完成旧source共享负载机械诊断及仓外successor准备；当前三轮、容量边界、稳定窗口与峰值仍无验收证据。BERT任务仍15.425953905%>15%。

下一次续接先查T5真实CUDA初始化栈是否可在冻结源码/receipt内确认，再对Qwen隔离storage patch做静态审查；main最终只在候选确实匹配全dirty source后才可计划验证。Day4–5 successor需要最终全树identity绑定并审阅稳定性守卫，代表B8/S256三轮PT/JT交替；容量只做独立显存规划/单步标定决策，不因B224候选自行启动。所有旧失败、预算、阈值、seed及receipt保留。详细增量在[本日结果](2026-10-26-torch-compat-release-day2-5.md)和[唯一看板](../architecture/refactor-board.md)。

## 2026-10-27 19:27 +0800 当前续接入口

发布未完成，以[看板](../architecture/refactor-board.md)与[本日结果](2026-10-26-torch-compat-release-day2-5.md)最新增量为准。Qwen最终加载guard定向CPU7/CUDA7通过，原no-hook已终态：rc0执行但完整logits FAIL10,338,462/29,171,712、max abs3.8125，有效位置同样超限；无LoRA/训练/自动重试。先用SiLU-only小case证明真实CPU/CUDA值与梯度的旧/新差异，再决定后继。完整CPU仍33pass/4fail/1skip，oneDNN r7静态缺binding不能直接跑。Vision native Lt r2、T5 ordered-sum仅冻结准备，原失败与预算保留；不要从静态检查推导设备或恢复通过。main core独立维护检查运行中，首JIT由root占用；其日志及终态应读取20261027-main-maintenance-guard-r3/core-attempt-01，旧PID不可直接重用。目标远端e5cffe1a仍为main765f1791f祖先，原dirty/index与所有失败/cache/checkpoint保留。

## 2026-10-27 18:42 +0800 当前续接入口

发布未完成，现场以[看板](../architecture/refactor-board.md)与[本日结果](2026-10-26-torch-compat-release-day2-5.md)18:42增量为准。Qwen CPU/CUDA公开加载定向各7pass，完整CPU仍33pass/4fail/1skip；新known-key错误保护待最终source复验，原no-hook还没有新成功收据，LoRA不得据此启动。Vision单候选双fresh1024pair exact，原8候选同期双fresh1024pair均复现旧527差异；隔离native稳定选择修复正在准备，未关闭原五格，原预算余1289.681100598s。T5新primitive已终态STOP调度标签，独立审阅表明实际ownership相同且JT两布局32/8hash、PT各1；不改旧STOP、不重跑改标签，家族余1491.236827937s，无full100。BERT synthetic CUDA三组grad exact，不解释原任务gap；旧r10全201grad已在旧source完成，不重复或改绑。所有旧失败/cache/checkpoint/dirty保留；后继仍需核对新cohort身份、串行JIT和真实receipt。

## 2026-10-27 18:10 +0800 当前续接入口

Day2–5 未完成，唯一状态源仍为[看板](../architecture/refactor-board.md)；新增证据以[本日结果](2026-10-26-torch-compat-release-day2-5.md)18:10节为准。fresh fetch仍为`origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是main `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`祖先；无整合或丢弃本地工作。原main no-hook Qwen cohort已在forward前失败，需先完成assign=True公开语义修复的CPU、真实CUDA及原no-hook gate，再考虑LoRA；不得把隔离候选历史结果转移给当前main。minimal-r2已实证旧语义缺陷，修复后r3仅局部改善，CPU regression-r4因工具链漂移在测试前停止，无pytest结论；r5命中原core后7项6PASS/1FAIL，meta设备分类前置FAIL，full未运行，不能启动CUDA/no-hook。

BERT CUDA-r4前置manifest失败、r5误把显存变化当映射变更失败都已保留；r6仅修工具身份判据，已PT/JT真实CUDA rc0、三组gradient逐位相同、无update，CPU差异在本CUDA案例未重现；旧PID42853已经终态，后继依收据与身份核验，不可重启旧cohort。Vision原absolute-r2-cache replay-r4已终态rc0，同进程32次exact但对历史continuous/fresh仍不同；原4200s家族余1334.947563242s，Lt候选数控制目前只有离线准备，没有已通过干预。GC direct-only CUDA24/24数学比较通过，saved CUDA仍CPU-only拒绝；公开False/owner/RNG/AMP/峰值与原模型门槛均未关闭。详细路径、SHA和后继窗口都在本日结果；继续保持首次JIT串行、不自动失败重试、不复跑full100、不重启旧Day4–5 waiter。

## 2026-10-27 01:40 +0800 当前续接入口

Day2–5 仍未完成，整改状态以[看板](../architecture/refactor-board.md)为准。主树在本次复核为 `integrate/cgq-transformers-native-multicard-target@765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，相对最新 fetch 的 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 领先 17、远端为祖先；工作树有 98 条 dirty/untracked 状态记录，全部保留。本轮未整合其他分支、未 stash/reset/clean，也未启动新测试或 GPU/JIT 任务。终态进程复核未发现活动的 pytest、run_test_suite 或 Jittor compatibility 测试。

刚完成的 `compat/tests/torch/test_torch_compat_module_hooks.py` 全模块回归位于未版本化 `/data6/cgq/jittor-lab/_state/release-day23/20261027-tied-parameter-lookup-full-r1/`：8 collected/executed、8 passed、0 failure/error/skip，JUnit 时间 0.056s、`returncode=0`、actual backend CPU、fallback 0。JUnit SHA256 `d4941b2bb83af82e29259b8f82ded44a95523d50672e3cb7892a80cf83a8c0c0`，raw log SHA256 `4c3e71b7f7537270bd7d27e5515baa3c005ca3645d994d0f42a6bb22a8e9769a`，receipt SHA256 `5774bfca86ce1e12039afd1db0d51ba8a77dca28183dc9d411e232327a9dba9b`。先行的 tied-alias 定向 r3 为 2/2 passed、0 failure/error/skip；JUnit SHA256 `74c2fc925e7ca25c08b3243704f4a62ba43d1b118659316ebfc95cbeba957ef0`。两份 receipt 只记录 CPU/fallback，未绑定 source-content fingerprint，所以这是定向兼容性回归证据，不是 Day2–5 发布门禁。

GC combined-r2 的 CPU paired-math 对拍现已由 root-review 复核通过，细节与边界见[2026-10-26 Day2–5 结果](2026-10-26-torch-compat-release-day2-5.md)的 2026-10-27 00:18 增量；真实 CUDA、owner-safety、其他 Day2–3 FAIL 和 Day4–5 正式性能/显存门槛仍未闭环。后续工作启动前重新核对对应候选身份、进程、cache holder 和 GPU；不得从本次 CPU 模块测试推断设备或发布验收通过。

## 2026-10-27 00:09 +0800 当前继续入口

Day2–5 仍未完成，唯一整改状态源是[看板](../architecture/refactor-board.md)，GC r2 详细证据见[2026-10-26 Day2–5 结果](2026-10-26-torch-compat-release-day2-5.md)的 2026-10-27 增量。主树仍为 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，原 94 项 dirty/untracked 保留；fresh `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 为祖先，未 stash/reset/clean/push 或整合 `origin/2.0-refactor`。

GC combined-r2 frozen source `b6ca987cddb259e67756cdcf540d2a70e891e0be1bf2f988684ad34497389889` 已串行 bootstrap 并完成原 63 项、priorview 22 项和 swap 1 项 CPU 回归，全部无 skip/failure；四个 native saved/direct positive case 通过 worker 内比较。PyTorch 2.6 CPU oracle 已生成独立四场景结果，但 root 尚未运行原阈值 offline comparator。当前候选阶段状态不是发布验收或 owner-safety 通过；无真实 CUDA 数值结论。首项后继是审查该 comparator 收据，只有 paired CPU 比较通过后，另行释放独立 real-CUDA 回归窗口；不得自动追加 runtime。

所有这些 stage receipts、JUnit 和 oracle 输出都绑定 r2 source/tool/core，保存在未版本化 `/data6/cgq/jittor-lab/_state/release-day23/20261026-gc-generic-saved-combined-r2-cpu-acceptance-r1/`。此前 combined-r1 的 priorview 失败及其他 Day2–5 原 FAIL 均保留；Qwen、Vision、T5 strict resume、BERT 15% 任务门槛仍未闭环，Day4–5 正式代表/容量格三轮同步配对、更新与峰值门槛也仍 OPEN。

## 2026-10-26 23:20 +0800 当前继续入口

Day2–5仍未完成；唯一状态源为[看板](../architecture/refactor-board.md)，本轮完整证据见[当日结果](2026-10-26-torch-compat-release-day2-5.md)。fresh fetch目标 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 为六个核验树HEAD祖先；main `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`、94条原dirty/untracked、空index及本轮三份文档更新均保留，未stash/reset/clean/push/整合`origin/2.0-refactor`。结构终态现场 `20261026-main-structure-2250/ROOT_TERMINAL_RECEIPT.json`；本轮live树状态在 `20261026-root-resume-134637/ROOT_REACQUIRE_1513.json`。

主树structure以rc0结束，1396 passed/4 skipped/700.57s；两项skip是未安装Jittor、两项是未安装pytest-xdist，部分entry由日志说明未被本session证明。该门禁没有改变release状态。

T5 public norm唯一诊断rc1/46.376770s，actual cuda:0/fallback0：y 32次exact；dx与dgamma各32种raw hash，离线JIT源码强绑定到竞争atomic reduction，但没有独立launch trace或干预，且dy为synthetic、原内部cotangent缺档。原1800s预算累计227.053411348/余1572.946588652，旧FAIL771/2088不变。数据与机制边界见结果23:20节及未版本化 `20261026-t5-public-norm-backward-r1/PROVIDER-REDUCTION-REVIEW.md`。

BERT r7 CPU limited control已终态、组和cache锁空：source779b、actual mapped core7fee，11 arrays，forward/master/bias grad/weight grad在原限值内，input_grad 9150/61440超限（max abs0.375）。离线raw-array误差覆盖80行，median0.01709/p95 0.078125/RMSE0.03755；显式BF16-rounding FP64 dot与PT 61427/61440 exact并全过原阈值、JT仍9150项失败。只能收窄到BF16 input-gradient contraction/reduction，不能定 cast 或加法顺序根因。一个新的有界CPU中间量诊断仅待离线冻结与root审核；原双seed gap仍15.425953905%>15%，没有自动重试/运行。

GC combined-r1冻结source05512469/tool01615801。bootstrap CPU core27b820b rc0、原63项PASS；priorview22为21pass/1fail，失败是swap调用remat mutation helper误增native saved version后导致saved-value replay拒绝。原失败收据保留且不重跑；combined-r2目前仅离线准备，须重绑新source/tools/artifacts后再单独释放runtime。正向场景、PT oracle、owner-safety、公开False/RNG/AMP/CUDA/峰值均OPEN。

下一步先审combined-r2静态修复和完整冻结清单；若通过，仅新r2候选bootstrap一次，再分别审原63与22（失败停）；BERT input-grad离线分解，T5两个固定归约分开做新候选前置。Day2–3其他原门槛（Qwen、Vision、T5严格resume、BERT任务）仍有FAIL，Day4–5稳定窗口、代表/容量格至少三轮PT/JT同步配对、更新、峰值统计仍OPEN。首次JIT仍串行，不能用共享GPU诊断数字声称性能通过。

## 2026-10-26 22:45 +0800 当前继续入口

Day2–5仍未完成；唯一状态源为[看板](../architecture/refactor-board.md)，完整收据见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。main `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`、94项dirty与空index保留；fresh `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`为八树祖先，未stash/reset/clean/push或整合2.0-refactor。

GC weak-frame r1 source651e1b69/core6adb8518，CPU bootstrap rc0；原63项**63pass/0fail/0skip**，四个positive场景、独立PyTorch 2.6 CPU oracle及48项原阈值paired comparison通过。真实CPU、fallback0、no_fuse0；source/core一致、组/cache holder/锁空。RUNTIME_FINAL_AUDIT `8831db7f`。这不代表prior-DataView adoption修复：该guard+22反例仍未运行；准备中的下一候选必须把r4增量应用到weak r1并重新绑定身份，再bootstrap后运行原63+22共85项。weak r1完整记录见结果22:45节；不据日志存活对象统计宣称释放或内存收益。

T5 fresh-repeat r2确认前向/RNG exact、梯度存在SELF_REPEAT差异，原771/2088与预算1800未改变；新public norm synthetic-dy诊断已离线冻结但尚未运行，输入/档案、synthetic dy、120s含清理协议见结果报告。内部cotangent缺档，不能作原轨迹根因结论，不重训50/full100。

BERT r6于14:27:47UTC因actual mapped core与seed哈希不同在数值前rc1，source779b、原PT oracle及r4/r5/r6失败均保留，无自动重试。归一化ELF/源依赖核查和r7新绑定仅离线进行；mapped-core守卫必须保留，任务gap15.425953905%>15%仍FAIL。Vision原FAIL及预算、Day4–5正式三轮/代表/容量/稳定窗口、Qwen full logits、公开False/RNG/AMP/CUDA/峰值仍OPEN。

## 2026-10-26 22:31 +0800 当前继续入口

Day2–5仍未完成，唯一状态源为[看板](../architecture/refactor-board.md)，收据与完整hash见[本日结果](2026-10-26-torch-compat-release-day2-5.md)顶部。fresh fetch e5cffe1a为八树祖先，无需整合，main765f1791f/94项dirty与空index保留；新现场ROOT_REACQUIRE_1429.json，根盘余71GiB/data6余6TiB，无清理/重置/自动stash/push。

T5 fresh-repeat-r2已终审rc1/SELF_REPEAT_DIFFERS，所有组/cache空：两fresh的404前向/参数/inputs/RNG/context逐字节exact，loss/logits exact，但对continuous有63梯度/34,017元素差异、max5.96e-8，对旧baseline有77梯度/1,385,954元素差异、max7.72e-5。实际CUDA/BF16/fallback0/默认3，仅一micro零update/零新hook-retain；原1800家族累计180.676641110s、余1619.323358890s。权威20261026-t5-fresh-repeat-r2/RUNTIME_REVIEW.json(c7d52640)。先离线准备固定norm输入/synthetic dy的public backward重复诊断，不能因未归档cotangent而声称原轨迹因果；原capture不放行、771/2088与旧baseline FAIL保持，不重训50/full100。

BERT r6已14:27:47UTC rc1、组空/锁空，重编译后actual-mapped-core与旧磁盘seed43f1不符，实际磁盘7fee20dd；仅started、零forward/backward/update，无数值结论。原工具错误文本不证明旧seed是成功primitive anchor。16/16失败终审7d7ea372、completion62acd20c，source779b/旧工具/22PT/r4cache exact，旧r4/r5/r6失败不改。下一方案只离线审构建依赖配置及新core绑定，不去掉mapped guard、不复制cache/自动重试。

GC weak-only已实现且独立审阅，source651e1b69，只有slot.frame改weak和过期检查，原63/strong owner/lease/public False不动，未混priorview r4；18TU+binding语法通过，真实bootstrap/63tests须分别调度与审收据。r4 DataView-adoption guard另有22新CPU反例、sourceb9f2ea67，仍离线。GC原r3 62/63、fresh/no_fuse多1frame、priorview-v3隐藏owner环与两个缺口均保留。BERT任务15.425953905%>15%、Vision原FAIL与余1341.905s、正式性能三轮/真实容量/稳定窗口及公开GC/RNG/AMP/CUDA/峰值OPEN不变。当前先看agent/真实收据后释放唯一JIT窗口，不按任何历史PID或“准备”字段重复启动。

## 2026-10-26 22:15 +0800 当前继续入口

Day2–5仍未完成，精确收据/哈希见[本日结果](2026-10-26-torch-compat-release-day2-5.md)顶部。GC stride-only r3 source2c208246/coredae33a76已原63 tests真实CPU/defaultfusion/fallback0实跑62pass/1fail，rc1/93.221s/组空/source-tools exact；原20stride与4计数下降本轮未现，但lazy-leases最终多1frame。单独原fixture fresh0→1，no_fuse=1诊断也0→1，不能猜融合根因或放行positive。priorview-v3已rc0并实测隐藏owner环：direct释放，saved保留callable/ndarray与(1,1,0)，仅清prior_view即全释放；累计3.719155585/120s。r4只有离线DataView-adoption拒绝修复及21新反例，未runtime、未改原63或main；slot弱frame仍仅设计。

BERT原独立PT CPU终审PASS，r4 JT因public torch.__file__工具假设rc1未模型，r5合法元数据修订后又import jit_utils重载rc3未模型/前向/backward。两失败与原cache保留，r5 source779b/tools a307/core和jit_utils字节exact、组空/锁空。r6仅准备相同r5绝对cache和未改worker的新输出一次重入，先看root调度与真实收据，不再次搬迁cache，不重跑原PT。

T5 fresh-repeat-r1 manifest缺键在模型前rc1，无新数值，44.709051474s已加入原1800家族，累计102.071584954、余1697.928415046。r2 manifest85932450继承原26字段与worker/adapter原字节，root已单独释放一次180秒fresh-repeat；须重新读取live身份、receipt和外部ledger，不能按旧FROZEN快照重复启动。原capture不放行，原baseline77gradient差异及历史771/2088 FAIL保留。main765f1791f原91项和本任务三文档增量保留，未stash/reset/clean/push/整合2.0-refactor；布局通过，结构原证据未重跑。

## 2026-10-26 21:46 +0800 当前继续入口

Day2–5仍未完成，状态以[看板](../architecture/refactor-board.md)为准，直接证据见[本日结果](2026-10-26-torch-compat-release-day2-5.md)顶部。fresh fetch目标e5cffe1a为八树祖先，main765f1791f/原91项与本任务三文档增量均保留、index空，未stash/reset/clean/push或整合2.0-refactor。新现场位于未版本化20261026-root-resume-134637。

GC r2的63 CPU tests已终态39pass/24fail/0skip，rc1/无timeout/组空/source-tools exact。20个stride-rank异常已定位replay把合法空contiguous sentinel交给显式stride setter；4个GC live基线下降暂不归因。独立r3仅改这一调用，source2c208246，原63tests不变，目前离线未runtime，不得把r2失败改写通过。priorview原工具native Var weakref失败、v2 direct-control在numpy物化前location断言失败，两次均未saved；累计2.202631391/120s，原失败保留，owner缺口OPEN。新工具只允许修物化/设备观察顺序，仍需独立终审，不能自动重试旧目录。

BERT r4独立PT CPU已32/32终审PASS/组空；JT CPU control正独占首编，worker3565828/start163146791，原900s上限且无自动后继。先重验其进程与completion，不按快照重复启动。其他agent仅离线准备/只读；T5 singlemicro原rc1/77gradient差异、原57.36253348s计费和余1742.63746652s保持，Vision余1341.905232899s保持。main生产源码未变，未重复原结构门禁。

## 2026-10-26 21:28 +0800 当前继续入口

Day2–5仍未完成，唯一状态源为[看板](../architecture/refactor-board.md)，精确数值与哈希见[本日结果](2026-10-26-torch-compat-release-day2-5.md)顶部。main765f1791f、91原dirty及本任务三文档增量保留；fresh fetch e5cffe1a为相关树祖先，无需整合，未stash/reset/clean/push/整合2.0-refactor，main生产代码未变。

optimizer shared-scalars隔离候选bb1a5fa2已五步63数组baseline bytes exact/PT原门槛PASS，三步scalar72→30；Torch19/native34均PASS/0skip，source/core/fallback0/组空，权威20261026-adamw-shared-scalars-r1/ROOT_FOCUSED_TERMINAL_AUDIT.json。未合main、无模型热态/峰值收益结论，正式三轮/代表/容量/稳定窗口仍OPEN。

T5 singlemicro r2 manifest1fc8baf0、原404 wrapper/原Trainer/GA2/RNG已实跑，baseline21:25:28 rc1：原404动态forward exact、257params/inputs/RNG检查通过、仅一micro零update，但77个gradient字段不exact。source efe22515/core6dfe0645及退出守卫通过、组空，57.36253348s计费，同一1800余1742.63746652。**不运行capture、不重试baseline、不改参考/预算**；原工具、FAIL和旧771/2088保持，离线三方审计正在进行，不猜具体反向算子。

BERT r2公共Linear CUDA复现同AMP64/no_grad FP32、grad-enabled BF16；原BERT只知non-BF16不可补造dtype。三文件779b修复及r3工具离线冻结；r4仅工具改为按device共享且顺序用独立correctness cache以减少重复core，CPU/CUDA/PT及原数值/零skip/设备守卫尚未实跑。原BERT15.425953905%任务FAIL和two-update零更新失败保留。

GC r2已离线冻结source a380d0c6/tool648bc5a2，原40+新23tests未跑；先.data view再adopt的静态GC隐藏环仍阻断，独立复现准备好，不能以63PASS验owner安全。当前唯一JIT窗口只给20261026-gc-generic-saved-r2/bootstrap一次900s，worker3558549/start163047387；先查真实process/receipt/cache，不能按快照重复启动。bootstrap通过后root审core/source/组空再单独放行priorview120s，不自动tests/positive/PT/复现或rc3重试。公开False/RNG/AMP/CUDA/defaultfusion/峰值OPEN。Vision原FAIL及累计/余1341.905232899s保持，无新模型。

## 2026-10-26 20:50 +0800 当前继续入口

Day2–5仍未完成，唯一状态源为[看板](../architecture/refactor-board.md)，精确数字/哈希与边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)顶部。重新fetch目标e5cffe1a为四树祖先，main起始765f1791f/source062c8e11、91项原dirty/index空保留。本次BERT队列已失败终态并释放main冻结，不再按下面20:08旧快照重启任何进程。

T5默认3两侧已完成，2803数组119个梯度不同、404动态事件exact，protocol/参数签名/源码守卫通过；旧771/2088 FAIL不变。仓外 `20261026-t5-backward-offline-r1/` 已核实checkpoint50全部257参数及完整inputs/RNG可直接准备单micro回放；先原Trainer baseline证明257梯度与两种loss exact，否则停止，真实收据root审阅后才捕获最小norm反向，不能手写loss缩放/重跑50步。新两阶段总1800秒、各900含清理，仅准备不运行。

BERT新prepared-765f1791f的PT两更新rc0/17终审PASS；JT在20:30:51首次fixed-eval的layer0 query BF16 guard rc1，尚无backward/update/配对比较，实际core99959ad7/source守卫通过、组空、queue main_source_freeze_active=false。原dtype未保存不可补造；静态compute_dtype_name漏BF16位64，最小公开Linear CUDA两forward工具正在复核，无修复或重试。原15.425953905%任务FAIL和所有scheduler/group字段保留。

Vision v5两fresh各32与1024cross全部exact，实际CUDA/fallback0/core128db287，旧fresh仍527元素不同、原769字段FAIL1和五格FAIL不变。独立profile模型rc0，但FileName附带shapes注释使sources=[]而postcheck FAIL；保留失败，不重跑。root离线解析59份源/421op，选中mea_fwd forward的batch32/1模板明确、forward无atomicAdd，同文件backward原子定义不是前向证据。最终预算2858.094767101/4200s、余1341.905232899s。权威新终审为 `20261026-root-resume-1231/VISION_V5_TERMINAL_AUDIT.json`，后继未分配。

当前串行首编窗口仅给 `20261026-adamw-scalar-minimal-r1/` JT tiny baseline，supervisor3525845/startticks162776456，20:42:30启动、900s单次；先看completion/receipt及实际PID，勿复制启动。PT五步63数组已rc0；JT新cache冷编译尚无完整数值/8scalar-launch结论，正式性能/容量/三轮门槛不变。baseline-tree源码保持062c8e11，不随main文档提交改动。GC generic候选离线18TU及当前生成binding语法通过，source96e6cc55/toolbb31a005，40CPU tests和48数学对照未运行；public False/RNG/AMP/CUDA/默认fusion/显存仍OPEN。Qwen既有独立候选通过范围维持原adjudication，不外推main。

## 2026-10-26 20:08 +0800 当前继续入口

Day2–5仍未完成，唯一状态源仍为[看板](../architecture/refactor-board.md)，完整数字与证据边界见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。本次重新fetch目标e5cffe1a，所查相关树均含该历史，无需整合；main HEAD8eb88dbde、91项原dirty和全部历史FAIL保留，没有push或整合2.0-refactor。以下旧PID/“未运行”均是历史快照。

T5 r2受影响原生CPU已15pass/0skip/rc0，actual CPU/core4d116d9d/source efe22515，receipt SHA e4dea024…21fe560与此前CUDA17pass/默认3局部32重复共同满足局部前置。当前实际运行的是 `20261026-t5-ordered-mean-r2-1854/step51-default3-prepared-r1/`：19:49:43启动，supervisor PID3475041/startticks162459728，worker3475053/startticks162459887；默认3、原seed821/GA2/BF16/名义100步schedule，每段3600秒，仅捕获step51，不是full100。20:07已实际至少21次训练更新，loss/grad_norm有限、LR按原schedule，initial probe在continuous-output/而非capture/，原路径检查已纠正；仍无step51/checkpoint50，不等于恢复通过。resume仅在连续rc0/passed/trace/合法checkpoint50后启动，无自动重试。**先重查实时process、execution、capture与cache，不再启动相同campaign，也不并行首编**；两侧成功才离线比较，并显式审查protocol/array/event/signature，comparator exit0不证明数值通过。仅观察既有campaign的finalizer已启动（PID3497250/startticks162559854，最晚13:52:43 UTC）；无框架/新模型/JIT/重试，双终态合法且组空才离线比较，权威终审将写campaign/root-final-audit.json，不能提前按该文件名假定完成。旧771/2088 strict FAIL保持，未合入main。

GC r4已真实私有CPU17pass/0skip，positive/独立PT/9解析比较PASS，原r3与add/sum策略FAIL保留；source997d1acc/core63aac1cf。普通无remat线性图零二阶不变，小图释放6allocation/48B、1replay/3reads不是完整峰值节省。公共False/RNG/AMP/CUDA/defaultfusion/模型显存仍OPEN。Vision v4新core128db287/jitutils6064ff已bootstrap PASS，但repeat-a在模型JIT180s bound超时，181.116s/rc-15，无完整eval65或新数值结论；原累计2756.63591191615/4200s，余1443.3640880838502s不重置，所有自有组空/cache无holder。v5在`20261026-vision-eval-replay-v5/`已冻结离线准备、11checks/verify-only通过，无framework/JIT/模型/队列；manifest d49a88f4，复用同cache/core，不重做bootstrap，首repeat-a900s、repeat-b180s/profile180s及清理reserve均包含在原预算。当前partial cache只读归档实测收费0.745608919s，原累计现2757.381520835/4200s，余1442.618479165s，仍未执行，须先确认T5终态/无holder并独立审阅每阶段收据；原769字段FAIL1及五格FAIL不变。

性能r6仍是共享1×8单轮机械pipeline PASS，正式三轮/代表/容量/稳定窗口未过。新增离线trace归因核验了optimizer.step的4728个4B scalar kernel（197×3×8，占kernel数59.14%），成为最小复现目标，尚无A/B因果或性能修复通过。证据 `20261026-root-resume-1155/R6_OFFLINE_ATTRIBUTION.json`，旧r6身份不重绑。BERT两update工具33离线checks及独立hook只读审查完成但无runtime；保留scheduler/group字段差异，原15.425953905% FAIL不变。本次文档提交使旧prepared-8eb88dbde完整source绑定失效，执行前须保留旧准备另建新绑定，不自动排队。Qwen既有独立候选范围见本日15:26节，公共GC仍OPEN。

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

## 2026-10-29 task-scoped handoff

The isolated r12 patch and its static source/index audit are retained under
/data6/cgq/jittor-lab/_state/release-day23/20261029-headonly-r12-startup-regression-candidate-r1-hold/.
The r12 startup test passed with r10 actual CUDA core and r12 test/helper
overlay; a complete r12 cold build was not performed. Main df45 full100 and
fresh-process 50-to-100 continuation are separate source-identity results:
603 final model/moment tensors differ from continuous training, with cause
unresolved. The Day2–5 report records metrics, exact identity and the retained
original failure. Do not treat the short synthetic CPU run or the mixed-source
node as a complete release, GPU-training or convergence gate.

## 2026-10-30 Handoff: bidirectional cross-CUDA `Tensor.to`

The main-tree test now covers cuda:0→cuda:1 and cuda:1→cuda:0. Both cases passed with real ownership of two CUDA UUIDs, finite expected values/gradient, correct placement, and fallback count 0. Source HEAD is `387a096ef85f44dd9d097bfaf78d5a36176abff5`; source fingerprint is `95eb54ba17b771ece4ed585c2036f19565075b66c858e599972fbee011f76558`. Full details and audit hashes are recorded in `refactor-wip/results/2026-10-26-torch-compat-release-day2-5.md`. Run package: `/data6/cgq/jittor-lab/_state/release-day23/20261030-torch-cross-cuda-bidirectional-r2-hold`.

The r1 isolated worktree preflight failure is preserved: the worktree omitted the main tree's dirty source overlays and pytest did not start. The r2 main-tree-bound run and independent terminal audit passed. The historical no-auto-peer fifth-case failure and three cases not run stay as mechanism evidence for the older direct-copy candidate; the current production host-staging `Tensor.to` path has passed both ordinal directions, without establishing the old failure's root cause. Continue with remaining functional Day2–5 gaps; defer Day4–5 timing comparisons until a stable paired GPU window is available. Keep existing failures and dirty state intact.

## 2026-10-30 Live window and cache incident

A read-only CUDA API introspection through the shim-enabled environment unexpectedly invoked Jittor startup JIT. Process group 2707825 and its compiler children were verified against the exact command and dedicated session, then terminated; no files were removed. Preserve `/home/cgq/.cache/jittor/torch-shim/cgq-92dab87ea3805bdf/jittor_cache` as a potentially partial, unverified build artifact. It is not a validated core, test result, or performance result.

A five-sample read-only window probe from 11:21:28 to 11:21:49 UTC found the Day4–5 target UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` at 100% utilization throughout and 17,993–18,711 MiB free. No measurement workload started. The existing performance gate remains open.
