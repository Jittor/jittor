# 2026-10-21 工作区快照与本机接续交接

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

## 2026-10-28 06:40 +0800 接续入口：native core已走完，先做pow消费者优先控制

当前无本任务框架/JIT进程。r8 native实际103pass/45skip/1xfail/fallback0，worker/controller均因metadata遗漏而rc1，receipt d2417388保留；root21保存审计20bee571已按源码确认唯一OPENBLAS键与12标准产物并采用实际执行，不重跑。7143基线为 `20261028-qwen-pow-r8-cpu-core-r1-prepared/attempt-01-native-core/cache-after.json` SHA8f357097；原文件无改删、组/JIT/oneDNN锁空。focused6沿root19采用且原失败保留。新native pow测试的物化顺序存在盲点，先root串行old-r7/new-r8 consumer-first CPU微控制，每侧新120s，不改源；然后用其实际终态绑定独立Torch core600s，再CUDA900及原完整no-hook。原FAIL/预算/LoRA阻断和发布 **Not complete** 保持，精确入口见[本日结果](2026-10-26-torch-compat-release-day2-5.md)。

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
