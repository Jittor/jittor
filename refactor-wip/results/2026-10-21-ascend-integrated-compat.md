# 新 910B3 多库综合适配记录

本记录对应[综合计划](../testing/2026-10-21-ascend-integrated-compat-plan.md)，唯一活动状态源为[整改看板](../architecture/refactor-board.md)的 ASC 任务。前半部分保留 2026-10-21 的 MAIN-r28/123 历史验收；后续各 cohort 按各自冻结源码区分，不相互外推。R65 真实 Qwen3-0.6B BF16 LoRA 对拍与 R68 最终主树 FSDP2 双卡 Linear 对拍均绑定提交 `9390fa0e84de651e7e1b01de4d142e04d4333f8b`、3336 文件源清单 SHA256 `c00986fb31e15e18860acf60e2ee0059c7dc63afb90ef0a286131d9fc7eb0f9a`；旧 R64/R63 仍只对其原冻结源码有效。下文保留原始失败和未运行项。

> **截至2026-10-26 R250整合：** q008真实Qwen六NPU阶段及两侧各71项strict0恢复仍成立，跨runtime原比较仍FAIL124/272。其272条已保存policy记录均为True，不能沿用旧轮JT False标签。source006的CPU精度阶段61P/3S、Native core103P/45S/1xfail、Torch core238P/1S均实际完成；完整structure先有收集失败，再因监护取消，首F已定位outer subreaper持有未reap的own zombie，未授完整门禁资格。R249/source004原252数组仍44F。R248两类真实梯度反例已取得；R250现已把逐edge类型、非loss hook和单次sum-to修复及维护回归写入主树，但新core尚未编译，不能借8725/旧源通过。无新源NPU或性能通过结论。

| 本轮验收 | 实际结论与源码范围 |
| --- | --- |
| 用户环境 | cgq 私有 Conda/Python3.11.17、CANN8.5.0、原生 Torch2.7.1/torch_npu2.7.1.post2 实卡 smoke；项目 dev 与独立 oracle 分离，安装目标均在私有目录 |
| 历史MAIN-r28回归 | source123 完整 structure 1430P/2 CUDA skips、layout P；1432 不同 case 与 source121 逐 node 相同；97.py+1.pyi AST3.7 P，既定 Ruff 相对 HEAD 新增0、减少12 |
| R190 CPU提交门禁 | Native core103P/45S/1xfail、Torch core238P/1S，合计341P/46S/1xfail；完整tests/structure连同Torch core共1794P/3S，637.98秒、rc0；layout与internal links通过。梯度管理另CPU7P/NPU7P；不授予完整模型恢复资格 |
| R170 CPU Core/structure | Core：Native 103 passed/45 skipped/1 xfailed，Torch 238 passed/1 skipped，combined 341 passed/46 skipped/1 xfailed，rc=0；额外 tests/structure：1794 passed/3 skipped，rc=0；layout通过。均为CPU门禁，不更新任何NPU模型结论 |
| 八模型及综合 Trainer | 各自 source65 等冻结证据上，八模型完整输出/271 参数+2输入全梯度；Trainer4updates/8micro、61跨runtime检查；连续与2+2恢复216完整状态及轨迹 exact，历史fixture为dropout0顺序；R51/R56另对dropout=0.15的tiny BERT LoRA做Native/Jittor各自fresh-process恢复 |
| 共享真实权重推理 | 0.6B既有冻结源码BF16完整推理P；source121的1.7B与4B各3输入/15完整数组/58,343,424 logits bitexact、cached/uncached greedy8 exact、actual NPU/fallback0；不是123重跑 |
| 最终主树 AMP | source123 no-loader baseline 10P/0skip；source ee0146… 上 R43 完成 BF16/FP16 × FP32/低精度输入 × Linear/LoRA × 2步共168阵列/5760元素 Native/Jittor 对拍：168/168 bit-exact、最大差0、预算未改。修正共同显式cast参考后，source manifest r2 正式单测 Jittor 10/10通过、Native 8通过/2项Jittor-only跳过 |
| 真实0.6B BF16 LoRA | q008绑定d193/3351，Native及JT各自4updates/2+2 fresh恢复六阶段实卡完成，各71检查strict0；跨runtime原比较仍FAIL124/272。R65/R182/R213及旧R183/R185保持各自历史源码和协议边界 |
| FSDP2 | q008 d193/3351两runtime各自训练恢复通过、跨runtime仍FAIL；R229旧源nested FP32通过、两BF16失败。source004的CPU混合精度44P，252数组数值仍44F；R250逐edge/hook/sum-to修复已整合但新core和数学待验。最终主树双卡nested、公共状态布局及重计算未验收 |
| 速度/显存 | NPU未验收。1.7B配对流程仅18交替calls后停止，generation steady/decode未完成、rc-2/acceptedfalse；4Bperf未运行，CPU控制器资格不能替代NPU性能 |

尚未达到全部适配退出条件。R65在提交 `9390fa0e84de651e7e1b01de4d142e04d4333f8b` 的3336文件清单上完成真实Qwen3-0.6B BF16 LoRA Native/Jittor full、prefix与fresh-process恢复，跨runtime 2187项比较无失败/无超预算，两侧各自恢复状态933项exact、梯度分段各1136项全通过，Jittor fallback为0；该协议固定顺序且Native deterministic为true、Jittor策略为UNIMPLEMENTED_FALSE，不能宣称两侧随机流或确定性策略匹配。R68在相同旧提交和清单上通过双卡Linear FSDP2 fixture。R130则绑定当前提交 `f5af805ab30aced4aefc214df257a0b575ca36a2`、3336 tracked files 和 source manifest SHA256 `31a590acb78ffcedd4de62f9bd8f84e5f84870a5506c9c046445308b5db51d9e`，三种 compute/input dtype 共126项比较零 mismatch、最大绝对差1.49e-8，两个rank的Jittor fallback增量为0，物理NPU映射为3/6。R130只验证Linear FSDP2输入叶梯度、参数梯度和SGD更新，不能外推到真实Qwen3反向、nested拓扑或恢复。R67的初始化前路径断言失败仍保留且已在R68修复；R68不覆盖真实模型、复杂分片/视图拓扑、恢复或重计算。R169历史更新差异已在R182限定单步协议中通过复验；还需完成最终主树多步完整轨迹及恢复、复验R71 nested child+root HCCL，验证activation checkpoint真实重计算及随机sampler、多worker/generator fresh-process恢复，并处理deterministic策略不匹配；把常用Trainer、Accelerate、Tokenizers、PEFT、safetensors的数据处理、训练包装和保存重载组合映射到当前源码。只对正确性通过的对应工作负载测吞吐、延迟、峰值显存和稳定性。当前没有合格的NPU速度/显存结论。raw CPU core四文件另有46P/9S/1个原样test_concat_random的300秒超时，未删case、缩规模或放宽超时，不称全部core通过。模型与fixture、性能与功能、旧源与最终源均不混用。

## 源码与环境

初始同步目标为 origin/wip/cgq-day23-fix-20261021，HEAD与当时已核验远端均为 305c2f0866f163b0549d47b2e4264c5d41a86385；本任务分支为 codex/ascend-integrated-compat-20261021。按用户要求，分支历史已有两次 origin/2.0-refactor 合并：a962af0119a6b1c6d519948bc605dc1023c63bc2（当时tip 897d9cc61）和 ea3c7db80f7761a6f1632d36472e08d848f120b7（后续六项修复，第二父提交 9da9be98bc543eacef07532471dd0fecc4205550）；当前 HEAD 的 ancestry 包含 origin/wip 基线及该已核验 2.0-refactor tip，因此没有重复造合并提交。2026-10-22 新 fetch 在连接 github.com:443 后 129349ms 超时，独立 ls-remote 亦无响应后经核验终止；这些失败不证明远端最新，当次只能确认分支包含此前验证到的 9da9be98。详细同步记录见 LAB handoff/remote-sync.json。

主机为openEuler22.03 LTS-SP4/aarch64、GCC10.3.1、Python3.11.17、8×Ascend910B3（每卡64 GiB）、驱动25.5.1。用户目录CANN Toolkit/910b ops8.5.0与原生PyTorch2.7.1/torch_npu2.7.1.post2的既有独立环境已完成矩阵/反向smoke；这只证明原生供应商栈，不证明Jittor适配。未改系统驱动/公共Toolkit或其他用户任务。

项目环境均在 `$JITTOR_LAB_ROOT/_state/ascend-compat/envs/`：dev使用Jittor独立shim，oracle使用真实Torch2.7.1+cpu/torch_npu2.7.1.post2，oracle-cpu是另建的真实Torch2.7.1+cpu环境且完全不含torch_npu。该ARM64 Torch wheel的版本后缀为+cpu，NPU能力由实际torch_npu扩展提供。oracle隔离检查已确认无Jittor/shim distribution、未加载Jittor、torch/torch_npu文件均来自oracle，限制一张物理卡时枚举为1。

| 依赖 | 两侧固定值/差异 |
| --- | --- |
| Python / NumPy | 3.11.17 / 1.26.4 |
| Transformers / Tokenizers | 4.56.2 / 0.22.2 |
| safetensors / PEFT / Accelerate | 0.8.0 / 0.17.1 / 1.10.1 |
| setuptools | dev83.0.0（根lock），oracle75.8.2（独立供应商栈） |
| Torch | dev是2.11.0 API声明的Jittor facade，oracle是真实2.7.1；不能据metadata将dev当作原生Torch |

两侧pip dependency check通过。uv0.13.4的 `pip check` 对dev部署的torch/flash-attn facade metadata报告缺少WHEEL；现有deploy仅生成METADATA/top_level，已单独记录，不能将其表述为原生wheel完整性通过。oracle无该问题。完整freeze与metadata/isolation JSON在仓外，dev不含torch-npu。

根 `uv --locked` 在验证全项目时因Python3.9/CUDA可选依赖的跨版本解析失败；保留原 `pyproject.toml/uv.lock` 字节，使用 `uv export --frozen` 导出原锁的版本/哈希，经镜像安装。下游独立解析最初漂移filelock/iniconfig/pluggy/PyYAML/tqdm共五项，随后恢复根lock，并把oracle相同项对齐；未把无约束安装结果冒充lock同步。安装日志、原失败和修复日志均保留。

## 初始执行与候选修复

CPU首次冷构建使用 `JT_BACKEND=cpu`、新JITTOR_HOME/cache_name、DISABLE_MULTIPROCESSING=1、use_parallel_op_compiler=0、MPI禁用、BLAS/OMP2线程与4核亲和，原生selftest已完成；随后串行完成首次ACL编译和两个矩阵gate，详见下节。

隔离工作树经修前host复现形成四批候选，现已统一整合38个代码/测试/技能文件且尚未完成实卡验收：原生ACL逐设备RNG保存/恢复；NPU真实discovery/device/sync/memory/AMP接口；生态NPU迁移与完整placement守卫、Trainer/resume入口及版本限定的私有检测；WORLD集体通信委托native ProcessGroup和NPU DeviceMesh/HCCL发布。纯host复现先保存原失败，再修改同一production owner并复验。host stub通过只证明路由/错误合同，不能作为设备计算、BF16、HCCL或FSDP2支持证据。

共享资产只读核实Qwen3-0.6B/1.7B/4B等config、tokenizer与全部safetensors shard；第一档可直接复用，不复制大权重。BERT/GPT2/T5/ViT/Whisper等先用固定tiny配置。初始文件系统约15 TiB空闲，本轮环境/代码用量很小，无需清理已有资产。

## 仓外证据索引

以下均相对 `$JITTOR_LAB_ROOT`，内容随新cohort追加，旧失败不覆盖。

| 证据 | 位置 |
| --- | --- |
| 指定远端同步 | `_state/source-sync-20261021.json` |
| 环境bootstrap和freeze | `_state/ascend-compat/setup-envs.sh`、`dev-requirements.lock.txt`、`oracle-requirements.lock.txt` |
| uv锁解析失败/镜像安装 | `ascend-compat/logs/setup-envs-r1.log`、`setup-envs-r2.log`、`_state/ascend-compat/repair-lock-overlay.log` |
| 依赖与隔离 | `_state/ascend-compat/dev-metadata.json`、`oracle-isolation.json` |
| CPU冷构建 | `_state/ascend-compat/runs/bootstrap-cpu-r1/execution.json`、`stdout.log` |
| NPU API修前/host路由 | `_state/ascend-compat/device-api-before/` |
| 生态迁移/守卫修前与host复验 | `_state/ascend-compat/eco-contract-repro-before.json`、`eco-contract-after.json` |
| 分布式WORLD路由修前/host复验 | `_state/ascend-compat/distributed/host-routing/` |

后续实际结果在本记录追加明确cohort、源码fingerprint、完整执行数与失败边界；最终不以启动、mock或部分tensor通过替代整个矩阵验收。

## CPU首次构建结果

`bootstrap-cpu-r1` 于本机CPU完成新cache串行构建和原生selftest，退出码0，耗时726.55秒。实际forward为(1,4,9)、gradient为(2,4,6)，13个公开子包导入通过，三步卷积+BatchNorm+SGD训练loss为0.32825479、0.20825621、0.10019163，最大参数变化0.128。该结果不代替Torch生态对拍。日志SHA256为 `f5522d4f5035e95e58ea5fd34be454bb1d37e647d124057b9dbba9a499bff3d4`。

## ACL首次构建与原生矩阵前置

`bootstrap-acl-native-sync-r1` 在可见单卡、sync_run=1、fallback=error配置下完成CANN8.5首次串行构建与两个原生gate，退出码0，275个核心/ACL编译项耗时933.11秒，整个进程992.73秒。float32 mapped matmul、2D/转置/3D batch/4D attention/广播堆叠矩阵正反向全部通过，实际tensor/grad placement在device、fallback守卫通过，2项实际执行、0跳过。原始日志SHA256为 `e2a6b1d8195d7e6d6e58dadd3a0ca165a2e096a141b87c44e69fd2f744a8b24e`。这不覆盖Torch frontend、生态、异步sync_run=0、BF16或通信。

## 统一候选与资源隔离复核

四批候选在主树及device/生态/distributed隔离树的38文件内容逐一一致，指纹保存在 `_state/ascend-compat/unified-source-r1.json`，补丁来源在 `integrated-candidate-r1.json`。本轮运行期间冻结这些源码；不以HEAD SHA掩盖dirty候选。未提交、未push。

统一候选分四个独立cache验收：主树物理卡5的常规异步Torch接口/AdamW-BF16/SDPA；distributed树物理卡3+6的原生逐设备RNG与矩阵回归；device树CPU接口及完整structure；生态树CPU双解释器模型对拍。固定亲和分别为188–191、180–183、172–175、176–179，每任务最多2个BLAS线程，core/算子串行编译，进程及启动tick写入各execution.json。其他用户在0/1/2/4/7上的任务保持运行。这里的已启动不代表验收通过，最终以实际退出码和完整结果更新。

在原TorchNPU oracle环境中，use_cpu与CPU tensor不能保证不初始化NPU：torch_npu全局optimizer patch和Trainer RNG checkpoint路径仍会进入NPU。旧CPU reference cohort因此保留为资源隔离失败，不能作为纯CPU参考；曾出现的瞬时NPU6进程缺少可确认的PID归属证据，不据此断言其所有者。随后独立建立不含torch_npu的oracle-cpu，显式空NPU/CUDA mask且不加载CANN，检查import/distribution及CPU checkpoint RNG内容。审计原始证据在 `ecosystem/resource-boundary-audit.json`。

`ecosystem/cpu-oracle-clean-r3/manifest.json`的15个阶段全部返回0：8个tiny模型的完整输出及所有适用梯度参考、Trainer四个update/八个microbatch、safetensors六个key保存重载、连续4与独立进程2+2恢复的216个数组完全一致（最大绝对差0）。十份checkpoint仅含CPU RNG，无NPU/CUDA条目。此处只计真实Torch CPU参考，不宣称Jittor或NPU对应格通过。

曾为前端baseline启动的独立冷构建没有到达运行期，因600秒预算不适合第二种构建配置而主动结束。其自有进程组及启动tick已核实，完整受影响缓存保留于 `_state/ascend-compat/interrupted-cache/baseline-frontend-rng-r1-cfgbed66048`，不再复用；该cohort既不计RNG失败，也不计前端通过。新的统一候选前端run使用独立缓存及5400秒预算。

## CPU 多库综合闭环（r2）

`ecosystem/jittor-cpu-r2/manifest.json`冻结38文件、23个执行阶段及187产物hash，SHA256为 `98e31cf09512b9d4fe98200dffe6c15b0eae0a3973126cfb192656f6f213881a`。独立JT/纯CPU oracle完成BERT/GPT2/Llama/T5/ViT/Whisper及LoRA Llama/Qwen2八格，271参数梯度加2输入梯度全部通过；最大前向绝对差5.0798e-7，最大梯度差9.7813e-7，沿用既有CPU前向2e-3/反向1e-2门槛。BERT/ViT只保留原声明中不参与输出的pooler梯度例外，frozen集合、依赖、线程及实际CPU均检查，fallback均0。

Trainer完成4 updates/8 microbatches及6个trainable参数全部梯度；61项跨oracle比较最大差1.0617e-7，safetensors六key保存/公开重载通过。连续4与fresh-process 2+2的216轨迹数组atol=rtol=0完全相同；完整轨迹另与oracle按原2e-4/2e-3预算比较，最大差3.7253e-8。十份JT checkpoint仅有python/numpy/cpu RNG，四个训练进程终态npu_count=0/available=false/initialized=false。该cohort对应其38文件指纹，不外推后续core修改或NPU。

## NPU 前端修前与定向修后

`foundation-frontend-npu-r1`完成275项冷编译，19 collected/executed、0 skip，16通过/3失败，退出1，日志SHA256 `264140b9738ea505e9fc4936b85578097764ca8746ef0445d18836cc2fa67da2`。通过项包括14个host/API合同及真实NPU RNG恢复、独立Tensor计算；PID2253168由npu-smi确认只在物理5。失败为真实BF16 autocast反向、旧fused AdamW数值和SDPA原生分派断言，均保留原日志。

修前独立probe进一步实测到BF16 autocast的mapped GEMM仍输出FP32、显式.float().sum()却被reduce推导成BF16，反向报mapped operands dtype不一致；FP32对照正常。证据为 `runs/bf16-autocast-before-r1/` 及 `bf16-autocast-before-r1-result.json`。原生mapped候选复用cuBLAS既有dtype推导/显式cast/梯度cotangent对齐；reduce候选使后端承诺preserve_type时真实保留浮点输入dtype，未调整数值容差。

两个旧测试没有显式指定NPU，独立Torch默认CPU，不能把runtime flag当作Torch默认设备。补齐NPU输入、模型状态/梯度/输出驻留检查且保留CPU reference后，`explicit-placement-fixtures-npu-r1`的SDPA正反向真实PASS，fused AdamW进一步暴露低精度参数配FP32 moments的合同冲突。独立TorchNPU `runs/adamw-native-oracle-r1/result.json`确认fused两步完整参数及状态精确匹配原固定fixture，未以版本差异改写期待值。

兼容层仅在首次有梯度的Torch Adam/AdamW slot初始化时，将moments转换为参数dtype；复用原生数学实现，原生优化器的FP32 master-state策略不变。`adamw-state-fix-npu-r1`三项完整执行、0 skip、退出0：实际NPU BF16 fused两步参数/状态exact，以及CPU/NPU FP16/BF16、late parameter group和原生初始化策略。日志SHA256为 `62e6fb54292150945665876ad4a49dbcd43abdbc074125dddf597ba2aa194eeb`。

## 原生启动次序与双卡 RNG

`native-rng-npu-r1`编译完成后在dlopen静态初始化期SIGSEGV，0 collected/0 executed，不计RNG失败或通过。addr2line/反汇编/重定位及成功cache对比已定位：use_cuda=1的Flag setter进入sync_all，遍历尚未构造的跨TU全局fetcher list；失败树device TU早于fetch TU，成功树相反。与seed callback/provider无关。原始日志SHA256为 `9243def61b79b9bcdd782bd4107095a70d94e0d514749369421079da81316f21`，证据在该run及 `distributed/fetcher-owner-r1/`。

修复将fetch与待释放队列归为function-local process-lifetime容器，显式cleanup仍释放所有VarPtr；不靠调整compiler枚举顺序隐藏问题。r2的原生RNG五项（uint64/两个分布lazy snapshot/CPU独立/双device）全部实卡通过，fetch两项、既有CPU queue一项、matmul两项也通过，fallback0；PID2368985同时且仅映射物理3/6。额外startup子fixture因/home与/beegfs源码拼写差异产生新冷构建，原pytest如实保留10 passed/1 failed，旁注标基础设施未完成，未宣称完整cohort通过。核实自有PID/启动tick后终止子编译，其整个新cache归档 `interrupted-cache/native-rng-npu-r2-child-e9a5c29ca8cc`；父有效cache保留。r3仍受子fixture路径helper优先读取parent环境的行为影响，未得到新硬件失败；其子cache亦完整隔离。最终fixture保留两个真实queue案例，startup与late cleanup归属fresh-worker协议，不再用嵌套冷构建作数学验收。

主树整合的54文件候选在 `main-source-r3.json`，包含上述修复、ACL公开npu/原生acl映射及真实模型工具；runtime源码在实际验收期间冻结。Qwen3-0.6B oracle-cpu固定LoRA初始化退出0（23.36秒），共享大权重只读，config/tokenizer/safetensors记录SHA；实卡真实模型和性能格尚未通过。

## 原生双卡完整复验（r4）

`runs/native-rng-npu-r4/acceptance-summary.json`对应43文件冻结源码与fresh进程，直接use_cuda=1启动，未加载原生Torch。10 collected/executed/passed、0 failed/skip，fallback0，进程14.09秒；5个RNG、2个fetch、1个CPU queue及2个矩阵案例全部通过，结束前显式cleanup连续两次亦通过。PID2410307/启动tick625550217经npu-smi确认仅使用物理3与6。新ELF仍为device静态构造早于fetch构造，证明修复不依赖编译排列。日志SHA256为 `6fb5c2e6b9e43acae2c361907f7ec2f16e5b3c04c3fee67eeac5349b6296ab47`。这项准入不覆盖HCCL通信。

## NPU 基础核心修后复验（r2）

`runs/foundation-frontend-npu-r2/execution.json`对应 `main-source-r3.json` 的54文件冻结候选（manifest SHA256 `5cc10482c724f5e3eacb8173e752aa4eef74238b246ff4686e1a5420177a9746`）。275项串行重编译878.38秒，整个进程986.84秒；19 collected/executed/passed、0 skip、退出0，含真实NPU BF16 autocast正反向、RNG恢复、独立Tensor、fused AdamW与SDPA完整正反向。运行PID2402038的周期采样仅见物理5，日志SHA256 `65cc9f7f76c229bf9300ad7bc46fccb4947cd88f07d70a519f1403213b434d78`。完整优化器目录另行验收，不以这19项替代。

`runs/bf16-autocast-after-r1`退出0、9.99秒，独立相同probe与修前成对：BF16上下文中mapped GEMM输出BF16，显式.float()和sum均FP32，两个FP32叶子梯度dtype与全部数值均正确，实际device=npu:0/ACL。FP32对照亦通过；没有放宽容差。日志SHA256 `0a047c0fa36451106b84eb62ee0cd0a29ee2917388f40a8a9ca9e1ba97e9a3bf`。

## 完整优化器补验收与真实模型诊断

`optimizer-regression-npu-r1`的29项断言全部通过、无skip，但旧式CPU/NPU循环的nodeid不含设备词，pytest硬件计数为0，整轮如实退出1；没有改写原结果或降低min_executed。相同54源码与cache的新cohort `optimizer-regression-npu-r2`执行完整29项并加入明确的真实NPU BF16 fused案例，30 collected/executed/passed、0 skip、退出0，23.54秒，日志SHA256 `ef900a2bb8bf8eea838effa1a9184327776d0eb3578299333991df92854a50b5`。所有原数值容差与硬件要求保持，原source54事后hash逐项一致。

真实TorchNPU的独立诊断 `float64-native-npu-r2`记录请求to(float64)后的实际dtype仍为float32，设备npu:0、无警告；完整原始结果在 `real-model/float64-native-npu-r1/result.json`。真实模型工具因此只把detached梯度范数诊断显式搬到CPU再做FP64 pairwise，严格检查实际device/dtype，算法名改为host_fp64_pairwise_256_v1；这不参与裁剪或optimizer更新，训练梯度驻留在诊断前另外验证。CPU独立参考的0/1/255/256/257/1025长度与NumPy FP64结果差<=1e-10且原值不变，26个workflow host合同通过；实卡真实模型仍待验收。

`main-source-r4.json`仍54文件，增量仅最终fetch测试fixture、可选完整梯度归档、host FP64诊断及整模benchmark输入/精度工具。性能入口具备batch/context/3预分配slot、固定decode步数、每slot完整输出和cache on/off审计；host形状/确定性/拒绝非法输入检查通过，未据此宣称任何NPU性能或真实模型数值PASS。

## 严格精度参考环境与共享资源边界

生态NPU r1/r2均在参考bootstrap失败、模型执行为0：HF32off经ACL compile option初始化GE/TBE，原oracle缺decorator，先前最小matmul成功不覆盖这条SDK路径。两次失败和去掉多余cache选项后的复现均保留，未删严格精度要求。随后只在lab新建oracle-npu-cann，原34个依赖精确freeze另加CANN8.5 metadata声明且本机base已采用的6项：attrs26.1.0/cloudpickle3.1.2/decorator5.3.1/ml-dtypes0.5.4/tornado6.5.10/absl-py2.5.1；完全离线复用缓存wheel，原oracle/dev/oracle-cpu未改。完整40包freeze与lock一致，pip check通过，SDK元数据/wheelSHA在ecosystem/oracle-npu-cann-setup-r1；仍需真实HF32off smoke和模型验证。

HCCL r1原生op-registration生成因hccl_reduce_op.h的包行constructor与既有单行parser合同冲突而失败，通信数值0。distributed树按真实header最小修复后，r2 rank0完成MPI-free HCCL三个低阶FP32原语、fallback0、物理3，93.12秒；后续真双卡仍待测。rank1启动守卫发现物理6有yizhang新任务，未启动或干预；只允许在空闲3建立独立rank1 cache，并清楚区分缓存预热与真实3+6通信。

真实预训练Qwen3-0.6B的CPU参考full阶段（qwen3-06b-cpu-r1/full）退出0、234.35秒，4 updates/8 microbatches，112个LoRA trainable参数每步完整梯度归档，全部finite且冻结base签名不变；adapter safetensors公开重载、merge/merged公开重载、greedy cache on/off及sampling流程均通过原门槛。full日志SHA256为9e11f0933652ec8dc9f82fa3fe83adc57c7d6c607f8ac8523d4e7640ab2a9593。该结果仅为真实Torch CPU参考，不计Jittor或NPU；fresh-process恢复尚需独立结束和exact比较。

真实Qwen CPU参考的first/resume/compare随后均退出0，fresh-process恢复加载model/optimizer/scheduler的独立signature exact；最终933项状态、样本游标、update轨迹以atol=rtol=0全部一致。另逐步比较full与first的1/2、full与resume的3/4，共448个完整梯度张量及448个dtype数组逐值exact、最大差0，证据为real-model/qwen3-06b-cpu-r1/gradient-resume-exact.json。full/first/resume原日志SHA分别为9e11f0933652ec8dc9f82fa3fe83adc57c7d6c607f8ac8523d4e7640ab2a9593、1d10faa19a1483c2a8d9bac7efa9f7b6b54fed3c202c667ad8b2d143837f2456、67e720c5904e44b93687b9a84565a15bf7f1b20b5a9256ebc33850f47f292153；不外推NPU。

CANN参考环境已通过严格HF32off smoke：新env40包freeze SHA256 f623c35a7bb250240778d7ff6fd11dc75281204542c7b5a81854c04505e4facc。补依赖后的首次失败进一步确认是深TMPDIR导致AF_UNIX path too long；保留长路径负向与lab/_tmp短路径Manager正向，未按GE泛化提示误判内存/FD。短路径真实NPU烟测63.65秒退出0，FP32矩阵完整输出和两输入梯度与NumPy1e-6通过，HF32两个开关False、AMPFalse、count1及910B3均实际读回。完整多模型NPU协议现用该env、短TMP及原冻结权重/误差门槛启动，仍逐格验收。

## 真正双卡 FP32 HCCL（r4）

`distributed/hccl-fp32-r4/acceptance-summary.json`在物理3与6重新同时空闲后验证原生TorchNPU及Jittor两套独立双rank。四个正常退出进程分别为原生2633788→3/2633792→6、Jittor2649752→3/2649753→6，双launcher均退出0，source58指纹全保持。all_reduce_sum/broadcast/all_gather/all_gather_object共16个rank-cases，完整32值输入及全部输出跨runtime exact；Jittor实际fallback尝试0，四进程均无libmpi。原生整过程316.58秒、Jittor46.60秒主要包含初始化，不作为速度比较。原日志SHA分别为a67c1cc5594a64d5dc10abd7808f3842a300bd97e5523469ca72df54d24e0416、ed2950aa06c81c63042b5a9a7d502666215114445a9fc62636bf68e227cb7274。此前r3删掉CANN官方PYTHONPATH而缺tbe的0通信失败完整保留，r4仅保留SDK两条官方路径且明确排除Jittor/shim注入。BF16、autograd、DDP/FSDP2和完整恢复另行验收。

CPU最终57文件的affected129 passed/5 skipped、native namespace+RNG16 passed、完整structure1398 passed/2 skipped均退出0；完整core仍在跑。初轮stub两个基线失败通过真实声明/唯一manifest修复，provider结构fixture不再缺实际getter；没有降低数学阈值。实际日志和原失败均在device-api/cpu-final-r4及cpu-unified-r1。

整模性能工具的原生TorchNPU回退次数改为null/unavailable，因为没有该工具可实际测量的vendor counter；Jittor仍用core计数并要求0。完整输出比较器拒绝把原生未测量次数伪装成0，21个纯host正负/CLI合同通过。main-source-r6的58文件仅在性能metadata层新增这一诚实口径，runtime与r5一致，尚无真实整模NPU性能结论。


## 严格原生 NPU 综合参考完成、JT 同协议运行中

`ecosystem/npu-matrix-r4/native-phase-summary.json` 的24个原生阶段全退出0，SHA256为4d449f8e97a22d0d5cbe782af9ac573fec7543106452d85137590a3736dbbd51。八格小模型包含271参数梯度与2输入梯度，实际NPU/HF32off/AMPoff，最大前向绝对差1.791033e-5、梯度1.003699e-4；Trainer的61项比较最大3.585592e-8。full/stop2/fresh-resume的216轨迹数组同运行时exact为0，完整原生NPU轨迹与纯CPU参考最大差5.70435e-8，未改原门槛。当前JT对应矩阵已开始独立冷构建，尚无JT NPU模型格通过；全批checkpoint RNG文件审计仍待终态，不能把原生24阶段外推成双运行时完成。

`device-api/fp16-gradscaler-main-r1` 的原生参考退出0、122.38秒：两次FP16 forward/FP32 loss与参数的真实AdamW更新、scale16→32及第三次实际inf反向触发跳步，参数/optimizer/scheduler保持exact、scale32→16。原生日志SHA256为10d9402f63c5c84dc59b5f0fcb41e7d325c0679b9886d1df3b930b24957fbf6d。首个JT worker在数学执行前因jit_utils重建需fresh进程退出3；失败保留，新r2固定canonical目录重启后完成其必要的核心重编译再做同协议比较，当前仍运行中。

性能交替调度候选目前仅在隔离ascend-model树：`real-model/perf-controller-host-r2/validation.json`包含19个host合同，验证20组成对调用（prefill/generate各10组）、ABBA顺序、前一个同步边界完成后才放行另一个worker、扣除准入等待及各类协议失败回收自身子进程。parent始终将performance_accepted标为false，必须另行完整输出对拍才能接受性能。此处没有NPU速度/显存结果，也没有合入冻结的main-source-r6。


## CPU 核心终态与真实 FP16 前置接口缺口

CPU57冻结批次的core-native正常退出0：351 passed、136 skipped、1 xfailed、1 xpassed，共489项、1455.10秒、0 fail/error。XFAIL为TestCore::test_swap，XPASS为TestLivenessAccountingProperties::test_dropping_a_graph_leaks_nothing_at_all，后者保留既有环境/teardown敏感标记，不据通过归因成新修复。另13项Torch核心由同driver接续运行；源码仍冻结，不把后续优化器接口增量算进此57源码证据。

FP16 main-r2完整核心重编译后正常退出1，1043.93秒、日志SHA256 a46cfe046e7486f318430f78031fd77fdb06016a262b1c0c4fb4889f4f211916；实际失败在AdamW公共构造不接受capturable=False，未进混合精度数学。根候选仅在public frontend消费False并为True/非法类型/参数组变更明确报错，不修改native Adam数学。main-r3复用有效cache15.54秒后暴露第二个已存在的state接口缺口：初次optimizer.state[p]在shim抛KeyError，原生defaultdict会创建empty entry。日志SHA256 d7eae20ce7b2de4e40c1d75935e7ccda15ef403299a34d029137c6f1c264b773，source-r7的59项事后hash全一致。尚无JT FP16数学或成对PASS；维护owner正在补统一entry存在性、映射迭代与save/load，以及真实Torch2.7 AdamW的decoupled_weight_decay元数据，不删原参考字段或放宽预算。

主树当前main-source-r8为63文件（SHA256 ab2e5c02ef1e496177f26e56fa5ac778235e3b0cc3e1659ca8e297c9d1575a44），纳入已实卡验证的HCCL final-r2与CPU26修后合同、官方SDK隔离说明、以及host验证的成对调度/cached-decode工具。layout、239篇活跃文档governance及git diff --check通过。性能工具host证据更新为controller-r4的30合同、comparator-r3的32合同，以及真实原生CPU tiny Qwen3的24组缓存decode验证（B1/2、S8/13、T2/8、3 slots）；完整decode logits与整段前向最大差0、greedy exact。该host验证不代表NPU模型或性能已通过。成对co-resident协议明确双模型显存条件与独立HBM口径，常规单进程每卡协议不变。


## CPU 终态、optimizer.state 与 FP16 成对验收

CPU57冻结批次接续的Torch core 13项全部通过。完整终态为affected129P/5S、namespace/runtime16P、structure1398P/2S、native core351P/136S/1XF/1XP、Torch core13P，各stage退出0；最终device-api/cpu-final-r4/final-results.json的SHA256为33438b0e6ffd39ab41d46bfeca767d08b93b2563ebd7bffcc06e899b633fe67e。保留原XFAIL/XPASS边界，不把teardown敏感XPASS当修复归因。

实际FP16失败推动的public AdamW capturable=False与统一optimizer.state映射修复未改变native Adam数学。映射维护parameter identity、空entry存在性、held view随step更新、load/save/delete/clear/late group及AdamW decoupled_weight_decay元数据；原生CPU12项对照中旧实现10失败，新实现12通过，扩大CPU回归133 passed/2 CUDA skipped。未知非parameter key仍明确KeyError，不据此宣称完整任意defaultdict语义。

device-api/fp16-gradscaler-main-r4在source-r12的71文件冻结后真实退出0，17.853秒，日志SHA256 deed866d1f5dcdeae95567dd4d467c2778fec3b41e37f1177ddb2e5aab1e2564。沿用原生r1相同fixture与原预算的成对比较36项全通过，最大绝对差0.0001220703125。两次有限FP16 Linear/FP32 loss、参数/完整梯度/moments、AdamW/StepLR及scale16→32均对拍；第三次真实Inf反向跳过参数、全部optimizer状态与scheduler，保持exact、scale32→16。JT实际ACL驻留、fallback0。此前r2/r3接口失败完整保留；该结果不覆盖真实模型FP16训练或静态dynamic=False GradScaler。

## 逻辑归约与整数列表索引的真实失败修复

Qwen生成首先暴露原生any/all后端不接受第三个keepdim参数，后续最小probe还发现全轴keepdim在public method/module路径未传递。修复归入ACL descriptor返回shape和既有Torch shape/reduction owner，保留原数学与CANN descriptor约束。device-api/truth-public-npu-r4的64个真实NPU值、shape、dtype、驻留检查全部通过：原生JT/public Torch、bool/含NaN float32、any/all、None/0/-1/(0,1)、keepdim双值及isin生成触发路径；fallback0。对应CPU原生32项与public Torch16项通过。原生JT默认dim为()，显式None仍沿CPU不支持边界，不以public Torch支持外推。误选独立native build配置而在600秒预算结束的旧cohort未到数学，整个cfg缓存已归档，不能混入通过项。

JT NPU GPT2与Llama已在原source54分别完成完整输出和全部28/21参数梯度，最大梯度绝对差3.3384e-7/3.5585e-7，实际ACL驻留/fallback0。随后BERT在input_ids[:,[-1,0]]触发真实列表索引失败。ACL GetItem仅复用既有core整数数组转换与Index/IndexPut数学，未扩大容差；原生NPU9个fixture对照与修后JT9个fixture均完成，八个浮点case完整source梯度和int64 BERT触发值exact。正式native-mode pytest与新的完整八模型/Trainer批次另行执行，不能将九个最小probe写成全矩阵通过。

## 真实 Qwen3-0.6B：生成完整，FP32全量数值未通过

原生real-model/qwen3-06b-npu-r1/torch-fp32在单卡NPU、HF32关闭/highest、FP32 eager、B1/S128、三个不同预分配输入slot、T8协议下139.902秒退出0，保存完整输出与cache on/off greedy。日志SHA256为ced6de76a17cd5e6379409d219d06c4591d7c95d376fcd0540507108e39a7d1e。修复any/all接口前的JT r1失败保持原日志；不覆盖为数学通过。

JT real-model/qwen3-06b-npu-r2使用source-r13的73文件冻结（manifest SHA256 9bb8ed0b1b53462b23f70421193ca606868131fe4f1f3525d807a352e78a9fbf），173.319秒正常退出0，完成加载、全量前向、三个slot完整生成、缓存/非缓存与原生greedy token逐值exact，fallback0。独立比较器仍退出1：58343424个float32 logits中94956个超出固定atol1e-5/rtol1e-4，maxabs0.00011157989501953125。比较证据qwen3-06b-npu-r2/compare/result.json的passed=false，日志SHA256 f044f468c41ed29cf206c076f5cf3feaecf5a7b6b83cafafb5b93f2d1b4a60bb；整项未设PASS。

CPU最小诊断device-api/qwen-rope-cpu-r1显示64个inv_freq中1个值相差1 ULP，首差来自1e6的float32 pow；相同freq上的cos/sin还有独立末位差。完整cos/sin最大差5.96046e-8，仍在原预算内，不能据此认定真实logits失败原因。完整CPU数组及comparison SHA256 c6e92b043613ff84c004f62846ec14ad2013dbc645856db757ce49ebf8d92659保留，下一步是独立真实NPU逐层定位。sample1的首步/稳态计时仅作编译诊断，未达到ASC-06至少十次交替配对门槛。

双卡BF16旧候选已真实复现JT HCCL dtype表缺bfloat16，而原生两rank四原语完整通过。补表及实际Tensor dtype证据候选目前只在distributed隔离树。新67源distributed/hccl-fp32-r5完成16 rank-cases全payload exact，native244.73秒、JT22.51秒、各退出0，实际dtype/schema2、物理3/6映射、JT fallback0/无libmpi与源码事后hash均通过。初始化耗时不作为性能结论。新表BF16及DDP/FSDP2另行实测，尚无对应通过声明。


## 新表 BF16 真双卡通过与 DDP 首个真实接口失败

distributed/hccl-bf16-r2/acceptance-summary.json（SHA256 91ab0ef91914035fac98cf857b284a129fe55962a9c41ae59fd9fabf31f014f8）完成两个runtime×两个rank×四原语共16 rank-cases，完整32值tensor/object跨runtime exact，五个实际Tensor dtype均bfloat16，JT fallback0、无libmpi，source67全hash稳定。原生228.19秒、JT29.72秒各退出0，PID与物理两卡映射实际保存；初始化耗时不作性能比较。三个小增量在root无运行worker窗口整合为main-source-r14的74文件，SHA256 aad5524eb467c8321aac930776b9bcaf21ba6d1928e909060186018cc072f7b8；整合前source73逐项不变，git diff --check/AST通过。no_sync证据候选此时仍只有host验收，不能随primitive通过外推训练。

DDP driver首次在world启动前因指纹清单漏未修改依赖而host preflight失败，0训练；新71项清单保留原67全字节不变，仅补入四个与HEAD相同依赖，旧失败保留。随后原生DDP full8两rank真实完成289.32秒、rc0，累积2/no_sync/clip/AdamW/StepLR完整轨迹已保存。JT full首次24.12秒退出1：optimizer.zero_grad经native optim/base将DDP state作为dict key时，SimpleNamespace不可哈希。两个rank同栈、HCCL已初始化、0updates，未达到任何训练数值或DDP通过结论；prefix/resume/FSDP2尚未启动。修复由DDP state owner先真实最小复现，未据此修改优化器数学。


## FP32 真实模型完整逐层定位（不替代默认验收）

真实NPU trace先核对310个参数的全部shape/dtype/字节SHA完全一致，两个运行时插桩后的logits分别与原未插桩结果bitexact。固定slot0/B1/S128的115个完整数组包括全部层hidden、KV、完整19447808个logits及第0层细分输出。baseline首个超原forward门槛的hidden为hidden/17；CPU RoPE频率只有一值1ULP差，统一它后第0层QKV/O与attention概率bitexact，但post_attention_layernorm的同一131072值输入仍产生16537个不同输出，maxabs1.9073486328125e-6，首差2ULP。CPU行尺度重构说明1–2ULP的rstd差足以解释本处输出，未冒充直接测量CANN内部中间值。

| 初始化频率 | RMS计算 | slot0超差logits数 | 定位含义 |
| --- | --- | --- | --- |
| 各自默认CPU初始化 | 当前融合入口 | 28431 | 默认FP32仍失败 |
| 仅统一完整频率buffer | 当前融合入口 | 19473 | 频率不能单独解释全部差异 |
| 各自默认CPU初始化 | 原HF class forward | 24487 | 取消融合也不能单独通过 |
| 统一频率buffer | 原HF class forward | 0，全部115数组bitexact | 两因素控制定位，不是普通from_pretrained门禁通过 |

最后两格在r16的78文件指纹上执行，trace-r6/r7使用strong sync逐一实化捕获的中间张量并验证真实ACL驻留/fallback0。旧r1漏输入搬运同步、旧r4只weak sync而使被融合中间量未实化，均被严格location守卫拦截；它们属于诊断脚本不完整，原错误和日志保留，不算模型数学失败。未启动的r5也不计执行。没有复制参考buffer到production，没有修改正确libm或放宽预算。默认原FP32三slot门槛仍未通过，控制定位不能作为ASC-06性能准入。

完整数组/元数据分析在device-api/qwen-trace-analysis-r1：full-array-analysis.json SHA1170ed954a59e85f19faf6097ef016211ff4cc7410971f8926dbd58a7f1a6fa2，row-scale-reconstruction.json SHAc6148050bf75092ca5034cb34da6b6dbc4aa7111d39e7ae4ba37b3d41dfa6d8a；root trace-r2/r3-controlled-rope及r6/r7目录保留独立PID/启动tick、实际物理卡映射、全部NPZ和原始报告。

## 完整生命周期归档与跨runtime训练比较器

旧真实CPU lifecycle.npz仅保存2×4×64的logits_sample，内部reload/merge虽用完整数组检查，但不足以证明训练后完整跨runtime数值。先保留真实缺口复现，再新增original/reloaded/merged/merged_reload完整logits/loss和对应输入，以及generation完整input/mask。旧sample与原数学、merge/reload预算均保留。新纯NumPy/stdlib比较器强制完整step/trainable/gradient/dtype/keys/finite/参数组和optimizer/scheduler/scaler计数，数值预算只读原fixtures-r1，不提供随意放宽的CLI覆盖；native回退次数须null/unavailable，JT须实测0。sampling只验证完整形状、范围、前缀与合同，greedy和所有整数输入exact。same-runtime独立补查first+resume的全部完整梯度/dtype仍要求exact。

三个文件的owner主机合同56通过；主树整合后既有unittest选择集28项通过，新的pytest选择集实际30项通过、0skip（两套runner分别记录，不把选择器计数混称一个全套）。主机合成合同通过不代表模型运行。相关scope和日志在device-api/release-hf-cross-runtime-r1及hf-complete-evidence-main-r1/r2，整合source-r16共78文件，SHA256 7ab8fb1101f3b2896e578142e2a2fd331fc8265e651a890820ea7404ecb8fafa。还整合DDP私有identity state小修复，其CPU35项通过，未改native optimizer数学。

## 七模型 NPU 通过与后续真实缺口

ecosystem/npu-matrix-r6的15阶段为14 PASS/1 FAIL：BERT/GPT2/Llama/ViT/T5/Whisper/LoRA Llama七模型的267参数加2输入完整梯度已完成；Qwen2 LoRA在SDPA enable_gqa的标量repeat_interleave路由生成不支持的reindex/0，被error回退守卫拦截，未产生其完整输出/四梯度。失败log SHA d76cfa125b44a97516764bdffe2e4d1aa7e7fce140b4fb677198ca618ee68783；Trainer/resume与最终20 checkpoint audit在这批未执行。

通用ACL lowering复用reshape/broadcast/reshape，修前7失败保留；修后独立native与JT最小9-case完整输出/源梯度通过，正式native-mode回归实际14项（CPU/NPU各7）全通过。另发现repeat=1会alias原输入的公共语义回归，已用真正inplace证实并准备native Copy小修复，旧候选尚未合入MAIN；原CPU冷编正常结束后再冻结新批。native十份checkpoint先行只读审计包含python/numpy/cpu/npu完整状态，最终仍需JT十份和20份合并审计；确定性dropout0/sequential恢复不能写成随机训练恢复通过。

DDP state identity修后真NPU能完成第一update的累积2/backward/clip/AdamW/StepLR，但DCP snapshot随后因公共get_state_dict返回整数ID而非Torch2.7的FQN键失败，0完整记录、未做全八步比较。独立真实CPU对照还发现setter未恢复optimizer lr，get/set_optimizer_state_dict未安装；底层Optimizer.state_dict返回整数本来正确，修复归canonical DCP owner，不能改optimizer数学或接受错误fixture。该问题仍未通过。


## 普通BF16真实推理失败及最新算子、DCP整合

inference-functional-grid-r1/06b-bfloat16-b1-s128在原始独立初始化、HF32关闭/highest、B1/S128、三slot/T8下，原生129.953秒/JT104.834秒各退出0。完整58343424个logits中14833706超出原固定atol0.005/rtol0.03，三个slot maxabs为0.75/0.875/1.09375；三个slot的输入、mask与缓存/非缓存greedy均exact，JT fallback0。比较器退出1，整项FAIL；sample1计时不构成性能结果。原日志SHA依次758eb3847f650e11b55067b31455b1c0a9443cd1dcd6ccadcdc9443970ad57f2、097eba44bdc157d6e0618a35f170f0b55c236e200b0bfe95058c9663011abaa7，比较log1140f9c5c698edd76a49983aec30865733acbc29fd32f9685cc5a963402cc123。

ACL repeat_interleave最终包含repeat1 native Copy语义，修后CPU8/8、native-mode16/16（CPU/NPU各8）全通过，最终4file patch SHA1ca39906ac4699881cd877eaa1114c3fd6278a697150525d285772fef8134c41，旧alias候选不合入。r7 source65新的八模型完整forward/271参数加2输入梯度和独立比较共16stage全部通过，含Qwen2 LoRA；其后Trainer在第0步前placement严格发现CPU，未执行训练数值/恢复/最终20份RNG审计。失败SHA67f3748cb3fdd42f9be61c433e1aece80e11f5198ed38f1ce67270673ca91e3f保留，不能以8模型通过覆盖Trainer。根在旧worker全结束后按before/after字节精确整合为MAIN-r17的81文件，manifest SHAc12ce7a03f2aeae240a8f83a63ecb088e3f37ca60db461755eb0f8435a5094ab。

DCP canonical owner修复FULL state的FQN映射、优化器setter和lr恢复，raw Optimizer整数ID接口保持原语义；不支持的sharded planner/storage和options明确报错。CPU46项/0skip及独立Torch2.7与shim-after的8完整数组47值、参数组metadata均exact。单文件增量patch SHA485983458419f2cca3dd9c2a2baf64e745c1493942408449ae2f47846e8706ac已整合为MAIN-r18（81文件，SHA623a32412cdaef181d8141541e671d041f87162973be92703bf867e315e329e2）。实际双卡训练仍待新75源cohort验证；因为训练proof source发生变化，原生参考也重新运行。此前两次DDP失败仍保留。


## BF16归一化单因素诊断完整bitexact

real-model/bf16-norm-only-diagnostic-r1仅在仓外entry关闭HF RMS shortcut，保持普通from_pretrained各自RoPE初始化与全部原权重，复用已封存原生native输出，不改误差门槛。JT PID3121454实际映射物理卡，完整58343424个logits在三个slot均maxabs0、0超差；所有输入、mask、cached/uncached greedy exact，fallback0。worker194.612秒/比较2.525秒各rc0；worker logSHA793239e5dd338a6ace798cfb444f765e0cc45cebefeed9ea4f37b4bbc9fb6d38、compare logSHA0f49af47ed72d80385b6684c06a9908efefc493cd0939dd295a44886186db3eb。MAIN仍r18未改；这是支持修复融合入口的单因素证据，默认门禁仍FAIL，须canonical修复后重新验收。正式ABBA工具已准备默认完整功能/source/assets匹配准入和10次prefill/generate/cached decode，尚未执行性能试验。


## DDP八步完成但完整状态schema仍失败

75源新FP32 primitive r6完成16 rank-cases、完整32值tensor/object exact，native249.43秒/JT22.51秒各rc0，实际dtype FP32、实际双卡PID、fallback0/noMPI与source稳定均保留。随后DDP r3原生FULL8 344.70秒/JT FULL8 60.98秒各rc0，两个rank各自完成8updates，DCP FQN路径越过此前失败。但正式完整对拍仍FAIL：native每组存在initial_lr，shim缺失；shim StepLR state多verbose字段。完整54项schema差异保留，未删字段或放宽。独立只读的1110数值leaf/2662元素在原2e-6/2e-5内无超差只是诊断，不能替代完整门槛。prefix/freshresume/FSDP2尚未启动；lr_scheduler canonical owner正按真实native public ctor/8step/freshload复现修复，LinearLR既有verbose合同不全局改变。

real-model/qwen3-06b-training-npu-r1的native普通FP32 full完成4update/8micro、112trainable的448份全梯度有限，base冻结、保存/加载/合并与完整lifecycle输出均已归档，worker274.657秒rc0；native prefix2亦246.716秒rc0，freshresume正在执行。该结果不证明JT真实LoRA或跨runtime训练通过。


## 默认BF16推理正式通过与当前严格边界

MAIN-r21包含87文件，manifest SHA256为5d81d4bb245018ed1edad9a462926f0d0e80863d4f18d1bf436e3be9416b5179。canonical HF RMS owner在ACL上保留原HF class forward的dtype与舍入顺序；4项真实CPU/NPU FP32/BF16全input/gamma梯度测试及反转全局标志的4项驻留复验均通过，CPU tensor实际仍CPU、NPU tensor实际仍ACL，fallback0。主树受影响集累计122个独立合同通过，结构全量另外执行，不能把原source57的结构结果外推。

`real-model/inference-functional-grid-r2/06b-bfloat16-b1-s128`在规范adapter overlay-r2、独立普通from_pretrained初始化、eager/HF32off/highest、三输入槽/T8下完成native316.716秒和JT58.778秒各rc0。完整58343424个BF16 logits三槽均maxabs0，原固定atol0.005/rtol0.03下无超差；全部输入、mask、cached/uncached greedy exact，JT实测fallback0。初始化耗时不作为速度数据。它替代早期默认BF16失败的当前结论，不覆盖普通FP32严格门槛；没有向production复制oracle RoPE buffer。

`real-model/inference-paired-perf-r1/06b-bfloat16-b1-s128`启动原生worker3262827，但entry中的npu-smi物理驻留查询超20秒，模型未加载、计时未开始即rc1；controller3262521随之退出。fresh快照确认物理5空且两PID终态后交给生态Trainer，未接受性能数值。新lab r2只把startup witness改为60秒有界等待，保留原物理PID、完整输出、10次ABBA配对门槛，准备未执行。

## FP32 DDP完整恢复通过，FSDP新失败已复现

`distributed/hccl-training-ddp-fp32-r4/acceptance-summary.json`在distributed source83完成两个rank的FULL8及prefix3/fresh-resume5。原完整梯度、参数、AdamW moments、StepLR字段和DCP schema按2e-6/2e-5跨runtime通过；同runtime连续与3+5恢复exact，no_sync第一micro的rank-local梯度实际不同，最终副本一致，JT fallback0。JT full/prefix/resume分别23.51/24.42/24.06秒，native prefix/resume303.61/258.44秒；native FULL明确复用r3/source75原参考，11项声明proof sources及worker/初始状态/oracle全字节未变，双manifest receipt保留，不伪装成新83重新计算。共12个rank phase报告、6个runtime×phase组及4份按rank保存的checkpoint。

FSDP2 FP32原生FULL8实际387.57秒rc0，JT FULL50.35秒在第一update后DCP快照报NameError，canonical `_all_gather_object`不存在，正确helper为`_native_all_gather_object`。失败原样保留。公共CPU fully_shard singleton+DCP真实最小复现修前失败，修后连同受影响集合105 passed/0 skipped；owner属于proof source，新的primitives与双runtime FSDP FULL都从新源码重跑，不复用旧full为新源验收。

## 原生真实LoRA严格loss标量边界

`real-model/qwen3-06b-training-npu-r1/fp32`的原生full4、prefix2、fresh-resume2各正常完成。连续与恢复的933个完整状态数组、448份全梯度及dtype全部exact；8个micro/mean loss标量因1–3ULP差异使原exact合同失败，未删除字段或调阈值。JT r1因缺少规范adapter distribution metadata而模型停在CPU，零update前严格placement失败；不能以原生梯度通过宣称JT LoRA通过。

独立`real-model/native-ce-determinism-r1`固定54×151936 FP32 logits/9 ignored labels，对torch.use_deterministic_algorithms(False/True/False)各20次真实NPU CE forward/backward，三种模式的loss均出现4.2824239731与4.2824244499（1ULP），各模式完整梯度SHA保持唯一且相同。该供应商栈复现解释原生标量exact不可复现的一项原因，但不据此降低验收标准；shim deterministic接口本身未实现，也未盲设为True。当前真实LoRA完整跨runtime仍未验收。

## 08:10 当前整合与验收状态（较早结果按其原基线保留）

MAIN-r22共冻结94个源码及相关文件，manifest SHA256为96629be006aba70aa7f369b72a6358fa43d88db8db251b3f4657c200741437c7。八个不重叠增量、20路径逐项核对整合前后字节，Python3.7 AST与diff检查通过；receipt在`main-integration-r22/receipt.json`。当前主树已完成HF主机合同59 passed、DCP12 passed、DTensor dtype4 passed、优化器相关62 passed，均0 skipped，真实CPU/fallback0且94指纹前后稳定。完整structure现为1402 passed/2 skipped（JUnit1404、0F/0E，397.74秒），与r21逐node比较原四失败全部转通过、无删除，新增两项实化同步守卫均通过；两CUDA skip全文及路径行号一致。包括上述选择集在内共1541个不同JUnit case、1539 passed/2 skipped；layout rc0，正式receipt为`device-api/main-final-cpu-prepared-r1/main-r22/complete-evidence.json`（SHA80e7b2f230709ebbf2354d2a6fca5066e346577c2bf0a787321010045745923c）；原HF第一次因lab wrapper缺tests路径而collection失败单独保留，补跑59通过。r21四个结构失败的修复保留CLI阻断runtime导入的实际负向控制、统一纯host SHA helper和新kernel注册canary，不删除测试。

`ecosystem/npu-matrix-r10/result.json`（SHAcf401b8d088928f47b6f8294ed46dc549611d4a0039d1d2ea0c4bb3f6ca3a434）在规范adapter overlay-r2与独立source65下，25个新执行阶段及24份明确复用的不可变native参考共49阶段全部通过。八模型完整271参数及2输入梯度全部对拍；真实ACL Trainer4updates/8micro/6trainables共48份全梯度、eval/predict/best checkpoint/safetensors/fresh reload完成，61项跨runtime按原atol2e-4/rtol2e-3通过，maxabs2.048909664e-8。连续与prefix2/freshresume2的216个完整状态/轨迹数组exact，跨runtime216项maxabs3.725290298e-8通过，fallback0。20份checkpoint的python/numpy/cpu/npu RNG均为非空完整uint8；不同runtime使用各自不透明格式，不宣称格式互换，dropout0/顺序fixture不证明随机训练恢复。所有该批own进程终态，compact summary在`ecosystem/npu-matrix-r10/validation-summary.json`。

Trainer评估真实暴露独立None索引失败，canonical ACL GetItem将单个None与既有tuple(None)统一；修前8case中2失败，修后原生/JT8case的14个完整输出/梯度数组bitexact。正式`test_acl_indexing.py`实际24 passed/0 skipped（7旧项、9整数列表、8 None项），已合入r22。保留旧异常及缓存诊断，不据此报告显存优势。

FSDP2新的FULL8曾在第一更新前完整梯度及clip norm bitexact，但15个更新参数全部超差。最小CPU复现定位公共AdamW重新绑定live param_groups时丢失构造器betas/eps/weight_decay；canonical frontend现保存并即时补齐公共defaults、保留显式group优先级、校验追加及恢复group，未改native Adam数学。修前10合同9失败、修后10通过，独立native协议40个完整参数/moment/step数组通过，34个bitexact、maxabs1.49e-8；受影响既有回归亦通过。DTensor修复canonical dtype helper导入和独立公共factory的Tensor frontend身份，CPU125项及实际NPU双角色四dtype factory均通过。相邻反射guard仍有一项旧hash不匹配，当前与恢复pre-dtype源码的payload相同，尚待审计有意公开API变化，不盲改固定hash。

当前FSDP后续批因另一用户开始使用原分配卡而避让，只终止验证过身份的本任务launcher/ranks，保留native数值输出及缺失最终退出确认的基础设施失败；不算paired通过，也不算数学失败。后续改用重新交接的空闲两卡，不干预外部进程。

真实Qwen3-0.6B FP32 LoRA的MAIN-r22首FULL4完成4updates/8micro、112 LoRA trainables与实际eval；随后生命周期sampled generation在float32[2,151936] argsort触发error fallback，worker198.237秒退出1，`real-model/qwen3-06b-training-npu-r2/fp32/jittor/full`保留完整日志和已生成产物。该批未完成lifecycle、prefix/freshresume或跨runtime，因此不标训练通过；训练78.7643秒包含观察器与冷状态，也不作性能比较。

正式推理性能r2虽然完成10对prefill与generate，但manual cached decode先后暴露实化同步、BF16 arg_reduce以及min_new_tokens EOS抑制语义差异，原批失败保留。工具现采用HF4.56.2同一FP32 logits copy及MinNewTokensLengthLogitsProcessor，在setup强同步后检查真实驻留，手动循环包含同一处理器与完整历史。`real-model/cached-decode-residency-owner-r3/jittor-npu-after`实际NPU三槽7cached steps加首token均与generate exact、完整输出有限、fallback0，74.8091秒rc0；这是sample1诊断，不是跨runtime性能通过。须在新源重跑匹配普通功能后执行至少10对ABBA及全部cached输出比较。

训练计时工具`ecosystem/train-step-timing-r7/qualification-summary.json`仅CPU资格通过：显式eager attention、双方相同preoptimizer及update边界强同步、4次全shape warmup加10次更新；native736/JT733完整状态均相对功能观察器exact。285项完整梯度/loss/模型参数跨runtime按原Trainer预算通过。负向控制证明读回守卫能拒绝accelerator的cpu/numpy/item/tolist/bool/float/int；尚未执行实际NPU10配对，不报告速度或显存结论。

只读审计确认当前公共checkpoint实现直接调用fn，注册Fidelity.APPROXIMATE，没有激活重算与内存节省语义。后续BF16基线使用不启用checkpoint的协议；若测试该近似接口，只能说明数值路径，不能对比其显存与原生真实checkpoint或宣称完整支持。普通FP32严格数值、真实BF16 LoRA、较大模型及有效性能仍为未完成项。

主树r22的74个Python源码按Python3.7语法完整AST检查通过；配置ruff对比52个HEAD已存在Python文件及22个新增Python文件，当前120条诊断、HEAD132条，无新增诊断身份，修复6类既有诊断。`main-final-lint-r22/comparison.json`保留逐项差异，未宣称整个仓库lint清零。

BF16原生argmax/min最小探针在`ecosystem/acl-bf16-argreduce-r1`完成native32项/128完整数组；仅加Python BF16白名单的JT候选仍16项无ACL执行路径，所有尝试被回落error守卫拦截。失败候选已恢复，未合入MAIN。第二门控位于原生ArgReduceOp构造器的IS_ACL dtype标志；双gate最小候选尚待真实NPU完整indices/values/gradient及CPU回归，不把本机CANN支持等同于Jittor已支持。

## 后续真实训练诊断与状态修复

`real-model/partial-lora-diagnostic-r2/full-array-comparison.json`只读比较失败JT FULL已保存产物与旧native FULL参考：4updates共448份完整LoRA梯度按原atol1e-5/rtol1e-3均无超差，最大绝对差8.586794137954712e-7；共有467个完整最终数值状态数组也在原预算内，但schema多出scheduler/verbose。初始完整7596951个数值中logits有622项超原FP32预算，maxabs5.7220458984375e-5。旧native source尚未证明可复用为新源正式参考，JT后续sampling/lifecycle失败，因此这只是定位证据，绝非完整跨runtime通过。

LambdaLR公共schema单独双解释器真实CPU复现：原生Torch2.7的5个合同全部通过，JT为4通过/1失败，唯一为多余verbose字段。最小修复只在LambdaLR构造后移除该legacy属性，数学递推不变；LinearLR及其他既有legacy合同保留，并新增完整8步/3+5恢复、callable对象状态/身份、输入checkpoint不变和legacy附加字段可加载的合同。MAIN-r23已整合96文件，manifest SHAfc9ac7f8ab46ad3e937eaa6294a238dc0355009e4f61c68cb11cc53cbc6db4db，受影响真实CPU71 passed/0 skipped、原生新增6 passed，LambdaLR全部新合同/StepLR/shared legacy/FSDP完整反射/legacy optimizer均通过，fallback0且96指纹前后不变；`device-api/lambdalr-public-schema-r1/acceptance-summary.json`保留命令与原失败。旧65/94训练计时fullschema失败都保留，新工具不得以共同键数值通过忽略该schema差异。

相邻FSDP反射guard的旧hash已按公开配置接口精确审计：恢复guard原commit f6a90e32的config.py能完整重建旧81c636...；HEAD305和distributed87的当前payload均为c8153c...，差异仅五个state-dict config构造器的显式参数/offload_to_cpu/_use_dtensor默认，与独立Torch2.7实例字段一致。DeviceMesh默认None在三份基线均相同，并非本轮引入的变化。只更新已有guard及解释，LAB候选真实CPU3通过；完整MAIN文件复验另行记录，不以盲替哈希消除失败。

真正空闲双卡上的`distributed/hccl-fp32-r9`新primitive完整通过：native264.287秒、JT22.514秒各退出0，16 rank-cases/完整32值payload exact，actual FP32/fallback0/noMPI/source87稳定。随后`distributed/hccl-training-fsdp2-fp32-r6`两个runtime FULL8均完成（native324.66秒/JT26.514秒），原atol2e-6/rtol2e-5全梯度、clip前后、全部参数/moments/完整groups与scheduler严格cross通过。该证据确认AdamW公共groups修复解决旧FULL数值失败；prefix3/freshresume5仍在运行，完整恢复尚不标通过，耗时包含初始化不作性能结论。

FSDP r6最终状态：native prefix3/freshresume5均完成（316.49/325.97秒）且对其FULL8完整状态exact，JT prefix3亦通过；JT freshresume在update4前失败，`compat/fsdp2/_state_dict.py::_checkpoint_tensor`把内部backend名acl直接传给公共Tensor factory，frontend正确拒绝。owner将仅修正ACL到公共npu及逻辑index映射，保留CPU/CUDA/preserve_dtype；不修改frontend接受内部命名。原两rank栈及`first-failure-summary.json`保留，18个own launcher/rank与driver已终态，空闲两卡正式交还。该owner位于训练工具11项SOURCE_PATHS，原check_oracle要求source_hashes完全一致，因此修后Native和JT三phase均重跑，不调整原判据复用异源旧参考。

## 09:05 新增梯度边界与本轮资源交接

BF16 arg_reduce 的Python与原生dtype双门控候选在生态隔离树source98完成正式native-mode 48 passed/0 skipped，完整indices/values/raw梯度和CPU回归均通过。公共Torch max(dim)/min(dim)另外按原生32 case的128个完整数组核验，仅108通过、20个梯度数组失败；indices和值全部exact，失败还涉及FP16/FP32重复值和NaN。只读定位发现ACL专用public shortcut使用会均分tie梯度的标量归约，遮住了dim接口应采用的首索引梯度语义；现有GatherACL.grad另有输出source形状错误，host最小复现已确认，实际NPU修复尚未执行。不能将raw 48通过写为公共BF16 max/min完整支持。原证据在ecosystem/acl-bf16-argreduce-r3/comparison.json（SHA1df461202c4741bfa5a45d478c59b5157990460a51174061ea6a544687777f2c），后续候选不改dim=None标量归约语义。

原生独立Torch/NPU排序参考在修正官方SDK Python路径后完成36/36：3 dtype、6场景、双序，rc0/64.56秒，完整数组archive SHAad4f94b4b1b5f458913be16370e976aecfa0f4bae433c583ea8b4106452e0a50，source96前后稳定。Jittor排序候选仍仅host/static通过，尚未真实NPU验收，不据原生参考通过宣称实现通过。此前诊断wrapper不同构建环境触及的整个JIT cfg已完整隔离，旧目录不再作为合格缓存使用；新的固定参数构建与实际runtime验证分别记账。原生首次缺TBE导入、零数学执行的基础设施失败保留，未安装额外包覆盖该错误。

新的FSDP checkpoint公共设备映射候选只将内部acl转换为公共npu及逻辑index，CPU十项路由/值合同和受影响回归通过。新source88下的r7在双方实际空闲的双卡交接后开始，六个完整world均从本轮新源启动，Native FULL/prefix/resume不复用异源旧参考，预计约18–20分钟。primitives只按四项未变proof source明确复用已通过的r9，完整恢复结果待本轮终态。

LambdaLR schema修复同步后的训练计时CPU资格r9完成：735个共有完整状态、252份全梯度与micro loss通过原预算，两个运行时各736份最终完整状态对自身功能观察器exact。原生source94参考的重用关系与新JT source96明确区分；不透明跨runtime RNG格式不作数值互换，runtime内严格恢复仍检查。新的双worker持久计时控制器开始真实CPU资格，协议为各14次功能shadow及14次交替计时更新、4次warmup与10次steady pairs，完整状态及梯度门禁保留；该工具尚无NPU性能结论。

## 09:23 FP32 FSDP2完整fresh恢复正式通过

`distributed/hccl-training-fsdp2-fp32-r7/acceptance-summary.json`（SHA40afc495b64607633420884c8b78ea5b5f9ff8a0a4939fb1b7194cefbe27047f）在新独立source88下完成Native和JT各FULL8/prefix3/freshresume5，六个world全部新启动、六个launcher均退出0，四份新checkpoint及完整12个rank-phase报告齐全。Native三phase为264.17/344.75/331.58秒，JT为31.24/29.53/28.48秒，均含初始化与观察器，不作性能比较。FULL完整梯度、clip前后、参数、moments、groups与StepLR按原2e-6/2e-5严格cross通过；两runtime各自连续8步与3+5恢复完整状态exact，JT六个rank-phase fallback0、无MPI替代。它闭环了checkpoint owner内部acl到公共npu及逻辑index的修复；两个生产文件及测试候选仍待受控合入MAIN，不把隔离树通过写为MAIN-r23完整恢复已重跑。

本轮r9 primitives仅按四项完全未变proof source明确复用，原87身份与报告字节在双manifest receipt保留，不将复用通信描述为新88重新计算。共1driver、6launcher、12rank的19组PID/start_ticks均终态，fresh npu-smi确认两张分配卡空闲后交还；handoff SHA为c127222eb5f7b14e0e39fe15aec8990a184a0e5c750689c7cadfbdccd62078dd。旧r6 freshresume失败完整保留。

配对推理controller的另一最小候选在每一pair前后、两个worker的计时区域外重新核对实际物理卡进程和PID身份；启动第二模型的保守HBM余量只用于启动阶段，不在两模型已驻留时重复套用。纯host修前5失败/2通过，修后7通过/0跳过，覆盖外部进程出现时阻断新admission及仅清理匹配原start_ticks的own worker。新测试Python3.7 AST及配置ruff通过，候选未改MAIN；`real-model/paired-resource-guard-owner-r1/candidate.patch` SHA09287694917213d74654008f1d2ea85f6048deaa4401376def19b5b4f27596cc。

另按官方确定性状态首次下发的限制，准备新的原生CE独立进程实验：两次分别在首个NPU初始化/算子前设置True，另一个fresh False控制，每进程40次同一固定CE输入。此前同线程warmup后False/True/False的60次结果保留，但不据其排除fresh True-first条件。新实验已在实际交接的空闲卡启动，结果待终态；不改变训练loss exact判据或公共shim实现。

## 原生CE首次确定性状态的独立进程复核

`real-model/native-ce-determinism-fresh-r2/result.json`（SHAad93fcd5d4bed60223e3f1afa3e9b60a048708e2953901f08d64fa0b5db172ed）已完成三个独立原生进程，共120次固定54×151936/9 ignored labels的CE forward/backward。两个进程各自在首个NPU初始化及算子之前仅设置一次torch.use_deterministic_algorithms(True)，各40次loss均唯一bits1082722718，完整梯度SHA均为1ffab42fe823ac052729c7de3464e52d0dd05d602b075abb302380a11fa7237b，进程之间也完全相同；fresh False控制40次出现1082722717/18/19三种loss bits。三worker均退出0、实际物理PID witness通过、BLAS实测恢复2线程，原输入SHAfc8006d64d4a3cdbf7d1c1c789a874109015f149db868e2f5308f2fafbffeca8与旧实验一致。说明此前warmup后在同线程切换开关不足以测试True-first条件，不应据旧模式排除原生确定性配置。官方关于首次下发状态的说明只用于提出实验，安装二进制的具体结果由本轮实测支持：[确定性计算API支持清单](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/determin_API_list.md)。

这仍是固定CE诊断，不能替代真实LoRA完整训练恢复。后续新增r4准备协议将原生参考明确配置为True-before-first-kernel，保留全部原数值预算与same-runtime strict0状态/轨迹要求，不改变production CE实现；未强制原生确定性的r3准备脚本及旧r1失败保留。Jittor公共torch.use_deterministic_algorithms仍标Fidelity.UNIMPLEMENTED、getter为False，本轮不伪装该API已支持，也不以此比较共同确定性policy下的速度。r4禁用尚无重算语义的checkpoint接口，尚未执行真实模型。

三个CE worker、driver及waiter共五组原PID/start_ticks均已消失，source96前后稳定，fresh卡空后交回；handoff SHA21f5ddb2cbf60ccf14a4bab31fdac0cc8cf5b9ab084c82fabc69bff8944c333e。新完整ACL空cfg275个编译单元已完成并保持全部原build ingredients exact；空mask下C++静态初始化因GetDeviceCount207004以原始rc=-6退出。外部post-terminal assessment严格区分完整ELF/stamp产物与尚未验收的实际runtime，不改写原始returncode；当前下一项为分配卡真实加载及argsort候选测试。

## 10:20 MAIN-r25整合及当前验收边界

MAIN-r24已受控整合argsort原生执行/任意cotangent完整梯度、BF16 arg_reduce双门控、FSDP checkpoint公开设备映射、public max/min(dim)及Gather梯度、逐pair物理资源守卫，以及持久Trainer训练计时工具六组增量。115文件指纹为main-source-r24.json（SHA c1373d36fbf2e053b55a387b731d4ae34df2527b9cb2f3ee6fbcf76234f8221a），整合receipt SHA41591470c18c0286febaff58c8e75bf8493122d3015eb8f69f6967c2f7a97765。Python3.7 AST与git diff --check通过。仅tensor.py的BF16白名单与argsort候选重叠，合成后逐项逆变换能重建原argsort字节，没有覆盖整文件而丢弃另一修复。

MAIN-r24实际生产路径、无overlay的CPU回归已完成90 passed/0 skipped，含FSDP/checkpoint62、DDP identity6、StepLR11及shared11。worker118.59秒，实际增量构建24.76秒，fallback0，115原指纹及5项额外生产依赖前后稳定。凭据distributed/fsdp-checkpoint-placement-owner-r1/main-r24-cpu/acceptance-summary.json（SHA a9a39f337216385bc9453e39ec298232fdfdf824c434f31016261d623d795ac2）。该证据没有被写成MAIN已重新完成实卡FSDP全训练，后者将在双卡交接后运行。

ACL argsort首MAIN生产90为66P24F0S，全部完整值、排列及任意cotangent梯度数值断言通过；24F均为FP32在NumPy取数后检查原变量location。fresh阶段探针证明确实是identity FP32 cast的D2H取数改变原Var存储位置：取数前values/indices/gradient实际device、dispatch acl:0；FP16/BF16 cast生成另一变量而未改变原存储。原90失败完整保留，探针report SHA101eeb24a91f33b9ddba2072b41a29e309e657b923e52127b41372df5873d25d。r25仅移动原驻留断言至同步物化后、任何host fetch前，未变更生产源码、90项矩阵、数学或阈值；数值AST同一性证明SHA72254f6a48620c56dd624b7ffd7c4753c6982955fe592de4fdcced7a7598442c。新manifest main-source-r25.json仍115文件，SHA101b411fcb7cf7a8ef35f00e95d4d5fc9f0ba7c082aae467d03d5258fc5400c8；receipt main-integration-r25/receipt.json SHA18732241cb1bd25dcbe785e805e820c22527516c1a4956de2b5b487c7f2a238a。

新MAIN-r25正式生产90项已90 passed/0 skipped/rc0，pytest9.50秒、worker21.24秒，actual physical5/fallback0，115源码前后exact；stdout SHA1752f6588a1ccc506e832a67b7ad2bc026e186f68d2e1edf571f311b7b6f5afa。公共反向、fresh双方36及sampling继续串行，完整structure并行在独立CPU cache运行；layout已rc0。未终态项目仍不标通过。

public max/min(dim)最终候选在ECO原104文件基线真实NPU上完成Native153/JT153全部通过，完整case inventory/值/索引/任意cotangent梯度齐全，actual ACL/fallback0；最终1D只经既有gather owner，避免原1D IndexPut561103失败。另Native CPU68/JT CPU68通过，global CPU默认而显式NPU输入的gather/max/min反向三case也全部通过；原CPU后置NaN六项语义差异保持修前修后equal_nan完全一致，单独保留边界，没有降低ACL NaN判据。原raw BF16 arg_reduce native-mode48P保留其独立98基线。两个修复已按最终字节整合MAIN；153旧树通过不会冒充MAIN原生全套重跑。

持久Trainer性能工具最终CPU资格ecosystem/trainer-perf-skill-r1通过：两个常驻runtime各14次functional shadow与14次交替更新，4warm+10steady pairs共28独占grant；735份完整state、252份full gradient及全部micro loss按原预算cross通过，runtime内各736最终状态对功能shadow strict0。20项host合同覆盖有界连接、own PID/start tick安全清理、dtype/schema/RNG元数据等负向控制。CPU资格result SHA787e0535ae39e12c6092b31ad0f43578f6c798e1ce91f6fc2750adb5c3c2919e。工具已迁入canonical生产路径及MAIN；尚未提供NPU训练速度或显存结论，原生确定性True与JT公开API False/UNIMPLEMENTED也不能混写为相同policy性能。

ECO为后续较大模型保留独立暖cache，已完整镜像MAIN-r25全部2853路径并核对内容、symlink target和mode，twin proof SHA092bdd83298a76f02b8b1e5d6c0ce2914ed9acfdb7563885ca4cb237a0619709；唯一增量test观察顺序，其他2852路径字节/mtime未变。新ECO unified-source-r19.json SHA20ab173c89ba4cbb37417533dd8fdd625b0f3087c96b0027b8a0689464166739的115内容与MAIN完全相同，但保留不同实际tree身份。旧全树完整备份保留，旧source104验收不自动继承。真实模型继续优先0.6B BF16 LoRA与1.7B/4B推理，实际设备交接和对应功能通过后才进入十对配对计时。

## 10:34 新115真实模型与采样关卡

同bytes MAIN-r25/ECO-r19的新tiny GPT2门禁已用fresh Native/JT重新完成，完整forward error0、28份完整梯度worst3.3383983e-7，沿用原5e-3/2e-2门槛。两进程102.66/90.99秒，JT含正常276单元增量检查；逐tensor同步后actual location=device、acl:0、physical3 witness/fallback0、双方实际BLAS2。新gate为ecosystem/current-source-tiny-r1/result.json（SHAc15cc2c4a71151fe9c8b6b8b8bf21f361352423f4b85c47b7b6dd837240801ac），保留ECO/MAIN双manifest及twin proof，不引用旧54文件tiny外推当前源码。

Qwen3-1.7B普通eager/pipeline0、BF16、B1S128已在同一115字节的实际ECO路径完成fresh Native/JT完整功能对拍：15数组全通过，三槽58,343,424个logits全部逐bit exact/maxabs0，完整input/mask及cached+uncached greedy8 token均exact；311模型张量和57份prefill张量actualACL0、PID physical3、JT fallback0、双方HF32关闭及实际BLAS2。Native145.84秒/JT197.32秒/compare2.53秒是含初始化和完整观察器的功能墙时，performance_accepted=False。summary ecosystem/larger-models-preparation-r1/17b-functional-summary.json（SHA310d06a0e27e1f08aaddfe89d82f04f062a3e17f84a4187afca46464f97bca02）。9个own PID/start ticks全终态、fresh SMI3空后正式交还，handoff SHA8f7758d38f06381fd66ff8a728c194b9d1044077ef8f2131a9a34387ff82d8e1；paired及4B尚未启动。

fresh Native36及JT36排序全场景档均生成，完整288数组合同通过。最初LAB comparer额外要求的6处FP16 NaN payload不同，定位为F32 host transport后再astypeFP16得到的重编码位，并非原NPU原始位；原failed comparison和两档完整保留。只修LAB不实的rawbits假设，仍要求NaN位置/类别一致，所有finite/Inf位及完整梯度exact；18项host含NaN错位/类别错误、finite一bit差、Inf符号及梯度差等负向均拒绝。新compare SHA144f72d41327fcf91b69514b47f78836f4526a8f5eae03fddee8a247e564f8c1，不改变生产排序、fixture或数学门槛。

fresh TopP sampling检测出独立的新失败：Native同协议82.82秒通过，JT在TopP warper的bool切片赋值触发CANN8.5 StridedSliceAssignV2 workspace query161002，bool输入不在该vendor算子支持dtype内；这不是argsort值或梯度失配。失败worker60.56秒退出1、115前后稳定、physical5/fallback0，日志SHA7d5a61ede3737e56c1416e0ddd3fa2efa38a16a16014d615790fb619f6930363。后续仅在既有canonical setter owner准备bool typed lowering，不为Transformers写私有绕路；候选未合MAIN，真实LoRA的sampled生命周期仍未验收。

完整MAIN-r25 structure正式终态1422P8F2S、1432个不同case、426.49秒；原两CUDA skip保留。8F均为新增测试/包装约定：trainer测试的collection加载与裸sys.executable/PYTHONPATH三项，LambdaLR测试跨test import一项，新增argsort后的源码计数/direct owner三项，以及generated MANIFEST漏两个argsort .h/.cc一项。三个owner已备7路径纯test/packaging候选，原policy不变，新增和迁移的所有真实负向保留；官方manifest仅加两include、无删除，原typed guard缺dispatcher/缺query两负向仍拒绝。scheduler五helper函数逐AST exact移入共享helper，两组原测试AST不变，Native17P/JT17P。下一轮须在当前worker和cache预热全终态后整合并重新跑实际完整structure，当前不标全套通过。

## 11:10 主树采样与最终配置回归

MAIN-r26已受控整合九个真实文件路径，120文件manifest SHA6e5f6bcccf3dc83114ee05fbcdcc461bf7cd12ba9e079bf710f1c99ca5f8ad35，整合receipt SHA5b88becc0a635fd49d275c7e3e355759ba12d4235a07a88b340a6dcf230a2b5d。其完整structure1430 passed/2 CUDA skipped、1432不同case，原8F全部转通过且无删除，408.53秒；布局通过。布尔setter仅bool基本slice走精确Bool→I8→Bool，非bool别名路径及原grad函数保持，None新轴按singleton-view处理。原生CPU9/JT独立CPU9、v1实际NPU+CPU16候选资格均保留；r26实际无loader正式MAIN测试18P0S/fallback0，6.70秒pytest/21.56秒进程。

新MAIN120采样完整链FP32[2,151936]、temperature0.8/topK10/topP0.9/softmax/multinomial通过，实际physical3/fallback0/source120稳定。与明确复用的原生115/physical5参考完整九数组通过，输入与mask exact，303872个概率、温度及TopK/TopP完整scores maxabs/maxrel均0。两个runtime采样ID各自合法且对应概率非零，未要求不同RNG实现产生相同token。comparison SHA17ef1cd64fb6cfa3e3d39b8a52a480aeee78a820eba5b5dc0ff0c7cc4b8ddc40；formalhandoff SHA5c07ca1794a14080901389bda98ab697626e5000dc096aaaed35026bfcbb1059，六PID全gone、cacheusers0、fresh3空后交接。

普通pytest在compat子目录会选择最近的compat/pyproject.toml，原严格marker配置未注册npu；原collection error保留且为0数学执行。MAIN-r27只增加该marker，121文件manifest SHA705ead0587b1464e5ec762cb190addcde3142549ad94f2a1fd92e9ebf03de436，原120字节全部exact；receipt SHA294fc8abd78658d903bb125a96e0f88eebf615ebf84b5062ffbf13f51ecbbe43。独立Native CPU环境已实际先复现旧配置collect rc2，再当前正常nearest-config collect18 rc0和CPU9P；没有用强制根配置代替最终修复。全部95 Python AST3.7通过，configuredruff163对HEAD175、精确(file,code,message)多重集无新增，修复12项既有诊断；证据device-api/main-r27-host-final-r1/receipt.json SHA0306618e547a280d58ecc25a48779615cc56e8564891b64057232b7af2194c80。最终r27全structure、相关原生Jittor core四文件和nearest-config CPU9另行在独立CPU cache运行。

资源在共享主机上发生变化：5上出现外部VLLM服务，3亦间歇出现外部Python，启动器均在0worker时阻断。原R4 0.6B LoRA只创建了启动记录，未执行训练；随后fresh全卡表显示6与3均空，Root按新lease使用6、ECO使用3。R5实际BF16 LoRA worker52706/start628322899在6进入真实模型前向，47.07秒退出1，尚未完成任何update：PEFT lora_A已产生BF16 activation，lora_B为FP32master；ACL快速mapped_matmul在既有混合dtype AMP策略前直接调用typed raw op而拒绝。日志SHA48f3dbb7a73ac6f72bd27f2e031abe02eff72eaf97dbd285dd57baaf50dbd901，source121前后稳定、实际physical6 witness成立。该失败不是采样问题，后续在canonical AMP/matmul owner准备最小复现与显式cast图；真实BF16 LoRA完整训练恢复仍未验收。

ECO最终r27镜像清单unified-source-r21.json SHA7cb255e90bb3afd3bf85e5c9992909ee103bc9fc2e520185e06f19587f7a779e，2855路径twin proof SHA5b84fbe9e949dfd620659a07c029df03dc8ed14ca81628fee58c92f93c9c6c80。组合preflight gate SHA290fe2cb68d519c05aaed91f23514e2a4c030979954ef798dad01f7e3a2e09a7保留fresh115 GPT2全28grad及当前120 bool18/完整sampling来源，明确fresh_current_source_GPT2_rerun=false；机器验证115到121唯一runtime改动为不可微bool setter，其余为test/packaging/marker，未把历史tiny写成121重新运行。新4B ordinary BF16 B1S128功能已freshNative启动，尚无终态比较；后续每模型新格完整功能通过后才能进入十对性能。

受单卡波动及新增真实训练阻塞影响，后段实际队列统一11:30停止新计算phase、11:42 hard-stop，为12点左右最终核验、文档、本地中文提交与交付留时间。所有外部任务均未收到信号、未读其environ；双卡MAIN复验保持NOT_RUN_RESOURCE，独立own88/r7完整通过与MAIN新world0记录分开。

## 11:41 最终 MAIN-r28：限定 AMP 实卡通过，真实 LoRA 遇资源阻塞

`main-source-r28.json` SHA256为`d3904da402f4a0c0b17d61102ec8bf704c97391f338415d7f56ac567e311510e`，123文件冻结。此次仅ACL AMP matmul owner与正式测试两路径，原121文件逐字节不变、C++增量0。仅在活动AMP且eligible mixed浮点输入时复用既有dtype策略插入可微cast，FP32 master及梯度dtype保留。

actual MAIN无候选loader实卡67.21秒：正式10 passed/0 skipped、fallback0；FP16/BF16 activation、FP32 master的两步Linear/LoRA目标子集84/84完整数组、2880元素bitexact。完整168原FP32 strict probe仍FAIL：164数组bitexact，4个FP32input LoRA input_grad非bitexact，其中BF16两数组45元素超原阈值、maxabs2.74658203125e-4；原阈值不改，不将限定84通过写成全168通过。关闭AMP时Native vendor支持的mixed公共扩展与Shim严格raw限制仍单列。

actualMain交接`device-api/acl-matmul-amp-mixed-r1/handoff3-main123-terminal.json` SHA256 `08a70620dfc2a384154520e8eb2f1f42d7bb3876f49ae501abd41c68d19e77c4`；完整比较SHA256 `e4ab96205cc52333f5a51f1218ce5c3c866d6c576952ec9651fa2f598c2ae3a7`。自有worker/cachewriter全部退出；物理3被UID10028的vLLM任务占用，未触碰。物理6短暂空闲后，Root新鲜admission又检测到占用，真实BF16 LoRA R6未启动driver、0计算worker/0update，保留`real-model/qwen3-06b-training-npu-r6/admission-blocked-r1.json`及raw snapshot，不以瞬时空闲推断资源预留。

最终host receipt `ecosystem/final-delivery-host-r1/final-receipt.json` SHA256 `e14d1cd1ed96e80f379122800f18b511fdddb70c14f0501889886c95ce61293c`：97.py及1.pyi通过AST3.7；配置Ruff当前163/HEAD175，按path/rule/message多重集合新增0、减少12。四私有环境完整包集合/版本/METADATA/RECORD未变，overlay21文件immutable、13.py等于最终MAIN。ECO两大模型实际源码仍121，未冒充123重跑。最终123完整structure/layout在CPU独立缓存回归中。

## 11:54 最终真实 BF16 LoRA 保存与 fresh 恢复闭环

最终 MAIN-r28/source123 的真实 Qwen3-0.6B BF16 LoRA continuous full（R6）完整返回0，4 updates/8 microbatch、112 FP32主参数，448份完整梯度及448 dtype标签均保存且梯度finite；普通评估/预测、greedy与sampling生成、adapter保存重载/合并等既有完整生命周期检查完成，实际NPU/fallback0。训练诊断94.66秒，端到端356.79秒，包含初始化/检查/文件I/O，未作为性能指标比较。

R6遵守原11:45新阶段截止，未再启动独立prefix或Native，原stop记录保留。随后复用其真实checkpoint-2启动一个全新Jittor进程（R7）恢复剩余2步：恢复载入状态exact、global step2→4；既有compare的`--first`原生支持continuous prefix，按atol=rtol=0核对933个完整最终状态数组及样本cursor、更新数、逐micro loss、LR和梯度范数轨迹，全部通过。额外对step3/4每步112份完整梯度及112 dtype标签逐数组核对，224梯度及224标签均bitexact；未用norm代替完整梯度。R7包含完整保存/生成/重载，端到端120.73秒。两实际worker和compare子进程及父driver均已退出。

最终证据`real-model/qwen3-06b-training-npu-r7/closeout.json` SHA256 `69c34f256d848389f2e751bd3275c2cad072b399355fa2e06bcbc2cbabd9b7b3`，strict0比较SHA256 `f3b42f3423ac1c5a5caaa8264d172d32f9d8463d3f420b9a1922f04446b22234`。这是dropout0/顺序fixture的Jittor自身真实恢复结论；未运行原生PyTorch同模型BF16完整训练交叉，不声称双runtime full/prefix/resume六进程协议完成，也不声称随机dropout/通用确定性恢复。

最后只读核验最终123文件与manifest全部相同，完整structure1430P2S/layoutP已结束；本任务进程扫描无存活工作进程，原生大模型对拍/最终双卡/NPU配对性能保留未完成项。私有安装和共享资产保持原范围，未杀外部进程或清理他人文件；交付索引和原始日志在仓外lab。本地中文提交只包含明确123源码/工具/测试/技能路径与3份报告文档，不push。

## 2026-10-22 910B3 生态续验：真实模型交叉、输入梯度定位与 FSDP2 资源阻塞

### Qwen3-0.6B BF16 LoRA 原生交叉对拍证据补齐

LAB 中已有的 R37 cohort 完成了同一真实 Qwen3-0.6B BF16 LoRA 在 Native torch_npu 与 Jittor shim 两侧的 full、prefix、fresh-process resume 交叉验证。该结果绑定 2500 文件冻结源清单 `real-model/qwen3-06b-training-npu-cat-fix-r37/source-manifest-r1.json`，SHA256 `aa56c083de1404dcfc744b31edc3dcb973f574d3dcb0d048c691d8ffa400d383`；不是当前 MAIN 的隐式新验证。4 次 optimizer update、两边各自的 2+2 fresh-process 恢复均有终态，所有实际训练阶段都有 NPU witness；Jittor fallback_count=0。交叉比较共 2187 项，失败 0、超预算元素 0，其中精确比较 1360 项、forward 138 项、gradient 684 项、采样 RNG 标识 5 项。Forward 沿用 atol=0.005/rtol=0.03，gradient 沿用 atol=0.01/rtol=0.05，没有改预算。证据见 `real-model/qwen3-06b-training-npu-cat-fix-r37/cross-validation-summary.json` 和 R37 cohort 内两种 runtime 的原始阶段报告。

R22 在当前工作树逐项核验该清单的2500个文件，全部 SHA256 与 R37 manifest 一致（0 mismatch；证据为 LAB 的 distributed/r22/current-source-manifest-check.json）。因此 R37 的 Native/Jittor 对拍对应当前树中这2500个清单覆盖文件的相同源码内容，不是跨不同冻结源码外推；这仍是 R37 的冻结执行记录，不是当前 HEAD 新启动的一次模型运行，也不覆盖清单外文件、清单外环境差异、随机 dropout、通用确定性恢复或性能结论。MAIN-r28 已有的 Jittor 真实 LoRA 4-update 训练、保存和新进程恢复仍是独立证据，见本报告 11:54 记录。

### 修复 HCCL 验证器的过时 facade 假设

主树 `agent/skills/jittor-distributed-verification/scripts/verify_hccl_stack.py` 曾断言 shim 运行时 `torch_npu` 不得存在；实际兼容层会发布带 `__jittor_acl_facade__` 标记的 facade，因此 R5 的 Jittor FSDP2 在进入模型前被错误拦截。本轮将断言改为核对 facade 标记、provider `jittor.backends.acl` 和精确入口 `compat/torch/installers/npu.py`。独立 R10 smoke 使用实际 NPU 5、实际 ACL dispatch 和零回落，核验该 facade 与 Jittor 安装上下文，没有加载 Native oracle；结果在 `distributed/main-tree-fsdp2-cat-fix-r10-facade/facade-probe/`。该结果验证了守卫条件本身，不等同于完整双卡训练。

### MAIN FSDP2 进度与资源记录

R5 的 serial single-device Jittor/HCCL 预热（rank roles 0/1 顺序运行、物理卡 5，明确 not_two_rank_proof）、FP32/BF16 本地 collective 检查与 DTensor 设备工厂均通过；Native torch FSDP2 full 在物理卡 5/6 实际完成 8 updates，rank 报告为 PASS。Jittor 的 R5 首次尝试在模型前撞上上述过时 facade 断言，完整失败输出仍保存在 `distributed/main-tree-fsdp2-cat-fix-r5/jittor-full-logs/`。R10 修正后的 facade smoke 已过，但没有把它冒充为 FSDP2 全流程通过。

R6 在卡 5/6 初始空闲后启动 Native full；运行 17.5 秒左右发现外部 PID 940643 进入卡 6，driver 只终止本 cohort 的进程组并以资源中止结束，未形成模型报告。R7 在启动 worker 前检查到卡 6 外部 PID 948533，未启动计算。新建的 R8 在 2026-10-22 的 fresh 快照中卡 5/6 均空，但首次 runtime admission 又观察到外部 PID 1125153 占卡 6；driver 在创建首个 runtime launcher 前退出，`compute_workers=0`。R8 快照及 admission 记录位于 `distributed/main-tree-fsdp2-cat-fix-r8/npu-smi-before-attempt.txt`、`npu-smi-before-torch-full.txt`、`npu-smi-after-block.txt` 和 `admission-blocked.txt`。R8 的预热材料与 Jittor 专用缓存从 R5 复制，Native oracle 缓存没有复用；因 admission 失败，R8 不作为验证 cohort 重跑。MAIN-r28 双卡 Jittor FSDP2 和恢复交叉仍待下一次全新、连续可用的双卡窗口。

### FP32 输入梯度差异的分支诊断

R6 输入梯度分支探针在 NPU 5、生产 shim、fallback_count=0 下记录了 74 个数组。72 个数组完全一致；只有 LoRA 两步的 `total_input_gradient` 不同，共 2784 个元素中的 77 个不同值，最大绝对差为 `2.74658203125e-4`。输出、分支梯度、各自的输入梯度贡献、参数梯度和参数更新数组都完全一致。逐项算术验证显示，在这个 fixture 中 Native 结果等于“先以 FP32 合并两支贡献，再舍入到 BF16”，而 Jittor 结果等于“两支各自先舍入到 BF16，再相加”。完整诊断见 `device-api/input-grad-branches-r6/comparison.json` 及两边的 `arrays.npz`。

这是对既有 MAIN-r28 严格 FP32 输入 LoRA probe 失败位置的窄化解释，不是修复，也没有调整数值预算。本轮未改通用 autograd 累积逻辑：现有证据只覆盖该小型分支 fixture，尚未证明一个全局改动在其它 dtype、视图和多分支训练中的安全性。FP32 / 部分混合精度输入梯度差异仍开放，修复后需要原严格矩阵和真实模型梯度重跑。

截至R40，尚未得到合格的主树双卡 Jittor FSDP2 完整训练、恢复交叉或 NPU 性能结果；共享卡资源变化期间未接触外部进程、驱动或公共环境。后续R42结果见文末。

### FSDP2 后续 admission 复核（R11–R14）

为避免覆盖 R8 失败结果，R9、R11、R12、R13、R14 均使用独立 LAB cohort 目录。R9 fresh 检查时卡 6 有外部 PID 1135400。R11/R12 的临时 shell admission 使用了错误的 GNU grep 扩展正则：R11 原始快照实际有卡 6 PID 1143642；R12 原始快照中的卡 3/5 当时为空，但 shell 仍误报阻断；两次均未运行 Python driver 或创建 worker。审计写入 `distributed/main-tree-fsdp2-cat-fix-r11-r12-guard-audit.txt`。从 R13 开始移除该外层 grep，直接让已审阅 driver 解析 fresh `npu-smi`，并保留其逐阶段检查及运行中外部进程守卫。

R13 在首个 runtime launcher 建立前由 driver 捕获物理卡 3 外部 PID 1151694；R14 捕获物理卡 6 外部 PID 1157188；两者都在模型计算前退出，worker 数为 0，完整 admission 及退出后快照分别保存在各自 `distributed/main-tree-fsdp2-cat-fix-r13/` 与 `r14/`。截至 R14 后的快照，卡 3/6 仍有外部进程、卡 5 空闲。没有对这些 PID 发送信号或读取其环境。

### R15：runtime-entry 外部占用中止

R15 在 fresh admission 时卡 5/6 为空，使用与 R5 相同的 2500 文件清单（SHA256 aa56c083de1404dcfc744b31edc3dcb973f574d3dcb0d048c691d8ffa400d383）启动 Native torchrun full。运行期间 4.36 秒后，live monitor 在物理卡 6 发现外部 PID 1177475，向本 cohort 自己的 process group 发送终止信号，driver 返回 125。该次已有 torchrun launcher，因此只记录为 runtime-entry 资源中止；stdout 为空、没有 rank 训练报告或可接受的 FSDP full 结果。R15 的 execution.json、foreign-npu-occupants.json、空 stdout 和 before/during/after npu-smi 快照均保存在 `distributed/main-tree-fsdp2-cat-fix-r15/`。中止后核验 PID 1177542 已退出、该进程组为空；外部 PID 1177475 仍在卡 6，未向它发送信号或读取其环境。

### R16：部分 rank 启动后因外部占用中止

R16 将物理卡 3/5 作为 fresh pair，Native torchrun full 启动后运行 45.34 秒；live monitor 在物理卡 3 发现外部 PID 1192793，driver 返回 125 并终止本 cohort 自己的进程组。两份 `torch/full/rank*-process-start.json` 已保留，但没有终态 `rank0.json/rank1.json`，torchrun 的 rank stdout/stderr 均为 0 字节，因此没有可接受的 Native full 结果。driver PID 1192335 与其 process group 在退出后核验为空；post-abort snapshot 显示 PID 1192793 仍在 NPU 3，未触碰。原 execution、foreign occupant 与完整快照保存在 `distributed/main-tree-fsdp2-cat-fix-r16/`。

### R21–R22：串行 JIT 预热与双卡 admission

R21 首次双 rank Jittor 启动时，两个 rank 同时从冷状态编译 jittor_core，约到 24/295；这不符合“首次 JIT 串行”的运行约束。我核验自有 launcher PID 1237382 的 start time 634440208 与 PGID 后，只向该进程组发 SIGTERM，并确认 PID 与组均退出。该段并行编译日志、执行记录和启动前缓存快照都保留在 distributed/r21/，不作为模型验证结果。

随后按 rank 顺序做串行 core import。rank0 在单进程下完成 295/295，日志标记 SERIAL_JITTOR_CORE_IMPORT_COMPLETE True，返回码 0，has_acl=True。rank1 的第一次 import 返回 1，提示 jit_utils was rebuilt，原始日志保留；同命令重试后在 rank1 独立缓存目录中串行完成 295/295，返回码 0、has_acl=True。两次均只把空闲物理 NPU5 设为可见设备，并先后运行；rank1 原始缓存先做了 LAB 快照，再从 rank0 完成的 core 缓存复制，rank1 随后仍实际执行完整重编译。此项只证明两个 rank 的 Jittor core import 预热成功，不证明模型算子、HCCL 或双 rank 执行。日志、cache snapshot 与边界记录位于 distributed/r21/serial-jittor-import-rank0.log、serial-jittor-import-rank1.log、serial-jittor-import-rank1-retry1.log、rank1-cache-before-serial-copy/ 和 distributed/r22/prewarm/prewarm-complete.json。

R22 使用 source manifest SHA256 aa56c083de1404dcfc744b31edc3dcb973f574d3dcb0d048c691d8ffa400d383、物理卡5/6及隔离 rank cache。启动前 fresh npu-smi 快照显示5/6均无进程；driver 再次准入后启动 Jittor full launcher，live monitor 在1.884秒后发现外部 PID 1443945 占卡6，按防护逻辑仅终止本 cohort 自己的 launcher 进程组，返回码125。launcher PID 1449813 及 PGID 已核验退出，rank0/rank1 日志均为0字节，没有 rank worker 进入训练，也没有 full/resume 报告。外部 PID 1443945 随后自然退出；约30秒后的 fresh 复查中，卡5仍空、卡6又出现外部 PID 1452009。没有向任何外部进程发送信号。完整记录在 distributed/r22/，包括 execution.json、foreign-npu-occupants.json、driver 输出、rank 空日志及启动前/运行中/事后快照。

因此，本轮完成了串行 core import 预热，但没有得到 MAIN-r28 Jittor 双卡 FSDP2 full、prefix/resume 交叉或性能结果；卡5/6下一次 admission 仍须使用新 cohort 与新的即时空闲快照。

### R23–R24：HCCL/DTensor 串行预热与双卡 admission

R23 在5/6 fresh 空闲快照后启动双卡 Jittor full。约10.97秒时旧 live monitor 将 PID 1476170 识别为外部占用并返回125。后续把占用 PID 与 cohort 的 rank 身份记录核对后，确认它正是本轮 Jittor rank0（物理卡5）；这是监控误判，driver 终止的是本轮自有 launcher 进程组，不是外部任务。rank0/rank1 日志分别有706和555字节：rank0 已开始 setup_hccl 的 MPI-free HCCL 算子编译，rank1 到达 Jittor import；没有进入模型训练或产生训练阶段报告。launcher PID 1476146 的进程组已核验为空，事后 fresh 快照显示卡5/6均无进程。该次仍暴露 HCCL 算子首次 JIT 需要预热，但不计作外部资源抢占或 FSDP 结果；记录在 distributed/r23/。

R24 随后复用相同的冻结源清单，在物理卡5上按 rank role 0、1 顺序运行 R5 的单卡 HCCL/DTensor prewarm worker，并分别使用 R19 的 rank0/rank1 独立 Jittor cache。两份报告均为 PASS：world_size=1 的 FP32/BF16 HCCL collective 与 DTensor factory 检查通过，实际物理设备为5，fallback_after=0，source_unchanged=true；rank0/rank1 返回码均为0。prewarm-complete.json 记录报告 SHA256（rank0 018f6abdc1197191e372b1acef28328516a41186c1dd53bed0ae51c41b8cffbd；rank1 9eacf3d773b7169ca7c2228aee0b1554e1fd69b7ac7ec04c108e852d763ae140）。这仍是两个顺序单 rank 的 warmup，不是双 rank HCCL 或 FSDP2 证明。

R24 双卡 driver 校验 current-source manifest 与 R5 Native baseline 后，在 runtime launcher 创建之前的 fresh admission 检查捕获物理卡6外部 PID 1493168，直接退出。没有创建 launcher、rank worker 或模型输出；admission 前后 npu-smi 快照与 driver traceback 保留于 distributed/r24/。不曾向外部进程发送信号。

本节扩大了串行预热范围到 core、HCCL collective 与 DTensor factory，但 MAIN-r28 Jittor 双卡 FSDP2 full、prefix/resume 交叉仍没有运行通过；R23 是旧 watchdog 误判、R24 是 launcher 前资源阻断，两者均无性能数据。

### R25–R29：多物理卡尝试与监控身份复核

R25 使用 fresh 空闲的物理卡3/5启动双卡 Jittor full，约10.64秒时旧 live monitor 将 PID 1524441 报为卡5外部占用并返回125。身份审计确认该 PID 是本轮 Jittor rank1；这是 watchdog 误判，不是外部任务抢占。两份 worker 均写出 process-start.json，但没有训练终态；rank0 的 HCCL ops 已加载并开始 communicator 初始化，rank1 到达 Jittor import。自有 launcher PID 1524412 的组已退出。R25 原始日志与 before/during/after 卡表保存在 distributed/r25/。

R26 建立新根目录，fresh 快照发现卡3已有进程，卡5空闲；因此没有启动 Python driver、launcher 或 worker。随后 R27 用 fresh 空闲卡5/6再次进入 driver，但 driver 在创建 runtime launcher 前捕获卡6外部 PID 1532400 并退出，rank worker 数为0。两次快照分别保存在 distributed/r26/、distributed/r27/。

R28 再用 fresh 空闲卡3/5进入双卡 Jittor full，约11.61秒时旧 monitor 将 PID 1610224 报为卡5外部占用并返回125。身份审计确认该 PID 属于本轮 Jittor rank1；这是 watchdog 误判。两 rank 都有 process-start.json，rank1 到达 HCCL communicator 初始化；无训练终态或完整参数报告。自有 launcher 进程组已空，事后卡表中3/5/6为空，证据在 distributed/r28/。

R29 用 fresh 空闲卡3/6启动后约2.14秒，live monitor 捕获外部 PID 1992911 占卡6，driver 返回125；rank日志均为0字节。自有 launcher PID 2013322 及进程组已核验退出；事后卡表仍显示卡6有外部进程，未向其发送信号。完整记录在 distributed/r29/。

这些 cohort 均复用 source manifest SHA256 aa56c083de1404dcfc744b31edc3dcb973f574d3dcb0d048c691d8ffa400d383 与 R24 串行单 rank prewarm 证据，但都没有得到 Jittor FSDP2 full 的训练终态；R25/R28 虽启动到 worker setup，也没有可接受的模型更新或恢复比较，原因是旧 watchdog 把本轮 rank 误判为外部 PID。R29 的外部占用没有匹配到本轮 rank 身份。MAIN-r28 双卡 Jittor FSDP2、prefix/resume 与配对性能仍未验收。

### R30：3/5 runtime admission 抢占

R30 在 fresh 快照确认物理卡3/5为空后启动 Jittor full launcher；运行中1.80秒，live monitor 在卡3发现外部 PID 2267027，driver 返回125并终止本 cohort 自己的进程组。rank0/rank1 stdout 均为0字节，没有 rank worker 或训练状态；launcher PID 2268069 及 PGID 已核验退出。事后 fresh 快照仍显示外部 PID 2267027 在卡3，未向其发送信号；记录保存在 distributed/r30/。

## 2026-10-22 续验：梯度累加修复、Qwen 对拍与 FSDP2 监控审计

### BF16 叶子输入梯度累加修复

device-api/input-grad-branches-r9/comparison.json 记录修复前的分支诊断：74个数组中72个完全一致，LoRA两步的总输入梯度有77/2784个元素不同，最大绝对差为 2.74658203125e-4。逐分支复算显示差异来自低精度分支合并时的舍入位置。R10 曾把 AmpGradGuard 扩大到所有梯度合并；该窄探针变成74/74 bitexact，但以当前真实 Qwen 比较暴露副作用：R38 的2187项全交叉中有2项 LoRA-B 梯度超出既有预算（层4、5 q_proj，最大差分别约 0.012847、0.014148）。保留这次失败，不改阈值。

随后将修复缩窄到 src/core/grad.cc 的 FP32 backward leaf 累加：只有目标是 backward leaf 且 dtype 为 FP32 时启用 AmpGradGuard，中间 activation 梯度合并仍保持原路径。当前源清单 ee0146… 下，R11窄探针74/74数组 bitexact；并在 R39真实 Qwen3-0.6B BF16 LoRA Native/Jittor full 对拍中，输入梯度、112个trainable参数的梯度、参数更新及训练/恢复状态均通过原比较预算。未据此宣称所有 FP32 或混合精度输入梯度组合均已覆盖。

### 当前源清单上的真实 Qwen3-0.6B BF16 LoRA 对拍

R39绑定2500文件 manifest SHA256 ee0146a457c1c8df09a277797c04be3044a0cffdcd8ffdbf93fa6aeb083fd419，与本次代码清单一致。Native torch_npu 和 Jittor shim 均在物理 NPU3 顺序执行 full、prefix（2 updates）和 fresh-process resume（续训到4 updates），各训练阶段均有对应实际设备 witness，Jittor fallback为0。full cross result 共2187项，失败0、超预算元素0；另外各自恢复态与梯度比较也通过，驱动记录的5个比较阶段返回码全部为0。比较沿用 protocol 中 forward atol/rtol 0.005/0.03、gradient 0.01/0.05，没有放宽预算。证据在 real-model/qwen3-06b-training-npu-cat-fix-r39/bf16-native-deterministic-reference/，终态为 execution.json，逐项结果为 jittor/cross/result.json。

该 cohort 的 Native 在模型前启用了 deterministic algorithms，而 Jittor 的 deterministic policy 仍记录为未实现；终态明确将 cross-runtime deterministic-policy matching 标为 false。因此这次通过结论针对固定顺序、当前训练 fixture 与采样协议，不能外推成随机 dropout 或通用确定性恢复已经兼容。阶段性 functional-gate.json 是运行前的 Jittor preflight，不能覆盖后续终态执行与交叉报告。

### FSDP2 主树双卡续验与外部进程守卫

当前源清单 ee0146… 上，R36 的 Native torch FSDP2 双卡 full 在物理3/5完成8个 optimizer updates，两个 rank 均有 PASS 终态。其后 Jittor full 在开始模型计算前以 rc3 退出：复用 cache 时传入了 symlink alias，导致 jit_utils was rebuilt and cannot be reloaded in this process。R37 冷预热期间另有真实外部 VLLM EngineCore 进入 NPU5；只终止了本 cohort 自己的进程，未碰该任务。

R38 将缓存路径 canonicalize 后，rank0/rank1 在物理NPU3上按顺序完成 Jittor core、HCCL/DTensor 单卡预热；两份报告均为 PASS、fallback0、源码未变。这不构成双 rank 证明。R38 正式 driver 的首次调用误传相对 manifest 路径，Torch worker 在读取 manifest 时 FileNotFoundError、driver rc1；没有模型计算，原日志保存在 distributed/fsdp2-r38/。

R39 使用绝对路径及新 cohort 启动物理3/6 Native full。运行约154秒后，live monitor 在物理6看到 PID1279830；该 PID 不匹配本轮 rank0/1（1302146/1302147）的身份记录，driver只终止本轮launcher/ranks，rc125。未形成完整rank训练终态；之后复查该PID已消失，未向其发送信号。R40 fresh admission 再用3/6尝试，6.78秒时捕获到物理6 PID1319054；ps 显示其为用户 yizhang 的 step_parts.py bert_base_train 性能探针。该 cohort 在 rank worker 建立前返回125；未向该外部任务发送信号或读取环境。稍后的最新卡表仍见同一探针命令的外部 PID1322005 占用NPU6，NPU3空闲，因此没有可用的双卡组合。R39/R40 原始执行、foreign occupant 和设备快照分别保留在各自 LAB cohort 中。

另审计了历史 watchdog 的 PID 分类：R23 PID1476170是本轮 Jittor rank0；R25 PID1524441、R28 PID1610224均是本轮 Jittor rank1；R31 PID288762/288763分别是本轮 Native rank0/1；R33 PID965070是本轮 Native rank1；R35 PID990566/990567分别是本轮 Native rank0/1。旧 monitor 只按外层 process group 识别自有进程，遗漏了 torchrun/Jittor rank 的独立进程组，因此把这些自有 PID 误报为外部并终止了自己的 launcher。原始 watchdog 文件保留，但这些事件不再记作外部资源占用。R22、R29、R30 等未匹配到 cohort rank 身份的事件仍保留为外部占用；R24/R27 为 launcher 前 admission 阻断。纠正后的 R36 driver 同时核对 rank process-start 记录与 PID/start_ticks/runtime/phase/rank/manifest/物理卡。

### R41：双卡准入阻断

R41 启动前的 fresh 快照显示物理NPU3/6空闲；driver再次准入时发现NPU6外部PID1364304，于创建launcher和rank worker前退出，未执行模型计算，也未向该进程发送信号。该次仍是资源阻断，不计为训练验证。原快照和cohort材料保留于 LAB distributed/fsdp2-r41/。

### R42：当前源 FP32 FSDP2 双卡训练与恢复通过

R42绑定当前2500文件 source manifest，SHA256 ee0146a457c1c8df09a277797c04be3044a0cffdcd8ffdbf93fa6aeb083fd419。Native torch 与 Jittor shim 分别在物理NPU3/6完成双rank full（8 updates）、prefix（3 updates）及独立 fresh-process resume（从prefix续至8 updates）；12份rank-phase报告全部为PASS。Native/Jittor完整训练轨迹交叉检查通过，固定阈值为 atol=2e-6、rtol=2e-5；两个runtime各自的full与prefix/resume恢复state、梯度和轨迹比较均按exact通过。逐rank记录核实world_size=2、实际NPU映射、源码哈希未变；Jittor各阶段fallback_delta=0，运行隔离检查为MPI-free。

复核证据位于 LAB distributed/fsdp2-r42/：torch/full/、torch/prefix/、torch/resume/ 与对应 jittor/ 下的rank报告，跨runtime检查为 cross-runtime-check.stdout.log，两侧fresh-process恢复检查为 torch-resume-check.stdout.log 和 jittor-resume-check.stdout.log。完成后物理NPU3/6均已释放。该cohort使用固定FP32合成fixture；不覆盖BF16 FSDP2、Accelerate包装路径、真实模型FSDP2、激活重计算或NPU性能与显存测试。

### R43：FP32/部分混合精度输入梯度全矩阵对拍

R43 使用当前 2500 文件 source manifest（SHA256 `ee0146a457c1c8df09a277797c04be3044a0cffdcd8ffdbf93fa6aeb083fd419`），固定 cohort 位于 LAB `device-api/input-grad-fullscope-r43/`。BF16/FP16 compute、FP32/对应低精度输入、Linear-chain/LoRA、SGD 两次更新共输出168阵列、5760元素，包含输入/损失 cotangent、输出、输入梯度、FP32 master 参数梯度、参数更新及 frozen base。Native 和 Jittor 均在物理 NPU3 有实际 PID witness，源码前后不变；Jittor backend fallback=0。两侧 arrays.npz SHA256 都是 `432aee32ef71c7316000c285252578b11668676ada6f432a38ade768eafd5bbb`；严格比较168/168 bit-exact、最大绝对/相对差均为0、strict failed arrays为空，低精度输入子集84/84 bit-exact，协议中的 FP32 数值预算保持未变。结果在 `device-api/input-grad-fullscope-r43/full-array-comparison-r5.json`。

原始 formal reference 的失败完整保留：Jittor 为8通过/2失败，隔离 Native 为6通过/2失败/2项 Native 不提供 Jittor raw API 而跳过，双方失败节点完全相同，都是 FP32 输入 LoRA 的输入叶子梯度与显式 `reference_x.to(dtype)` 路径逐位不一致。按 fixtures-r1/protocol.json 的既定 FP32 gradient budget 复核，FP16差异全部在预算内，但 BF16有23/48个元素超预算，因此没有把 exact 改成更宽的 allclose。根因是显式 cast 图把两支贡献分别舍入，不等于 autocast FP32 leaf 的 Native 累加顺序。修正后的参考按计算 dtype 权重量化，分别算LoRA base与线性链输入梯度贡献，以FP32合并后按compute dtype舍入，再恢复输入梯度dtype；输出和参数梯度仍保持原 exact 断言。该公式先在隔离 Native 与 Jittor 各自2个失败节点上均逐位通过，再绑定只更新测试文件哈希的新 source manifest r2（SHA256 `4ccba929463e1b829abb360adfba6e649bbb88ea15117899c2c0dc3e9e565da0`）复跑完整套件：Jittor 10/10通过、fallback=0；Native 8通过、2项Jittor-only raw API检查跳过；两侧退出码均为0，源前后不变，所有比较阈值未改。

### 2026-10-23续验：当前2620文件清单 R47/R49

#### R46/R47：Accelerate FSDP2 checkpoint语义与惰性HCCL gather

R46 的 Native/Jittor full 训练和 `native-torch-npu-oracle` 全轨迹对拍先通过；prefix 也把模型、优化器、调度器和rank RNG写出，但 fresh resume 在 rank1读取 `rank1/scheduler.bin` 时失败。原因是当时的验证夹具把 checkpoint 根目录拆成 `rank0/`、`rank1/`，而 Accelerate 的 shared-filesystem `save_state` 将模型/优化器/调度器写在主进程的共享目录、只把 RNG 按进程编号分开。该失败属于夹具目录语义错误，原失败和缺失文件证据保留在 LAB `distributed/fsdp2-bf16-accelerate-r46/`，不计为适配层恢复失败。

R47 修正夹具为真实共享 checkpoint 根目录，并由rank0写一份训练快照，R46输出没有覆盖。R47绑定 source commit `2aa5191c1943c06168d67263e53bc7dad5c1a3d2` 的2620文件 manifest，SHA256 `5e9ff90b92603dadf8145b91c57468d62edece5d8f5ede3e5e51272d56f7fcf5`。Native torch 与 Jittor shim 在物理NPU3/6分别完成 BF16 Accelerate FSDP2 full 8 updates；严格 full 交叉门禁 `atol=2e-6`、`rtol=2e-5` 通过。Jittor prefix 保存于 update3，fresh进程加载后继续至update8；恢复状态、梯度和训练轨迹以exact通过。共享 checkpoint 中有完整模型、优化器、调度器、rank0/1 RNG和训练元数据。Jittor各阶段fallback_delta=0，MPI-free，source unchanged。可复核文件位于 `distributed/fsdp2-bf16-accelerate-r47/`，门禁摘要为 `cross-runtime-check.stdout.log` 与 `resume-training-check.stdout.log`。

该问题的代码根因是 FSDP FULL state/optimizer gather 基于延迟 HCCL op：非rank0经过CPU offload后丢弃图，rank0后续序列化才执行隐藏collective，破坏rank间collective顺序。`compat/torch/installers/distributed.py` 现在在full-state CPU offload前让每个rank `jt.sync_all(True)`，分别覆盖模型态和优化器态；回归测试 `compat/tests/torch/test_checkpoint_full_state_materialization.py` 检查同步先于offload，两项测试已通过。R47 的完整保存恢复复验覆盖该修复。

#### R49：当前源码上的真实Qwen3-0.6B BF16 LoRA Native对拍

R39使用的2500文件清单与当前2620文件清单不同，因此新建R49，不沿用旧源码上的通过结论。R49使用同一共享Qwen3-0.6B权重和固定训练/评估数据，在物理NPU3顺序运行 Native torch_npu 与 Jittor shim；两侧都执行4个optimizer updates、8个microbatches、prefix（2 updates）和fresh-process resume至update4。Source manifest SHA256仍为 `5e9ff90b92603dadf8145b91c57468d62edece5d8f5ede3e5e51272d56f7fcf5`，source commit `2aa5191c1943c06168d67263e53bc7dad5c1a3d2`，文件内容在每个阶段前后都经driver校验。实际参数放置为 `npu:0`；Jittor fallback计数0；Native oracle与shim进程、缓存和环境隔离。

R49 的 full跨runtime报告比较2187项，失败0、超预算元素0，沿用protocol BF16 forward 0.005/0.03 与gradient 0.01/0.05预算。Native、Jittor各自的full对prefix+resume状态比较均为933 tensors exact，训练梯度分段比较各1136项 exact；两侧均有112个trainable参数、4步训练和checkpoint2/4产物。Jittor实际设备和完整梯度健康度由functional/report核验。完整材料位于 LAB `real-model/qwen3-06b-training-npu-cat-fix-r49/bf16-native-deterministic-reference/`；顶层 `execution.json` 的五项比较返回码全0、`all_strict_comparisons_passed=true`。NPU3在阶段后已释放。

R49中Native在首个NPU kernel前启用 deterministic algorithms，而Jittor deterministic policy仍为 `UNIMPLEMENTED`；cross-runtime deterministic-policy matching明确为false。R51/R56后来在tiny BERT LoRA、dropout=0.15下分别证明Native和Jittor各自的full与prefix+fresh-process恢复完全一致，但不比较两个backend的随机数流，也不覆盖Qwen随机训练、随机数据增强、worker或显式generator状态。R47/R49均未接受性能测试，也不据其fixture给真实大模型FSDP2或吞吐/显存下结论。

### 2026-10-23 提交前源码验证

在当前源码树上重新执行仓库级检查，所有日志与新建 CPU JIT 缓存均保存在 LAB 的 `_tmp/validate-ascend-20261023/` 下，不使用 NPU：

- `bash tools/check_repo_layout.sh` 通过，文档治理检查覆盖 298 个 active Markdown 文件，仓库布局检查通过。
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q compat/tests/torch/test_checkpoint_full_state_materialization.py` 在预热 ACL 构建缓存上通过，2 passed in 0.10s。
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 使用隔离 CPU backend 缓存完成，1558 项收集并执行，1556 passed、2 skipped，588.26s；没有结构测试失败。完整输出为 LAB `_tmp/validate-ascend-20261023/structure-final-cpu.log`。

另有一次显式 ACL 环境的冷缓存 native 收集探测在导入 `jittor_core` 时 SIGABRT，未进入 checkpoint 回归断言，也未执行 NPU 算子；完整堆栈保存在 LAB `_tmp/validate-ascend-20261023/structure-cpu-warm-collect.log`。该尝试不计为 ACL 通过或测试失败，最终结构套件在无 ACL、无 NPU 可见性的 CPU-only 配置下通过。

### R50：当前2620文件树输入梯度矩阵与 FP32 FSDP2

R50 分为 LAB device-api/input-grad-current-r50/ 和 distributed/fsdp2-f32-r50/ 两个独立 cohort。两者使用 R47 的 2620 文件 source manifest（SHA256 5e9ff90b92603dadf8145b91c57468d62edece5d8f5ede3e5e51272d56f7fcf5）；该 manifest 的 source_commit 元数据仍写着较早的 2aa5191c1943c06168d67263e53bc7dad5c1a3d2，但本次独立对当前 HEAD f9578ed95d9b0102c55701b6d2c79a1c19561939 做了2620个文件逐项核验，hash mismatch为0。每个训练worker还在运行前后复核自身清单。

#### 当前源码 FP32/部分混合精度输入梯度矩阵

Native torch_npu 与 Jittor shim 在物理NPU3先后运行同一个 BF16/FP16 compute矩阵，覆盖 FP32/compute-dtype 输入、Linear-chain/LoRA、两次 SGD 更新及完整输出/梯度/参数快照。Native 正式回归为8 passed、2 skipped（仅Native没有Jittor raw mapped_matmul API）；Jittor正式回归10 passed，backend fallback=0。两侧归档经原协议完整比较为168个数组、5760个元素，168/168数组逐位一致，最大绝对/相对差为0；低精度输入子集为84个数组、2880个元素，同样84/84逐位一致。没有调整原比较预算。具体运行、完整 arrays 与报告在 LAB device-api/input-grad-current-r50/，汇总比较为 full-array-comparison-r50.json。

首次 Jittor 启动时，规范环境中的 JT_BUILD_CACHE_NAME 覆盖了legacy cache_name，日志显示进程误入 foundation-frontend-npu-r1 命名空间并开始编译。该任务在NPU计算前由本任务停止，return code -15、设备witness未通过，不计为验证；启动日志与部分缓存保留在R50 LAB cohort。修正后的运行显式设置 JT_BUILD_CACHE_NAME=main_tree_fsdp2_cat_fix_r1，复用R47在同一2620文件哈希上实际通过的构建缓存；rank0/rank1的build_config、jittor_core、build stamp、jit_utils_core四项哨兵哈希前后完全一致。

#### 当前源码 FP32 direct HCCL FSDP2 双卡 full/resume

Native torch_npu 与 Jittor shim 在物理NPU3/6分别完成 FP32 FSDP2 full 8 updates、prefix 3 updates及fresh-process resume到8 updates，六个 phase return code 全为0，12份rank-phase终态均为PASS。Native/Jittor full轨迹交叉 oracle 按固定 atol=2e-6、rtol=2e-5 通过；两侧各自full与prefix/resume的状态、梯度和训练轨迹比较均按exact通过。每个rank的实际物理卡映射均被核实，Jittor fallback_delta=0、MPI未加载、源码未变化。R50结束时的NPU3/6快照均无进程。完整phase、rank、跨runtime、恢复和缓存证据在 LAB distributed/fsdp2-f32-r50/，摘要为 result-summary.json。

R50使用固定FP32合成linear fixture；本次结果补上当前源码FP32 direct HCCL/FSDP2双卡证据，但不代表真实Qwen FSDP2、Accelerate混合包装、更多分片/视图拓扑、activation checkpoint、随机dropout恢复或性能/显存资格已通过。R47的BF16 Accelerate FSDP2证据仍单独保留，不与R50互相外推。

### 2026-10-23 R51/R56：Trainer dropout 随机状态 fresh-process 恢复

R51 与 R56 使用当前分支 HEAD d43f12d1078912ad2974995c69c64b11ed79411c 的2620文件清单。R51 manifest SHA256 为 6776a161d91292d1117c173d0d4a6b2d7b831e91babb78b5052b76620bc42705，R56 manifest SHA256 为 1dac711a4e39457429de829bc4131e711d941b34eaac6fc5eeabd0920588ef3a；两者均记录唯一工作树改动为本文件所述的 Trainer 轨迹/RNG 观测。R51 Native full、prefix与fresh-process resume阶段均成功，单runtime比较232项（含8组optimizer边界CPU/NPU RNG状态）零容差exact；但R51整体cohort仍是FAIL，因为当时的旧Jittor wrapper错误地要求 sys.modules 中不能出现 torch_npu，与当前源码有意发布的Jittor ACL facade冲突。该失败和日志保留，没有把R51改标为成功。

修正后的独立R56只运行Jittor，并将R51 Native结果作为只读参考。R56以当前2.0.0 adapter overlay及核验过owner/provider的ACL facade完成full 4步、prefix 2步、fresh-process resume至4步；每个worker均由live PID witness确认映射至物理NPU3，source before/after一致、已验证Jittor build-cache哨兵未变、fallback计数为0，cohort状态PASS。Jittor full与拼接后的prefix+resume在232项轨迹数组上逐元素完全一致，包含每个optimizer边界的CPU/NPU RNG状态、每个microbatch loss、梯度、参数与优化器/调度器状态；atol=0、rtol=0，mismatch为0。训练fixture为dropout=0.15的tiny BERT LoRA、顺序采样、dataloader_num_workers=0。R52–R55 wrapper/adapter版本校验失败阶段均完整保留在LAB；R56修正后再启动，没有覆盖失败输出。

这只证明两个backend各自在该fixture中的随机dropout状态可随Trainer checkpoint精确恢复，不证明Native与Jittor使用相同随机流，也不证明随机Sampler/数据增强、多worker预取、显式generator、真实Qwen dropout或deterministic algorithms兼容。证据与完整输出位于LAB ecosystem/trainer-dropout-resume-r51/ 和 ecosystem/trainer-dropout-resume-r56/，R56 cohort终态为 cohort.json，最后阶段设备与进程证据为 jittor/resume/terminal.json，数组比较为 jittor/full-vs-fresh-resume-comparison.json。

### 2026-10-23 R57：当前最终HEAD FP32 direct HCCL FSDP2 双卡复验

R57绑定当前干净HEAD ee5dee4371eea38eb3e1f49d86b38503a7b6dba9，复核同一2620文件源码集，manifest SHA256为99809589b46947350793d2b4a6faee7a2d6738db9728caa6a73a5eaf02f573e1，当前文件hash mismatch为0。复验前物理NPU3/6均空闲，R47/R50验证的rank缓存无活跃使用者；只复用已验证的Jittor构建缓存，没有并发首次JIT。

Native torch_npu 与 Jittor shim 在物理NPU3/6完成full 8 updates、prefix 3 updates和fresh-process resume至8 updates，共6个phase、12份rank终态均PASS且return code为0。Native/Jittor full轨迹交叉检查按atol=2e-6、rtol=2e-5通过；两侧各自full与prefix+resume状态、梯度和轨迹exact通过。每个rank的进程身份/实际物理卡映射均与分配一致，worker校验源码前后未变化；Jittor各phase fallback_delta=0、MPI未加载。结束后的当前NPU3/6快照均无进程；构建缓存哨兵与上次已验证的R50结束值一致。所有执行、rank日志、环境隔离、资源快照及汇总保存在LAB distributed/fsdp2-current-final-r57/，摘要为result-summary.json。

R57仅封闭当前最终HEAD上的FP32 direct HCCL FSDP2双卡复验，不把固定FP32 linear fixture外推至真实模型、BF16 Accelerate包装、activation checkpoint、FSDP2输入梯度差异或性能/显存。该运行耗时不是受控性能测量。

### 2026-10-23 R58–R63：当前树 FSDP2 输入梯度三精度对拍与 SGD 修复

R58–R63 均为独立 LAB cohort，旧输出没有覆盖。R58 的 Native harness 在模型计算前把 mp_policy=None 显式传入 fully_shard，违反当前 API 的默认策略路径；该轮没有计算或数值输出。R59 改为 FP32 完全省略 mp_policy 后，Native 已完成 FP32/FP32 输入用例，并进入 BF16 compute/FP32 input 的 backward；保存 x.grad 时共享测试 helper 直接调用 BF16 .numpy() 而抛 TypeError，未落盘数组且没有启动 Jittor。仓库 _values helper 随后改为将 BF16 精确转成 FP32 再序列化，CPU smoke 对样例值逐项相等；此转换不改变 BF16 数值，也未调整比较预算。

R60 的 Native 双 rank 三精度 full 阶段通过并保存完整报告。Jittor 在首个 FP32 用例完成前向/反向、进入 SGD optimizer.step() 时因 compat/fsdp2/optimizer.py 漏导入 momentum_initialized_flags 而 NameError，数值比较未执行。对照 origin/2.0-refactor 文件确认其已有该导入；本分支将其补到现有 SGD import，并在 compat/tests/fsdp2/test_fsdp_optimizer_math.py 增加分片 SGD 首次 momentum buffer seed 的回归用例。该定向回归在隔离 CPU Jittor 环境 1 passed，日志在 LAB distributed/fsdp2-sgd-cpu-regression-r1/pytest.log。

R61 使用 source commit 571afb2bda09672d1c179654fc12e76091f1848e 加三项明确工作树 overlay（共享 BF16 serializer、FSDP2 SGD import、回归测试），2620 个跟踪文件哈希 mismatch 为0，manifest SHA256 ac37ddd68e126a3e59f97024e13172a56f3f8bb1fed67fe70ac295a95cffdbac。Native 与 Jittor 分别在物理 NPU3/6 完成 FP32 compute/FP32 input、BF16 compute/FP32 input、BF16 compute/BF16 input 三个 case 的输出、输入叶梯度、全参数梯度和一步 SGD 更新；两边4份 rank报告均 PASS，Jittor fallback为0，MPI未加载，设备进程退出后NPU3/6为空。R61 首次总驱动没有完成 cross-runtime gate，因为比较器把报告中的 torch.float32 与预期标签 float32 直接比较；全部运行时数组与报告已保留。

R62 修正 dtype 标签后离线比较又暴露比较器没有递归处理 initial_parameters、parameter_gradients、updated_parameters 字典；R62以 TypeError 停止，仍未作数值比较。R63 在独立副本中补齐 dtype 标签归一化和嵌套字典叶比较，只读取 R61 两侧 rank 报告，事前/事后报告 SHA256一致，没有重新启动模型或使用 NPU。比较覆盖90个数组/标量，失败0、mismatch0；沿用 atol=2e-6, rtol=2e-5。所有三种输入梯度 dtype 在 Native 与 Jittor 两个 rank 上均匹配；各自 rank 间参数梯度和更新一致，rank-local 输入梯度随输入不同。R63 是对实际 R61 双卡输出的离线比较 cohort，不把 R61 原始驱动的 checker 异常覆盖成成功。

完整执行、原始失败、源清单、设备及进程日志位于 LAB distributed/fsdp2-input-gradient-r58/ 至 r63-compare/。当前分支后续提交的源码内容逐项包含 R61 三个 overlay；R63 不提供更复杂 FSDP2 拓扑、真实模型或 checkpoint 恢复的外推证据。

### 2026-10-23 R64：当前提交上的真实 Qwen3-0.6B BF16 LoRA Native/Jittor 完整对拍

R64使用当前提交 `a3c61912f9b00344ca338ada25e47614058804e0` 的2620文件源清单，SHA256为 `d80452b7546f6b67321a355409b54365b5d23d3316f48f8385f06c11b2da5c3a`；复用共享Qwen3-0.6B权重、固定训练/评估数据和R49协议，按Jittor shim后Native torch_npu顺序隔离运行在物理NPU3。两侧各完成4个optimizer updates、8个microbatches、2-step prefix及fresh-process resume至step4；训练更新无跳过，冻结base保持不变，adapter保存/重载及merged generation流程通过。6个训练阶段均有PID到物理NPU3见证，Jittor fallback计数0。

两侧full与各自resume的933项状态比较均exact，协议、样本游标、update计数exact；每侧full与prefix/resume梯度分段对比各1136项全通过。Native/Jittor全流程交叉比较共2187项，失败0、超预算元素0；继续使用既定forward atol=0.005/rtol=0.03与gradient atol=0.01/rtol=0.05。所有阶段退出码为0。运行后NPU3为空闲状态，Jittor复用缓存哨兵哈希未变。Native在首个kernel前启用deterministic算法，但Jittor deterministic policy仍为UNIMPLEMENTED_FALSE，因此该结果不证明两侧确定性策略或随机流匹配；本轮性能未验收。完整结果、逐rank/phase日志和final-acceptance见LAB `real-model/qwen3-06b-training-npu-current-r64/`。

### 2026-10-23 R65：最终提交真实 Qwen3-0.6B BF16 LoRA Native/Jittor 对拍

R65绑定提交 `9390fa0e84de651e7e1b01de4d142e04d4333f8b` 与3336文件源清单 SHA256 `c00986fb31e15e18860acf60e2ee0059c7dc63afb90ef0a286131d9fc7eb0f9a`。复用共享Qwen3-0.6B BF16权重、固定数据与R49协议，在物理NPU3先后运行Jittor shim和Native torch_npu。两侧均完成4次optimizer update、8个microbatch、step2 prefix及fresh-process resume到step4；112个LoRA梯度档案完整且无缺失/非有限值。所有11个runtime/比较阶段rc=0；full与resume的933项状态比较两侧分别exact，每侧梯度分段比较各1136项通过。Native/Jittor完整输出、参数、梯度、更新和恢复交叉比较2187项，0失败、0超预算，使用既定forward atol=0.005/rtol=0.03和gradient atol=0.01/rtol=0.05。Jittor实际NPU放置通过、fallback=0。Native在首kernel前deterministic=true，Jittor策略仍为UNIMPLEMENTED_FALSE，所以不声明确定性策略或随机流匹配；未测性能。证据目录为LAB `real-model/qwen3-06b-training-npu-current-r65/bf16-native-deterministic-reference/`。

### 2026-10-23 R67/R68：最终主树 FSDP2 三精度双卡输入梯度与 SGD fixture

R67已完成双rank单卡串行预热，但Native driver两个worker都在Torch/NPU初始化之前因临时路径字符数超过108而触发harness断言；原日志和失败未覆盖，且不算算子/FSDP2失败。R68改用短路径和隔离输出，在提交 `9390fa0e84de651e7e1b01de4d142e04d4333f8b`、3336文件manifest SHA256 `c00986fb31e15e18860acf60e2ee0059c7dc63afb90ef0a286131d9fc7eb0f9a` 上重新执行。R68 fresh admission确认NPU3/6空闲、源码哈希3336项全复核、缓存空闲并克隆到两个rank私有cache；各rank串行单卡预热通过，但该预热本身不是双rank证据。Native torch_npu 与 Jittor shim 双rank full阶段均rc=0，launcher及rank报告对应物理NPU3/6，HCCL/MPI-free；Jittor fallback delta为0。Native torchrun在收尾输出rendezvous heartbeat warning，但两个rank报告、process-final记录通过且launcher rc=0。运行后NPU3/6进程表为空。

交叉比较覆盖 `fp32_compute_fp32_input`、`bf16_compute_fp32_input`、`bf16_compute_bf16_input`，比较126个数组/标量、mismatch=0，沿用atol=2e-6/rtol=2e-5，最大绝对误差1.4901161193847656e-8；24项SGD `updated=initial-0.01*gradient` 断言通过。FP32与BF16 compute/FP32 input的输入叶梯度为FP32，BF16 input梯度为BF16，参数梯度均为FP32，Native/Jittor一致。两rank数据不同，rank间参数梯度与更新相同，输入梯度按rank数据不同。完整摘要 `final-acceptance-r68.json`，比较器与进程日志均保存在LAB `distributed/fsdp2-npu-r68/`。这仅关闭最终主树小型Linear fixture的三精度输入梯度/SGD双卡复验，不代表真实模型、更复杂FSDP2分片/视图拓扑、checkpoint恢复、激活重计算或性能通过。

### 2026-10-23 R69：activation checkpoint CPU shim 语义探针

R69在私有 dev 环境、独立冷 JITTOR_HOME/XDG cache 下串行构建并运行CPU探针；绑定提交 `9390fa0e84de651e7e1b01de4d142e04d4333f8b`，探针相关生产文件哈希见LAB `activation-checkpoint-cpu-r69/preflight.json`。通过 `torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)` 计算 $x^2$ 后反向，输出梯度为 $(4,-6)$，与解析值一致；Python callback在forward后调用1次、backward后仍为1次。shim同时发出warning，说明直接运行函数、不保留checkpoint重算或节省activation内存。由此确认当前API的数值梯度路径可用但Python函数级backward重算未实现。Jittor引擎另有基于图liveness的内部释放/重建机制；此探针没有测量该机制，亦未测NPU峰值显存或模型级checkpoint。完整日志和摘要位于LAB `activation-checkpoint-cpu-r69/`。

### 2026-10-23 R70/R71：FSDP2 复杂拓扑 harness 修正与保留失败

R70 的临时验证脚本在静态检查时发现缩进错误，未启动 NPU worker。修正后的 R71 先按 rank 顺序完成 NPU3/NPU6 串行预热；一次 Native 双卡运行随后因结果序列化变量被覆盖而未生成有效报告，该失败与完整输出保留。修正序列化后，Native 在“子模块分别 fully_shard 后再包根模块”的嵌套拓扑首个子模块 unshard 时触发 HCCL 拓扑错误（rank num 为 2、topology rank list size 为 1），没有数值结果。该拓扑不作为兼容通过证据，也没有通过改比较预算掩盖失败；原始材料保存在 LAB distributed/fsdp2-complex-r70/ 与 distributed/fsdp2-complex-r71/，R72 改用 root-only FSDP2 包装继续定位。

### 2026-10-23 R72：root-only FSDP2 视图拓扑对拍及 BF16 反向差异

R72 使用分支式 LoRA 形状模型、非连续输入叶 leaf[:, ::2] 与 root-only FSDP2，在物理NPU3/6分别完成 Native 和 Jittor 双 rank full；两侧各自输出、反向及 SGD 阶段正常结束。跨运行时按既定 atol=2e-6、rtol=2e-5 比较324个数组/标量：FP32 compute/FP32 input 通过；BF16 compute/FP32 input 与 BF16 compute/BF16 input 共16个数组超预算，集中在输入叶梯度、base.bias、base.weight 和 lora_b.weight；72项 SGD 更新公式检查通过。预测、loss、head梯度逐位相同。BF16 输入梯度最大绝对差为一个 BF16 ULP（1.52587890625e-5），各参数差异及原始数组见 LAB distributed/fsdp2-complex-r72/cross-runtime-comparison.json。本轮将差异定位到 tanh backward 的低精度中间舍入；未改数值阈值，R72整体保留为 FAIL。

### 2026-10-23 R73/R74：ACL 低精度 tanh opmath 修复与 FSDP2 复验

R73 在直接 BF16 tanh primitive 上以相同输入比较 Native torch_npu 与 Jittor ACL。前向输出逐位相同，但64个梯度值中18个不同、最大绝对差为0.001953125。候选 FP32 opmath 后仅舍入一次与 Native 梯度逐位相同，逐步 BF16 算术则与修复前 Jittor 完全相同；该探针确定反向精度差异，并保留完整运行日志与数组于 LAB diagnostics/tanh-low-precision-r73/。据此将 src/ops/unary_op.cc 中低精度 tanh backward 的 FP32 opmath 条件从 CPU 扩展至 ACL；CUDA 路径维持原算术分支。计算从已保存且舍入后的 tanh 输出构造反向图，避免用重算 cosh 改变低精度导数，并保留更高阶梯度图。

R74 使用当前修改后3336文件源码清单，manifest SHA256 2847d54980320456bc8086a678dac2f242bf4c49a5c19f17a8886ebf2e479a97，源码仅有上述 tanh 文件工作树改动。串行rank预热和NPU3上的直接BF16 tanh回归均通过；直接 primitive 输出与Native参考逐位相同，梯度也逐位相同。随后 Native torch_npu 与 Jittor shim 在物理NPU3/6完成 root-only FSDP2 分支模型 full：FP32 compute/FP32 input、BF16 compute/FP32 input、BF16 compute/BF16 input。输出、loss、非连续输入叶全梯度、所有参数完整梯度及一步 SGD 更新共324个跨runtime数组/标量在原 atol=2e-6、rtol=2e-5 下全部通过（mismatch=0）；rank终态设备映射正确、源码前后不变、MPI未加载，Jittor fallback_delta=0。终态和比较结果位于 LAB distributed/fsdp2-complex-r74/ 的 cross-runtime-comparison.json、两侧 full/rank*.json 与 rank*-process-final.json。R74 关闭该复杂分支/视图 fixture 中 R72 的 BF16 梯度差异，不外推至 FP16、真实模型 FSDP2、FSDP2 fresh-process 恢复、checkpoint 重算或性能/显存。

### 2026-10-23 R75：当前提交输入梯度全矩阵复验

R75 绑定当前提交 `bad478395edb621850dddf215ad91916d1ea52cb` 的3336个跟踪源码文件，manifest SHA256为 `c71d967178d76bdedec44d20a056d068530bec54ca75afbd7e4237461abd49db`。在物理NPU3使用当前源码运行Jittor shim，并以不可变Native R43归档为oracle；Native数组归档SHA256在复验前后均为 `432aee32ef71c7316000c285252578b11668676ada6f432a38ade768eafd5bbb`。Jittor正式回归10/10通过、fallback增量为0，运行进程被live PID witness确认只放置在NPU3，源码运行前后哈希不变。

R75完整比较覆盖BF16/FP16 compute、FP32/compute-dtype输入、Linear-chain/LoRA、两次SGD更新的168个数组与5760个元素；168/168数组逐位相同，最大绝对和相对误差均为0。低精度输入子集84数组、2880元素也逐位相同；原数值预算未改，`strict_failed_arrays=[]`。此结果是当前提交对R43不可变Native oracle的当前源码复验，不将不同提交上的Native/Jittor运行称为同一次源码构建。完整source manifest、运行日志、数组及比较保存在LAB `device-api/input-grad-fullscope-r75/`。

### 2026-10-23 R76：FP16 tanh backward NPU 对拍

R76绑定与R75相同的当前提交和3336文件manifest，在物理NPU3分别隔离运行Native torch_npu与Jittor shim。FP16输入为8×8、范围[-10, 10]，使用固定上游梯度；输入、上游梯度、前向输出、反向梯度四个数组全部逐位相同，mismatch=0、最大绝对误差0。64个输入中32个处于饱和区（|x|≥5），36个前向值为±1，饱和区32个梯度为0；Jittor fallback增量为0，源码清单前后不变，运行PID实际映射NPU3，结束时NPU3无进程。对比摘要为LAB `diagnostics/tanh-low-precision-r76/comparison-attempt5.json`，终态为 `jittor-attempt5/terminal.json`。

前序Jittor attempt遇到jit_utils首次重建后的进程重载门槛，attempt3在编译阶段被本任务的短时runner提前停止；这些日志和失败输出均保留在同一LAB cohort。用预热所用的`/usr/bin/g++`并按Jittor提示重新启动后，attempt5完成NPU计算并通过；这些准备失败不计入NPU算子失败，也不替代attempt5证据。R76只证明FP16 tanh primitive，不代表FP16 FSDP2或模型训练链路已通过。

### 2026-10-23 R77：当前源码 Accelerate BF16 FSDP2 训练与恢复

R77绑定提交 `b4b344b07` 的3336文件manifest（SHA256 `048aff8211c031b630f18ad14bb0439903fa6c52180822541028f2f7eb3a012c`）。与R74串行预热manifest相比，文件清单只有本报告和ASC看板两份文档不同，生产代码清单完全相同；R77又在当前manifest上依次于物理NPU3、NPU6运行单rank预热，再执行两卡任务。Jittor复用R74已验证的共享cache，cache probe SHA256前后保持 `28fa0074e0dc6d4de34835e874ebba9425536e5f0a1f09bd8638d6b84c76c569`。

Native torch_npu与Jittor shim分别在物理NPU3/6完成Accelerate FSDP2 BF16 synthetic linear fixture的full训练，各8 updates；Jittor另完成update3 prefix与新进程恢复到update8。Native/Jittor full cross-runtime gate按既定 `atol=2e-6, rtol=2e-5` 通过；Jittor full与prefix/resume的模型、优化器、调度器、rank RNG及训练轨迹按exact通过。共享checkpoint包含模型、optimizer、scheduler、两个rank RNG和training metadata。四个launcher阶段return code全为0，rank设备映射均核实为NPU3/NPU6，Jittor fallback增量0、MPI未加载，worker源码前后不变；结束时NPU3/6空闲。完整结果在LAB `distributed/fsdp2-bf16-accelerate-r77/`，摘要为 `result-summary-r77.json`。

R77关闭当前源码上合成linear fixture的Accelerate BF16 FSDP2 full/prefix/fresh-process恢复证据；它不是Qwen FSDP2，不覆盖更复杂的嵌套child+root包装（R71 HCCL错误仍保留），运行耗时也不是性能测试。

### 按原计划仍需完成

1. **真实模型 FSDP2 与恢复。** R65已完成真实Qwen3-0.6B BF16 LoRA单卡full、梯度/参数更新、保存重载及fresh-process prefix/resume对拍；R77完成当前源码synthetic Accelerate BF16 FSDP2 full与恢复。R90/R92/R96的真实Qwen3-0.6B BF16 PEFT/Trainer双卡Native FSDP2 one-step均通过；R91/R93/R96的Jittor FSDP2也完成one-step且fallback=0，但Native/Jittor全输出、梯度和参数更新严格对拍失败，初始LoRA参数与样本一致。R96显示首个可见差异在首层attention context（进入o_proj前），并向后续层放大；仍需定位QK matmul/softmax/context matmul差异、修复后重新通过数值gate，再补真实模型fresh-process FSDP2恢复。R71 nested child+root HCCL错误仍未修复，R76也未覆盖FP16 FSDP2。
2. **激活重计算。** R69证明当前checkpoint shim不会因Python checkpoint函数而在backward重新执行，也没有模型包装与NPU显存收益证据。需落实支持边界或实现路径，对普通和重计算模型比较完整输出、梯度及参数更新，再采集真实NPU峰值显存。
3. **随机训练恢复与确定性。** R56/R65只覆盖既有顺序采样和各自dropout训练状态恢复。还需验证随机sampler、数据增强、多worker预取、显式generator及Qwen fresh-process恢复，并明确Native deterministic=true与Jittor UNIMPLEMENTED_FALSE的差异；目前不能声称跨后端随机流一致。
4. **日常生态组合矩阵。** 继续覆盖Transformers、Tokenizers、PEFT、Accelerate和safetensors在数据处理、训练包装、梯度累加、保存/加载、错误路径中的组合；真实Qwen单卡主链和已有FSDP2 fixture不代表所有常用组合。
5. **正确性通过后的性能/显存验收。** 对已通过的相同模型、数据、dtype、序列长度和更新配置成对测量吞吐/延迟、峰值显存及稳定性，把首次JIT、首步和稳态分开报告。目前仍没有合格的NPU性能或显存结论。

### 2026-10-23 R78：HF Trainer FSDP2 配置探针（CPU-only）

R78以当前提交 `327f6d4bfbc7add8630c3354068caf4067cfe258` 和已安装 Transformers 4.56.2 / Accelerate 1.10.1 做配置探针。构造 `TrainingArguments(fsdp="full_shard auto_wrap", bf16=True, fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"], "use_orig_params": true, "sync_module_states": true, "cpu_ram_efficient_loading": false})` 后，Accelerate 插件报告 `fsdp_version=2`，auto-wrap和transformer类映射均正确。验证同时确认：`TrainingArguments`本身设置`ACCELERATE_USE_FSDP`、wrap策略、类名和`use_orig_params`等环境项，但不会设置`FSDP_VERSION`；实际FSDP2 launcher必须显式设置`FSDP_VERSION=2`。

原始JSON位于LAB `real-model/qwen3-06b-fsdp2-trainer-r78/config-probe-r78.json`。这是无process group、无模型加载、无NPU计算的CPU侧配置验证，只解决后续Trainer入口配置问题；真实Qwen3/PEFT双卡FSDP2训练、梯度/更新对拍和fresh-process恢复尚未启动，R78不作为FSDP2功能通过证据。


## 2026-10-23 R89–R97：真实 Qwen3 BF16 PEFT FSDP2 对拍与 attention 定位

### 源码和已通过的检查

本组运行绑定分支 `codex/ascend-integrated-compat-20261021`、基线提交 `d2748c52ade0d55b2215c06084efaeb736115613`，3336 个 tracked 文件源清单 SHA256 为 `5de7ef03c7fba86af1889641ef497cda0098fc8bb3c009e0c91809def13ccee5`；模型运行时（更新报告/看板前）的 tracked-source 工作树差异 SHA256 为 `61965853b0732e54db7b45efc7e16186be6d1d7ad57a412245ea415410267014`。运行完成后仅追加了本报告与看板文档，生产代码未再变化。该 cohort 的源码差异仅涉及 `compat/torch/amp.py`、`compat/fsdp2/shard.py` 与两个定向测试文件；R89–R97 的 harness 与模型输出均保存在 LAB，没有改写模型权重或已有 cohort。生成目录 `kernel_meta/` 保留为未跟踪内容，不属于源码清单。

AMP/FSDP focused suite 日志在 LAB `real-model/qwen3-06b-fsdp2-trainer-r89/unit-tests/pytest-amp-fidelity-final.log`：81 passed、0 failed、0 skipped，2.34 秒；覆盖 disabled autocast 对外部 FSDP `amp_reg` 的挂起/恢复，以及 FSDP forward mixed-precision register 作用域。`git diff --check` 通过。R90–R97 没有因此扩大数值比较预算。

### 各 cohort 终态

- **R89**：串行 ACL 预热通过；原 Native driver 使用全局 `npu-smi info` 时遇 30 秒查询超时，rank 尚未产出模型报告，未记录 foreign occupant。该次是监控失败，不是训练通过/失败结论。
- **R90**：Native torch_npu 双 rank、HCCL、global_step=1 完成，112 组梯度和112组LoRA参数均完整，returncode=0。Jittor rank 各自落入新的 rank-local JITTOR_HOME，意外并行编译 295 个对象；确认NPU3/6为空后只终止了这次任务的编译进程，保留Native成功结果和完整Jittor日志。R90 Jittor没有模型结论。
- **R91**：把Jittor worker改为在rank-state初始化后恢复串行预热过的共享R65缓存，并引用R90冻结Native结果。两侧同一manifest、同一样本ID，Native/Jittor两rank都完成一次HF Trainer FSDP2更新；Jittor物理NPU3/6映射正确、112组梯度和参数快照完整、fallback增量0、returncode=0。比较汇总为2个rank、2组logits、224组梯度、224组初始参数、224组参数更新，`passed=false`。每rank logits 1/1、梯度33/112、更新量56/112超过既定预算；初始LoRA参数112/112 exact。logits最大绝对差分别1.5和1.8125，最大梯度差0.0888671875，更新差最大约`6.0e-5`（原更新预算`atol=2e-6, rtol=2e-5`）。两侧loss分别3.8984375和3.890625。没有放宽预算。
- **R92**：Native双卡逐层输出采集通过。Jittor阶段的driver在rank1尚无早期身份文件时把本cohort PID短暂报成foreign occupant，returncode=125并中止；foreign记录与rank日志均保留。没有Jittor数值结论。
- **R93**：worker在运行时初始化前写入精确PID/start ticks/rank/source manifest/device assignment，Jittor-only双卡训练returncode=0，fallback=0；复用R92同manifest的Native输出后严格比较仍失败。逐层输出显示差异随网络深度累积：层0只有少量差异，层1/2开始增加，层3明显扩大，后续层继续增长。
- **R94、R95**：额外模块输入诊断分别遇到`npu-smi info -t proc-mem -i <card>`超时，driver只中止本cohort；两次结束后NPU3/6均为空。R95虽将轮询间隔改为5秒，但运行循环仍显式传15秒timeout，不能记作30秒timeout修复。
- **R96**：真正把显式逐卡timeout调用修到30秒，并以5秒间隔轮询。Native与Jittor双卡阶段均returncode=0，真实Qwen FSDP2各完成一个update，设备归属正确、无foreign occupant、Jittor fallback增量0。跨运行时gate仍与R91相同失败。新增的输入快照确认：首层输入、input layernorm、Q/K/V投影输出逐位一致；进入`o_proj`前的attention context已有差异，rank0有49个元素超过诊断阈值、rank1有1个。其后post-attention layernorm及MLP输入/输出差异进一步扩大。这把首个可见偏差定位到eager attention context路径，尚未区分QK matmul、softmax或context matmul。
- **R97**：在Native worker增加QK matmul、softmax和context matmul追踪；启动前NPU3/6均为空，之后Native阶段因assigned NPU6的proc-mem查询超过30秒被fail-closed中止，returncode=126。该cohort没有Jittor阶段或attention trace结论，日志已保留；中止后设备表为空。

R91/R93/R96报告、逐rank tensor archives、对比JSON、launcher执行记录和日志分别位于LAB `_state/ascend-compat/real-model/qwen3-06b-fsdp2-trainer-r91/`、`...-r93/`、`...-r96/`。R92/R94/R95/R97的失败现场也在对应目录，没有覆盖历史结果。R91/R93/R96的summary现标明比较已完成但`passed=false`，真实模型FSDP2 fresh-process恢复未运行。

### 数值差异解读与边界

R91保存数组的梯度向量相关系数约0.9904，但非零梯度符号约4.1%不同；LoRA-B参数更新符号约4.2%不同。AdamW第一步更新幅度接近学习率`3e-5`，小梯度符号翻转会转成接近`6e-5`的参数差异。因此目前证据更支持“前向/梯度数值漂移经更新放大”，不能把它说成优化器实现已完全等价。输入、样本、初始adapter、FSDP策略、step/world-size/HCCL和设备放置均核对一致；Jittor无回落。完整logits和梯度仍超出既定门槛，当前BF16真实Qwen3 FSDP2组合为**未通过**。

R93/R96的层内快照显示首层Q/K/V投影精确，首个可见差异在`o_proj`输入context；因此下一步应在无监控丢失的运行窗口里单独查QK matmul、FP32 softmax及context matmul。R97专门的attention内部trace没有完成。此结果不外推到单卡R65、synthetic FSDP2 R77，也不因这些旧场景通过而覆盖本次失败。

### 分支同步与后续工作

本地历史包含与`origin/2.0-refactor`的合并提交`9390fa0e84de651e7e1b01de4d142e04d4333f8b`；当前HEAD为`d2748c52ade0d55b2215c06084efaeb736115613`。2026-10-23再次按HTTP/1.1尝试fetch `origin/wip/cgq-day23-fix-20261021`和`origin/2.0-refactor`，GitHub:443连接129.6秒超时；旧tracking refs分别为`305c2f0866f163b0549d47b2e4264c5d41a86385`和`ca56aa802c2acf9a22bca32d1bee3276381ed203`，只能视作上次成功fetch的快照，不能声明当前远端最新。失败已追加到LAB `handoff/remote-sync.json`；本次没有reset、stash、丢弃本地提交或合并任何新远端提交。

R118–R129复核后，attention前向快照以及QK matmul、FP32 softmax、context matmul和o_proj在固定上游梯度时的反向都逐元素一致；待定位的是跨decoder反向传播产生的上游梯度差异。真实Qwen3 BF16 FSDP2仍需在原预算下通过完整logits、梯度、更新对拍，再验证fresh-process optimizer/scheduler/RNG恢复；此外还需完成nested child+root HCCL、FP32/混合精度输入梯度、激活重计算、确定性/随机训练恢复和Transformers/Tokenizers/PEFT/Accelerate/safetensors组合。只对正确性通过的对应配置测配对吞吐、延迟、峰值显存和稳定性。目前没有合格NPU性能或显存结果。

## 2026-10-23 R118–R129：BF16 FSDP2 梯度与 attention VJP 复核

### 源码与验证边界

R118、R119 与 R125 使用同一 Qwen3 源码 manifest SHA256 `56031fce397906fcd1a6d64d4b226c20994a50fc4316a93376367d7b8a4c60d7`，记录基线提交 `e67d8924d276e019168fe95493101f46572ff9b5`、3336 个 tracked 文件；运行时工作树 diff SHA256 为 `0abf8131e90e95532b5b8a927372f5ddf4df916aa6f7588e5d5ca2c56a8e511f`，未跟踪 `kernel_meta/` 排除。R118 对 `cross_entropy` 增加了 FP32 `amp_reg` 保护，并在串行 ACL focused suite 中 4 passed、0 failed、0 skipped，日志为 LAB `real-model/qwen3-06b-fsdp2-trainer-r118-ce-amp-register-fix/unit-tests/cross-entropy-amp-register.log`。测试验证普通 BF16 autocast 与 autocast 退出后仍有效的全局 AMP register 两条路径；R118/R119 均未放宽既有数值预算。

### 真实 Qwen3 双卡训练对拍

R118 与 R119 的 Native/Jittor 两 rank 均完成真实 Qwen3-0.6B BF16 PEFT/Trainer FSDP2 的一次更新，设备映射、初始 LoRA 参数、输入样本和源 manifest 一致；Jittor fallback 增量为 0。两轮正式 gate 都是 `passed=false`：每轮比较 2 组 logits、224 组梯度、224 组初始参数及 224 组更新后参数。logits、参数梯度与初始参数通过；每 rank 仍有 54/112 个更新后参数数组超预算（合计 108/224），最大绝对差约 `5.99965e-5`，原预算为 `atol=2e-6, rtol=2e-5`。超差集中于各层 attention 的 LoRA-B 参数。该差异与近零梯度的符号变化会被 AdamW 首步更新放大相符；这不能据此归因为 AdamW 实现不一致。

R118 的 CE FP32 保护显著改善 Trainer logits 梯度：rank0 最大绝对差 `1.4552e-11`，rank1 `4.5475e-13`；R119 复验维持该水平。R119 的 16 组 QK matmul、softmax 与 context matmul 前向快照逐元素相同；其注意力反向诊断仍有 12 个数组超诊断预算，集中在 rank1 首层路径。decoder 输出梯度快照显示误差从 decoder.27 的 exact 状态向较低层累积：rank1 decoder.1 最大预算比 `0.9473`，decoder.0 为 `1.1628`，3 个元素超诊断预算；rank0 的这些层边界诊断均未超预算。R119 的正式梯度数组仍在原门槛内，但参数更新 gate 未过。

### R125 单算子 VJP 结果

R125 从 R119 rank1 的真实 layer0 `self_attn.o_proj` 输入、上下游梯度快照和同一 Qwen3 safetensors 权重取数，在 NPU6 串行运行真实 Native torch_npu 与 Jittor shim。两侧 forward 输出各自都与全模型快照逐元素一致，Native/Jittor forward 也完全一致。固定同一个上游梯度时，BF16 线性输入 VJP 在两边逐元素完全相同（每个 VJP 比较 32768 个元素，changed=0、max_abs=0），并精确复现各自全模型的输入梯度；但 Native/Jittor 送入该算子的上游梯度有 13845/16384 个元素不同、最大绝对差 `0.0234375`，全模型 `o_proj` 输入梯度有 28305/32768 个元素不同、最大绝对差 `0.0166015625`。因此当前证据排除了该 BF16 线性反向原语本身，差异在进入它之前已经存在于反向上游梯度；R119 层间快照显示其随 decoder 反向传播逐步累积，尚未定位到具体产生差异的 block 内算子。

R125 Jittor/NPU6 receipt 为 PASS、fallback 前后均 0、PID 实卡见证通过、前后设备为空；Native receipt 同样通过且隔离运行。结果与摘要在 LAB `real-model/qwen3-06b-o-proj-backward-r125-single-rank/`，对拍 JSON SHA256 `c4b8004892fcd57d69408dfa8e458710f04fa4a8cb22c6748c5036237bc488f9`，Jittor receipt SHA256 `1549927e7cb73bf4a7be846c3c7e7e4f9ffe2c00eb10bdfb10c6a334bab235d`，Native receipt SHA256 `489b05ba1f47117df161b7c67f7e2587559b29aff5df62196a211f150f61340f`。R118、R119 正式比较 JSON SHA256 分别为 `12b4b5bce7342ec955e4bd0c20361c0ca9dbab63965e3dc40ab6f8cae0668947`、`048aec09bfaeb74987927155286c68a10d25724f4897abd6b7a1053fb2fb4cab`；R119 注意力反向诊断 JSON SHA256 为 `109039901344efe4220146ba40fbc6bb1ba20abcb30495da4b25c9672f9de160`。

R121–R124 为准备单卡 probe 的启动器/输出目录错误，均在完整线性数学比较前失败；每个现场保存在对应独立 LAB cohort，未覆盖旧输出。R124 曾成功做 NPU6 空闲预检并加载 Jittor shim，但探针因目录已由启动器创建而提前退出，postflight 设备为空。只有 R125 是有效的 Native/Jittor VJP 结果。

### R129 attention primitive VJP 结果

R129 复用 R119 rank1 的真实 QK matmul、softmax、context matmul 输入与梯度快照，在 NPU6 串行运行 Native torch_npu 和 Jittor shim。当前源 manifest SHA256 为 `6fe5c733e398615d658a2de7e267b4120e7bd5dbde9e01da7c0cb5b0487852a8`，对应 3336 个 tracked 文件；运行时工作树 diff SHA256 为 `4a866414b968d0d6a9e5378683fe14ffd377a7ce18ac57a5652f6b0d30d201b7`。Jittor fallback 前后为 0，Native/Jittor PID 均由 NPU6 实卡快照见证，前后设备为空。

R129 的 35 项诊断检查全为 `changed=0, max_abs=0`：三个算子的前向与 R119 Native 快照、两个运行时的前向相同；在固定 Native 或固定 Jittor 上游梯度时，QK matmul 的输入 VJP、softmax 的输入 VJP、context matmul 的两侧输入 VJP 均在 Native/Jittor 间逐元素一致，并复现相应全模型梯度快照（Jittor 的 QK 右输入梯度在原模型 trace 中未保留，因此该项只做跨运行时比较）。但送入这些算子的全模型上游梯度并不相同：context 输出梯度 28305/32768 元素不同、最大绝对差 `0.0166015625`；softmax 输出梯度 3594/4096 不同、最大差 `0.0625`；QK 输出梯度 1734/4096 不同、最大差 `0.0005493164`。因此本次继续排除了这些局部反向原语；不同发生在它们的上游梯度生成路径，尚未定位到 transformer block 的具体累加或反向节点。

R129 对拍 JSON SHA256 `1ab426851d4fd8b72a789e6fca38d0dc536d94959e893817da012048611d5a42`，上游梯度比较 JSON SHA256 `88e711355b7639acf2c999112b5c8f208eee42ecbfe5e368f774a32eede4927c`，Jittor receipt SHA256 `aa7450f308cdde95bf8ddd5e0b018cc70393b6476aa1b9caf9d58c9a6d8778f0`，Native receipt SHA256 `17b439c84d339f4d4db8b69074037781eba920484b8060a43c97982f12ede7ed`。全部原始输入、运行日志、脚本和结果保存在 LAB `real-model/qwen3-06b-attention-vjp-r129-single-rank/`。R126 的源 manifest 已过期，R127/R128 的 probe 键名错误；这些失败分别留存在独立 cohort，且均未产出有效算子数学结果。R129 才是有效对拍。

### 截至 R130 按原计划仍需完成

1. **修复并通过真实 BF16 Qwen3 双卡 gate。** 已证明 CE FP32 保护让 logits 梯度过关，attention 前向及同上游梯度的 `o_proj` BF16 VJP exact；尚需定位跨 decoder 反向逐步累积的差异，并在不改变 `atol=2e-6, rtol=2e-5` 更新预算下通过完整输出、梯度和更新 gate。只有之后再做真实 FSDP2 fresh-process optimizer/scheduler/RNG 恢复。
2. **补充 FSDP2 拓扑与真实模型覆盖。** R130 已在当前提交的双卡 Linear fixture 覆盖 FP32/FP32、BF16/FP32 与 BF16/BF16 compute/input 三组合；仍需修复并复验 R71 nested child+root HCCL，并将数值结论扩展到对应真实模型场景。
3. **验证重计算和随机恢复。** 对 activation checkpoint 比较重计算语义、输出、梯度、更新和 NPU 峰值显存；补随机 sampler、显式 generator、多 worker 及新进程恢复，并记录 Native deterministic 与 Jittor 当前策略差异。
4. **完成多库日常组合矩阵。** 覆盖 Transformers、Tokenizers、PEFT、Accelerate、safetensors 的数据处理、训练包装、梯度累加、保存/加载和错误路径；单一 Qwen 主链及 synthetic fixture 不能外推为完整兼容。
5. **最后测性能和显存。** 仅对同一正确性 gate 已通过的配置，成对记录吞吐、延迟、峰值显存及稳定性，分开首次 JIT、首步和稳态。目前仍无合格 NPU 性能或显存结论。

## 2026-10-23 R130：最终源码双卡 FSDP2 输入梯度复验

### 源码与执行

R130 使用当前提交 `f5af805ab30aced4aefc214df257a0b575ca36a2`，source manifest 覆盖3336个tracked文件，SHA256为 `31a590acb78ffcedd4de62f9bd8f84e5f84870a5506c9c046445308b5db51d9e`。所有Native与Jittor rank receipt均报告source unchanged；实验及缓存保存在LAB `_state/ascend-compat/distributed/fsdp2-npu-r130-final-source/`，没有覆盖R68等旧cohort。

按串行首次JIT要求，先后在物理NPU3和NPU6建立各自私有缓存预热，两个rank均returncode=0，耗时1099.13秒和1098.58秒；预热总收据为PASS。Native Torch与Jittor shim随后隔离运行，均使用双rank物理卡3/6映射，rank receipt均为PASS、`mpi_loaded=false`、`source_unchanged=true`。Jittor rank0/rank1的fallback增量均为0，最终设备快照映射为物理NPU3/6；Native与Jittor阶段结束后assigned cards为空，没有foreign NPU occupant。Native执行耗时277.60秒，Jittor执行耗时46.82秒；这只是该验证器的执行记录，不构成性能或显存比较。Native torchrun曾记录一条rendezvous keepalive超时warning，但两个rank均生成PASS final receipt，stage returncode为0且完整对拍通过；原始warning保留在launcher日志。

### 对拍结果

三种case为FP32 compute / FP32 input、BF16 compute / FP32 input、BF16 compute / BF16 input；每种均比较prediction、loss、输入叶梯度、FP32参数梯度及SGD更新。Native与Jittor两rank共126项，全部通过，mismatch为0，最大绝对差为 `1.4901161193847656e-08`；预算保持 `atol=2e-6, rtol=2e-5`。这使R130的synthetic Linear FSDP2输入梯度覆盖在当前最终源码上通过，但不外推到真实Qwen3、nested child+root HCCL、fresh-process恢复或完整多库训练。

核心产物校验值：comparison JSON SHA256 `a80a5c857c3c3af27e52922fc651a9c29f835c78d2fdf665e53019646f9f0797`；source manifest SHA256如上；预热driver日志SHA256 `226b3d2ed41883cb54f87fbe0914411ff3566986ff979cfe6f0b8cdb95862808`；Native/Jittor launcher日志SHA256分别为 `e1cafd2c968f1f07842b9bcf677930031474179f30a9331d476c570a57800f69` 和 `f085255eae0678d0b12179552f5c5aee48fa8a055f11681b9058858b0d39e3ad`。四份逐rank process final receipt SHA256依次为Torch rank0 `a862ef0fb6b22fda16380bd6b37eb6dfaa0d451fa462c609c0f90b1a05cedb4b`、Torch rank1 `94d2c1a449c99f79a2960b61553673a176bb23c925abe6914267f74cc8bf5405`、Jittor rank0 `4fa027d9c6fbad0a801d73a4987768ebd17c86afb981cdb089be974982758397`、Jittor rank1 `c8213c51af44d21ed4109dfa343e48b835119f8194a2c023a1fab7b26b1af31e`。\n

## 2026-10-23 R131：当前源码真实 Qwen3 BF16 LoRA 双卡复验与反向定位

### 源码、预热与执行

R131 cohort 保存在 LAB real-model/qwen3-06b-fsdp2-trainer-r131-attention-layer-trace/。source manifest 覆盖3336个tracked文件，记录分支 codex/ascend-integrated-compat-20261021、源码提交 8232bc5c025be1886858cc3c8339b0b1bbfd30d3，manifest SHA256为 862458f46bdad625a559781727da3c74ea00abfdb454aa03ed62d9e5af8e7ab3；该提交仅增加R130文档，运行代码对应 f5af805ab30aced4aefc214df257a0b575ca36a2，含BF16弱标量opmath修复。R118/R119失败运行绑定较早的 e67d8924d276e019168fe95493101f46572ff9b5，因此R131也检验了该修复在真实模型上的效果。 R131 per-rank report及smoke summary的source_commit字段实际为null，这是worker读取commit键而manifest使用source_commit键造成的元数据缺口；执行收据、worker source_manifest_sha256及逐文件source unchanged检查均保留，源码身份以source manifest为准，后续worker需修正该字段读取。

按首次JIT串行要求，物理NPU3和NPU6分别进行world-size=1的Qwen3 BF16 LoRA前向/反向预热，returncode均为0，耗时1229.32秒和1223.04秒；收据均报告源码未变、MPI未加载、fallback前后为0，预热总状态PASS。这只是序贯编译/模型路径预热，不作为双卡正确性证据。正式阶段先Native Torch后Jittor shim，均由R131 driver新鲜检查卡3、6，绑定正确物理卡，阶段结束后卡为空；Native和Jittor阶段returncode均为0，耗时424.61秒和183.16秒，monitor gap均为0、foreign occupant均为空。Jittor两rank fallback增量为0。四个rank均完成global_step=1，每rank含112个LoRA参数梯度，模型、fixture、初始adapter、输入样本及source manifest一致。环境版本为Transformers 4.56.2、Accelerate 1.10.1、PEFT 0.17.1、Tokenizers 0.22.2、safetensors 0.8.0。上述阶段耗时仅为验证执行记录，不构成性能比较。

第一次full-driver启动在运行任何模型或NPU阶段前失败：driver错误要求prewarm-complete.json内含该文件实际未写出的source_manifest_sha256字段。原始日志保存在driver-preflight-attempt1.stdout.log（SHA256 0b6c7b01b4b3b62bac067afe961f68dd7dbcbd8d8bd720d77ad9f832a0e7fff9），完整尝试索引在full-stage-attempt-ledger-r131.json。保留原始预热收据未作篡改；attempt2 driver改为验证manifest路径、逐rank成功收据和源码不变标记，并用独立文件名记录attempt2输入。重跑前重新核验3336个源码hash与NPU3/6空闲状态，preflight通过。

### 完整训练数值gate

R131复用既有R119比较器的数值预算，仅更改cohort schema和输出文件名；比较器代码SHA256 63281216328c5a5b68fb4f35b09023a35390e36671572df93bb2573234aad30c，与R119父比较器逐字节相同。预算为logits atol=0.005, rtol=0.03、参数梯度 atol=0.01, rtol=0.05、初始参数逐位相同、参数更新 atol=2e-6, rtol=2e-5；没有放宽预算。

两rank合计比较2组logits、224组参数梯度、224组初始参数和224组post-update参数。logits、梯度及初始参数均通过；更新gate仍为passed=false，每rank54/112个LoRA-B更新数组超预算，合计108/224个，最大绝对差 5.999648965371307e-5，最大预算比约29.99。失败数组为layer0–26的q_proj/v_proj LoRA-B，layer27对应更新数组未超预算。附加符号诊断显示，每rank这56个q/v LoRA-B梯度数组中有54个包含符号不一致，总计1560个元素；每rank严格更新阈值下有1659个元素超差。这与小梯度附近的Adam首步更新差异相符，但尚不能单独作为成因证明。严格比较JSON SHA256为 0950a8f1e3be3173036a703c84246a6b4bc15031a20b90971437ec48ae40323f；LoRA-B符号诊断JSON SHA256为 38000eb4dec1051212313dbf4ba58dc3ad3f530fd1efa4517dfc208817c2116d。

### Attention与block反向定位

Native/Jittor在decoder layer24–27采集的QK matmul、softmax、context matmul共8类前向张量，每rank共32个数组均逐位相同。trace状态显示每侧均捕获全部28层QK/softmax/context调用，计数完整。Jittor对qk_right的中间梯度虽调用了retain流程，但最终grad为missing；因此24–27每层只比较到其余7类保留梯度，trace coverage gate明确为false，不能将结果表述为完整层级定位。

在可比较的7类梯度中，layer27两rank全部逐位相同；layer26、25、24两rank均有差异。decoder.27边界输出梯度逐位相同，而decoder.26边界梯度最大绝对差分别为rank0 7.62939453125e-6、rank1 3.814697265625e-6，仍在既有梯度预算内。o_proj及MLP gate/up/down projection输入梯度在layer27相同，在较低层逐渐出现小差异。当前证据把首个可见差异收窄到layer27反向传播至layer26的路径，attention内部首个可见差异位于layer26；尚未分辨layer27内的MLP、归一化、残差梯度累加或具体算子。attention对拍JSON SHA256为 190835a8803e1511c3c204f82ba0885bc33bca0a327e38ecb406ed682df97947，诊断脚本版本与coverage规则保存在LAB。这里的差异清单不使用通过预算，也不改变正式gate。

R131重复了每rank54/112个更新数组失败，因此f5af805中的弱标量opmath修复单独不足以解决真实Qwen3严格更新差异。当前没有真实FSDP2恢复、激活重计算、随机训练恢复、完整多库组合或合格NPU速度/显存证据。

另外，R71旧现场的Native Torch在hcclCommInitRootInfoConfig阶段就因topology exchange无法连接peer而退出，nested FSDP worker未运行；这不足以归因到Jittor nested child+root实现。该场景仍需在新cohort做两runtime的fresh验证。

### R131关键证据校验值

source manifest SHA256为 862458f46bdad625a559781727da3c74ea00abfdb454aa03ed62d9e5af8e7ab3；prewarm-complete SHA256为 73c540fcfb3547829ac81e1e3dc376700c92bc0161b73492c00e2eeea276d75b；attempt2 Native/Jittor launcher execution receipt SHA256分别为 fb0397a840f2daa8b90a6ec7371cb948c79574721fc725316ea45d840046bc35、f400d182bac843324837a8954b0982b48e555b83d51d78f585722fdaf8069241；R131 smoke summary SHA256为 d376b6c6a9746207822e5613adbe7ce21dcf1ed1f445b7b19d1cda9c92197605。rank0/rank1预热execution SHA256分别为 b004f1d087a753f4e983da0f7cf65483b61301a0988aac82553808875c0697d0、aced039f4ee9967d41ae34ecc94cb51d5b1ecb9991854ea2ed6cf0cb2c296671。原始日志、模型、缓存、arrays和脚本均留在LAB，没有覆盖R118/R119或更早cohort。

下一步保持原严格预算，针对layer27的decoder输入/输出梯度、input/post-attention layernorm、self-attention与MLP模块输出梯度及残差合并点增加采集，并修复Jittor qk_right.grad捕获缺口；定位首个具体反向差异后再改production实现和重跑完整Qwen3 gate。之后仍需做真实FSDP2新进程恢复、nested child+root HCCL、activation checkpoint与随机状态恢复、多库组合；只有相应正确性场景通过后再测性能和显存。

## 2026-10-23 R132：当前源码真实 Qwen3 BF16 LoRA 双卡反向细化

本节状态：双卡执行和反向诊断完成，严格更新gate未通过；日期：2026-10-23；基线提交：994896c8cbdb684f9f6990489a6f967a497906a0；Owner：Ascend兼容验证；Review when：更新gate通过、qk_right梯度覆盖闭合或首个差异节点结论变化时。

### 源码、预热与执行

R132 cohort 保存在 LAB real-model/qwen3-06b-fsdp2-trainer-r132-layer27-backward-trace/。source manifest 覆盖3336个tracked文件，绑定当前提交 994896c8cbdb684f9f6990489a6f967a497906a0，SHA256为 55e5408f07c8a641c1c9e3c0e10898c6d1aeda5cd19fe6b41eb7aa2df2314a71。与R131冻结清单对比，仅报告和ASC看板文档变化，执行所用生产代码与R131一致；worker SHA256为 4bc359dc9ecbf8081668a936bf28d80e0cb6423984e7397d1e03d3693c1f7852。模型、日志、脚本、缓存和数组均留在LAB，未覆盖旧cohort。

R132先后在物理NPU3、NPU6做world-size=1串行预热，prewarm receipt为PASS（SHA256 52eb3f5810b497e5d40551a48f75f8b0e2736a9036aca973611cfd50f86de145），只证明预热路径，不是双卡证据。预热启动曾有一个路径拼写失败，发生在模型/NPU运行前，原始attempt日志保留。随后Native Torch和Jittor shim分别在物理NPU3/6完成双rank full stage，returncode均为0，monitor gap为0、foreign occupant为空，rank设备映射符合分配；Jittor两rank fallback增量均为0且MPI未加载。Native/Jittor阶段分别耗时403.01秒和182.28秒，这些只是执行时间，不构成性能比较。Native日志另有一条rendezvous keepalive warning；完整stage与rank receipts通过，warning原样保留。两阶段结束后assigned cards为空。

### 严格完整训练 gate

两侧使用相同Qwen3-0.6B权重、LoRA配置、输入样本、冻结源码和原有数值预算：logits atol=0.005, rtol=0.03，参数梯度 atol=0.01, rtol=0.05，参数更新 atol=2e-6, rtol=2e-5，初始LoRA参数要求逐位相同。两rank累计比较2组logits、224组参数梯度、224组初始参数及224组更新后数组。logits、梯度、初始参数均通过；正式更新gate仍为 passed=false，每rank54/112、合计108/224个LoRA-B更新数组超预算，最大绝对差约 5.99965e-5，最大预算比约29.99。没有放宽阈值。正式比较JSON SHA256为 abcaa9cc11f6a47be78b39fb827cf9363ae5dfb7899b9bc13d9987e1dcafab42。

附加符号诊断显示，每rank q/v LoRA-B梯度数组中54/56个包含符号不一致，两rank合计1560个符号翻转值；这与Adam首步小梯度更新差异相符，但不能单独证明根因。符号诊断JSON SHA256为 74675ad9c69d342cc9623d0dad68a84656ad6f0f6ad0c845111106af7d5a7de0。

### Attention与layer27模块反向trace

Native/Jittor的decoder layer24–27中，QK matmul、softmax、context matmul八类前向张量每rank共32个数组逐位一致。attention反向trace每层只比较到8类梯度中的7类：Jittor qk_right.grad仍缺失。报告显示retain和register-hook均已注册且无注册错误，但最终梯度状态仍为missing，因此attention coverage gate为false；不能据此宣布attention反向完整或认定具体算子根因。可比较的layer27 attention梯度两rank逐位相同，layer24–26存在差异。attention trace JSON SHA256为 a2ed967ce58e2adc90b0d63d3abf57be5b87873f80744ca3aa184c10e6fe59e5。

修正后的layer27模块trace比较器v4显示模块采集coverage通过，但其attention coverage仍为false。两rank的MLP分支、o_proj及多处模块梯度精确相同。按反向模块顺序记录的差异包括rank0 `q_norm.input` 有51200个值中的1个变化，最大绝对差 5.820766091346741e-11；rank0 Q/K/V projection `.input` 三个key各报告7126/25600个变化，rank1三个key各报告4380/16384个变化，rank1最大绝对差为1.9073486328125e-6。现在已确认同一rank的Q/K/V forward均接收同一个 `hidden_states` Var，这三个 `.input` hook key实际保存同一份共享输入累计梯度，不能解释成Q、K、V各自的branch-local VJP。block输入边界梯度也有差异。模块trace比较JSON SHA256为 3f65445acfdf1b665252967596f267536c0abd5e64d08f2f24ed6903e6f2f89d。以上是差异定位观察值，不是通过预算或因果结论；R132未改生产实现。

### R132证据与下一步

source manifest、prewarm receipt、Native/Jittor execution receipt及完整日志均绑定本cohort；Native/Jittor execution JSON SHA256分别为 eec03a1137ac72d6244ba30fe3a4a45ca8445dd478720b730d307fc72430dd33、d0b868a3a642ba2e28f24269f63437ec91de5b7bfced30b5b6a9165c96ca49f2，完整launcher日志SHA256分别为 fc09c37c1aeff1111c7be8f1e8e4c3274090f3f98e373a7f6728984268c4bcb0、3a9437408fe5b198b87357f02661f75094e32dbc8747e19d99567762e64822bd。比较器与原始诊断产物没有替代严格gate。

R133随后已对Q/K/V base与LoRA分支做定点重放，因此不再把R132的projection `.input` hook值当成单支路梯度。下一步仍需修复或解释Jittor `qk_right.grad`捕获缺口，并沿layer27到layer26及更早层追踪完整图的upstream梯度、共享输入梯度累计、rank规约和optimizer/update；找到差异节点并通过原严格更新gate前不改预算、不声称根因。之后仍需真实FSDP2 fresh-process恢复、nested child+root HCCL、activation checkpoint重计算、随机sampler/multiworker/generator恢复及确定性策略、多库日常组合。只有各对应正确性gate通过后才测相同场景的速度、延迟、峰值显存与稳定性。


## 2026-10-23 R133：Q/K/V 分支 VJP 与 LoRA 参数梯度定点对拍

本节状态：局部分支诊断完成，Native/Jittor比较80/80数组逐位一致；不关闭R132完整训练更新gate。执行源码提交为 `884f4dc014452e548a9bba8e1c92278be870d2ee`，source manifest覆盖3337条Git路径（3336个普通文件及1个符号链接），SHA256为 `29b716b01d5c00ffdf5468b1cea2033231308532c134b8b6f5cfbd2ade0f5481`。该提交之后的工作树变化只涉及报告和看板；本次没有改生产实现。输入归档SHA256为 `79a2c67677da2187f7538e80246393fcc77a1e6f6a48dc55eebfb2bbc6f5b33d`，Qwen3-0.6B base safetensors SHA256为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。固定R132 rank0/1激活、Q/K/V输出梯度、R132共享输入梯度、layer27真实权重及初始LoRA A/B；将Native和Jittor捕获的两种upstream分别交给两个运行时重放。

### 执行与资源边界

R133在物理NPU6完成最终Native/Jittor两阶段，execution receipt确认阶段PID实际出现在NPU6、identity PID匹配、foreign occupants为空、结束时assigned卡无进程，两个returncode均为0；Jittor fallback增量为0。Native环境为Torch 2.7.1+cpu/torch_npu 2.7.1.post2，Jittor侧Torch shim版本为2.0.0。正式带LoRA参数梯度的attempt05耗时42.38秒和27.97秒，这只是正确性执行时长，不是性能比较。详细stage记录位于LAB `real-model/qwen3-06b-fsdp2-trainer-r133-qkv-projection-vjp/attempt05-lora-parameter-grads-card6/`。

早期attempt01在NPU5出现其他用户的 `VLLMWorker_TP` PID 2533300后被driver中止，本任务没有向外来PID发送信号；该PID当时同时出现在NPU3/5。attempt02和attempt03各自在NPU初始化前收到Jittor `jit_utils was rebuilt and cannot be reloaded`并返回3，日志与execution JSON均保留。attempt04在固定私有cache路径上完成完整首次Jittor构建和branch VJP，attempt05复用该私有缓存扩展LoRA参数梯度。各attempt、日志和缓存均未覆盖；attempt04首次编译耗时不是性能结论。

### 固定upstream的局部梯度结果

attempt04对两rank、Native/Jittor两个upstream、Q/K/V base/full分支计算64个projection输出与输入VJP数组，64/64逐位相同，最大差为0。attempt05额外计算Q/V LoRA A/B参数梯度16个数组；含projection输出、输入VJP和LoRA A/B梯度的80个数组均逐位相同，最大绝对差0。Native与Jittor结果归档SHA256均为 `4585ffada7878ad23edd81d16486f26cfd1d487d796f19df91153442eea1d04f`。在同一backend内将R132 Native与Jittor两套upstream分别重放，projection和LoRA A/B梯度数组也相同；两套捕获的upstream在BF16分支中未造成局部梯度数组变化。

回看R132原始输入，Native/Jittor Q/K/V `grad_output` 只在rank0的q_proj一个值和k_proj两个值不同，最大绝对差1.862645149230957e-9；rank1三组grad_output均逐位相同。同一rank的q_proj、k_proj、v_proj输入激活逐位相同，三条 `.input` hook key保存的梯度也逐位相同，证实这些key是共享hidden-state的累计梯度而非三份分支梯度。按R133输入归档形状，Native/Jittor共享输入累计梯度差异为rank0 7126/25600个值、最大3.814697265625e-6；rank1 4380/16384个值、最大1.9073486328125e-6。R133用Jittor捕获的upstream重放Q/K/V后，分支和与R132 Jittor共享输入梯度逐位一致；用Native捕获的upstream重放时，分支和与R132 Native共享输入梯度仍有同量级差异。局部跨runtime VJP/LoRA参数梯度精确相同，但原始完整图共享梯度的累计行为仍值得沿图定位；本结果不证明根因，也不替代完整训练gate。

### 结论边界与证据

R132严格完整训练更新仍有108/224个LoRA-B数组超原预算，最大绝对差约5.99965e-5、最大预算比约29.99。R133把Q/K/V projection和局部LoRA A/B梯度算子本身从当前首要嫌疑中移开，但没有验证完整decoder上游反向、跨rank梯度规约、Adam状态和完整参数更新，因此不能据此宣称Qwen3训练通过。Jittor `qk_right.grad` capture coverage缺口也仍存在。

summary JSON SHA256为 `81d815c261377f7caaee8d470566d2e26ce37110f9e7533f3af3ff43bd24054e`；input manifest、源清单、两阶段execution receipts、launcher日志和完整数组分别保存在上述LAB cohort。attempt05 Native/Jittor execution JSON SHA256分别为 `d162b16f912520be0a709f26d2c5f157f5e4cbce9fc7dc667bd3062c9bca951c`、`38540ddbee7e96de85f9c9dbb589db2da367527f4999d07bde771dee6eca9aea`，launcher日志SHA256分别为 `ef36d617a48bed1bb855cbce5ec7bb3a3a7ae592d24549064efcbee7fb9eb17d`、`3ceaba274a1d03229c95f53d456a5a4a2b95f64d646215b25cf4b826218937f7`。

### 按原计划截至R133仍需完成

1. **真实 Qwen3 完整训练差异。** R132 的完整BF16双卡更新仍有108/224个LoRA-B数组超原预算。R133已排除固定upstream下Q/K/V投影及局部LoRA A/B梯度不一致，但还需沿layer27到layer26及更早层采集完整上游梯度，检查共享梯度累加、跨rank规约和Adam状态/update；修复后按原严格预算重跑完整logits、梯度、更新和恢复对拍。Jittor `qk_right.grad` 捕获缺口仍需解决，当前attention覆盖不完整。
2. **真实 FSDP2 与 HCCL。** R130当前主树双卡Linear的FP32、BF16/FP32输入和BF16/BF16组合126项通过；仍缺真实Qwen3 FSDP2 fresh-process恢复及nested child+root HCCL场景的双runtime验证。R71 Native停在communicator初始化，需另建cohort复验。
3. **激活重计算。** 需要验证backward是否真实重执行checkpointed block，对拍输出、梯度和更新，并采集实际NPU峰值显存；现有路径没有模型级重计算/收益证据。
4. **随机状态和确定性恢复。** 继续覆盖随机sampler、数据增强、multiworker预取、显式generator及fresh-process恢复，并解决或明确Native deterministic与Jittor未实现策略的差异。
5. **日常多库组合。** 补Transformers、Tokenizers、PEFT、Accelerate、safetensors在数据预处理、梯度累加、训练封装、保存/加载及真实FSDP2中的组合用例；单项或synthetic通过不外推成整栈通过。
6. **性能与显存。** 只对同一模型、数据、dtype、序列长度和更新设置且通过正确性gate的场景，比较稳态吞吐/延迟、峰值显存和稳定性；当前没有合格NPU性能/显存结论。

## 2026-10-24 R134/R135：当前源码 Qwen3 BF16 LoRA 对拍与双卡资源中止

### R134 双卡 FSDP2 预热尝试

R134以HEAD `8a644a69b5bbb1870c9efd9767caff7c24703d5c`作为commit anchor，manifest绑定运行时工作树文件（包括本轮尚未提交的梯度顺序补丁），manifest SHA256为 `4710cce2a360d94cffc902c6852c3384fa21a064ca082239c33ea8daf7d16`，覆盖3337条Git路径（3336个普通文件和1个符号链接）。启动前物理NPU3和NPU6均为空。rank0在物理NPU3首次JIT期间运行685.86秒后，监控在NPU3发现外部进程 `VLLMWorker_TP` PID 2869407；driver只终止了本任务的进程组并以rc=125退出，没有向外部PID发送信号。该次没有完成模型预热或FSDP2训练；原始文件、日志和执行收据保存在LAB `real-model/qwen3-06b-fsdp2-trainer-r134-torch-grad-order/prewarm/prewarm-rank0-attempt1/`。资源中止后留下的173个当前源码JIT对象（约6.4MB）只在R135 manifest复核后供单卡流程复用。foreign occupant JSON、execution JSON和stdout SHA256分别为 `ddd99687b1c7513f3075a5571c54c41781ea6958c4fa8ab256351b5e209e7326`、`51aa80de50ae24a4e5573977b894d7fa5da04da4bec65575820abd4bed365ebb`、`e240c4214b738d3fc9d8e7200916726a7f97cdfddeb3447d19b1fd5b311cc2e2`。

### R135 单卡 Native/Jittor full、分段训练与恢复

修复内容是在Torch兼容入口调用 torch.autograd.grad 和 Tensor.backward 期间启用线程局部的BF16 shared-Var梯度贡献排序策略，按forward op id逆序稳定消费贡献，并在调用结束后恢复此前状态；不改变非Torch兼容图的默认累加顺序。针对 1 + (-1) + 0.001953125 的BF16共享梯度回归测试覆盖两种入口，结果为 1 passed, 31 deselected（44.31秒）。ACL/NPU6定向probe中 torch.autograd.grad 与 Tensor.backward 都返回精确0，实际设备为NPU，fallback delta=0；原始JSON位于LAB diagnostics/bf16-qkv-autograd-order-r1/attempt06/result.json。

R135 建立独立cohort `real-model/qwen3-06b-training-npu-current-r135-torch-grad-order/`，源码提交和manifest与上述R134一致。只使用预检为空闲的物理NPU6，Native torch_npu和Jittor shim分环境、串行执行；每阶段的PID到NPU放置见证均通过，阶段结束卡为空。场景是Qwen3-0.6B BF16 LoRA、固定样本顺序、16条训练样本和4条评估样本、batch 2、梯度累积2、4次更新；不属于FSDP2双卡结果。

Native和Jittor各自执行full四步、first两步保存checkpoint-2，再以新进程从checkpoint-2恢复到step 4。两个运行时各自的full与first+resume比较均通过：933个状态/张量项在零容差下完全一致，样本游标与更新契约一致；四步梯度归档及分段归档逐项精确一致，失败比较和超预算元素均为0。Native/Jittor参数实际放置报告均为`npu:0`，映射到物理NPU6；Jittor fallback_count=0，Native torch_npu没有可用的shim fallback计数。

严格跨运行时结果 `jittor/cross/result.json` 比较完整初始探针、前向、梯度、112个可训练LoRA张量、最终状态和生命周期字段，共2187项，全部通过，failed_comparisons=0、outside_budget_elements=0。沿用protocol SHA256 `ab11ec4b11b95e23ca72075ea7e9012ca91801adf7197fb7cc39c298bc8f9592`，BF16 forward预算为atol=0.005、rtol=0.03，gradient预算为atol=0.01、rtol=0.05，没有放宽门槛。最大forward绝对差为0.0043417215；最大gradient绝对差为0.030277052，出现在microbatch 5的gradient-health L2 norm项，其相对差为0.0124474，仍在既定预算内。Cross JSON SHA256为 `31180bd75fc0cd5779c380df06ac1cf79c24c3ad49cc88d13b8a352b1339058e`。Native/Jittor各自完整输出报告SHA256分别为 `9dd3d51b78ca72561b56ca9d9e6bd3f15afae418b8cb1f679dbd3fe23b975e54` 和 `6ae6bb9fe3ed2f97c62dcbdbf3605c564053dab661bf7dd9ada6bdbcf1f7a53e`。各运行时的fresh-resume比较JSON SHA256分别为 `7a8605d1f4a6d6de511e2aae68b66b087bdd8dab36078f041b855fe409ffb9ec` 和 `e6593d0ae7fe5307222a55e157751f974e447faf333c3575f82de54aeb448afd`；Native/Jittor full-gradient比较JSON SHA256分别为 `baf635770b642c9676e37cfeb9b66bcb7983694e0488e4f1d10176fce4577cbd` 和 `ece3270cc98b0a9d062b571aee2be52585e3abb9264b13214e2c7660e0dd1851`。

Native环境版本为Transformers 4.56.2、Tokenizers 0.22.2、PEFT 0.17.1、Accelerate 1.10.1、safetensors 0.8.0。Jittor full train_loss=3.842697、eval_loss=4.474481；Native full train_loss=3.842866、eval_loss=4.479791。结论依据完整数组比较器和原数值协议，而不是仅按loss接近判断。本cohort没有性能接受标记；执行时间不作为速度或显存结果。该固定顺序单卡结论不覆盖FSDP2、随机sampler、多worker、激活重计算或不同源码冻结点。

## 2026-10-24 R137：当前源码真实 Qwen3 BF16 LoRA 双卡 FSDP2 严格更新对拍

R137 使用当前提交 df495e7c0ab7fc874a3536963954b9af7f83ef21，source manifest SHA256 7f6b138e12dcc998bcef6620fcac1f9af68c13aebcd36220788abe6049016af4。在 NPU3 和 NPU6 分别完成 serial JIT/backward prewarm 后，Native torch_npu 与 Jittor shim 以独立环境、串行阶段完成真实 Qwen3-0.6B BF16 LoRA 的双卡 FSDP2 单次 optimizer update。两侧 launcher returncode 均为0、monitor gap为0，rank0/rank1分别绑定物理NPU3/NPU6；Jittor fallback增量为0。四份rank报告均为global_step=1、world_size=2、HCCL、FSDPVersion=2，mixed policy为 BF16 param/reduce/output，模型、初始LoRA、fixture和样本一致。R137没有做fresh-process optimizer/scheduler恢复，因此不覆盖R135的单卡恢复结论。

未改动既有预算：logits forward atol=0.005/rtol=0.03，parameter gradient atol=0.01/rtol=0.05，post-update delta atol=2e-6/rtol=2e-5，初始参数要求exact。逐rank比较logits、112组参数梯度、112组初始参数和112组参数更新。两rank logits逐位一致；224组梯度均在预算内，最大绝对梯度差0.009765625；224组初始参数均exact。完整更新gate为 passed=false：每rank 52/112组LoRA-B更新数组超预算，合计104/224组；最大绝对更新差 5.999376662657596e-05，最大预算比约29.988。差异集中在BF16低幅LoRA-B梯度：每rank观测到824个梯度符号翻转，且这些元素都落在更新超预算元素中；这与AdamW首步在零初始化LoRA-B上将微小梯度差放大成近两倍学习率的更新差相符，但尚不能据此断定首个生产反向差异节点。训练loss两侧均为3.8984375。比较JSON SHA256为 1b52d9009460282ae57943380e34001ede4c92c42b5e6ddb34e770f9eac044dd。

诊断部分记录398组前向激活数组并逐位相同；438组激活梯度只作定位用，不计入通过门槛。已捕获的decoder.27与decoder.26边界反向值相同，decoder.25一带开始出现小差异，随后向更早层累积；这把搜索范围缩小到上游反向，但还没有识别到具体算子。R137仍有 qk_right.grad Jittor捕获缺口。下一独立cohort R138已准备 layer-26 模块输入/输出反向hook及BF16参数/输出、FP32梯度规约配置，严格数值门保持不变；本报告更新时该cohort尚未运行模型，因物理NPU3被外部VLLM进程占用，必须等双卡重新fresh后再启动。R137原始失败、完整张量和日志留在LAB real-model/qwen3-06b-fsdp2-trainer-r137-grad-order-dual36/；执行时间仅记作验证记录，不作为性能结论。

## 2026-10-24 R138：反向模块trace完成；FP32 reduce配置未实际生效

R138 以独立cohort qwen3-06b-fsdp2-trainer-r138-fp32-reduce-gradtrace-dual36 运行，绑定当前源码commit df495e7c0ab7fc874a3536963954b9af7f83ef21及source manifest v2 SHA256 13d0abf8e059a316da4baf3120e7c121ebb6952de0020526424a47486dc876a8。第一次rank0预热在2.56秒时因沿用R137 manifest而触发源码哈希断言，发生在模型计算前；随后保留原manifest和失败receipt，另建manifest v2。rank0和rank1的Jittor初始化无工作重试分别以rc=3退出，随后各自预热通过，耗时1223.40秒/1219.68秒，物理卡为3/6，loss均为3.540607，fallback增量0、MPI未加载。aggregate收据明确标记not_two_rank_proof。

Native双卡阶段rc=0、耗时491.24秒；Jittor双卡阶段rc=0、耗时172.04秒。两个阶段均为global_step=1、world_size=2、HCCL，PID见证分别正确映射NPU3/NPU6，未记录foreign occupant。Jittor monitor_gap_count=0且fallback_delta=0；Native记录一次30秒npu-smi proc-mem查询超时，monitor_gap_count=1、monitor_degraded=true，超时细节在torch-full-launcher/npu-smi-timeouts.jsonl。故Native计算结果已保存，但监控完整性不足，不能把R138记为完全合格验收。

R138配置请求的是BF16参数/输出、FP32 reduce，但四个rank的实际报告均为param_dtype=torch.bfloat16、reduce_dtype=torch.bfloat16、output_dtype=torch.bfloat16。原因已定位到当前Transformers Trainer/Accelerate接线：Trainer创建Accelerator时未把TrainingArguments.fsdp_config中的mixed_precision_policy传给FSDP plugin，后处理只同步了limit_all_gathers和activation_checkpointing；Accelerate从训练环境创建的plugin按bf16=True使用BF16 reduce。因此R138不是FP32 reduce对照，不能用于判断FP32 reduce能否修复更新差异。下一cohort会在Trainer构造后、train开始前显式给已创建的FSDP2 plugin赋MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32, output_dtype=bfloat16)，并把plugin中的实际对象/三种dtype作为启动断言和报告字段。

R138严格比较仍使用原有预算：forward atol=0.005/rtol=0.03，gradient atol=0.01/rtol=0.05，post-update delta atol=2e-6/rtol=2e-5。两rank logits逐位一致，224组参数梯度和224组初始参数通过；更新gate仍失败，每rank52/112、合计104/224个LoRA-B更新数组超预算，最大绝对更新差5.999376662657596e-05，最大预算比约29.988，没有放宽门槛。train_loss均为3.8984375。比较JSON SHA256为0534574d6a18d2756c720ddcd7d9873d3411ea41c5adbc1c1fdd96a19546ef35。

新增hook后，诊断数组增至454组前向激活和494组激活梯度。layer26新增采集到的前向值逐位相同；MLP路径所捕获的梯度为exact，而self-attention Q/K/V投影和input_layernorm路径可见很小的梯度差，最大约9.54e-7，远小于gradient gate。现有capture仍缺Jittor的trace.layer24至layer27.qk_right.grad，且局部小差异尚不能确定首个生产反向算子。R138所有原始输出、配置、timeout和失败attempt保存在LAB real-model/qwen3-06b-fsdp2-trainer-r138-fp32-reduce-gradtrace-dual36/；该cohort不作为FP32 reduce结果或完整监控验收。

### 当前仍需完成

1. 建立R139（或后续新cohort），显式设置并断言Accelerate FSDP2 plugin实际为BF16参数/输出、FP32 reduce；保持所有严格预算，重新跑真实Qwen3双卡Native/Jittor完整update并检查monitor、device placement、fallback和完整对拍。
2. 用修正后的FSDP2配置继续分析LoRA-B更新差异，并补足Jittor qk_right.grad capture；若原预算仍失败，定位生产反向差异后修复并用新cohort完整复验。
3. 完成真实FSDP2 Native/Jittor fresh-process optimizer/scheduler恢复，并覆盖global state、LoRA参数/梯度、样本游标和下一步更新；现有R135恢复证据是单卡，R137/R138仅有一次双卡update。
4. 验证双卡nested child+root HCCL，以及FP32/BF16 compute、FP32输入梯度等组合；结果按各自完整输出/梯度/参数更新门槛验收。
5. 验证activation checkpoint重计算下的输出、梯度、更新和恢复，再采集峰值显存；另覆盖随机sampler、generator、多worker与新进程随机状态恢复和确定性策略。
6. 补齐Transformers、Tokenizers、PEFT、Accelerate、safetensors在数据预处理、梯度累加、Trainer封装、保存/加载和错误路径中的日常组合用例，不从单项或synthetic通过外推整栈。
7. 仅在对应正确性门通过后，比较同配置的速度、延迟、峰值显存和稳定性；目前没有合格的NPU性能结论。

## 2026-10-24 R139：实际 FP32 reduce 策略双卡对拍；LoRA 更新仍失败

R139 在独立cohort `real-model/qwen3-06b-fsdp2-trainer-r139-fp32-policy-hook-dual36/` 运行，当前源码commit为 `df495e7c0ab7fc874a3536963954b9af7f83ef21`，source manifest SHA256为 `0d2bd1c1524e9a62ba0b1aa56bcf370d7b4f7b266e0f7cb4435376e3b8fd01c0`。Harness在Trainer创建后、`train()`前，将Accelerate已创建的FSDP2 plugin policy显式设为BF16参数、FP32梯度规约、BF16输出，并对四个rank在训练前后读取实际policy对象并断言三种dtype。这个实验确实覆盖了FP32 reduce，但赋值由诊断harness完成，**不代表标准 `TrainingArguments.fsdp_config` 已自动接通，也不是生产Transformers/Accelerate配置修复**。

真实Qwen3-0.6B BF16 LoRA、同一初始adapter和固定样本执行双卡FSDP2单次更新。Native和Jittor均为global_step=1、HCCL、rank0/rank1映射物理NPU3/NPU6；Jittor fallback增量为0，无foreign occupant。Jittor stage returncode=0、monitor gap=0；Native计算stage returncode=0，但两次npu-smi查询超时，`monitor_gap_count=2`、`monitor_degraded=true`，所以Native结果已保留用于数值比较，但该cohort不满足完整监控验收。

原有门槛未改变：logits atol=0.005/rtol=0.03，参数梯度 atol=0.01/rtol=0.05，初始参数exact，更新差 atol=2e-6/rtol=2e-5。两rank logits exact；224组参数梯度均在预算内，最大绝对差0.00970458984375；224组初始参数exact。参数更新仍 `passed=false`：104/224组LoRA-B更新数组超预算，最大绝对差 `5.9993762988597155e-05`、最大预算比约29.9879。全图诊断保留454组前向数组和494组激活梯度；layer27与layer26反向边界一致，layer25开始出现小梯度差异并向早层扩大。比较JSON SHA256为 `b0645500f8a385907d2c55fd8ca4c406494aa60c9b5dc4dbbec989b811425e4e`。

对R139归档梯度、初始参数和更新做了CPU侧AdamW首步公式复核，没有启动NPU。每rank 112个全量梯度数组的Native/Jittor L2范数分别为7.2546255352和7.2354765679；`max_grad_norm=1`对应裁剪系数分别为0.1378430899和0.1382078970。按训练学习率3e-5、weight decay 0.01、Adam默认β=(0.9, 0.999)、ε=1e-8及各自裁剪后梯度计算，两个运行时保存的56个LoRA-B更新数组均与AdamW首步式吻合，最大残差分别约8.08e-12和8.00e-12。每rank LoRA-B梯度有1126/344064个元素符号不同。该归档复核说明当前失败与运行时梯度及其全局裁剪系数差异相符，没有显示AdamW首步公式偏差；它不替代FSDP2多步优化器状态或恢复验证。

## 2026-10-24 R140：固定upstream的QK/context matmul局部VJP逐位相同

R140 cohort位于 `real-model/qwen3-06b-qk-matmul-vjp-r140-card3/`，绑定与R139相同的source manifest `0d2bd1c1524e9a62ba0b1aa56bcf370d7b4f7b266e0f7cb4435376e3b8fd01c0`。它从R139 rank0/rank1数组取layer24–27的前向trace，固定使用对应Native全图输出梯度，分别在Native和Jittor上重放QK与context matmul，不执行模型更新。Native在物理NPU3运行29.72秒，Jittor在NPU3运行991.20秒；两个阶段returncode=0、monitor gap=0、owned PID见证通过、无foreign occupant，阶段后卡为空。Jittor约953秒用于重编295个JIT单元，属于本次诊断准备时间，不作为性能测量。

两边局部matmul前向输出以及左右输入/权重VJP共48个数组全部逐位相同，最大绝对差为0。另有32组局部梯度与归档全图Native参考逐位相同，Native和Jittor两套局部结果均通过。比较摘要为152项检查、`passed_exact_diagnostic=true`，comparison JSON SHA256为 `196ea7dcdd1caa78f7d5b9635f426b370098dd1d834038eca96817ef378ad270`，execution summary SHA256为 `29e9e2b3a319757ee12744bb972aa62a289bcd5d7e0485ac0f73773c6bb8131d`。

这个实验把相同输入和Native upstream下的QK/context matmul VJP从当前首要嫌疑中移开；它没有证明完整attention反向一致，因为R139 Native/Jittor在layer24/25捕获的部分matmul输出upstream梯度本身并不相同。Jittor全图的 `trace.layer24..27.qk_right.grad` 仍未捕获。R140不做生产代码修复或训练gate通过结论。

### 截至R140仍需完成

1. 修通标准Transformers Trainer/Accelerate `fsdp_config` 到FSDP2 plugin的mixed precision policy传递，并在不做harness后置赋值的独立cohort中断言实际BF16参数/输出、FP32 reduce；R139仅证明显式赋值的诊断配置确实生效。
2. 继续定位layer25附近首先出现的全图反向差异。R140已证明固定Native upstream的QK/context matmul局部VJP一致；下一步需采集并比较输出投影/残差/归一化分支的实际upstream和输入梯度，补足Jittor `qk_right.grad`捕获。找到生产算子差异后修复，不通过放宽预算接受更新。
3. 修复后以新cohort重跑真实Qwen3双卡Native/Jittor完整输出、224组梯度、初始参数和更新对拍；同时保证Native和Jittor监控无gap、NPU映射正确、Jittor fallback为零。
4. 完成真实双卡FSDP2 fresh-process optimizer/scheduler恢复，核对模型/LoRA状态、优化器与scheduler state、数据游标及恢复后的下一次更新；R135恢复证据仍仅覆盖单卡。
5. 补齐nested child+root HCCL，以及FP32/BF16 compute、FP32输入梯度等目标dtype组合的完整双runtime验证。
6. 验证激活重计算下的实际重执行、输出/梯度/更新/恢复和峰值显存；覆盖随机sampler、generator、多worker预取、确定性策略及新进程随机状态恢复。
7. 完成Transformers、Tokenizers、PEFT、Accelerate、safetensors的日常组合场景，涵盖数据处理、梯度累加、Trainer封装、保存/加载和错误路径；单项或synthetic结果不外推为整栈通过。
8. 只对各自正确性gate已通过的对应配置比较速度、延迟、峰值显存与稳定性；当前仍没有合格的NPU性能结论。

## 2026-10-24 R141 前置：Trainer FSDP2 policy bridge CPU 合约通过，NPU 待验

针对 R140 第1项，已在 adapters/jittor_adapters/transformers.py 增加仅适用于已锁定 Transformers 4.56.2 的 Trainer 初始化补丁：Trainer 创建 Accelerate 后，从 TrainingArguments.fsdp_config 读取 mixed_precision_policy，按实际 FSDP 版本构造 Torch policy 并应用到活动 plugin；支持的 dtype 字符串与字段做白名单校验，plugin validator 异常时恢复原 policy。Transformers 5.5.3 不应用此补丁。适配器单测覆盖 FSDP2 BF16/FP32/BF16 映射、无配置保持默认、非法 dtype 不污染 plugin、5.5.3 不改写；适配器测试17项通过。

真实 Transformers NPU 训练尚未验证此接线。隔离执行 compat/tests/torch/test_ecosystem_runner_contract.py，设置 JITTOR_TORCH_SHIM=1、清空 ASCEND_RT_VISIBLE_DEVICES，并将 HOME、JITTOR_HOME、XDG_CACHE_HOME、TMPDIR 全部指向新 LAB 验证目录 trainer-policy-bridge-20261024。首次 Jittor core 244/244 编译约714秒后，runner contract 18 passed、0 skipped、退出码0；完整日志位于 validation/trainer-policy-bridge-20261024/runner-contract.log，退出码记录为 runner-contract.exit-code。该结果只证明 CPU runner/注册合同与环境隔离，不证明 NPU 实际 plugin policy、FSDP2 更新、梯度或恢复正确，也不覆盖 R139 的失败门槛。新缓存保留在该 LAB 目录；此前误触到的既有用户 Jittor cache未清理。

下一步应以本次代码提交冻结新 source manifest，复制 R139 的真实模型协议建立全新 cohort：Jittor 侧移除 Trainer 构造后的 harness policy 赋值，并在 train 前后断言 TrainingArguments 配置自动生效；Native 侧仅为对拍显式设置等价 plugin policy。启动前通过 driver 对物理NPU fresh admission；若占用则不启动。该cohort仍需保留完整输出/梯度/初始参数/更新的原预算、零回落、物理映射及无监控缺口要求。R139更新gate和R140全图反向定位状态保持不变。


## 2026-10-24 R145：标准 Trainer 双卡首步完整对拍

R145 cohort 位于 LAB `real-model/qwen3-06b-fsdp2-trainer-r145-shared-cache-paired-dual56/`，绑定分支 `codex/ascend-integrated-compat-20261021`、源码提交 `4ad7e5a738d1886111b81e4ac811f135be81e4f0`，source manifest SHA256 为 `97caadcba617299b34fa20a2991a7bc9f5121251b49ba512452736de5d03b0a1`。Native PyTorch 与 Jittor shim 使用同一真实 Qwen3-0.6B BF16 LoRA 输入、权重、初始化参数、Transformers 4.56.2/Accelerate 1.10.1/PEFT 0.17.1/Tokenizers 0.22.2/safetensors 0.8.0 组合，在物理NPU5、6分别完成标准HF Trainer FSDP2一个优化步。Trainer policy bridge 在train前后都报告实际生效的 BF16 参数、FP32 reduce、BF16 输出配置，Jittor侧没有harness后置赋值；两侧每rank112个LoRA参数、global_step=1、train_loss均为3.8984375。Jittor实际参数留在各进程的NPU设备，driver映射rank0/1到物理卡5/6，fallback增量为0。

Native阶段returncode为0、耗时534.17秒，但monitor gap=1且driver标记degraded；Jittor阶段returncode为0、耗时172.67秒、monitor gap=0。两阶段均无foreign occupant。Native的npu-smi rank1汇总曾同时看到[6,5]，driver身份记录仍将该rank绑定到分配卡6，故本cohort不能作为clean-monitor最终验收。阶段耗时仅为执行记录，因参数更新gate失败，不作速度或显存结论。

按既有预算比较两组logits、224组梯度、224组初始参数及224组更新数组：logits、原始参数梯度、初始参数全部通过；更新gate失败，每rank52/112、合计104/224个LoRA-B数组超预算，最大绝对差为 `5.9993762988597155e-5`，最大预算比29.9879。门槛仍为logits `atol=0.005, rtol=0.03`、梯度 `0.01/0.05`、参数更新 `2e-6/2e-5`，未作放宽。每rank有1126个LoRA-B梯度值出现符号差异，差异集中在接近零的值；这不是单独的根因证明。CPU重算保留的NPU梯度、初始参数和更新后数组显示，各运行时的更新都符合各自的首步AdamW公式，最大残差 `9.8690e-10`，大于 `1e-8` 的残差为0；Native/Jittor梯度范数分别为7.25463和7.23548，clip系数分别为0.137843和0.138208。因此AdamW公式实现未解释跨运行时更新偏差，梯度及全局裁剪差异仍需追踪。比较JSON SHA256为 `2c3b108db7a036caff5f9942b564d305125b52e8ac80b97a7774dbe46fa958a3`，公式复核JSON SHA256为 `97d249aebcc74fe5d71b22e0dceb348fffba940c7d41dd42c7df9f3d8ed3c6fa`。

## 2026-10-24 R146：第25/26层反向采集与差异收窄

R146 cohort 位于 LAB `real-model/qwen3-06b-fsdp2-trainer-r146-layer25-linear-vjp-dual56/`，source manifest与R145逐字节相同。只增加第25层投影输出cotangent的观察hook，并在第26层采集Q/K/V投影输出与输入梯度；没有修改模型计算或门槛。Native、Jittor双卡阶段均returncode=0、monitor gap=0、degraded=false、foreign occupant为空，driver映射rank0/1到物理NPU5/6；Jittor fallback增量为0。阶段耗时分别373.15秒和68.68秒，只记作诊断执行记录。四个rank均完成一个Trainer优化步，policy在train前后断言成功。

R146沿用R145相同的严格对拍结果：logits 2组、梯度224组、初始参数224组均通过，更新104/224个LoRA-B数组仍失败，最大绝对差 `5.9993762988597155e-5`、最大预算比29.9879。cross-runtime comparison SHA256为 `87445d023d6998ac30d66a2c3b4eb97a01650b6c548d12b04c2385347d1f6339`。Jittor的 `trace.layer24..27.qk_right.grad` 仍缺失，因此attention内部梯度覆盖仍未闭合。

第25层探针在rank0观察到o_proj/down_proj输出cotangent最大差分别为 `2.3842e-7` 和 `1.1921e-7`；rank1分别为 `7.6294e-6` 和 `4.7684e-7`。对应输入梯度差异仍低于诊断预算，但rank1的gate/up输入梯度最大绝对差为 `1.5259e-5`。第26层Q/V投影输出cotangent在rank0逐位相同，rank1的V逐位相同、Q最大差 `2.9802e-8`；Q/K/V的 `.input` hook数组在每个运行时和rank内彼此逐位相同，却跨运行时有小差异（rank0最大 `9.5367e-7`、rank1最大 `2.3842e-7`）。这确认这些hook读到的是共享hidden-state梯度累计值，不能把它们解释成Q/K/V各自的branch-local VJP。当前证据把差异进一步指向第26层反传中的共享梯度累计/上游路径，但还没有分离具体分支、归一化、残差或累加顺序，也不能归因于单个matmul kernel。layer25诊断JSON SHA256为 `e3a430ad3ee837ab390318b490b5139c5f312f5a1cc12f909290dd5d76a07276`；layer26 localization JSON SHA256为 `59ac1b4902698d78460f5fefea27275c1f252d6326979cc5ac4c9ca8911560c4`。

R145/R146都证明标准Trainer policy bridge已在真实双卡训练中生效，但都没有通过LoRA参数更新gate。下一轮首先保留当前严格预算，补做第26层Q/K/V branch-local VJP重放及输入梯度累计顺序诊断，并关闭qk_right梯度采集缺口；确认具体差异后才改生产实现，再跑新cohort完整输出、梯度、参数更新与恢复对拍。R145 Native监控缺口意味着最终clean-monitor主树双卡验收仍需另建cohort；R146虽然监控干净，更新gate仍失败，不能替代最终正确性验收。

## 2026-10-24 R147：分支VJP探针在模型计算前失败

R147保存在LAB real-model/qwen3-06b-fsdp2-trainer-r147-layer26-qkv-branch-vjp-card6/，source manifest绑定当前源码005d13499891f3cc0de5d6b0f40b0dde8b619518。首个Native stage因probe读取manifest[source_commit]、而生成的manifest字段名为commit，在模型计算前以returncode=1退出；driver收据没有process identity或NPU进程见证，前后卡6均为空、无foreign occupant。原失败未覆盖，launcher日志SHA256为9c75e2903ba924a5613e12a65364beb3afdd5f1c2f9d1908cc570f9549f45aba。该cohort不包含模型验证结果，R148使用兼容字段名的新probe。

## 2026-10-24 R148：layer26 Q/K/V分支级VJP重放

R148 LAB cohort为real-model/qwen3-06b-fsdp2-trainer-r148-layer26-qkv-branch-vjp-card6/，复用R146保存的输入激活及上游cotangent，使用同一个固定输入分别在Native PyTorch和Jittor shim上重放Q/K/V base与LoRA投影，再对照原捕获的共享hidden-state梯度。源码提交为005d13499891f3cc0de5d6b0f40b0dde8b619518，source manifest SHA256为a594987ac36815095592def7e25661a77dfcef8d6682c1c4c962fdd899f84167，输入归档SHA256为e62a2f3565ae966864a42c996dd0714d2245f63182b6dfa776cf0d28fe59c9c3。

Native和Jittor阶段均在物理NPU6观察到本任务PID，returncode均为0、无foreign occupant、退出后卡为空；monitor gap分别为1和2，因此这是局部诊断运行，不作为clean-monitor验收。Jittor fallback delta为0。Jittor首次串行编译295/295个core单元约1196秒，完整stage约1274秒；此时间仅为诊断运行记录，不能作为性能数据。

64项跨运行时分支VJP/前向及LoRA参数梯度比较全部逐位相同；48项分支前向回放对照R146捕获也全部逐位相同。探针直接按Q→K→V求和得到的8组aggregate与原捕获shared gradient不逐位相同，最大绝对差3.8147e-6。运行时元数据确认R146对应激活梯度实际为BF16，虽然NPZ用float32保存；NumPy-only BF16逐次舍入重算显示K→V→Q和V→K→Q两种局部顺序在四种rank/upstream组合中都能逐位复现各自捕获值。Native与Jittor的R146 shared capture本身仍有少量差异：rank0有9个元素不同、最大绝对差9.5367e-7；rank1有13个元素不同、最大绝对差2.3842e-7。局部重放与求和顺序拟合不能证明完整模型autograd实际采用哪个累计顺序，也不能单独解释LoRA更新失败。

分支VJP对拍JSON SHA256为987f247ced347a8a073c368048e291beef17fed0cf952c3782b3913c7befe081；NumPy BF16累计分析为6916d8c525b9dac5f95f112fcc512238a58d75d64f23add433bf63daffa51679；跨运行时累计分析为336a15d8697b1d1ecc2350aa64720d76ed6ed20b1ef31e4037a2aa8e530e6182。Native/Jittor原始数组与driver执行收据、脚本哈希均保存在上述LAB cohort。

一个CPU归档分析命令误用了会自动加载shim的dev Python，短暂触发用户HOME下既有默认Jittor cache的CPU编译；没有调用torch.npu或产生模型结果，进程组随后被中断并确认退出。该cache目录早于本次任务存在，为避免删除其他运行可复用的旧cache而保留；具体路径、PID、清理的本任务临时文件及处置记录见LAB同目录tooling-side-effect-incident-r148.json。之后NumPy分析统一使用python -S和显式NumPy路径。

## 2026-10-24 R149：layer27 Q/K RMSNorm backward定点重放

R149 LAB cohort为real-model/qwen3-06b-fsdp2-trainer-r149-layer27-qknorm-vjp-card6/，以R146 rank0/rank1捕获的Q/K RMSNorm BF16输入、输出cotangent和反向输入梯度为参照，使用模型原Qwen3RMSNorm模块分别运行Native及Jittor shim。模型为Qwen3-0.6B，norm epsilon为1e-6；输入归档SHA256为15f9076b006e32803e1786bea2f2a6517974b3493e295863c6a9d92a2920e261，source manifest SHA256与R148相同。

两个阶段均经各自启动前的fresh empty check后在物理NPU6实际执行，PID与launcher匹配、returncode=0、无foreign occupant、退出后卡为空；Native/Jittor monitor gap各为1，Jittor fallback delta为0。阶段时长40.06秒和27.59秒仅记执行收据，不用于性能比较。16项前向输出比较逐位相同；输入和上游cotangent均逐位相同；两边各自的重放输入梯度都逐位复现各自R146捕获，8项norm权重梯度比较逐位相同。跨运行时输入梯度差仅出现在rank0：Q norm 1个元素、最大绝对差5.8208e-11；K norm 2个元素、最大绝对差1.8626e-9；rank1的Q/K输入梯度逐位相同。

R149将可见差异收窄到BF16 Q/K RMSNorm backward的极少量数值结果，但没有证明这是104/224个LoRA-B更新数组超预算的完整根因。重放与各自训练capture相符，forward及权重梯度相同，因此没有依据修改生产RMSNorm实现或放宽比较预算。比较JSON SHA256为c9d2b2fb892445c275c97790dd855f581d3b1a3c32df4ba1f46f94df11ea750a；完整输出和执行收据保存在R149 LAB cohort。

R148/R149没有修改生产源码，没有更新R145/R146失败门，也没有形成NPU性能结论。Jittor完整图中的trace.layer24..27.qk_right.grad仍缺失；layer25附近的softmax/QK/context梯度差异尚未用固定激活和cotangent完成分支级重放。

## 2026-10-24 R150–R161：注意力与ACL rsqrt反向定位

R150和R151分别因旧源清单未包含本轮dirty文件、以及runner缺少R146捕获目录变量，在模型计算前退出；两次均无NPU tensor结果，失败日志保存在各自LAB cohort，未复用覆盖。R150 launcher日志SHA256为5fccc73dd45b37c553a83864364fbb1c3f1d6d6f4cbeb9dd14413b7c0；R151 launcher日志SHA256为c1e7f493e1246aca7b3bffcae48ce13db9975089945047468956b668fae13812。

R152的layer25 attention VJP重放误把Softmax输出当作BF16，而R146 hook保存的是FP32 softmax输出、随后才cast为BF16；该轮结果只证明重放公式不匹配。R153改为FP32 softmax并保留cast边界，layer25 attention前向/VJP和固定Native upstream的比较共52项逐位相同、最大差0，物理NPU6执行且Jittor fallback增量0；comparison SHA256为b5b422f6450ffd6d074f0fc49def83d32e35a3a4270b85d6ec6cfe7f41c8d4ca。R153排除了这组局部attention算子重放，不能推出完整模型梯度一致。

R154对R146 layer26 Q/K RMSNorm实际模块重放，前向、cotangent、weight梯度一致，少量BF16输入梯度有最多3个元素、最大差2.98e-8。R155中两边rsqrt输入、输出及上游梯度一致，最早差异出现在ACL rsqrt反向，variance梯度最大差2.79e-9；仓库ACL公共rsqrt当前由1/sqrt组合实现。R156直接调用底层ACL unary rsqrt触发fused backend fallback并被零回落门禁拒绝，因此未采用。R157/R158的自定义Function诊断消除了rsqrt/variance/square路径上的分歧，但普通backward中x_fp32累加仍有极小差异；相关comparison SHA256依次为R154 3872d7aec26eac808dcf2aa704e0c8ac6269ca48f102c126682862375e6ec83e、R155 1b15d60028c1b535dceaa49cc1998f65e9b2ca1a09973626847be3655a82a0d5、R157 df79f524d09d24d9341bf63972635bce0e5fd824eeb30219784582173c9e7961、R158 785c1bf4d511f7377ffcac111683225300568742894de36081626737d92a920d。R156被拒绝候选日志SHA256为818d572be32ee59a6f4828d419652b83f5fc8e5d58b0b4631ea1b839d25d124e。

### R159：Torch ACL rsqrt生产兼容路径

在compat/torch/native_api.py中，ACL输入的torch.rsqrt保留既有1/sqrt前向，以Jittor native.Function显式实现Torch的VJP次序；FP16/BF16梯度先升FP32、按次序计算后cast回原dtype。底层ACL unary rsqrt没有启用。该路径保留既有ACL前向实现、不启用未注册的底层unary rsqrt，并以zero-fallback为硬门槛。

R159绑定源码提交005d13499891f3cc0de5d6b0f40b0dde8b619518，tracked-file清单SHA256为34c6974f3fa80afe6162887f8e8715d9668cc2c109e5747efe07e06ac685be54，输入归档SHA256为7988ec6786dadd362695f7a4d057f7d7fdcdf515314decac338a45ca550f3cd1。Native和Jittor均在物理NPU6执行、无foreign occupant、阶段后卡空，Jittor fallback增量为0；两阶段monitor gap各2，因此只作为局部诊断证据。比较Qwen3 layer26 Q/K norm四组输入/上游张量时，前向输出16项和weight梯度8项逐位相同；Jittor replay对Native训练捕获的输入梯度有3个BF16元素不一致：rank0 K-norm两个元素、rank1 Q-norm一个元素，最大差5.96e-8；rank0 Q与rank1 K逐位相同。该结果相比R154显示局部梯度差收窄，但没有执行optimizer更新，不能证明LoRA更新gate通过或关闭R145/R146全图差异。最终comparison SHA256为a4223cc7c08f32b087317b361ba9b2c3ea4397b16fad47fc79e3879f3b9755eb；cohort SHA256为c361cd3dee90465c349015b2060780ead2f8bc898345714ef90afe2802596f4d。

### R160/R161：ACL rsqrt回归测试

R160在真实NPU上运行新增测试并得到2 passed，但driver随后把Path对象当文件句柄调用close，未能写出execution-r160.json，因此按driver收尾失败保留，不作为合格运行收据；其日志SHA256为118c7014cffbbb2718f2aeb2dc8bf77143a1d8935c5375714bae26398b232e6a，后post-check确认NPU6为空。没有覆盖R160输出。

R161使用修正后的独立driver和新cohort复验compat/tests/torch/test_rmsnorm_mixed_dtype.py中的ACL torch.rsqrt VJP合同，FP32与BF16共2项通过、14项按-k筛选未执行。driver fresh-check NPU6为空，运行中观察到本任务进程在物理NPU6、无foreign occupant，returncode=0、monitor gap=2、阶段后卡空；测试断言Torch次序梯度、输出与梯度实际ACL放置及fallback增量0。source manifest SHA256为907ba8e77fb18a10f380f1ed5d4909aadf6f62f1210bb7a03bd33ebaadbb1504；pytest日志SHA256为247c0dd73222d1ef367e5644805edb5da40f76cc4f553b8ca103d2546e510de5；execution JSON SHA256为3b9c0f0980c6715e00dffc455e0918e7390e1d7478cc1716c1d77e5153f27bf4，cohort SHA256为7832c0c075b6b92880414f94f935c8261a16a811df8c4d352ee9679cbb5cae07。该回归只覆盖局部算子VJP，不替代真实模型训练、双卡FSDP2和恢复验收。

原始数组、逐阶段execution、driver快照、失败日志、manifest及比较脚本均保留在LAB的real-model/qwen3-06b-fsdp2-trainer-r150至r161各cohort目录；环境与缓存留在LAB。
## 2026-10-24 R166–R169：BF16 rsqrt VJP合同与完整模型复验

### R166–R168：校准ACL Torch rsqrt VJP次序

R166使用隔离的原生Torch 2.7.1/torch_npu 2.7.1.post2环境，在物理NPU6对16,384个BF16元素做直接VJP探针。Native autograd与候选次序 (-0.5 * upstream) * output.pow(3) 逐位一致；旧的FP32 widening后再cast公式有3,990个元素不同，最大绝对差0.125。该结果只确定直接BF16 rsqrt输入的算子合同，不代表Qwen3整模型通过。结果JSON SHA256为0003d599b4cc87aee2f02fe789954888dee63191e6d9daf7b6db3bab911742bf。

R167在Jittor ACL合同测试里复现了dtype边界：FP32通过，但未显式cast的BF16 cube导致6/8个检查元素不匹配，因为Jittor的pow路径先提升到FP32。R167的driver preflight另有一次路径错误，未启动NPU；正式运行失败及原始pytest日志保留在LAB。R168将cube显式cast回输入dtype后，在物理NPU6通过FP32/BF16两项、14项按筛选未执行，returncode=0、无foreign occupant、fallback增量0。R168 source-manifest SHA256为ebf827ad58dcd2d1c1f0c41f36aba78b9222874bde82e75a7667c8f79c065acd，pytest日志SHA256为0551a487679795b68c9fda4f16cdff7d420c0c0eb12ae6d7864b012cba7c47f4，execution收据SHA256为f4a32fd8e3921d483acb7fee4e78e8980be0f4ea16dcb0e13f349728c15389dc。

### R169：真实Qwen3-0.6B BF16 LoRA双卡Native/Jittor对拍

R169建立新cohort，目录为LAB real-model/qwen3-06b-fsdp2-trainer-r169-rsqrt-pow3-full-lora-dual56。源码HEAD为005d13499891f3cc0de5d6b0f40b0dde8b619518，source-manifest SHA256为bc72d64e003bdef53cc9930d69999d272e40fc973f4fe57e0b1401af93aa88e7，执行前tracked工作树diff SHA256为1f5e8dacda414bf87d9c5ecb82fc56a8e9817a40af8fb4d47568cd4b72c4dbf3。使用共享Qwen3-0.6B权重、既有LoRA初始化、相同tokenizer与Trainer FSDP2单步协议，Native oracle和Jittor shim隔离；R169本身不覆盖fresh-process恢复。

两个rank的串行预热最终都通过。预热期间rank0曾遇到一次Jittor jit_utils同进程重载失败；另一次fresh检查发现物理NPU5有外部VLLM占用，driver仅终止本任务自己的预热组并以125退出，没有触碰外部进程、没有模型计算。外部占用离开后，rank0 attempt3与rank1 attempt2各自预热通过，完成收据明确标记not_two_rank_proof=true。正式运行前卡5/6为空。

Native双卡阶段rc=0、耗时415.29秒、monitor gap=0、无foreign occupant；rank0映射物理卡5，rank1参数实际设备为npu:1（映射物理卡6），但npu-smi记录rank1 PID也出现在物理卡5，因此不能宣称Native进程只使用卡6。Jittor双卡阶段rc=0、耗时174.46秒、monitor gap=0、无foreign occupant，rank0/1分别只观察到物理卡5/6；两rank参数实际设备均为各进程的npu:0，fallback增量均为0。两阶段结束后卡5/6均空。阶段耗时仅作执行记录；更新gate失败，不作为速度数据。

对比严格沿用R146门槛：logits atol=0.005/rtol=0.03，梯度0.01/0.05，参数更新2e-6/2e-5。两组forward数组、224组完整参数梯度和224组初始参数全部通过；梯度最大绝对差0.015411，最大预算比0.5621。参数更新失败102/224数组，每rank 51个，全部为LoRA-B；每rank共有1,462个元素超预算，最大绝对差5.999600216455292e-5，最大预算比29.9890。Loss比较逐位相同。update-cause分析在每rank 344,064个LoRA-B梯度元素中找到1,370个符号不同；这与零初始化LoRA-B首步AdamW更新对微小梯度方向高度敏感一致，但还没有定位完整图中造成梯度漂移的具体算子。梯度采集点位于Trainer反传之后、全局裁剪/optimizer之前，不能把差异简单归因到AdamW实现。

R169捕获的诊断梯度显示trainer.loss梯度一致、logits梯度最大差不超过1.46e-11；decoder.27/.26输出梯度一致，差异随后沿反向链逐层放大到decoder.0（rank0最大差0.00244140625，rank1为0.029296875）。这将现象收窄为上游反向链上的累积差异，但不能据此归因于某一个attention、RMSNorm或matmul kernel。Jittor qk_right诊断hook仍不完整。

R169与R164在每个rank内的Native快照862个数组、Jittor快照865个数组均逐数组相同。Transformers 4.56.2当前Qwen3RMSNorm源码先把hidden_states提升到FP32，再在FP32 variance上调用torch.rsqrt；R168校准的是直接BF16 rsqrt输入合同，所以没有改动这个模型路径，也没有改变R169训练数组。这解释了R169没有改善，但不解释完整训练梯度漂移的根因。不要把R168算子通过外推成Qwen3模型通过。

比较JSON SHA256为6405221ca69cdffb22626ea5f1612d8af51350b4925270a40450a0bc0d00a396，update-cause JSON SHA256为4ba2e45e17c97ef87c5ba63a00949bfb454087a9fb1eb7bc59444181245c6fe0；Native/Jittor execution收据SHA256分别为8d1d602d026877732c994950b07202b61b73921d4a8dbfdd44ae6f0893b1e814和8536ac9218ff7933d86250bcc240af788f6051720bc5893291d8863c209bd8a1。四份rank tensor快照、完整进程/NPU监控、模型输出、梯度、参数与失败项均留在LAB该cohort目录。

### 工具误触发记录

本轮一次只读模型源码检查误用dev环境的Python import，触发Jittor在用户默认cache做CPU core编译；之后立即终止本任务独立进程组。未运行Qwen3模型或调用本任务的NPU计算。仅清理该进程在精确时间窗和精确配置目录中新生成/改写的75个文件，保留更早的缓存文件；进程和缓存处置细节记录在LAB该cohort的tooling-side-effect-incident-r169.json。后续仅使用文本读取和python -S离线JSON/NumPy分析。

## 2026-10-24 R170：Torch兼容CPU回归修复与结构门禁

本轮没有创建新的NPU模型cohort。代码快照基于HEAD 005d13499891f3cc0de5d6b0f40b0dde8b619518，包含7个兼容实现/测试文件；按文件路径与内容计算的R170 source-manifest SHA256为52e2434d83512060dcfa77b426e8c61b78e1a4c89b7b5f13a18fddafdb1aed4a。它仅标识本轮CPU修复代码，不可用作R169模型运行的源码范围。

修复内容：
- method_api显式导入_PYTHON_SCALAR_DTYPES，修复Tensor.to(int/float/bool)类型入口未定义；frontend Tensor(existing_tensor)在未显式传requires_grad时保留原生clone/view的梯度状态。
- C++ _fast_binary的float64张量/Python float除法快捷路径改为物化无梯度float64 0维Var，与Python慢路径采用相同操作数精度；原bit-exact测试通过，数值阈值未放宽。
- 将optimizer_api中的参数状态映射、state_dict与load_state_dict实现拆入optimizer_state.py，原模块继续再导出原对象；optimizer_api从981行降为558行，新模块456行，未修改结构门槛。
- 保留R159 ACL rsqrt Torch VJP改动和回归用例；R169快照与R164逐数组相同，故R170不声称该改动修复真实Qwen3漂移。


对R169既有cross-runtime comparison与update-cause JSON做了标准库只读归类，未重新加载NPZ、未import Jittor/Transformers、未调用NPU。两个rank各自112组梯度门均为0 mismatch；LoRA-B共有344,064个梯度元素，其中1,370个符号不同。每rank的51组更新失败数组只来自self_attn.q_proj LoRA-B（26组）和self_attn.v_proj LoRA-B（25组），覆盖layers.0–25：0–24层各两组、25层一组、26–27层为零；这些失败参数的初始值均为零。符号差异计数从layer0的91、layer1的92逐步降到layer25的1，layer26/27为0。在目前已采集的projection input cotangent中，layers.27/26逐位相同，首次非零差异出现在layer25：rank0最大4.77e-7、rank1最大1.53e-5；随后向低层增长。该分布与小幅反向差异向低层累积、再穿过零初始化LoRA-B首步AdamW更新的符号边界一致，但不能据此定位源算子或判定AdamW实现有误。离线结果文件为LAB同cohort的update-localization-r169.json，SHA256为66c81835b09c9880749fd0c497fe3ba7f093bc0e85fc9425b5e1eb1a188d82ee；下一次探针应优先观察layer26到25边界及后续Q/V输入梯度和共享hidden-state累计，并补齐qk_right捕获。


验证：
- tools/run_test_suite.py --tier core退出码0：Native 103 passed、45 skipped、1 xfailed；Torch shim 238 passed、1 skipped。合计341 passed、46 skipped、1 xfailed。
- tools/run_test_suite.py --session torch --tier core -- tests/structure退出码0；其中tests/structure目录覆盖1797项，1794 passed、3 skipped，耗时588.58秒。该运行器使用LAB独立JITTOR_HOME、CPU backend和JITTOR_TORCH_SHIM=1；本结果为CPU结构门禁。
- bash tools/check_repo_layout.sh通过：298个活动Markdown治理与仓库布局均通过；git diff --check通过。
- R170离线分层定位写入报告/看板后，使用同一Torch CPU运行器对tests/structure/test_docs_structure.py复验，254 passed、1 skipped，退出码0。
- 以上结果不包含昇腾设备计算、HF真实模型训练、参数更新或恢复验证；R169的102/224更新差异和Native rank1卡映射问题仍原样保留。

## 2026-10-24 R171/R172：Qwen3 attention context与RMSNorm VJP定位

两个诊断cohort均绑定当前主树提交 `d2910102697bbcd6fab9993f8b72c6d3aa0312d8`、source manifest SHA256 `6b72327b1139f6dad49a792c4e78b065680f74d8c073332186b6d80c7ac2ecb7`，使用R169捕获的Native/Jittor激活与上游梯度；均为单卡局部重放，不执行完整模型更新，不改变R169原门槛。

### R171：layer25 context matmul VJP

LAB cohort为 `real-model/qwen3-06b-attention-context-vjp-r171-card3/`。物理NPU3依次运行隔离的Native oracle和Jittor shim；两个成功阶段均returncode=0、monitor gap=0、无foreign occupant，Jittor fallback增量为0，阶段后卡空。Jittor首次attempt仅因harness重复复用同一context manager在VJP前失败；修正为每个case创建新scope后，在新attempt完成，不覆盖失败日志。

对两个rank、Native/Jittor两套快照重放layer25 context matmul，固定相同输入和cotangent后，12组输出、左右输入VJP数组全部逐位相同；各运行时重放也精确复现其自身保存的全图输出及输入梯度。离线全图快照显示context operands/output前向逐位相同，但context输出收到的上游cotangent已有差异：rank0最大2.3842e-7、rank1最大3.8147e-6；context-left梯度跨运行时最大差分别为3.0518e-5和4.8828e-4，context-right最大差为7.6294e-6。由此可排除这组固定输入下context matmul局部VJP为差异起点，证据指向其上游反向链；这不能定位到具体attention分支或共享梯度累计。比较JSON SHA256为 `1d98a44c6344bda44f02061233808bab28ccaf6b1eef4ca18de1ee9b74ca00ab`。

### R172：layer26/27 Qwen3 RMSNorm VJP

LAB cohort为 `real-model/qwen3-06b-qnorm-vjp-r172-card3/`，对真实模型layer26/27的Q/K RMSNorm，使用R169 rank0/rank1的Native/Jittor输入与cotangent做Native/Jittor交叉重放。Native成功结果是attempt2；attempt1因driver漏传权重快照参数、在设备初始化前退出。Jittor attempt2因driver后续环境合并覆盖暖缓存路径而启动冷编译，随后只终止本任务自身进程组；保留该日志和执行记录。修正缓存路径与attempt映射后，Jittor attempt3复用R171缓存，在NPU3用20.26秒完成。成功Native/Jittor阶段均returncode=0、monitor gap=0、无foreign occupant，阶段后卡空；Jittor fallback增量为0。运行前检查确认物理NPU3无进程且利用率/AICore为0。

32组同输入比较中所有RMSNorm输出均逐位相同；仅6组输入梯度数组有非零差异，共12个元素，最大绝对差5.9605e-8。两运行时的各自重放对照全图保存梯度时，也各仅有3组输入梯度数组出现同量级的3个差异元素，最大值同为5.9605e-8。该结果说明当前局部RMSNorm路径基本复现各自全图，但其微小输入梯度差异不足以解释R169中LoRA-B更新的主要偏差；没有修改生产代码或放宽预算。comparison JSON SHA256为 `c1cab953a794a284c74498142183eb17bc536ca099149e0c9d145f8d303a3a05`，runner summary SHA256为 `c28a3322737a1d9d59eeccd15132a5c9481cfc9971c75877343065e1c0c4fffa`。

R171/R172所有数组、完整日志、NPU监控、失败attempt、探针与比较脚本及缓存均保留在各自LAB cohort。下一步仍需采集layer25 context上游的attention输出投影、残差及相关共享hidden-state累计边界，补齐全图 `qk_right` 梯度观察；在明确生产差异之前不做数值门槛调整。

## 2026-10-24 R173–R176：layer25/26反向链与RMSNorm VJP继续定位

这些新cohort使用R169保存的四份Native/Jittor rank快照和共享Qwen3-0.6B权重，绑定当前提交 `fd03f62972e2245e24d8498cd25f819139f8e6f8`、3338个tracked文件及source manifest SHA256 `117cbf200c91a9ed801a563ec7e144d27fa8a13384b0cb7ec811de88699a1e34`。该提交只新增前一轮报告/看板文档，运行时源码与R171/R172使用的 `d2910102697bbcd6fab9993f8b72c6d3aa0312d8`相同。全部是物理NPU3单卡局部诊断，不执行完整模型更新。

### R173：layer25 attention output projection

LAB cohort为 `real-model/qwen3-06b-oproj-vjp-r173-card3/`。从共享 safetensors 独立提取 `model.layers.25.self_attn.o_proj.weight`，确认BF16权重形状为 \(1024\times2048\)，在Native与Jittor上分别用相同context输入和o_proj输出cotangent重放Linear前向与输入VJP。两个runtime的8组同输入输出/梯度数组逐位一致；两边重放也都精确复现各自全图保存的8组数组。成功阶段均returncode=0、monitor gap=0、无foreign occupant；Jittor fallback增量0，阶段后卡空。

准备过程保留了三个未触及NPU的失败记录：权重提取预检最初误写为方阵，修正为实际1024×2048；首次driver使用文档提交前的d291 manifest而在fresh admission前退出，随后重算全部3338 tracked file hashes，确认仅两份文档变化并绑定fd03；首次离线分析因JSON metadata key未去除NPZ namespace而退出，修正索引后完成。日志和执行收据均留在该cohort。comparison JSON SHA256为 `bcaa900d607a0950baaa1323036295f157aa70237470e3b5eb75b872cc92a5bd`。

### R174：layer26 QK matmul与缺失qk_right重放

LAB cohort为 `real-model/qwen3-06b-qk-layer26-vjp-r174-card3/`。对layer26保存的qk_left/qk_right及qk_out cotangent分别由Native与Jittor重放。12组同输入输出与左右VJP全部逐位相同；每个runtime重放Native保存的两rank qk_right梯度都精确，重放其余已保存全图输出/梯度也为零差异。原Jittor全图 qk_right hook仍缺失，因此其右侧梯度只记为同输入局部重放值，不冒充全图捕获。

离线核对R169 layer26 attention trace：QK operands、qk_out、softmax输入/输出和context operands/输出前向全部逐位相同；qk_left、qk_out、softmax_in/softmax_out、context_left/context_right/context_out已保存梯度跨runtime也逐位相同。Jittor qk_right全图梯度未捕获。R174把QK matmul及已采集的softmax/context梯度移出当前首要差异源，但没有覆盖Q/K RMSNorm VJP。comparison JSON SHA256为 `531a4c5adc62e6f86d2122c6e30ac009ad34fc44b14a00920302d536c2c39b42`；offline attention frontier SHA256为 `1d41404dc3f7e162106f6b602d9d7ffbd0ce6dd86228388448e953515559dc92`。

### R173离线梯度前沿

同批CPU NumPy-only归档分析未加载Jittor/Torch，也未调用NPU。layer26 MLP各模块梯度跨runtime逐位相同；可见非零差异集中在Q/K路径和input_layernorm梯度：rank0 k_proj输出梯度最多3元素/5.96e-8，rank1 q_proj最多2元素/2.98e-8；共享投影输入梯度分别最多9元素/9.54e-7及10元素/2.38e-7。每个runtime内block→MLP/down_proj及self_attn→o_proj对应cotangent逐位一致。layer25 block输出cotangent已各出现少量差异（rank0 4元素/1.19e-7，rank1 2元素/4.77e-7），并沿norm/attention支路向context梯度传递。该分布与R172层26/27 Q/K RMSNorm的微小输入梯度差异吻合，但仍不能证明它们足以造成整个LoRA更新gate失败。layer25与layer26 frontier JSON SHA256分别为 `e656a3f49340517d429ec05fe9e57c694f053e5e2fc56949719a93936a8452ab`、`c6488b3e55b886ffef1dce093f1146aab5f54b29baf7f15d7b21055cffb4b682`。

### R175/R176：RMSNorm手写VJP候选，没有采纳生产修复

LAB cohort分别为 `real-model/qwen3-06b-rmsnorm-manual-vjp-r175-card3/`、`real-model/qwen3-06b-rmsnorm-vjp-order-r176-card3/`。Native和Jittor普通Qwen3 RMSNorm路径复现各自全图输出逐位相同；同输入比较的输出也逐位相同。标准Jittor与Native对照的16个输入梯度数组中有6组、共12个元素不一致，最大差5.9605e-8。

R175加入只在诊断worker里的手写输入VJP，将归一化乘法分支与均方/rsqrt分支显式组合；16组梯度中仍有6组、8个元素不一致，最大差仍为5.9605e-8。R176再比较五种Pow梯度乘法与mean缩放顺序，五个候选逐数组相同，结果仍是6组、8个元素不一致。forward没有改变。该候选没有通过逐位门槛，未修改生产代码、未放宽数值预算，也未推断其足以修复R169更新失败。R175 comparison JSON SHA256为 `b391f6b1e82f014dee48879981ede7f5531ad2c38bb7fa1a0fbfaaa5a1d2b91c`；R176 comparison JSON和runner summary SHA256分别为 `534f1f634123e9db8dc76c7406e004dd4c92f67df7e195c11766dea7fb974411`、`16adb46ab33aeb019e06a8c5440e2ea915f3406d899e63a641bc51f936409833`。

R173–R176 Native/Jittor均按顺序在NPU3完成，成功阶段monitor gap=0、无foreign occupant、阶段后卡空，Jittor fallback增量为0。首次冷JIT复用R171同源缓存，不与其它任务并行；完整tensor、报告、失败准备日志及缓存均留在相应LAB cohort。没有新增生产源码修复或完整模型验收。

## 2026-10-24 R181：生产Pow2路径的Qwen3 RMSNorm局部复验

R181绑定提交 fcf3571d024e7394babd8f4c601dab633d30bb1c、3339个tracked文件和source manifest SHA256 8dfc65f172de224748aa990ba49af71bb1f9bd1566c6cd757d0a197cb2381edd。LAB cohort为 real-model/qwen3-06b-production-pow2-vjp-r181-card3/，读取R169既有Native/Jittor rank0/rank1快照与共享Qwen3-0.6B RMSNorm权重；不覆盖R169/R180输出。比较文件SHA256为8a05b43250e0a533405f04ef21f91acc547e2fd818c57201d200084d175074b4，runner summary SHA256为e25f5e0420dcb5d5b96e1c20c66b7bfcbdf496c9034439d87d5eafe2b4c0925e。

### 本地CPU门禁

新增回归测试覆盖 Tensor.pow(2)、torch.pow(tensor, 2) 和 tensor ** 2，检查输出保持x*x且反向逐位等于(upstream * 2) * x.pow(1)，1项通过。独立CPU冷构建耗时712.608秒，单测耗时5.50秒；复用缓存重跑日志为1 passed in 0.29s。按tools/run_test_suite.py --tier core串行执行并保存完整日志：Native 103 passed、45 skipped、1 xfailed；Torch shim 238 passed、1 skipped；合计341 passed、46 skipped、1 xfailed，退出码0。core白名单不含这条新定向测试，因此两者分别记录，不混为一个测试集合。仓库要求的完整 tests/structure 复验退出码0：1558 collected/executed，1556 passed、2 accelerator skipped，耗时430.34秒；check_repo_layout.sh通过，298个活动Markdown治理和仓库布局均通过；本段报告更新后，tests/structure/test_docs_structure.py复验16 passed。

### NPU3局部生产路径对拍

隔离的原生Torch oracle先运行，Jittor shim随后运行；二者均通过物理NPU3新鲜空闲检查，returncode=0，分别耗时32.33秒和26.20秒，monitor gap=0、无foreign NPU进程，结束后卡空。运行进程实际映射均为物理卡3；Jittor fallback增量0且运行前后源码哈希不变。Native oracle使用oracle-npu-cann与CANN专用PYTHONPATH，Jittor使用当前源码和私有adapter overlay；两者未并行。

生产Pow2路径的标准输出在16组固定保存激活重放中逐位匹配Native，16/16组无差异、最大绝对差0；输入梯度同样16/16组无差异、最大绝对差0。样本覆盖两套来源快照、两个rank、layer26/27的Q/K RMSNorm。R180手写候选中显式保留x.pow(1)的f_pow1_grad也在16/16组匹配Native；另五种乘法/缩放顺序各有6/16组、共8个元素不匹配，最大差5.960464477539063e-8。比较预算未改。

### 结论边界与保留项

这是生产Torch shim Pow2算子入口在真实Qwen3保存激活上的局部前向与VJP复验。它尚未重新执行R169全模型训练，故R169的102/224更新失败仍是当前全模型状态；还没有验证优化器更新、恢复状态、双卡FSDP2、nested HCCL、激活重计算或性能/显存。不得把局部逐位一致外推为完整模型通过。

准备过程中有四次只影响本地harness/日志采集的非NPU失败：定向测试输出采集的PowerShell变量展开错误、语法预检路径变量为空、首次布局门禁日志路径遗漏-card3，以及首次完整structure日志路径遗漏-card3。pytest本身完成且摘要为1556 passed/2 skipped；修正日志路径后同一完整structure门禁退出码0，原始输出与执行记录均保留在该cohort对应的capture-preflight-failure-001.*、harness-preflight-failure-001.*、layout-launch-failure-001.*和structure-preflight-attempt-001.*文件。正式Native/Jittor阶段均成功。没有改变NPU进程、数值门槛或公共环境。

R181的完整NPZ、报告、manifest、运行日志和NPU监控全部保留在LAB cohort；CPU单测与core日志位于 _state/test-suite/pow-two-target/。

## 2026-10-24 R182–R190：真实单步更新、恢复故障及裁剪修复

### R182：真实Qwen3-0.6B BF16 LoRA单步数值通过

冻结HEAD为 `ee2f3e9345085b6bff1ded9519ea10af2c62723f`，3339个tracked文件manifest SHA256为 `9843decf48326baedd8ba946281fdac767d6a1bbda8c326fd19375477a6617a1`。LAB cohort为 `real-model/qwen3-06b-pow2-full-lora-r182-dual56/`。Native和Jittor分别新进程运行，Native 387.842秒、Jittor 165.369秒，均rc0、monitor gap0、无foreign occupant；Jittor fallback增量0，结束后卡空。Native rank0 PID1326918仅见于卡5，rank1 PID1326919参数为npu:1、但PID同时见于卡6和卡5；Jittor PID1345297/1345298分别只见于卡5/6。因此数值比较通过，严格Native排他物理映射仍未验收，阶段耗时不作性能。

完整logits、224组参数梯度和224组初始参数逐位一致；224组更新参数均满足未改变的atol2e-6/rtol2e-5，最大绝对差7.2759576e-12。forward预算仍为.005/.03，gradient仍为.01/.05。这只关闭本冻结源码、该单步协议中的更新差异，不能外推4–10步、完整恢复、随机策略或性能。comparison SHA256为 `2da35fdb3e28c2b4bee9bc486d87e24f059f8ae70b6b2ead6239c7c070fbd8ff`；runner summary为 `d46b06d7e48dc41ed3b25e74a8bde1a7afcd949e891dfe7288f5794dee7a8661`。

### R183/R184：保留Native恢复失败与Jittor裁剪失败

`real-model/qwen3-06b-fsdp2-resume-r183-dual56/` 的Native continuous2、interrupt1、fresh resume到2三个阶段均rc0、无foreign、monitor gap0。恢复仍FAIL：连续Adam step2、恢复step1；每rank 112/112组最终LoRA参数不同、最大绝对差2.4981113710964564e-5，483组trainer状态中224组不同。末条sample及scheduler匹配不抵消Adam状态失败。原checkpoint实际包含112组step1状态及56组非零moment，不能把失败归于未保存。

独立oracle-cpu审计原样安装的Accelerate1.10.1函数发现：保存使用FQN getter，加载却用fresh optimizer空数字键state_dict作为DCP模板，结果未请求moment。候选改为公共get_optimizer_state_dict模板后，真实checkpoint的112组状态、下一次更新和step2与直接恢复reference strict0。审计属于CPU定位证据，adapter候选仍待主树整合及真实模型NPU恢复；审计文件位于 `real-model/r183-native-audit-20261024/`，不计作新的NPU阶段。

R183原Jittor阶段先因LAB缺少diagnose_matmul helper失败；R184补齐helper后，真实训练在clip的0维coef乘4096元素梯度处报cannot broadcast x [] to [4096]，rc1、61.948秒、monitor gap0、无foreign。两次失败及原日志保留，不覆盖。

R183/R184 worker仅采集每phase首次logits/梯度与训练返回后末态，没有完整第二步轨迹；initial_parameter_values在checkpoint加载前采集，不能称为恢复初态。后续需即时保存每micro/clip/update状态的新harness。

### R185：修复裁剪后完成首步，默认SHARDED保存失败

基于ee2f3e9加 `compat/torch/grad.py` 和现有梯度测试两处dirty源码，全部3339文件manifest SHA256为 `0d58f68e0dc8a9239d23bda622dc69d47e02b66dab02d40c7b141f6520b50616`。每个local梯度形状显式broadcast设备端裁剪系数，未引入host item或改变裁剪数值预算。

新cohort `real-model/qwen3-06b-fsdp2-resume-r185-dual56/` 的rank0/1分别串行预热1249.193/1265.942秒，实际卡5/6、fallback0、源码未变，两者退出后才双卡运行。正式Jittor continuous rc1、166.634秒、monitor gap0、无foreign，PID1717076/1717077分别在卡5/6，结束后卡空。已完成首步optimizer update，loss3.8984、grad_norm7.25462532043457、lr3e-5；随后Accelerate默认SHARDED保存调用FileSystemWriter触发NotImplementedError。未改FULL来绕过。execution SHA256为 `24dc877d76949e9d5da3c259ac4cdc6578897accb23bbfc6e69ef035d632363a`。

本cohort把fallback scope改为覆盖实际trainer.train；Native文件显式绑定复用R183不可变结果，不算新Native运行。worker失败前尚未即时落盘完整逐步数组，因此不提供完整轨迹通过结论。首次JIT固定g++、python-config、cache_name及MPI-off，后续复用需保持相同构建配置。

### R187/R188：梯度裁剪CPU/NPU回归

CPU现有梯度管理文件全部7项通过、0skip、38.53秒，日志位于R185的 `grad-management-cpu.log`；新增4096元素用例另有R184独立1P/6 deselected记录。

`device-api/clip-gradient-regression-r187-card5/` 因漏带预热编译器路径触发jit_utils重建，未进入测试或NPU计算；R188 attempt1在恢复正确配方后按保护要求rc3请求fresh-process重试。原日志保留。R188 attempt2 rc0、118.741秒，pytest7 passed/0skip、95.11秒；18次clip调用记录包含4096元素实际ACL0输入与norm，CPU子例记录真实CPU。PID1770032只观察到卡5，monitor gap0、foreign0、fallback0到0、源码前后与R185 manifest一致、结束后卡空。

R188 attempt2 result SHA256为 `f7371869329fbf2c60744316e259585114fcacb038adb1a47edd699ba44374b0`，execution为 `d3ed9bf56ed34fc92e62b22fdf11219bad4dc6fd86bf8c4f0aef6873fb38b3e9`，pytest/stdout为 `514c51f59c7f2bcc6ef9bcd1eed46534d284af1866d03055ab41cd7b9a655fb2`。

### R189：真实ACL确定性策略接口复验

在同R185源码manifest上运行现有test_npu_deterministic_policy_toggles_backend_and_restores_state。attempt1因启动未source CANN、未找到ACL compiler而导入失败，未进入NPU计算；修正启动环境后，`device-api/deterministic-policy-r189-card5/attempt2/` rc0、23.802秒，1passed/0skip。测试True/False/True均核验真实backend flag、ACL求和，拒绝warn_only及错误类型，finally恢复原False。额外witness context为ACL0、sum4，fallback0到0，PID1826641实际卡5、monitor gap0、foreign0、结束后卡空、源码未变。

result SHA256为 `9980bcccb1331cf8d7d8a8369dca120f661c496cd8e97031fd3faf7aaff0bb26`，stdout为 `4e13bd0efd0b04edf397401014c70418fcc1f1c1a16a492b5812f0f01a828442`。合并后源码已具备ACL策略setter/getter，本文历史UNIMPLEMENTED_FALSE只描述原冻结源码；这不证明完整训练算法可复现或双方性能策略一致。

### 源码同步与后续准入

2026-10-24 17:53的fetch及独立ls-remote均成功，wip为305c2f0866f163b0549d47b2e4264c5d41a86385、2.0-refactor为ca56aa802c2acf9a22bca32d1bee3276381ed203，均为当前HEAD祖先；既有9390fa0合并包含该2.0 tip，无需重复合并。18:26恢复时fetch与ls-remote各50秒超时，跟踪ref未改变，但不能据此声称远端仍然最新。两次收据与既有失败历史均留LAB handoff/remote-sync*.json。

R186完整4updates/accum2/split2、两runtime各continuous/interrupt/resume的逐步harness只完成静态/离线正负测试，尚未模型执行。SHARDED存储、FQN优化器adapter、sampler/ForkRng候选仍处于独立worktree，未合入本段计算源码、未授予NPU通过。所有原始日志、NPZ、checkpoint、manifest、缓存与机器路径均为LAB未版本化产物。

### R190：裁剪修复的提交门禁

本轮没有新增模型计算。主树保持ee2f3e9加裁剪实现/测试及报告/ASC看板四个明确文件冻结，CPU运行前后文件SHA一致。Native core为103 passed/45 skipped/1 xfailed、465.73秒；Torch core为238 passed/1 skipped、567.90秒；合计341 passed/46 skipped/1 xfailed，两个session退出码均0。后续命令tools/run_test_suite.py --session torch --tier core -- tests/structure覆盖完整structure目录并同时重复Torch core，合计1794 passed/3 skipped、637.98秒、退出码0；不把这个总数误写为纯structure目录的case数。布局检查通过298份活动Markdown，内部链接检查通过284文件，git diff --check通过。

门禁日志及逐文件hash位于LAB未版本化 `device-api/clip-and-policy-closeout-r190/gate-results.json`；core.log SHA256为 `04f28b11473fa4619bbdfa6fe5936b13790337c2cfedf9500d2344b04c13af7b`，structure.log为 `7694aac38580f415f9544ffac6a1bd5b4182282bc697ad66db7135273e27210f`。仓内旧kernel_meta只包含本任务已退出PID引用和空临时目录，核验路径及PID均不活跃后整体归档到本cohort，保留文件hash和原路径收据，没有暂存生成物。

后续DCP/adapter/PEFT修复尚未包含在本次门禁源码中；其主机合同通过不能替代整合后的真实CPU/HCCL张量、完整模型及新进程恢复。R186工具在执行前审阅中已恢复完整16train/4eval、原batch2/accum2/4updates/split2和SequentialSampler，split为双卡epoch边界；确定性政策也必须在新运行显式设置并保存真实值，旧两行重复/默认随机sampler候选没有模型执行。完整生命周期、batch1及随机中间cursor仍单独待验。

## 按原计划截至R190仍需完成

1. 以整合后冻结主树完成真实Qwen3 BF16 LoRA 4–10次更新及完整逐步输出/所有梯度/更新对拍；修复默认SHARDED保存及Native优化器加载，双方fresh恢复核验model/Adam/scheduler/RNG/data cursor与下一次更新，same-runtime strict0。
2. 完成最终双卡FSDP2严格rank物理映射、no_sync累积、真实分片、保存恢复、FP32及混合精度输入梯度组合；R182只授予该单步数值资格。
3. 完成nested child+root HCCL及所需dtype组合，不从Linear或单步真实模型外推。
4. 实现真实activation recompute并验证重执行、闭包参数梯度、RNG preserve与显存；完善Generator合并合同、sampler、多worker预取和随机数据fresh恢复；训练确定性仍需完整fresh验证。
5. 对受本次变更影响的Transformers/Tokenizers/PEFT/Accelerate/safetensors组合做最终源码回归，保留历史模型矩阵的准确冻结身份。
6. 仅对应正确性通过后测配对至少10次NPU推理/训练吞吐、延迟、峰值live/reserved显存及稳定性，区分JIT/首步/稳态；当前没有合格NPU性能结果。


## 2026-10-24 R192–R204：检查点CPU恢复与nested实卡复验

### 源码与同步边界

R194绑定d2974a3d7fc93aff42333b82b347437bb94048f0加当时四文件候选，
3343 tracked文件清单SHA256为06f37890c2186a79e4b6ff6debd4cbbce882e6b48ecdb261b569c44aac6e060f。
之后整合检查点状态修复eaa12822f和Adam叶身份/closure梯度模式修复a266f384b。
R195 combined-r2及四个CPU singleton cohort绑定a266f384b50cc627022d1b8e203af6f1dfbbdc39加六文件候选，
涉及legacy同对象别名、FULL物化测试、默认DCP加载、原生Module加载及safe-globals发布。
R195的3344文件清单SHA256为e981c0073815a5c5a3fa6194f7a0ced438874f48ec7765c82248e32c61591072；
R197同一代码集合的独立清单为ec8c8d0fb97fa2feb9179c4d582fb775cefd2ce9d6136dd27bb311dad1a887cd。
全tracked清单以UTF-8/NUL读取Git路径，目录symlink使用SYMLINK前缀加readlink字节哈希，保留中文路径。

新HTTP/1.1 fetch在45秒超时，不证明远端最新。LAB记录为
_state/ascend-compat/handoff/remote-sync-20261024-111716-r195.json（文件名UTC）。
已知wip tip 305c2f0866f163b0549d47b2e4264c5d41a86385与2.0-refactor tip
ca56aa802c2acf9a22bca32d1bee3276381ed203均经merge-base确认是当前HEAD祖先。
保留既有合并、本地提交和候选，没有reset、stash或push。

### R193/R195：原失败后的真实CPU修复验证

R193为114P/8F，原日志保留。修复覆盖optimizer缺省状态及参数组预校验、
不存在的FSDP命名hook、Adam更新错误记录AD历史、native Var没有frontend copy_，
以及旧safe-globals发布遮蔽实际受限解码器注册表。
复用canonical owner：公开named_parameters配合法FSDP刷新、所有optimizer先计划再加载、
Adam公式及写回no_grad而closure enable_grad、native Module及DCP使用共享存储copy owner；
legacy私有checkpoint名字改为同对象canonical别名，移除重复旧实现。
扩大复验先160P/2F，继续修DCP加载no_grad及补齐FULL host替身refresh记录；
combined-r2最终163P/31个CUDA skip、13.78秒、rc0。skip均属于仅CPU窗口的加速设备限制。
新增回归要求原生与前端参数的对象、存储指针、detach别名、旧快照、叶身份及后续全部梯度正确，
没有豁免真正非叶的requires_grad错误，也未改数值预算。

独立PyTorch2.7.1+cpu执行相同Adam/AdamW新增12个CPU fixture case全部通过，未加载Jittor/torch_npu。
共同正向CPU offload恢复下一步，完整loss、全部参数/输入梯度、model及Adam状态24项/62元素，
沿用atol2e-6、rtol2e-5，0超预算、最大绝对差7.450580596923828e-9。
Native整个StateDictOptions文件原23tests/4errors/1skip保留；
固定2.7.1在部分partial/strict负向场景有KeyError及畸形参数组接受差异，
未放宽Jittor更严格的原断言，也未将Native整个文件记为通过。

证据位于LAB _state/ascend-compat/device-api/checkpoint-integration-r193-cpu/及
device-api/checkpoint-integration-r195-cpu/。
后者combined-r2/focused.log SHA256为044337b1892d3bb9f090fc461d2ec839aca5c143467958f41e17b8e25d343959；
cpu-native-options-comparison.json保存全部24项指标，progress-summary-r204.json汇总实际cohort。
目录symlink曾在清单准备时误当普通文件读取；错误发生在worker前，保留后修正，未用不完整清单运行。

### R198/R201/R203/R204：实际Accelerate SHARDED及PEFT fresh恢复

四格串行复用已归还的显式CPU暖cache，实际固定Accelerate1.10.1与PEFT0.17.1，
复制当前完整adapter包并核验source/overlay/build recipe/worker前后一致。
每格独立运行warm、continuous、interrupt、fresh resume及比较器；
完整输出、输入梯度、全部适用参数梯度、model、optimizer和scheduler即时落盘。
默认FileSystemWriter/Reader真实读写native manifest和payload，没有切换FULL规避SHARDED。
fresh恢复后两个Adam状态均推进到step2，fallback增量0。

| Cohort | 实际场景 | strict0完整数组比较 |
| --- | --- | --- |
| R198 | 非PEFT、SHARDED、FP32、offload0 | 97项通过 |
| R201 | PEFT round2、SHARDED、FP32、offload0 | 101项通过 |
| R203 | PEFT round2、SHARDED、FP32、offload1 | 101项通过 |
| R204 | PEFT round2、SHARDED、BF16 policy、offload0 | 101项通过 |

这些是world1 CPU本地API/存储/两步恢复，true_fsdp_initialized为false，
不覆盖双rank分片交集、HCCL、910B3放置或真实Qwen4–10步合同。
R204当时只验证BF16 policy配置下的恢复；singleton lifecycle未启用，不能据此宣称真实BF16计算。
BF16原始位断言另含signed zero/NaN、空张量与模型buffer。
证据分别位于LAB _state/ascend-compat/real-model/下
dcp-cpu-r198-sharded-fp32-offload0/、dcp-cpu-r201-lora-sharded-fp32-offload0/、
dcp-cpu-r203-lora-sharded-fp32-offload1/及dcp-cpu-r204-lora-sharded-bf16-offload0/，
以各自driver.json、recovery-comparison.json和完整NPZ为准。

### R200/R202：FULL恢复仍有BF16原始位缺陷

R200在runtime导入前因TMPDIR过长被fixture拒绝，原尝试保留。
R202使用短LAB临时路径，warm/continuous/interrupt均rc0；
fresh resume在BF16 buffer原始位断言失败：0x7FC1变0x7FC0，其余三项一致。
离线读取原pytorch_model_fsdp.bin确认float32 payload高16位仍含0x7FC1；
保存未丢该payload，数值cast回BF16会规范化NaN。
独立Native2.7.1的float32→BF16 cast复现此变化，
直接BF16 torch.save/load则保持dense/view/empty/scalar全部原始位。
继续修复canonical原始位编解码，不改fixture或放宽逐位断言。
证据在LAB real-model/dcp-cpu-r202-full-fp32-offload0/，
独立审计在real-model/r183-native-audit-20261024/r202-bf16-native-audit.json。
固定Accelerate在FULL world1强制offload False，不把两个请求值计为两种多rank实现。

### R192/R194/R197：Native nested通过，Jittor预热失败保留

R192两个Native rank在新增observer处失败：公开Shard字符串为S(0)，旧包含Shard的字符串判断错误；
尚未到原R71 HCCL点，原51个输出文件字节不变。
R194改用公开placement.is_shard并在断言前fsync元数据，数学和communicator调用不变。
独立Native实际四child加root，FP32/FP32、BF16/FP32、BF16/BF16三格的完整
forward/loss、六组参数梯度、输入梯度及SGD更新rc0，66.919805秒。
PID2009609仅物理5，PID2009610仅物理6；13个采样、gap/foreign/wrong均0，退出后空闲。
两rank参数梯度和更新一致，36项SGD公式检查最大差0。
这是Native参考通过，尚无对应Jittor数值通过结论。
LAB distributed/nested-hccl-preflight-r194-dual56/native-qualification.json
SHA256为9fb33a5df265b4a9b22ce0b75b6b2989ffc4ecc88a6356a3d452fb8f38fb72d5。

R197同一候选源码开始rank0→rank1串行预热，首个单卡阶段39.773890秒后rc1：
nested Sequential的BF16路径在mapped_matmul观察到BF16输入与FP32权重。
world1 HCCL已初始化，正式双卡未启动；gap/foreign/wrong均0，卡5已归还。
LAB distributed/nested-fsdp2-r197-dual56/保留完整stdout和failure JSON；
不能用手动cast模型或改变dtype绕过FSDP mixed policy路径。
绑定Native参考及数学AST只解释复用条件，不替代Jittor执行。

### R205/R206：FULL原始位修复及真实singleton SHARDED恢复

统一BF16编解码已提交：新存储使用uint16原始位，旧float32载荷按高16位解码，
Torch archive与safetensors复用同一owner，避免数值cast规范化NaN。
真实CPU定向13P；新增Torch archive夹具先暴露persistent-id元组顺序和缺少version的问题，
修正为实际格式后，独立PyTorch2.7.1+cpu完整读取dense、非连续view、scalar及empty四种夹具，
每种BF16原始字节完全一致。原失败日志和夹具保留，未修改原数据与逐位断言。

R205非PEFT/FULL/FP32与R206 PEFT round2/SHARDED/BF16 policy，均完成独立warm、
continuous、interrupt、fresh resume及完整比较，各阶段rc0，源码及完整adapter包前后不变。
两格分别97项和101项严格一致，两个Adam状态均推进到step2，fallback增量0。
两格的true_fsdp_initialized实际为true，区别于R198—R204未进入singleton lifecycle的历史结果。
R206记录了BF16 policy及真实生命周期，未单独观测每个子Linear的计算dtype，
不把master参数FP32及输出FP32当作BF16计算证明。

证据在LAB _state/ascend-compat/real-model/dcp-cpu-r205-full-fp32-offload0/、
dcp-cpu-r206-lora-sharded-bf16-offload0/的driver.json、recovery-comparison.json及完整NPZ；
BF16夹具在device-api/checkpoint-integration-r195-cpu/bf16-codec-r3/。
这些CPU singleton结果不替代HCCL2分片交集、910B3执行或真实Qwen多步合同。

### R209：optimizer提前构造与状态迁移仍有两个实际回归

整合FSDP旧参数身份/历史slot迁移及ACL affine reindex后，真实CPU专项247 collected、
216 executed：214P/2F/31 CUDA skip，29.23秒；原singleton FULL checkpoint及新增mixed policy均通过。
两个失败是AdamW初始化没有建立参数身份缓存，以及flat模式旧Parameter在断言时仍被保留。
真实CPU tracing确认后者由参数收集函数递归visit闭包循环引用造成；
原始失败、完整stdout和源码清单保留，不用gc.collect或放宽原生命周期断言取通过。
全tracked 3347路径，UTF8/NUL及symlink规则完整覆盖，
source-before.json SHA256为0f404da928d962ee9b7c4f87836d78ca4b090be46c92e0259391599c7e7cfe2c。
证据在LAB _state/ascend-compat/device-api/checkpoint-integration-r209-cpu/，
整合AdamW初始化与闭包修复后，R209 attempt-002真实CPU218P/31 CUDA skip，
249 collected/218 executed、14.13秒，原两项失败与新增回归均通过；原方法数值/生命周期断言未改。
完整源码清单SHA256为0979b8f9ed863aff713642ef2de3cd7097808188191ba7931a4cb095ab645f2a，前后不变。

### R196：NPU原始位save越过reindex，fresh load仍失败

attempt-001缺少显式JT_BACKEND=acl，在导入前发现cache recipe偏离暖缓存；
只核验并停止本任务已记录进程组，未执行数组，保留停止receipt及冷缓存。
attempt-002使用正确cfgbed66048，原始UInt16 reindex在backend禁止回落时失败，
fused reindex/void无原生分派；void是算子variant，不是UInt16 dtype映射错误。

复用现有native_indexing字节复制owner修复有界非负affine reindex forward，
不经过Cast，原fixture与严格逐位预算不变。候选88项host通过、真实SDK两翻译单元语法通过，
整合后attempt-003首次实际增量JIT串行，物理卡5的save rc0：
14 leaves实际ACL/index0/npu:0，含BF16 signed zero、NaN payload、整数、empty及真实DTensor metadata；
fallback 0→0，15个严格parent采样，gap/foreign/wrong均0，退出后健康空闲。
执行记录SHA256为2ca630d4690eddd12d456f6c6a5ccae4bb075eb69fc595d0cbd6b737cdbca5f2。

随后fresh load rc1：初始化uint16[5]目标触发CANN aclnnExpand不支持DT_UINT16。
原stdout、完整保存载荷和失败JSON保留；6个采样gap/foreign/wrong均0，退出后健康空闲。
执行记录SHA256为560b10a59db4060be4fce737f13f39e7e0ba7303b63d7fc9c2230dd73549d930。
这不是完整保存恢复通过，也不是双rank训练结论。
storage dtype字节广播复用相同copy planner，146项host及真实SDK语法通过；
常用浮点/BF16保留原Expand路径，UInt16/UInt32/Int16/Complex64采用原始字节复制。
整合后的attempt-004在同一0979b8f9完整源码上save与fresh load均rc0，
14 leaves完整原始字节strict0，两个阶段fallback增量0；15/7个严格parent采样，
gap/foreign/wrong均0，退出后健康空闲，source/code/overlay前后不变。
比较JSON SHA256为a59efb4d5969091f7a74f834408c5212c7be53e751200b873e48285bab9b9294；
save/load执行记录SHA256分别为9cf22aa4a16f122811501863ea58ba77947480e6eecaaecd54fa58bd7dab46b1、
9d95d190dc0db176948071a79d6b85f9fed376b2dd8d1824d7efab773ea1a738。
没有改变fixture dtype、数值cast或跳过广播；世界1读写不授予双rank训练或性能结论。

证据在LAB _state/ascend-compat/checkpoint/dcp-native-acl-r196/cohorts/attempt-001/、
attempt-002/、attempt-003/、attempt-004/；host在checkpoint/acl-reindex-r198-host/及acl-broadcast-storage-host/。
R209 progress-summary.json绑定了两个NPU执行记录和原始CPU失败；首次SDK语法失败同样保留。

## 按原计划截至R209仍需完成

optimizer缓存/闭包CPU回归及ACL字节结构算子NPU roundtrip已在对应冻结源通过。
接下来完成nested三dtype最终主树对拍、默认SHARDED双卡保存恢复及R186真实Qwen多步完整轨迹；
Native/Jittor各自continuous/prefix/fresh恢复strict0、全micro输出/梯度/更新、严格物理映射均须通过。
R186/R208准备及R196单rank存储通过不代表真实Qwen或双卡模型运行通过。
真实激活重计算、训练确定性、random sampler/显式generator/多worker恢复、
eval/predict/best reload与PEFT safetensors保存加载合并继续待验。
至少10对性能和实际live/reserved显存仍须对应正确性准入，当前没有合格NPU性能。
最终提交门禁须在整合源码上重跑，R190不能授予后续源码通过。
本次续跑HTTP/1.1 fetch仍45秒超时，已知wip及2.0-refactor引用仍为本地祖先；
新remote-sync receipt未授予远端最新结论。


### R212：本次整合提交门禁入口

在源码停止修改后，按仓库要求执行CPU native/Torch两种core、Torch模式完整tests/structure、
布局及文档链接检查；所有原始stdout、逐项退出码与完整tracked清单保存于
LAB _state/ascend-compat/device-api/checkpoint-closeout-r212/。
该目录gate-results.json记录本次实际执行结果，source-check.json核验门禁期间源码未变化。
R190历史门禁不替代本次执行。


## R213及FSDP候选整合验收（2026-10-24）

本节区分独立Native oracle、Jittor对应运行及待验候选，不将b41960f35上的结论外推到新整合源码。
两目标远端在本机使用HTTP/1.1 ls-remote重新核实：wip/cgq-day23-fix-20261021仍为
305c2f0866f163b0549d47b2e4264c5d41a86385，2.0-refactor仍为
ca56aa802c2acf9a22bca32d1bee3276381ed203，两者与server tracking一致且均为本分支祖先，无需重复合并。
server本轮fetch仍20秒超时，不能声称fetch成功；fresh远端tip证据来自本机实际成功查询。
LAB handoff/remote-sync-client-verified-1792846918166974556.json记录方法、输出与祖先核验，
SHA256为08bb78b794ba477c5ae8c0c1537a493c99c6b8028dc91b05d4bc988d38bee184，旧receipt保留。

### R213：真实Qwen Native连续、中断与新进程恢复通过

n001连续训练完成4updates但监控gap3，仍保留FAIL，不复用作新门禁。
n002重新从continuous开始，独立oracle-npu-cann、同一冻结worker、共享原始权重/adapter/fixture，
16条训练和4条评估fixture、每卡batch2、累积2、112个LoRA参数、split2/total4。
monitor使用CPU176–179，launcher/workers使用184–191；每阶段核验真实mask与PID/starttime，
每次assigned物理5/6 Health OK，查询上限30秒、poll5秒，最终仍要求零gap。

| 阶段 | 两rank各自完整NPZ | 实际耗时（秒，仅诊断） | 结果 |
| --- | --- | --- | --- |
| continuous | 30 | 215.485266 | rc0、qualified、19 samples、gap/foreign/wrong均0 |
| interrupt | 16 | 224.017636 | rc0、qualified、21 samples、gap/foreign/wrong均0 |
| fresh resume | 16 | 208.871526 | rc0、qualified、18 samples、gap/foreign/wrong均0 |

124份完整轨迹、71项合同和57,078项逐项比较全部通过；所有带数值预算的指标atol=rtol=0，
所有可计算max_abs均为0。检查完整micro输出/梯度、裁剪、参数更新、Adam状态、scheduler、
CPU/NPU/Python/NumPy RNG、数据游标和fresh进程；loaded2边界沿原协议暂不比较RNG，
首个恢复iterator及以后重新包含RNG。全部112个trainable参数及frozen不变检查通过。
每rank实际参数npu:0且只枚举一张卡，父端真实物理放置分别5/6；Native无shim。
deterministic True在torch_npu导入及显式tensor前设置。所有自有rank身份均已终端。

源码为b41960f3586cbc1737e2f56a2e566baf42f31f02，完整3347 tracked源清单SHA256
4b4781f7bee966ef05b6e446cf8f48d49294a1a0c5d388e137d844ab0ef8732e。
实际输出为LAB r213-n002/qwen3-06b-fsdp2-trajectory-r186-dual56/，
工具索引位于real-model/qwen3-06b-native-trajectory-r213/cohorts/n002。
comparison-native-r213.json SHA256为327834e8cdbcb90dfe1471b2d0bb78855226aedf252578d591d716f48ce850c9，
summary-native-r213.json为8f78f3ec6a4ac9375d4b9878bf7f4f22fa3de3199475740b1a86e051aa4d075b，
parent-audit-n002.json为b706efc091b7d1ceaf31efbfda051687eccc024532b7cc0c8152eb8f9542db7a。
本节只授予NATIVE_ORACLE_PASS；Jittor、全cross-runtime、eval/predict/generation和性能均未据此通过。

### 无optimizer反向、旧alias与空闲FSDP状态修复

b419真实CPU六进程最小复现确认：ordinary有/无optimizer四参数梯度均正确；
singleton FSDP没有optimizer时used梯度缺失，有optimizer时正确；旧Parameter alias读取None，
而current shard已有正确梯度。完整输出、输入梯度及有optimizer参数梯度与普通模型strict0。
修复显式登记当前shard到既有弱holder registry，通过实际lifecycle/module/state/entry校验选择
本次forward full参数，复用一次原生AD及原reducer；不扫描live graph或伪造optimizer。
旧alias getter/setter跟随当前entry，idle state不额外unshard或修改已有grad/optimizer flags。
frozen full gradient保持None，flat原owner仍自然补零；不放宽原nonflat scatter计数。
私有seam显式登记，旧public helper真实签名和原public golden保持。
DTensor的checkpoint_shape/offset回到既有kwargs通道，metadata语义及clone/detach/roundtrip保留。

候选attempt001收集失败、attempt002的114P/4F/31skip以及完整基线对照均保留。
两项frozen计数和DTensor公开签名差异在MAIN独立单项及同顺序对照中也实际复现，
本轮已修复；不把baseline失败留作最终通过的例外。
attempt003真实224 collected、193 executed，193P/31 accelerator skip、101.01秒、rc0：
原19项、8项idle、mixed31、accum14、rebinding11、preexisting31、structure13、
checkpoint host13、state_dict14、options23、placement10、native module3、full materialize3全部通过。
原19测试SHA、全部旧数值/计数断言及public golden c8153c4b...未改。
五canonical模块真实来源/SHA、全3347 MAIN和7个暖cache文件SHA/mtime均核验，
own worker为空。证据在LAB distributed/fsdp-public-grad-candidate-host/cpu-candidate-attempt-003/，
candidate-delivery.json SHA256为487be954585e1861424d4b515d5253a8a2b2d3a884258ea7ec994afbb5d93640。

主树已按序保留提交整合为ea6648aba21250d69c34611ea8635881a9b25f2e、
bfb53ce18cf3ba06fada2ef9d11cf2f14915a1e8，未push。
上述候选CPU证据不授予新主树NPU/world2；本次整合core/完整structure门禁及NPU复验另记实际结果。

### R208-v3/v4：默认SHARDED双卡训练仍未完成恢复

v3首次warm的Linear pre-hook在lazy Var尚无mem_ptr时读location得到none而失败；
不是据device标签认定已实际NPU。v4仅在原observer读取metadata前，使用既有native targeted sync
物化当次inputs和weight，原dtype/backend/device/location强断言保持。
不做cast/readback/global sync；Native分支不变，剔除此两行后整份worker文本/AST等于v3。
39项host合同通过，两个rank serial warm实际34.941602/64.360861秒，均qualified/gap0；
三个Linear全部输入/权重已记录ACL0、npu:0和device驻留，fallback为0。

v4默认SHARDED、FP32 model、offload0实际Native continuous/interrupt/resume分别
179.465847/190.466885/199.060232秒，均rc0及严格资源qualified。
Jittor continuous48.505684秒、两rank各7份NPZ，rc0且qualified。
Jittor interrupt35.670427秒、两rank各4份NPZ及PASS报告、launcher rc0；
但监控尾部记录PID2494992在归属验证前消失，将其列为foreign，因此qualified=false。
此前观察到该PID的本任务精确身份及物理5仍不能覆盖此次缺失验证；原FAIL保持，
不将消失PID一概重分类为own。gap0、wrong0、退出后健康空闲不抵消foreign失败。
fresh resume和完整Native/Jittor比较均未启动。新cohort需用明确completion handshake
保持worker活着完成最终live归属核验，再按协议退出与fresh post检查，不忽略竞态。

实际证据在LAB distributed/dcp-training-r208-dual56-v3/及-v4/，
v4 runs/fp32-model-offload0-001/保留全部stdout、trace、完整NPZ、存储和execution。
v4绑定b419完整源清单3fb2acf3832fccb65583ab8f5624cf949c227fbde70bfa0388fb3f4ccfe11045，
worker c777e03b164eb83aab1b29bfa25a349f5225d6fa0849c3879d2cea66c81f2beb。
这些存储/训练结果不授予真实Qwen、最终新主树或性能通过。

### 重计算与后续退出条件

原公共checkpoint真实CPU六场景诊断中，closure两格输出/全梯度正确但fn仅调用一次；
preserve_rng_state三格在调用前TypeError，no_grad通过。完整原失败与数组保留于
LAB recompute-rng-audit-20261024/activation-recompute-v1-real-b419/。
原生private region候选5b9fb27a0在隔离worktree完成6TU及两生成binding语法、
11项host检查和29个待运行节点；尚未构建候选core或跑Jittor，未合入主树。
另Native CPU显式Generator8格和自定义backward随机4格实测支持默认流仅在forward replay恢复、
显式Generator继续消费的合同；不修改既有Generator协议或断言。

接下来须完成最终主树nested三dtype完整NPU梯度/更新、默认SHARDED双卡fresh恢复、
R214真实Qwen Native3/Jittor3的全部轨迹交叉，及真实重计算、随机sampler/generator/多worker恢复、
日常eval/predict/best reload与PEFT safetensors保存加载合并。
对应正确性通过后才做至少10对NPU配对性能与live/reserved峰值显存；当前仍无合格NPU性能。
旧2026-10-21截止epoch不再沿用，本轮未指定新截止时间或goal预算。


## R216：FSDP修复整合后的主树门禁（2026-10-24）

本节对应主树bfb53ce18cf3ba06fada2ef9d11cf2f14915a1e8及当次两份证据文档改动；
八个FSDP相关文件在运行前已与候选交付逐字核对，未将候选CPU结论直接写成主树通过。
本次实际门禁结果如下，原始日志与命令位于LAB device-api/fsdp-closeout-r216/。

| 门禁 | 实际结果 |
| --- | --- |
| native core | 103P、45skip、1xfail，rc0 |
| Torch core | 238P、1skip，rc0 |
| 13文件FSDP/检查点专项 | 224 collected、193 executed，193P、31 accelerator skip，rc0 |
| 完整Torch模式tests/structure | 1658 collected、1658 executed，1656P、2 accelerator skip，rc0 |
| 布局、内部文档链接、git diff --check | 全部rc0 |

Native首次进入本次显式空NVCC flags对应的CPU缓存分区，实际串行构建244个core编译单元及所需算子；
其构建和测试墙钟不作为模型性能。两种core合计341P、46skip、1xfail。
专项保留原19项无optimizer/alias测试、8项idle负向与既有mixed precision、累积、rebinding、
preexisting grad、checkpoint storage/state/options/placement等完整断言，未修改数值预算或公开golden。
全部3349 tracked文件使用严格UTF8/NUL路径和symlink payload绑定，门禁期间未变化；
source-check确认tracked集合相同，父端结束后再次逐文件核验，原门禁进程组已空。

source-before.json SHA256为c8d50f751401505ec34595d54a7b664985115d030ef0f1d06ba5b06f067b3303，
gate-results.json为58a9a40311de167c8866c0c1a1120fe558f90069bfccf4d6e5ea92d177612604，
source-check.json为065507293a64977eb06906f3f9547462da1196b6d68a64200aeba7bbd42c8bdd。
本节授予上述主树CPU范围，不授予NPU、world2、真实Qwen六阶段或性能通过。
本次证据文档更新后，按文档范围单独复验；实际记录在LAB device-api/evidence-docs-closeout-r218/。

### 新cohort的完成握手与运行入口

R208-v4中断阶段退出归属失败原样保留。新的共享完成握手位于LAB
distributed/resource-completion-handshake-v1/：原worker自然返回后保持上下文存活，
父端以原严格资源函数完成最终live身份和放置核验，再放行正常解释器/runtime清理。
记录真实inner与supervisor退出码、ACK和pidfd；清理异常、非零退出或产物改写仍拒绝。
原classify、stop_owned、资格谓词逐字保留，最终资格为原条件与新完成证明同时成立。
源码与fixture/site/tool实际内容由接入driver在seal前及正常退出后重新核验，
不以identity中的manifest SHA替代磁盘实查。

CPython3.11纯stdlib串行host-attempt-006实际28项合同通过，包含正常atexit、
提前零退出、非零SystemExit、错误身份/nonce、消失和复用PID、foreign/gap/wrong placement、
产物改写、清理信号/异常/超时及post非空。原前五次失败保持。
host-result.json SHA256为649cf8b57f14ce4d3d162e28e01bd9662e22e5f65e3a7d0dd94976442eef74e8；
最终helper-source-pins.json为d21e9a2288156b313a71d4419e252c53cf7d7c0fe15014a37800e2ef468f5b29。
父端实际重核pins、各case及原monitor字节，parent-review-001.json
SHA256为1a1d1fea6c911afc99a0e3feba8d22ccc58253253b3ce06228af38e07e0de789。
这些是host协议证据，真实NPU握手仍待运行。

R217默认SHARDED续跑包保留v4 worker及比较器原字节；16项离线/host与7项最终源码负向通过，
新site完整17源码加8metadata，RECORD按实际文件重建。source-host-delivery.json
SHA256为e78182d0a08896874a9d7b9c6b82c5a3fb7eaeb89fd9a132007b33bc13ffe863。
R214真实Qwen正式六阶段接入同一helper，原worker、完整8batch预热、比较器/policy/hook不变；
原25项与新增17项host共42P，final-preparation-completion-source-only.json
SHA256为05c30bca2fb7f301e09db25a5ae6d444b411af7a16ffd58d5abe9bbd2207a249。
两者均未据此启动或通过新模型cohort；预热仍保留各自原严格资源合同。

隔离重计算候选5b9fb27a0保留原29项断言，首次真实CPU coldcore构建已启动，
记录于LAB recompute-rng-audit-20261024/activation-recompute-private-v1-runtime/
coldcore-v3-5b9fb27a0-r1/firstwarm/；尚无候选运行通过结论，未合入主树。
其CPU188–191和独立JH/cache与主树结构检查分离，仅一个候选coldcore JIT；
其它NPU/Jittor新模型任务等待该首次编译终态。


## R217实际双卡恢复失败与R219/R220 Store修复（2026-10-24）

本节新增真实执行证据；前述R208-v4退出归属失败和R217源码准备记录继续保留原时点范围。
R217的runs/fp32-model-offload0-001绑定干净主树4abc114fea79f18f35398ae5c649c623efd00dcb、
完整3349 tracked清单f5142375e2ba8a40f15dd132c8cde8c212148effb4411c2703c7c371b25ac324。
原worker/compare、数值预算、Native deterministic True协议与严格placement/fallback条件未改。

| 实际阶段 | 结果与资源验收 |
| --- | --- |
| Jittor rank0/rank1各自world1串行预热 | 两阶段rc0；只授予world1预热 |
| Native continuous、interrupt、fresh resume | 三阶段rc0、qualified |
| Jittor continuous、interrupt | 两阶段rc0、qualified，完整保存完成 |
| Jittor fresh resume | rc1、qualified=false、NOT_SEALED，未进入恢复更新 |
| 完整Native/Jittor交叉 | 未运行，整个cohort保持FAIL |

前七阶段均真实完成NORMAL_EXIT_PROVEN，monitor gap/foreign/wrong全部0，
worker/supervisor正常退出0且postempty通过；说明新的完成握手已实际通过world1及HCCL2，
不再仅有host模拟证据。失败阶段postempty通过不抵消功能失败。
两rank原始failure.json均为DCP load validation failed，原因是
multi-rank checkpoint requires a native process-group Store。
前端_io_context已取得同一有效Store，save显式传递，但native.load无store形参且重新查空default Store。
此错误发生在READY之前；监控的“未seal就退出”为后续现象，不改变实际根因。
70份实际NPZ、所有日志/完整已有梯度和checkpoint均保留，失败rank各留initial-0000，
没有凭缺失的恢复更新宣布通过。parent-terminal-audit-001.json SHA256为
4e44c03497955da4362a7b5bcc8068644fda883e526df54813f2d63c640aefd8；
全部3349源运行期间不变，结束后原父进程及精确rank身份均已消失。

R219在隔离worktree以原有host MemoryStore/runtime doubles加真实manifest/payload I/O，
先在旧代码复现默认group和非连续global ranks subgroup两项失败，再完成候选21P/0skip。
首轮默认tmp位置不合本任务约定的记录保留，随后在LAB私有HOME/tmp/basetemp重现旧2F并验证候选。
中文候选提交d47812b83f2a73c65026605bb430ea86847a5c4c只修改两个生产模块和原host测试文件：
native.load新增keyword-only store=None并交给既有_context，compat.load传入context[3]。
_Context、Store/communicator生命周期、模板与payload校验、全rank一致拒绝及复制逻辑未改；
不重新初始化通信、不修改默认Store、不伪造world或放宽数值条件。
delivery.json SHA256为f784da849713b9346442d066e81afa4969f763b527cc9bf7a85468e7708707ba。

主树R220 CPU实际范围为4abc加上述三文件字节，完整3349清单SHA256
ecff83350a13d07ec933cedb3ef61a56e1686fce544f8fd17c3839698de53c96。
门禁期间源码和tracked集合不变，原始命令、junit与日志位于LAB
checkpoint/dcp-load-store-main-r220/，gate-results.json SHA256为
636869f93eb426e42d8430154e1ceb9059ab0e1c46c591b5ced29164d382dff0。

| R220实际CPU门禁 | 结果 |
| --- | --- |
| Native core与Torch core | 103P/45skip/1xfail及238P/1skip，合计341P，rc0 |
| 13文件compat FSDP/checkpoint专项 | 232 collected，201P、31 accelerator skip，rc0 |
| 原生tests/serialization/test_sharded_checkpoint.py整文件 | 30 executed，24P/6F，rc1 |
| 布局、内部链接、diff检查 | rc0 |

原生六项失败均在完成原training_step后保存时由native_forward_pending拒绝，
尚未执行本次改变的load路径。另以git-show提取4abc原checkpoint.py，在同一真实native runtime
加载执行旧模块，其他native源码逐字等于4abc，原六项再次全部出现相同错误；
该证据是精确模块基线复现，未伪称另一整棵冻结工作树通过或失败。
原完整30项和旧基线六项stdout/junit均保持。已发现native FSDP的_prepare_backward/_finish_backward
现只有定义而无调用路径，后续按其canonical owner定位；不删除pending状态拒绝来通过保存。
因此R220扩大回归整体仍含6F；Store CPU证据不授予双卡修复或真实Qwen恢复通过，
需要新cohort实卡验证，原R217失败不得改判。
本次提交的完整结构、布局与链接门禁单列在R220 structure-final-001/；
其冻结清单包含本节及ASC条目更新，CPU运行到结构门禁之间仅两份证据文档改变，
另外3347个tracked文件逐字核验保持一致。

本次恢复任务前server HTTP/1.1 fetch实际rc0，fresh client ls-remote核实wip tip
305c2f0866f163b0549d47b2e4264c5d41a86385和2.0-refactor tip
ca56aa802c2acf9a22bca32d1bee3276381ed203均与tracking相同且已为HEAD祖先。
无需再次造合并；未reset、stash或push。完整receipt在R220 remote-sync.json，
此前fetch超时记录保留，不回写成成功。

## 私有真实重计算CPU原型：失败修正与29项通过（2026-10-24）

隔离候选5b9fb27a0完成246TU首次coldcore构建后，原quick第一项重复target测试真实失败：
第二个目标梯度为None。2e574d63b仅在private checkpoint_region边界去重targets后，
调用既有native VJP并按原索引分别包装结果；普通grad_impl与清理语义、原29断言均未改。
新cohort增量构建后quick5P，原失败源码归档及原core独立副本保留。

随后完整focused29实际28P/1F，唯一失败在weakref.ref(jt.Var)准备阶段，
尚未调用_flatten，不能据此推断展平实现泄漏。3c4742967d只修正该测试的真实观测：
使用纯读取native holder registry，禁自动GC，加入真实Var自循环阳性对照，
并在spec存活、尚未执行GC时要求删除x/flat后holder精确回基线，零容差；
finally恢复原GC开关。其余28方法AST、生产模块、原core字节不变。
source/host变异检查能拒绝visit自循环和spec持有Tensor，但不据此替代runtime。

新lifecycle-focused-v9-3c4742967d-r1真实CPU结果为29 collected/executed、
29P/0F/0skip、rc0。原闭包/全部梯度、duplicate/nonleaf/nested/alias、
allocator2激活释放、默认与显式RNG及custom backward随机断言均实际执行。
完整3351源清单前后clean且unchanged，core SHA256保持
2bee25c71d3636e7b390439cd22ceb519203b277528d57e9988e04667650e444，
canonical15项环境检查全true；原worker/PGID结束为空。
outcome.json SHA256为7728353a5098ee51c89a336f97fb035c09258c35049f26f77e39067707631544；
artifact-manifest.json为65fcea8a960c76c3aea600a22cf5c27db56e9401a2196ce8df0fc7c8a08c4517。
这些通过由原全数组/精确计数断言支持，PASS未另行输出全部数组或holder序列。
上述产物位于LAB recompute-rng-audit-20261024/activation-recompute-private-v1-runtime/，
原weakref失败及完整输出不变。

此结果仅授予private原生CPU原型29项合同；尚未整合主树，
公开Torch checkpoint/wrapper、noexcept有效误用负例、既有AD/RNG回归、
真实Module/FSDP组合、NPU重计算与实际显存收益仍待验收。


## R217 cohort002：Store修复后的实际HCCL2完整恢复（2026-10-24）

本节对应主树中文提交56fc6faa7bd43444a962620ca2e287d425b45c8a，
完整3349源manifest SHA256为40a0e815b89963d8fc4967283aa22a6dc8481e3fd238899a31f965e1383397f5。
运行前重新检查健康空闲两卡及磁盘；原001失败和全部工具/数组原字节保留。
新runs/fp32-model-offload0-002完整执行两rank各world1串行warm、Native三阶段、
Jittor三阶段，共8阶段全部rc0/qualified，gap/foreign/wrong全部0，
各阶段NORMAL_EXIT_PROVEN、worker/supervisor真实退出0、postempty通过。
最后Jittor fresh resume实际完成恢复更新，原Store缺失没有再发生。

完整compare原38检查全部通过，含6项frozen/trainable合同与32项完整数组比较，
共694指标（454数值及240原始位/非数值字节精确项）。
Native与Jittor各自continuous-prefix/fresh恢复预算atol=rtol=0；
数值exact最大差0，raw BF16特殊位和scheduler非数值状态按原字节合同通过。
跨runtime最大绝对差：forward1.4901161193847656e-08，
gradient9.313225746154785e-10，update3.725290298461914e-09。
原R208比较器保留既有forward/gradient预算，未事后修改；
父端额外核验所有数值forward/gradient的max_abs均不超过计划FP32 atol1e-5，
无需使用rtol余量，不能把原比较器的较宽字段隐去或描述为改过FP32预算。

78份实际NPZ完整保存，包含完整输出/输入梯度/参数梯度、参数、optimizer、
scheduler、checkpoint存储和原始BF16特殊位；不以loss切片代替。
comparison.json SHA256为733dc61ffeabdec3bb40588648679a0b0aa61ce23dd7c88bccfa6c62c82b3d9d；
summary.json为b8d3920ffc17c7724f79456a19b94c950a956f5060ebb5723068d08232676f38。
父端parent-terminal-audit-004.json为
0cf861f7537b30098265118289c4f8d86c867d99339285defe114f7e6477f27f，
逐份绑定78份NPZ、8阶段exit/资源、完整源不变及原父/各精确rank身份均已消失。
三次父端诊断打印分别误按list读取named checks、未区分metadata check及max_abs=None；
原observer错误另存，不更改正式summary/comparison，亦不作为模型失败或补造数组。
全部原始证据位于LAB distributed/dcp-training-r217-dual56/runs/fp32-model-offload0-002/。

资格限定：tiny FP32 PEFT Linear、HCCL2、默认SHARDED model/offload0；
不授予真实Qwen、nested三dtype、随机dropout或NPU性能。
后续R222/R224改变了native Python文件，需要为最终主树重新绑定新cohort，
不把本节3349源通过直接改写成后续3350源硬件通过。

### 当前确定性接口的实际范围

本次14份deterministic-before记录全部requested=True、Python实际True；
Jittor backend实际True，final观察同样True。
Native在torch_npu导入前设置True；Jittor在import后、显式tensor前设置True，
不声称其import内部初始化已受该设置控制。
当前compat core.py getter直接调用backend_get，bool setter同步已有工作后调用backend_set；
ACL backend真实使用aclrtGet/SetSysParamOpt(ACL_OPT_DETERMINISTIC)，错误和未知值抛出。
因此旧交接的setter未实现/getter False或UNIMPLEMENTED_FALSE不能用于当前行为判断。

core.py仍把整项use_deterministic_algorithms静态fidelity登记为UNIMPLEMENTED，
同处detail又说明真实runtime setter及warn_only拒绝；该登记只记录属性，不替换实现。
warn_only=True仍明确不支持，完整PyTorch确定性语义及广泛随机恢复未验收。
本轮受控fixture各自严格恢复通过不能外推所有算子、随机训练或性能场景。

## R222—R224：修复native FSDP optimizer生命周期及公共延迟导出（2026-10-24—25）

扩大原生回归发现R220的六项pending错误后，先保留同runtime旧4abc模块六项实际复现，
再核对历史ae1a9bf1f确曾存在canonical接线；不凭路径git-log未命中推断从未接线，
也未在未核实后续commit时指认具体丢接提交。
独立中文候选23e0d15648d15f3fe9155037d24895da766bcc50恢复已有
_prepare_backward→full targets sync→legacy grad→_finish_backward以及pre_step的_assert_step_ready。
FSDP ID排除两条legacy collective、extra input拒绝保持；
整个DDP readiness fast分支AST不变，普通optimizer math无副本。
新增11项source/host合同旧5F/6P、候选11P，只作为接线前置。

R223首次整合实际native checkpoint+FSDP为47P/1F：
原30项checkpoint全通过，之前六项失败已消失；nn FSDP17P，
唯一失败是fresh child从jittor.nn.parallel导入FullyShardedDataParallel时ImportError。
中文候选e4f789052cb853b3d75140d500346d0d6153dbb4只用PEP562延迟返回canonical同一类，
__all__列名，未知属性拒绝，owner导入错误原样传播；
不eager加载FSDP、不引入Torch/compat、不复制实现。
原nn/test_native_fsdp.py整文件不改；新增7项host旧4P/3F，合并候选18P。

R224在56fc加上述三文件的实际3350源上重新验证：

| 实际CPU门禁 | 结果 |
| --- | --- |
| 原生serialization checkpoint及nn FSDP完整两文件 | 48 collected/executed、48P、0skip，rc0 |
| Native/Torch core | 103P/45skip/1xfail与238P/1skip，合计341P，rc0 |
| compat FSDP/checkpoint专项13文件 | 232 collected、201P/31 accelerator skip，rc0 |
| 布局、内部链接、diff | 全部rc0 |

48项覆盖SGD/AdamW、累积、完整梯度/更新及恢复、pending/partial/stale拒绝、
失败forward释放、布局损坏拒绝和真实fresh child的lazy同对象/native来源/无Torch无compat合同。
这些是CPU对应范围；native多rank实现原CUDA/NCCL限制未改变，不宣称native多rankNPU。
source-before.json SHA256为a24eae82583a4f48b7e7d77fa52913f50dec51da92ee097770cb3e7fc8aaf086，
gate-results.json为dae0d8f1cb040e1ca21f4a14097843597fda0f2d16ca04182233bbffb9671256，
native-checkpoint-junit.xml为800470cd5cef76847dc76770b7236bb92fafb62d2d63511fed55a7e1161eccf9。
全部3350源和tracked集合运行期间不变，source-check rc正常；R223及其唯一失败原样保留。
证据在LAB checkpoint/native-fsdp-optimizer-main-r223/和-r224/，
提交的完整结构/布局/链接门禁单列R224 structure-final-001/，
从CPU冻结清单至该结构门禁之间仅ASC看板与本报告两文件更新，其余3348文件逐字相同。

R223的server fetch再次25秒超时，未开始整合/计算的失败attempt保留；
fresh client ls-remote核实两目标tip仍分别305c2f086和ca56aa802，
与tracking相同且已为HEAD祖先，故无新增远端合并。
不把此前R220成功fetch或本次client核验写成R223 server fetch成功，
不reset/stash/push，也未干预他人进程。

### 重计算生命周期的新增失败与隔离候选

private 3c的CPU29P未覆盖非法active scope生命周期。有效v11子进程实际确认：
region非LIFO最后引用销毁与RNG跨线程最后引用销毁均自然SIGABRT(-6)，
原CHECK抛std::runtime_error穿过noexcept析构触发terminate。
监督rc0只表示证据收集完成，功能安全保持FAIL。
results.json SHA256为3597f407b8db44210d6d4346f8bb41f152e93b5c72d7dad871a5574d90a799aa，
位于LAB recompute-rng-audit-20261024/activation-recompute-private-v1-runtime/
lifecycle-negative-v11-3c4742967d-r1/。
v10的/proc解析监控失败保留，不用其stderr补称有效SIGABRT验证。

独立候选4d81d3720a将可转移metadata与active State分离，active资源由owner线程C++词法
callback lease/RAII持有；handle析构不再leave或跨线程解锁，显式finish/exit先检查owner/LIFO。
复用GraphLockSuspend使Python callback可等待另一线程销毁metadata；
RNG callback返回前恢复outer并释放entry，之后才构建原VJP。
e019ac41ec进一步让GenericAlloc返回nullptr时直接保留原PyErr返回，
避免进入会LOGf的PyObjHolder(nullptr)。
仅private worktree七文件，原29数学/全数组/释放/RNG断言均保持，新增9节点共38；
3头生成绑定、host控制流和source变异通过仍只作source证据。
原core2bee不作为新候选二进制；新增量构建和38项实际运行须单独记录，
本节不授予该候选runtime/NPU通过，不合入主树。


## R225—R230：真实Qwen失败定位、nested复验与共享参数修复（2026-10-25）

本节区分已封存的73083ccb0、3350文件NPU cohort与R230追加测试后的3351候选源。
没有沿用旧执行窗口。以下LAB证据均未版本化，位于 $JITTOR_LAB_ROOT；
完整数组、日志、缓存与原失败不放入主仓库。

### q005：Native全恢复通过，Jittor完整四更新后保留失败

R214/q002共享helper签名缺参、R225/q003实际outer monitor缺参原失败保留。
R226修复实际调用后，q004 Native三阶段与原比较器全部71检查、57078指标strict0。
q004 JT首micro前因错误要求实现类名必须为SequentialSampler而失败。
R227保持公开factory与采样数学，改查公开类精确身份、Python整数全集、长度及完整0至15顺序，
另存真实实现类名；没有给实现类改名或放宽数值比较。

新q005源清单SHA256为e1356d8856b3d6f2652c666abbc1f1b70d99b15b3c2c465c49f058e3d941bf87。
串行两rank全8batch/112梯度预热通过。Native连续/中断/fresh恢复分别248.372、274.901、
196.460秒，monitor_samples分别21、22、16，均qualified/rc0、gap/foreign/wrong为0。
JT双rank通过公开sampler身份与完整顺序，执行8micro/4updates，fallback增量均0；
原冻结完整清单断言失败，159.215秒、18份样本、gap/foreign/wrong均0。
正常seal被正确拒绝，外层CompletionError不是首个模型错误。
JT interrupt/resume未运行，整轮不能验收。

两rank frozen-before均310项、after均311项；原310项完整shape/dtype/hash全部未变，
唯一新增lm_head.weight与原embed_tokens.weight完整值hash相同。
这指向共享注册点去重身份，不写成已证明冻结值被优化器改变；
原JSON未记录两个属性的Python id，真实身份错误由R230 CPU最小复现另证。

父端原比较器未改，读取全部可用数组：Native71/71P、57078指标strict0、最大差0。
JT连续训练已有58份可比capture，原检查均FAIL。初始数值、首micro forward/loss与梯度数值exact，
但公开grad shape与完整materialized shape不一致；optimizer group额外448项来自m、values、_torch_steps、_torch_adam_params四个内部字段，Adam slots另有local形状及step存储类型差异。
首update第二micro起有真实梯度差，不把metadata问题当成全部解释。
现有forward32指标8失败、gradient4480指标1376失败、update4704指标2288失败；
最大绝对差依次3.125、0.17578125、3.2904209e-5。
预算仍forward0.005/0.03、gradient0.01/0.05、update2e-6/2e-5，各自恢复0/0。

首update112梯度×两rank共1146880元素的只读代数诊断显示：
JT第二micro与Native第二micro减首micro两rank平均值的最大残差4.656613e-10；
直接Native差0.17578125。这是定位累计梯度丢失的证据，尚非修复验收。
原完整失败、每micro数组、参数/Adam/调度器状态均保留。

入口为LAB r227-q005/qwen3-06b-fsdp2-trajectory-r186-dual56/；
_state/ascend-compat/real-model/r227-q005-original-diagnostic-20261025-r1/summary.json
SHA256为8e7b8cfd9ff672a02ee8b850f65ed5dad2110201dff5e774a7ece8b774fb1d45，
完整原比较为9b0cd2644edaf845b6806d0c8e47b11658081289bfae719cb7f05ce06bdcb0a1，
代数诊断为e51f20459450f7a662bf4f31b02ae18ca0186d0d9a5bd95a1d2e56fae1f9c94e。
父端terminal审计1fe7399353e759209dd0744191d4e6bab5c4709cc46d21a0f397337c51c3d4c2
绑定2529原文件、198份NPZ、全部源不变及24精确身份消失。
父端首次proc CLI错误与samples字段误读保留，数量以原execution.json的monitor_samples及纠错003为准；
原模型和资源结果未重写。

### R229：FP32 nested输入梯度复验通过，BF16仍不通过

父端在R228上卡前发现同名局部completion_artifacts递归解析自身，
实际调用因expected_root参数TypeError；旧19P/10P没有覆盖execute实际注册闭包。
保留原包与负例，新R229只改局部命名和入口标签。新增14项host真正调用execute至
真实completion.install，执行实际注册闭包的完整source/site/tool/shared/Native校验及原artifact owner，
成功与损坏负控均通过。

R229 fresh准入后串行两rank world1三dtype预热，再运行原HCCL2三case。
清单SHA256为8a289885060400e4eb67eae4a177e17d91312ea7bfce3bb244ffb143c9a23c23。
三阶段52.902、75.052、53.678秒，monitor_samples为6、9、5，均qualified/rc0；
实际物理放置正确、fallback0、gap/foreign/wrong0。
正常seal、worker/supervisor退出、postempty与完整源不变均通过。
Native仍是独立冻结R194参考，不称当前MAIN Native重跑。

原比较器共324项、90数值数组；固定False/highest/HF32off/autocastoff、
atol2e-6/rtol2e-5未改。FP32 compute/FP32 input为30/30P，
两rank完整输入梯度最大差0，其余最大差7.450581e-9。
BF16 compute搭配FP32或BF16 input各15/30P，各15项超预算；
各case最大差0.00048828125，输入梯度差0.0000152587890625。
72项SGD公式检查全通过，整体比较仍FAIL。
全部六份preassert完整leaf、本地参数梯度、None/布局/放置信息均保存。

LAB入口为_state/ascend-compat/distributed/nested-fsdp2-r229-dual56/attempts/attempt-001/。
父端terminal审计1f84aa47c0365794fa202ced2b8ee4d297a4d5f85da0511f460ee5fe9b1b5280
绑定416文件、10精确身份均消失、3350源不变及缓存无owner。
FP32结论只归此冻结源、拓扑和协议，不外推R230、新模型、混合精度或性能。

### R230：统一同state tied参数所有注册点的holder切换

先在未修改的73083原主树及核验CPU暖core上真实运行public fully_shard，
wrap后embedding.weight与head.weight身份断言失败，worker自然rc1；
before/after完整值、注册id、shape/dtype与traceback保留于LAB
_state/ascend-compat/distributed/fsdp2-tied-parameter-cpu-red-20261025-r1/。

随后整合四canonical owner：收集同state全部注册点，逻辑storage仍单entry；
init/unshard/reshard/optimizer/FULL与SHARDED load统一薄委托同一setter，
所有注册点接收同一holder。没有Qwen retie特判、忽略lm_head或改冻结断言。
Native自有FSDP原alias拒绝边界不变。跨已单独wrapped child-state共享还需独立
forward生命周期合同，本修复没有把它记为支持。

新增9项真实CPU全部通过：flat开关、冻结/可训练、root直接别名8格与容器负控；
验证wrap/forward/backward/SGD/手动unshard-reshard/公开FULL及SHARDED load后的身份、
单entry、optimizer去重，并比较完整output、全部适用梯度和更新。
扩大原13文件首轮200P/31skip/1F：host AST fixture未提供新canonical setter依赖。
原失败保留，fixture改取实际setter，保留原数值/padding/scope全部断言并追加
第二注册点load后同holder断言；复验201P/31原accelerator skip。
原生检查点/FSDP完整两文件48P；native/Torch core341P/46skip/1xfail，均rc0。
以上仅为CPU对应范围；新3351源NPU与完整Qwen恢复仍待，结构/布局/链接门禁另记下文。

代码与原日志在LAB _state/ascend-compat/distributed/fsdp2-tied-parameter-main-r230/。
初始source-delivery SHA256为6921ac9c29b45196a3a7c52418b3124044d0118ba8edb7734015f4f2633799ff；
最终CPU清单source-before-v2.json为9020ae6433c84ebb540faf48e82b779da189320f707f6e32650b7dd68e851fa5，
包括全部3350 tracked及明确新增测试共3351，不遗漏未暂存新文件。
fresh fetch见LAB _state/ascend-compat/handoff/resume-remote-1792941503/；
两目标tip仍305c2f086与ca56aa802且均为HEAD祖先，没有重复合并、reset、stash或push。

### 私有重计算和显存证据的范围更新

private e019/cfdb新core的原数学/全数组/RNG/释放及生命周期专项实际38P；
两析构误用在新候选真实进程中正常拒绝并rc0，旧SIGABRT(-6)保留。
原autograd七个完整文件56 collected、43P/13原CUDA skip。
最近complex64 gradfunctional整文件7P，99份完整NumPy数组逐字节验证，
父端审计b81e620072a65c4a53fdb037096b7786418957c54c156678412dc5cbb6a2f521。
剩余六A文件与B至F组未完成；私有core未合MAIN，不授予公开checkpoint/HF/FSDP重计算。
Native CPU checkpoint34协议与wrapper17协议仅提供参考，预期拒绝、no_grad已分列；
不是34/17项Jittor功能已实现。

Native NPU显存基线28公开调用、生命周期204调用实际通过；
4MiB分配、别名持有、释放、缓存保留与empty_cache/reset分开记录。
父端审计分别e0b97bb51504e547913382294cd704a51a53568ba208dd9751d34ab0db8e25b0、
b62d26e89af34227f67a5fe85642466735f7802ee87ca2126bec6549bcf28ec7，
入口为LAB _state/ascend-compat/performance/。
JT独立warm-r1在import阶段rc3；warm-r2因路径改变触发冷构建，由父端核验精确自有组后取消。
完整缓存差异、原SIGTERM退出与副本保留，均未进入数学或公开显存API，
中断缓存未获准复用。仍无JT显存接口实卡验收或合格速度/峰值显存比较。

### R230最终提交门禁

最终完整tests/structure实际1674P/2原accelerator skip，pytest470.04秒、控制器474.464秒、rc0；
layout检查298份active Markdown通过，内部链接284文件通过，diff检查通过。
命令、完整JUnit和日志位于LAB _state/ascend-compat/distributed/fsdp2-tied-parameter-main-r230/final-gates-001/。
results.json SHA256为8c71e413c186b5b30ecfa60af6e7c67ba6a75682908eff3d3d5147ffaee78464，
structure-junit.xml为4b02d5dc17264d2a825db2d9110ecae740a982b7ed20809d247d65c4db7c4aa2。
门禁期间3351份文件逐字不变，source-before.json为0c42475857f37717b0cbe968bd6a7e487fede83fb57439df1a3ac2b9add29069。
结构完成后仅ASC-01及本报告补记实际门禁结果与完整状态差异，其他3349份源/测试保持相同；
提交前再次检查最终文档layout、链接和diff，不把文档追加视为新的模型/NPU复验。


## R231—R232：公开累计梯度视图的实际红绿修复（2026-10-25—26）

本节细化此前q005的“少首micro贡献”诊断。R231在C3源上用两个独立binary NativeCPU
参考与8个真实singleton shim FSDP进程，覆盖flat、逐micro同步/显式no_sync后同步及
首micro是否完整materialize梯度。普通同步时，公开full_tensor第二micro有差异，
但完整输出及两次参数更新与Native byte-exact；显式no_sync后同步的post-sync梯度也exact。
只读v3直接读取Var底层存储，四份真实梯度与两次更新全部byte-exact，
而两个第二micro的公开full_tensor错误；public grad与state last-grad确为不同对象。
因此上节数组现象不能直接表述为AD数学漏掉首micro，更准确的已证缺陷是公开读取错选buffer。
原q005 NPU全部差异是否由此解释仍未证明。

原v2通过to_local尝试读底层仍经过相同resolver，其OTHER_OUTCOME与全部结果保留。
v3直接storage诊断SHA256为a408ddef4b58a061e85b4674285c6b32a05b6802ae23e1c4bdf5c619bcd3de64；
12 worker身份与自有组均消失、3351源不变、MAIN clean的最终审计为
15f583fb0d66a768ba660f763ee7106a26abedf48b1545ab5d7ffaba753519a2。
证据位于LAB _state/ascend-compat/distributed/fsdp2-gradient-accumulation-public-r231/。

R232先新增真实public singleton回归，原生产代码下整文件14P、4个新增case全部F：
每个失败均为第二micro完整4元素full gradient不符，最大绝对差2.34375。
raw、full/local、完整prediction/loss与独立NumPy闭式导数在断言前保存。
失败保留在LAB _state/ascend-compat/distributed/fsdp2-public-gradient-main-r232/grad-red-001/，
没有-k/-m过滤、缩规模或放宽预算。最初父端控制脚本拼接elf语法错误在runtime前退出，
原脚本与单独失败收据保留，新v2控制器通过compile后才首次启动测试；不算模型失败。

唯一生产改动在compat/fsdp2/grad_sync.py的_fill_fsdp_grads_from_grad_map：
optimizer-owned条目复用累计public slot后，entry.last_grad及state.true_fsdp_last_grads
同步指向该已发布对象，full_tensor/to_local resolver不再读最新单micro临时值。
没有新增Tensor计算、改变AD/collective/Adam公式；其余15个函数AST逐字不变，
逆替换能恢复整个原模块。新模块SHA256为
f18556c1fb8726990ef8a7f939b1442b85b67120ffa51cfcaa6a991dd6e5ca3b。
同一回归文件复验18P/0skip，同时验证同一buffer身份、独立完整导数、full/local视图及裁剪norm/值；
比较预算和四case原字节保持，原始小数组未覆盖。

| R232实际CPU范围 | 结果 |
| --- | --- |
| 原累计梯度整文件及四个真实public回归 | 18P/0skip，原14P/4F保留 |
| 同state tied完整生命周期整文件 | 9P/0skip |
| 原FSDP/optimizer/checkpoint专项13文件 | 236 collected、205P/31原accelerator skip |
| 原生checkpoint/FSDP完整两文件 | 48P/0skip |
| Native/Torch core | 341P/46skip/1xfail |

另外保持R231原训练worker SHA256
69cf820ace2446d1b369f9e216a17e01f2cced8a4f44224757fefeef898cb33b不变，
新起2个独立binary NativeCPU与8个真实shim singleton进程。
全部100份共同完整输出/loss/post-sync梯度与两次更新的shape/dtype/字节exact；
common array比较SHA256为51f8e96d5a8719f1e378d9c091937618115806c76f56580a0c3bd17910bcc32f。
显式FSDP no_sync且观察首micro的两个case仍各缺两份相对unsharded Native的首grad观察，
原差别完整列出，不把共同数学通过外推Native FSDP2全部发布合同或NPU通过。
其入口为LAB fsdp2-public-gradient-main-r232/independent-public-matrix-001/，
在完整3351源清单上执行，未借用历史PASS冒称新runtime。

本轮server fetch返回124，原日志保留；fresh本机Git HTTP1.1 ls-remote实际rc0，
两tip仍305c2f086与ca56aa802，与tracking相同且为HEAD祖先。
记录在LAB _state/ascend-compat/handoff/remote-r232-20261025-r1/；
不将client核验写成server fetch成功，不reset/stash/push。
最终结构/布局/链接门禁及新源NPU六阶段复验分别记录，尚未凭CPU结果授予硬件资格。

### R232最终提交门禁

完整tests/structure实际1674P/2原accelerator skip，pytest472.34秒、控制器476.600秒、rc0；
layout298份active Markdown与内部链接284文件均通过，完整3351源在门禁期间逐字不变。
LAB fsdp2-public-gradient-main-r232/final-gates-001/results.json SHA256为c86e48d55dee09ddd6bfa28f772b4575c7df057d53c8e5d1254f82dc124f206e，
structure-junit.xml为e5ea5deeb225fb945f41c70163a97372e1b39ffcd02782deb579116a897ccb34，
source-before.json为190ba8bfc52deae86e65663a281d7d8b8cc09448e8d1f60a94a3e12e2d5bf3f9。
门禁后仅ASC-01与本报告补记实际结果，其余3349源/测试不变；最终文档再查layout/links/diff。
本轮CPU通过不替代新主树NPU：后续使用原R227完整六阶段入口、新q006 cohort、新源码清单与fresh资源；
保留原比较器3f8d91a1898889788a0be45ad7457b0d1d589461b6763277ecaf2d3620f65bb8及全部固定预算。


## q008与R240—R247：真实六阶段完成，跨runtime仍失败（2026-10-26）

q008绑定MAIN d1934d7de97eee196133100822ee4bb1f1684ee5、3351 tracked文件，
完整源清单SHA256为30842b7c351bb498386a8827b9de9fc170bd3883ba019ef9c21b8eb7599572e3。
实际物理3/6，保留原4updates、8micro、累积2、step2中断及112个LoRA参数。
Native continuous/interrupt/resume分别281.387/206.822/253.629秒；
JT分别239.665/188.975/201.877秒。六阶段均qualified、自然rc0，
gap/foreign/wrong-placement为0；JT实际NPU且fallback0。
这些是功能运行耗时，不是交替至少10对的性能结果。

原比较器3f8d91a1898889788a0be45ad7457b0d1d589461b6763277ecaf2d3620f65bb8未改。
Native、JT各自71/71检查通过，恢复预算atol=rtol=0；
跨runtime仅6条件检查P、124捕获检查F，global仍FAIL。
8960份完整梯度比较记录全部在原预算内，并不代表bit-exact或tiny梯度同号；
10304份更新记录中5312F，最大绝对差3.290420863777399e-5；
64份forward记录中16F，最大差3.125。
原shape/storage dtype、live optimizer step和raw param_groups字段差异也全部保留，
不通过更改observer、归一化metadata或放宽预算取得通过。
完整comparison SHA256为4d5ef65bcc7539a72d5da7e8416187fdd820c70a95ef4c0d2a3bbe08966ac64d，
位于LAB r227-q008/qwen3-06b-fsdp2-trajectory-r186-dual56/，旧leaf名字不代表物理放置。
只读诊断入口为_state/ascend-compat/real-model/q008-full-original-diagnostic-20261026-r1/。
terminal审计0f10e8ca5cb50832509dbfa475aa3222984cbdea8990f077b56c2e037c3a6742
确认34精确自有身份和组均退出，4662文件/264NPZ封存、源不变，原比较失败rc1保留。

R240对92NPZ及92sidecar完整四步回放：各自postclip梯度到m/v/参数的AdamW
最大参数残差不超过8.631e-9，复现原跨轨迹差异；不支持clip被覆盖或AdamW公式错误。
step1两micro、两rank的完整输入、logits/loss和112梯度byte-exact；
step1更新后112个FP32参数的显式BF16 RNE像全部相同。
step2首micro输入和logits/loss仍exact，但111/112梯度数组已不再bit-exact，
最大差2.44140625e-4，229个相反符号、395个一侧为零坐标。
因此首次已观测偏离在第二次更新前的backward；尚不能仅凭RNE像排除微小FP32 master影响。
交付_state/ascend-compat/real-model/q008-first-adam-clipping-diagnostic-r240/，
delivery SHA256为032d78236c80cb7aab9d8f396d30a662da08834e7f41d9a5410adc4b6de63896。
R241 source-only进一步确认末端BF16 grad cast存在、无bias LoRA GEMM走C++ VJP；
更早edge/merge舍入与AMP回放仍待实测，不能把候选写成根因或修复通过。

独立Native CPU stochastic reference使用tiny BERT LoRA、dropout0.2、随机sampler，
在epoch中途step1保存，fresh恢复后完成原4updates。
76轨迹文件/30523数组atol=rtol=0全部通过；完整样本顺序、RNG及next draw保留。
入口_state/ascend-compat/checkpoint/random-training-resume-native-cpu-package-20261026-r1/，
parent-closeout SHA256为15372b9880f78dd18b0f8809ab29ae4b16131a7d93a9c0e9478e9b011ab84d1d。
此处仅Native CPU，JT随机恢复、NPU及多worker尚未取得相同验收。

R238 actual004原数学worker在CPU helper调用公开拒绝的npu.is_initialized时退出。
R243只修维护helper：已加载编译器和不可变BuildConfig均明确非ACL、
已知backend、两执行flags明确0，才以non_acl_build证明无初始化NPU；
Native保留真实query，ACL/未知/冲突在任何NPU调用前拒绝，公开npu API拒绝不变。
新增维护回归host红绿3F到8P，R242独立34分支检查P；原缺flag误放行及修正证据保留。
actual006真实导入cfg6af/core66cb后，helper通过，compiled has_acl/config.has_acl均false、
config.backend=cpu、use_acl/use_cuda0、fallback0、无torch_npu。
随后原worker在torch.use_deterministic_algorithms(False)处因native provider无callbacks退出，
模型未开始，新数学数组0、整文件24tests NOT_RUN。
handoff SHA256为7ced286f840564f753cca256e491a2f962485f8a0af40d20e65671cb4b609ac7；
两轮均正常失败退出、监护actions空、所有自有身份/组退出、lease已释放。

R247将修复落在原core runtime/backend owner：没有两callbacks时Jittor强制策略默认disabled，
允许查询false及重复设置false；true和残缺provider仍拒绝，
完整provider实际setter/getter/readback及异常传播保持。
不声明已关闭外部库的独立选项，也不声明CPU强制确定性已实现。
实际两C++函数提取后host编译，原8P/3F到11P；维护Native/provider及Torch回归已加入，
但新core编译和真实回归仍待，旧core66cb不能用于证明C++修复生效。
证据根_state/ascend-compat/distributed/backend-default-determinism-r247/及cpu-runtime-evidence-main-r243/。

本轮server fresh HTTP1 fetch实际rc0，两tip仍305c2f086与ca56aa802且为HEAD祖先，
没有重复合并、reset/stash或push。记录见LAB handoff/resume-r243-1792952681/。
本节追加及ASC更新只更正证据状态，不授予尚未执行的CPU数学、新源NPU、公开重计算或性能资格。


### R247独立新core：编译完成，生命周期阶段仍FAIL

新私有空cache实际完成244/244编译并加载CPU core，
SHA256为8725ba3d974ebc30ce6138bd1b0a52c5bc41681b5bd7fa3b2fd510efc0c5c9d7；
canonical stamp为f84ea206e5ba13c4659361afb4c9bb8abac43e6f38479b795590831e2c2cca94，
build_config为24844adcaf8691fabdd4ee767af2a93ef6e5b8f1c00b4bee3e3bacafc877fa34。
实际CPU/nonACL、use_cuda/use_acl0、fallback0、无torch_npu，旧66cb未变。
但ready时一个直属python child仍为S态，原supervisor拒绝seal并对本任务组SIGTERM，
worker rc-15，整体FAIL；此artifact尚未授予数学运行资格，不能仅按二进制已生成算通过。
原monitor1605采样/27168004字节及完整失败保留。

child从jit_utils加载附近持续存在；最后compiler/linker已结束后才ready，
所以不是仍在编译或zombie。其形态与tracer预fork服务一致，但原采样缺FD/maps/wchan，
没有把推测补写成原阶段已确认角色。
终态982精确身份全部退出、组及cache owner空、源码3352和site/tools末检P。
入口_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/new-core-prepare-001/，
handoff SHA256为6e6aa638b68cce9116ff8a0ac92da0bf0929dc0cf404f12402fc29dad1d8b963。
下一步仅受控诊断该helper；原失败、旧/新core及stamp不改，不通过放宽未知child守卫放行。

### R248独立Native CPU：反向edge舍入参考

固定Torch2.7.1+cpu/e2d141dbde55c2a4370fac5165b0561b6af4798b，
BF16/FP16的custom Function、普通mixed乘法、每支独立cast和共享一次cast，
原16case/94检查/662完整数组全部通过。
custom backward原始返回FP32且内部没有dtype cast；Native在各edge转换后再合并。
BF16首坐标原1.00390625与-1分别成为1与-1，最终0；
仅末端cast预测为0.00390625。FP16对应1.00048828125与-1，
最终0而末端cast预测0.00048828125。共享一次cast再分支的对照则符合先合并再cast。
这些只证明该版本Native CPU的实际graph合同，不证明JT/NPU或q008根因。

原full observer在Node.register_hook捕获校验后edge；JT目前没有该公开Node hook。
保留原662协议，另建无Node辅助observer的独立数学诊断，
实际Native16case/62数学检查/500数组通过，
全部500共同数组的dtype/shape/data/NPY bytes与原捕获exact，
证明去该辅助观察对Native数值中性，未改数学/RNE/leaf或Tensor hook断言。
不以新500分母冒称原662协议或Node hook支持通过。
两次均正常worker/controller rc0、握手完整、全部身份/组/descendant退出，
Native site22019与tools24完整末检P，无编译、JT、torch_npu或NPU。

入口_state/ascend-compat/real-model/native-edge-gradient-rounding-r248/；
原parent-closeout为b07352ce8966fc9a39342d94261311f1d9cc1c972a9a3f051cc9f5577bd5a076，
新增math-only closeout为abb779f890f8b605ea1985925e4d3a6d438f5f37a4c234b73ee26e515801597f。
两份shim候选仅源码准备，尚未运行；公开Node hook和新core准入边界保留。

### R247后续生命周期准入与R249已复现pointwise修复

原new-core-prepare冷构建的rc-15失败保持原样。另立helper-probe，仅零Tensor导入，
实际确认直属服务child的jit_utils映射、双向pipe/read等待和CLOEXEC；
canonical cleanup后精确child身份退出，未把该诊断算数学或自然退出通过。
随后只在真实签名逐次吻合时识别idle trace服务，未知child/编译器仍拒绝，
实际测试入口不调用cleanup，原PID generation、全部后代及组退出要求保持。
该后续四阶段绑定3352文件源清单603f262146003b648d4d65b02439db60ae7258914df95853550ad0d8025c548f：
Native runtime整文件6P、Torch backend整文件15P、生态runner整文件24P，均0失败/错误/跳过，
worker自然rc0、seal/ACK完整、监护actions空。数学worker也完成6case/252数组后自然rc1，
原report FAIL且无内部traceback。135个owned generations全退出，组/cache owner空，
源码/site/tools及新8725/旧66cb二进制末检通过。

实际数学red明确24处dtype错误：四个BF16样本的branch-add、tanh输入/输出、
三种公开tanh均应FP32却成为BF16；child返回FP32、GEMM为BF16、master梯度FP32，
公开autocast关闭且AMP还原正确。Tensor.mean、nn.SiLU、F.silu的dtype未错。
原比较器因report FAIL原样拒绝；额外只读NumPy诊断沿原atol2e-6/rtol2e-5，
252数组中75数值失败、108 byteexact，FP32两个rank共84数组全在预算内。
BF16 rank0 lora_a GEMM在branch-add之前已有值漂移，SiLU dtype正确但值也漂移，
所以没有把全部数值失败归因于pointwise，也没有修改参考或数值预算。

R249只应用AMP内部FSDP词法标记、既有binary/native precision委托与Tensor.tanh委托，
保持公开autocast关闭和GEMM原计算偏好，并保留FSDP外显式native AMP旧行为。
原7文件源码候选中尚未复现的mean与NN SiLU修改不应用；
现有mixed-precision维护文件新增12个真实公开singleton FSDP回归，覆盖嵌套两dtype、
flat/nonflat、child/root异常和同实例重试。当前只有AST3.7/git diff检查通过，实际回归待验。

证据根_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/：
post-core-prepare-001/result.json SHA256 c1afb3571aec431c93d49ea456939ba3b19bd468126d113cd1b49464338d5a80，
final-terminal-audit为cffd22ce1726bf41ee68a8422299c8ee3ac7748cc807f1b39b932e4e75c6e63a，
数学report为df596211e00c6b0346b8086e0945fe010ed0623a39c8472b5764fe646534a8eb，
完整NPZ为b2cdc6700a94b04031756e8d0c514b2aab00cfd944cc3457423ae88b1a692dfa；
只读numeric-diagnostic-002/result.json为38fd36fc8b70e539ee5c6c7e3ddc5f13d80713fd48a72c2d9e1b18f4310cf7d7。
修复及未应用原候选保存在fsdp2-pointwise-main-r249/与r246/，新运行必须新cohort。
本次fresh HTTP1 fetch rc0，wip305c2f086和2.0-refactor ca56aa802仍为HEAD祖先；
记录handoff/resume-r249-1792958748/，无重复合并、reset/stash或push。

R249首轮7550源实际mixed-precision整文件43项为31P/12F、0E/0skip，
12个失败全部是Tensor.add把FP32降成BF16/FP16；同轮先检查的operator_add和torch_add已通过。
这不是监督故障：worker自然rc1、actions空、81个owned generations及全部组退出，
源码3352/7550、site/tools与新旧core完整末检通过；原失败/JUnit完整保存，math和comparer未启动。
入口pointwise-fixed-prepare-001/final-terminal-audit.json SHA256为
e611d8defdb4e5377354510cf56be69887fcf2666018330604910941270c7114。
其后只补Tensor.add到既有_add的薄委托，alpha保留关键字语义并登记fidelity；
同12个公共case增加alpha=2和完整数组值检查，不改原6case/252数学worker或比较预算。
这份补充尚待新cohort复验，不能把首轮失败写成通过。

R249第二轮d557源实际整文件39P/4F、0E/0skip；余4项均为child/parent异常用例的
Tensor.mean在outer keep_reduce保留时降精度，新增8个正常嵌套/flat用例已P。
不是单凭首轮普通作用域mean dtype正确就外推全部作用域。该轮同样自然rc1、actions空，
45个owned generations及全部组退出，3352源/15dirty、site/tools与新旧core完整末检通过，
math/comparer仍NOT_RUN。pointwise-fixed-prepare-002/final-terminal-audit.json为
6ded7d90a02907ad883f6e87135f383d54a1a38aaf1efb2e38e5def6806eb490；
handoff-result为f72891a8f9945a475d9e6fa92d01e1710f2b28a6f686f01374d3ddbd34338ad1。
据此才将原先未应用的mean owner纳入修复；只清低精度偏好而保留native keep_reduce，
让FP32及低精度输入均保持各自dtype，FP32内部累积策略不改。
公开Tensor.mean与torch.mean使用一致策略，同12case追加两种低精度输入的dtype/完整均值检查。
这些新补充待第三个独立cohort实测，NN SiLU候选仍未应用，原预算/数学worker保持。

R249第三轮00b07源整文件43/43P、0F/E/skip，原数学6case/252数组report PASS，
原24dtype错误全部消失，两阶段均自然rc0。原bb2独立Oracle NumPy比较仍FAIL：
99/252数组超原atol2e-6/rtol2e-5，四BF16组合分别23/26/24/26F，FP32两个rank各42全在预算内；
两个updated/head.bias超预算，其余更新在预算内。comparer自然rc1、actions空，
30个owned generations/全部组退出，3352/source00b07与16dirty、site/tools末检P，core新旧未变。
入口pointwise-fixed-prepare-003/final-terminal-audit.json为
64bfaf172c06527675d719535eac9a0d695b04261390d9b95e317b44d917befd，
数学NPZ为24eaf43b27293e0986e0244187779b3688f916d21c552fcdea652eb643207df4，
原比较为9581a20f37994bc5bb136b10c3e21fdcc1f5c853e3c6795c2f108163d861fa5f。

只读完整数组三向诊断确认四BF16 base输入及36份初始参数均相同，
旧base对Native逐字节一致，003却新增2/16或6/16坐标偏离，最大1.953125e-3；
biasless LoRA输入/输出新旧逐字节相同，既有偏离仍在，head输入已改变不能直接归因。
该证据及generic CPU日志与native内部prefer32被新binary guard清零的源码路径一致，
但原捕获缺compute-weight/乘积中间值，不称唯一根因实测。
对应LAB r238-marker-gemm-numpy-comparison-001/delivery.json为
5fedde4cf10cee551fe332229aed423367e8d47b95bf6f4562000da4a4e2e970。
据此将FSDP标记改为线程局部精度栈，仅保护其自身隐式compute偏好，
Native kernel显式进入不同precision时保留其累积要求；公开autocast策略未改。
新增公开singleton FSDP Linear舍入/完整梯度回归，整文件现在44项，尚待实测。
独立Native Torch2.7.1 CPU已实际通过explicit BF16 cast及autocast两case/16完整数组：
输出1.0234375、输入/权重梯度2.0625、bias梯度2.046875；
linear-midpoint-native-cpu-001/完整NPZ为47517123791e3f3f1902f6c48e86db354a3b4177163cd70e3a65d5b1c36b9941。
此前mean-native-cpu-001也实际通过4dtype×3autocast的12case/60数组，
NPZ为45f2497e7262c23b7a899b4606e3b522db7bb4e4c54d32b75c58d4bbbce44b82。
两份新增Native参考均不声明CPU FSDP或NPU通过；原失败与预算保持。

R249第四轮d11c/source004实际整文件44P/0F/E/skip，含新增公开FSDP Linear中点及完整梯度回归；
原a835数学6case/252类型合同P，原bb2仍44/252数组超预算（四BF16为13/8/14/9，FP32各42全P）。
全部prediction、loss和updated参数数组均在原预算内，最大差分别7.45e-9、1.86e-9、6.11e-7；
残留input-gradient两项、head.weight梯度两项、lora_b.weight梯度四项及中间/辅助数组失败。
独立stdlib读取NPZ确认四个BF16 base output的完整NPY bytes恢复与Native一致，
不是用最终输出通过掩盖中间差异。math NPZ为37e3db3987e2b295d8fb462a2eba15a6a745a3874187de87c4dabf402fdc5d14，
原比较f8061c2bc5f436c5fc7b83006745d6313fdcedb5fb6f0aeb7494eea3b1183e5e。
72个owned generations全退出，source3352/16dirty/site/tools末审P、core新旧未变、锁释放；
pointwise-fixed-prepare-004/final-terminal-audit为e220c084c4343e93f5ce47ec7c5cfcc979a5343fd3a328b3ae8c20cf076e24e1。

004的8个auxiliary失败全部为nn.SiLU/F.silu，输入base-return完整NPY bytes两侧相同，
且各runtime的module/function输出互相byteexact；这些detach辅助输出可隔离到SiLU数值路径，
不据此解释主训练图梯度。源码最小compat候选及完整双runtime低精度路由验证仅在LAB准备，未应用。
入口R238/004-silu-offline-001/delivery-v2.json为c8015950098dad9c6c2f8793cf9e9777154b88d5c6cc2ca13a4c76e19d6340bb。

随后提交门禁首阶段真实20项16P/1F/3skip、0E，唯一F为已有test_bfloat16_linear_rounding的
普通FP32 master Linear公开CPU BF16 autocast用例：1.015625而非1.0234375，原差0.0078125；
该断言之后的梯度尚未执行。三个原skip为oneDNN unavailable、CUDA unavailable和CUDA closure contract。
worker自然rc1/actions空、333个owned generations及全部组退出、source004/sites/tools末审P；
Native core/Torch core/完整structure按原门禁NOT_RUN，没有缩选择、改期望或重试。
final-gates-prepare-001/final-terminal-audit为2d546003c1fa6e69bd045e199a5e5128bc64a2253710e0a5a97adff1621eca37。
据此将公开autocast保护也限制在其自身当前隐式compute偏好，保留Native累积器显式FP32请求，
复用既有effective-device/dtype/register owner，不复制Native数学；该补充待新cohort实测。

### R248真实JT CPU反例：协议未完成，不能写成500数组通过

专属单次运行绑定source004/core8725，第一bfloat16/custom/both_lr已计算并保存完整目标、
两条raw FP32 VJP及BF16叶梯度。两raw VJP与Native全值一致，实际叶梯度八坐标全部等于
merge-FP32-then-cast预测，并全部不同于Native each-edge预测；首坐标1.00390625+(-1)，
JT为0.00390625/bits3b80，Native为0。它是实际数学反例，但不是Qwen唯一根因证明。
原NPZ没有任何incoming hook key；不能从缺少low_bits推断hook收到FP32，原错误推断已纠正并保留审计失败。
worker在读取incoming key时原样KeyError，协议completed_cases0、仅30数组、无report，
没有补值、删hook、改变数学或凑到16case/500数组。原始NPZ为9b8560a5030f5836a9377d7291fa3c731490a73032c2fd1b018b68fd5cd72ca9，
独立witness为7c9d7471b1a67c70a6f3fc5a609fa91cb70fc63f6a5bb190130e834b22a5269a。
controller/worker自然rc1、actions空、15owned全gone/groups空、无unraisable，source004/sites/tools与core末检P；
真实审计8b4679eea25105b43e01ba5078d9cf6f0f543cf7df9ef2c5c5cd0e2492870701。
入口_state/ascend-compat/real-model/native-edge-gradient-rounding-r248/math-only-shim-parent-prepare-001/。

源码核对发现上游已整合tip ca56aa802的grad.cc有非loss完整梯度hook执行段，而当前本分支缺失；
per-edge dtype校验上游仍未补。两者分别准备候选，未写入主树、未建立新core或授数学通过。
独立Native Torch2.7.1 CPU又实际完成4case/11数组hook参考，确认注册/移除、loss恰一次及
requested nonleaf同时/单独请求的raw capture差别；NPZ为3e5161864c8f5fb851b1c0f959a85beea2df3c1b3edd55fc65a7e30fcae70e73。
所有NPU模型、重计算、随机恢复和性能边界保持；新NPU prebuild0/1→prewarm0/1→full包仅源码准备。

### 提交门禁的公开autocast修复与oneDNN分类阻断

source005/06bcef的首阶段实际包含原3个precision文件及整44项mixed，
64项为61P/0F/0E/3skip；公开autocast midpoint与FSDP中点/完整梯度都已执行通过。
但pytest真实rc1：唯一other是原reason optional oneDNN library is unavailable，
现有strict require-execution策略据other>0拒绝该run，所以仍记FAIL，后3stage均NOT_RUN。
原skip/严格阈值/选择/900 timeout均未改变，1740累计closed身份及组退出，
source3352/16dirty/sites/tools/core末检P；final-gates-prepare-002/final-terminal-audit为
35f0ae1b3e55be51d5ae61912e82ad00330880702e89dbfb31e6224fc940c4ee。

canonical gate_scope原名单已接受mkl及构建缺cmake等同类环境事实，却漏oneDNN名称；
在同一名单补onednn，不改skip本体和pytest_policy严格other阈值。
维护回归增加真实oneDNN reason应归environment及未知optional library仍归other的对照。
stdlib执行实际classifier AST，原reason由other到environment，两个未知reason保持other；
未导入框架，不把此host分类检查算完整pytest门禁通过。
证据_state/ascend-compat/distributed/fsdp2-pointwise-main-r249/onednn-skip-classification-001/。


### R249最终CPU门禁：三个通过阶段与两轮结构失败分开保存

source006完整3352文件清单SHA256为95c10cbfa5a4e1579a2589f03164e18d99f1b7a5ea3861fa567626d2afe40330，
HEAD为d1934d7d、当时18dirty，实际使用8725 CPU core。precision-neighbors为64项61P/3skip；
Native模式core为149项103P/45skip/1xfail；Torch模式core为239项238P/1CUDA skip。
三个阶段均自然rc0、seal/ACK及退出审计通过，oneDNN原因归类修复已实际生效。
原四阶段003的structure则在benchmarks.asv收集时失败，1666已收集、0执行，不能称1666通过。
003完整终态审计为6e87c4b0c033de208ca84bd08b9c297808a3853d8bb4ddc8b6f8ba7b9e83d1f2。

标准tools/run_test_suite与python -m pytest在仓库cwd执行；本轮私有cwd启动器缺仓库根搜索路径。
新structure-only入口仅恢复该路径，实际benchmarks来源和SHA确认是主树包，原收集阻断消失。
完整选择实际开始执行后，原global compiler守卫触发，监督只终止本任务精确owned目标；
worker rc-15、两次停止动作，无最终report/JUnit。触发foreign的PID在原异常前未落盘，
不能事后推断其归属或将本轮改成通过。失败及fce9b00e7f90b737fd40a8a2d5478655bfcb9428de2a6afc7db6ae9205256c59终态审计保留。

独立首失败诊断保留完整1676项收集，仅用-x取得原失败，实际执行74项73P/1F、0skip、rc1，
不授完整structure资格。失败是test_child_process_contract.py::test_a_timeout_ends_the_grandchildren_too。
同一grandchild的21次真实监护采样从S/原父进程转成Z/outer coordinator，并在失败前已是Z；
outer subreaper只在结束时reap，测试的kill(pid,0)因此仍可查询该PID。
这不证明活进程漏杀；原测试、2s超时和200次等待均未修改。
监督末端对该已知owned Z的额外SIGTERM动作也保留，不冒称actions空。
首F终态审计a9582d7c555cfe3ef27ddd431a07d4fd75346af93467d4af8c7e3fe298eb3e88；
3159个累计精确代际全gone、组/缓存owner/编译器为空、原二进制字节未变。
证据位于LAB _state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/对应003、structure-only和first-failure-diagnostic各独立目录。

### R248多轴真实反例与R250生产整合

独立Native边界参考另实际完成12case/228数组，涵盖重复输入slot中的None、单Node双输出完整VJP、
低dtype sum-to先于目标dtype转换；完整NPZ为3bae4729482727c1a41c520512245ff4269242cdb3aca6a6cfeb4a7a21c595e7。
这是轻量Native CPU数学参考，不补造未捕获PID的完整连续生命周期资格。

随后Native四例实测FP32叶形状(1,1)、低dtype raw VJP形状(2,2,1)，leading和singleton axes一次归约。
BF16/FP16 half-ULP fixture的Native梯度分别为0.00390625和0.00048828125，
分两次低dtype归约的事后控制均为0；两full-ULP控制分别为0.0078125和0.0009765625且两法一致。
原backward直接返回raw，不做观察端归一化。Native4case/96数组NPZ为
76e75505d8d0d7b08eb00d3edab0c677694f5bbb6441e22c5933fc06407e74d9。

source006/core8725上的真实JT独立复现不用Tensor hook，完整4case/88公共数组保存。
两个half-ULP叶梯度实际均为0，两个full-ULP控制与Native exact，且每case backward恰一次；
所有incoming/raw和事后控制的公共数组严格字节相同。原总断言自然rc1，
NPZ为87172db074513b20dd8f4961d886bef639fb4d58fc2e797c9185111c6b2f660e，
report为2fc581bb754af250249a2f06114ee2e840667228766198e308b48dacd192789f。
严格88比较共8数组F：4项是两个真实叶梯度的values/bits差异；
另4项loss/bits的payload相同但Native为0维、JT为1维，values均0维且相同。
两侧bits采集实现不同，原数据和严格FAIL保持，不通过reshape或豁免取得通过。
该新协议不替代原R248 500/662，也不声明Node或Tensor hook已支持。
终态审计18c12b6d5dcc1890c4a01525c631f43d6f009cf852159b0a30c606c140bb4cd9证明33owned全退出，
完整源码/site/tool及新旧CPU二进制末检通过。

据两类真实反例，R250现已修改grad.cc/h和compat autograd：
Torch反向使用可嵌套恢复的TLS逐edge dtype策略，完整VJP先按记录形状归约再转换目标类型；
同一普通op的多output先组成完整VJP，不把不同input slot的贡献混为一条edge；
跨edge累积保护已归一类型，Native默认路径保留。非loss Tensor hook回补的是已整合上游tip已有十行实现。
Python sum-to把leading/singleton axes合并为一次归约，原reshape保留。
五个既有维护测试文件加入独立判别回归，原用例AST保持。没有冗余的raw dtype补cast，
因为当前ReduceOp本就返回raw dtype；Python callback前后AMP作用域另待实测，未顺手修改数学。
应用与精确八文件SHA见LAB _state/ascend-compat/real-model/autograd-edge-main-r250/main-integrate-001/result.json，
SHA256为3e7a02c11a1e3f48295e026fab003260541f0b6b48e3e9344ec3c6bb00cad742。
目前仅AST3.7、patch检查和git diff检查通过，新C++编译、新原始数学协议和NPU均NOT_RUN；
旧8725或66cb不能用于证明新的C++行为，后续用新私有canonical core和新cohort验证。

### 新增SiLU参考与q008确定性证据更正

原SiLU worker/fixture在独立Native CPU parent下实际通过64records/256完整数组、0错误，
覆盖四dtype、函数/模块、公开autocast和内部偏好数学参考；实际Torch2.7.1+cpu、
-I/-B、私有HOME/TMP、BLAS2、原READY/seal/ACK和自然rc0，未导入JT/torch_npu。
NPZ为4453ba5668f838a59448c7f081cb23c4823db73ceb4bf6626f15f5715759a42a，
delivery为ae8b1203ab1a5146a2fbef48191461d211899043f32507b8971dd72585c12775。
内部FSDP行只是Native禁用autocast数学参考，不是Native CPU FSDP或JT/NPU资格；
SiLU生产候选尚未写入主树。

重新读取q008两runtime/三phase/两rank全部272条policy记录：
JT136条的Python及真实backend readback均True；Native136条Python True，backend字段null是未观测而非False。
ACL source SHA与q008完整manifest相同，现有process set/get及core readback已经实现。
旧R65等协议中的UNIMPLEMENTED_FALSE只保留其原源码范围，不能套到q008。
固定torch_npu还会按算子懒同步context/compile/HCCL owner；
这些owner及warm/asynchronous切换仍待真实小实测，不用Python flag推断。
核验交付为LAB _state/ascend-compat/distributed/acl-determinism-owner-review-r233-001/delivery.json，
SHA256 d2e486ce7322ee98cc5dd8089a492386336333a30674b4503762a6a0ba420a77。

本轮服务器HTTP1 fetch35s超时原样保存，随后从用户本机对同一origin执行git ls-remote实际rc0，
两目标最新tip分别为ca56aa802和305c2f086，与服务器已存在且已合入HEAD的tip一致，无需重复合并。
记录LAB _state/ascend-compat/handoff/resume-r250-1792966232317484809/local-origin-verification.json，
SHA256 c37e1de011db74c86a5432fbbe712d9db629f99a055ca90df344721a16ffb686。
没有把服务器fetch超时改写成成功，没有reset/stash/push。


### R251：新core与原500实际通过，168捕获暴露真实view/storage差异

本节记录2026-10-26续跑。source2为3352文件、HEAD d1934d7d、26处dirty，
完整manifest SHA256为4e4ef1f858b1a005d1b9e745882de579e61e9deae71b4d9fb91f943d5b370dc5。
后续R251已修改C++与Python，因此以下source2 PASS不授予新源码、NPU或模型通过。
source2全部tracked bytes、symlink及未提交patch已在LAB保全，
入口_state/ascend-compat/real-model/tensor-view-main-r251/source002-preservation-001/；
tar SHA256为b1921ddc78e7944ddd3410d36a50779913e68b5a02c9deabc5a8e3fba6a03a2d。
没有reset、stash、覆盖旧失败或push。

cold005在全新私有canonical CPU cache真实完成244/244编译，实际加载core
408e0aa8478f967e87dc673cc0e7aaf5f66ebc081e138bd27734f0741efa164a，
原stage自然rc0、actions空、seal/ACK完整，CPU/零回落及新per-edge符号通过。
末审980个owned/controller代际全退出，source3352/tools240及实际四产品字节通过。
入口_state/ascend-compat/distributed/autograd-edge-next-core-r239-005/，
receipt为99e72d1b29909e41187fe410bdbe13aa86bdb8c6d30b73a0b7a341832655a37c，
final-terminal-audit为6bab83d55fe694a13822530655b997c9197de6414ac361111f819d0b5e1f4a11。
前面001/003/004的实际资源守卫误停保留。执行器改以本任务真实flock、私有0700目录和
fresh普通外UID隔离，外部cmdline只作尽力记录，不要求独占共享服务器；
同UID未归属编译器、root/能力异常、明确受保护路径引用仍拒绝。
这是LAB调度修正，未修改数学或容差。

随后source2/408e实际运行原16case/62check/500数组worker，
worker7f09与原数学、Tensor hook及观察器保持原字节。
原报告PASS，独立Native500严格dtype/shape/整NPY字节比较全部exact，
missing/extra/mismatches均空；两个完整NPZ的SHA也同为
722200955ba0afb08890bfdeaaadbb690ab43784862beee1bd51cbbcc821fc9b。
原stage及parent自然rc0、actions空，48个owned/controller全退出，
入口_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/new-core-math-prepare-001/，
final-terminal-audit为96d507531295fb624bcf2a30a1fa7809ff628f40ee4d368c008ccaec3412b2ee。
该math-only合同仍未包含Node.register_hook辅助协议，不宣称完整Node hook能力。

同源随后实际完成原AMP12case/168成员/96metadata，所有12个内部数学检查均True，
FP16 raw sum-to的FP32梯度为1.0009765625，BF16大值控制为65536，三个forward autocast模式都一致。
但原严格验收仍FAIL：公开Tensor.view(torch.uint32)落入遗漏dtype的NumPy uint8 fallback，
导致60个FP32 bits成员实际为uint8及扩大shape；另外12份forward/output和12份backward/incoming
的真实stride为1，而Native为0。108个可解析原NPY全部byteexact，原168成员、206条比较错误、
validator拒绝、worker -15及监督的一次own SIGTERM均原样保存，不把完整捕获或数学通过冒充总PASS。
入口为上述目录amp168-prepare-001/；原NPZ为
ebbfcb924091604e0a313fde0fabbceaf3794215cef9cf5d039eea8f6ca35d46，
failure末审为1f7c204bb002bf7bb53bc7a3802abe9a4a7a5593fb65b88f9dc7fbcc2ff45cca。
45个owned/controller全退出，累计closed4414；没有放宽UInt8解析、删metadata或转换观察值来过关。

依据真实失败，R251将Torch dtype view委托已有native reinterpret_view，停止该公开view的梯度，
移除静默uint8/CPU copy fallback，并按Native合同拒绝改宽且最后stride不为1的输入。
C++广播在构造时捕获Torch前端/反向policy，选择已有真实零stride和share_with；
隐式Python number仍保留融合，duplicate以三态准确保存原意图，Native默认scalar融合不变。
同宽同形状reinterpret复制真实storage_strides，不能用contiguous/copy或虚报stride代替。
补充维护用例覆盖UInt32/UInt64、0D/2D/1D、双向alias写回、无梯度、
真实sum-VJP输入及contiguous但unit-axis stride0的边界。
候选v3为728940255fa0f4026cd04dfe6b3855b4b8563cee37a16dbb5b47654da6a4750a；
整合记录在LAB real-model/tensor-view-main-r251/。
实际三份C++ syntax-only均rc0、Python AST3.7及git diff检查通过，
新canonical core、原500/168、维护门禁和NPU均待当前源码实测，不能沿用408e。

另有独立Native参考，不授Jittor通过：单卡910B3 False→True→False owner探针正常rc0，
72完整数组/36tensor/13读数齐，64跨repeat/phase完整payload比较strict0。
Python flag切换后，process/context在首个实际算子及同步前仍保留旧值，
warm后分别为0、1、0；当前context与调用线程稳定。
ACLOP deterministic全为ZERO_SIZE_NO_VALUE，不能写成False，JIT_COMPILE真实disable；
没有process group，HCCL明确未测，亦不回填q008未观测字段。
入口_state/ascend-compat/distributed/acl-determinism-owner-review-r233-001/native-owner-probe-run-002/；
result为e11ba92fbbfa7a9a655bbb5f82c27e0044ef216163a32f143f18d4c210117a24，
NPZ为0598818c4629bd1ee1ef08aded13fae2a5c492445da1345a34250a3b1a247a6d。
24个已观察身份全部退出、相关组空、卡已归还。
run001在数学前因深TMPDIR使CANN依赖SyncManager的Unix socket路径过长而rc1，
数组/owner读数均0；原失败保留，run002只换LAB浅层私有TMP，并未改数学。

Native CPU另外真实完成6case/120数组的ambient/custom_bwd合同：
未装饰backward服从ambient，custom_bwd按forward保存状态重入并恢复，
cast_inputs FP32保存关闭；raw低dtype sum及FP32目标梯度保持正确。
入口_state/ascend-compat/real-model/r250-custom-bwd-native-cpu-reference-r233-001/cohorts/attempt-001/。
UInt32/UInt64及zero-stride的Native参考见LAB r238/unsigned-bitcast-native-prepare-001/
与cpp-stride-narrow-review-001/。这些证据不支持凭早期源码假设新增全局AMP guard；
本次未修改AMP数学，原Qwen跨runtime124/272失败、source004数值44/252失败、
最终双卡、公开重计算、Jittor随机恢复与NPU性能均保持待验收。


### R251 后续：新 core 的公开视图实测与 copy/Tape 修复（2026-10-26）

source3 的 cold006 已真实编译 244/244 并自然返回 0，stage 为 CPU_STAGE_PASS、actions 为空、terminal_sealed 为 true；实际新 core 为 68acf96f，独立缓存末审确认所有已观察进程代际退出、相关组和缓存 owner 为空、JIT 锁可实际取得。该结论只授予 source3 的 CPU 构建资格。

随后在该实际 core 上完整执行两个公开 view/storage 文件：17 项全部执行，9 通过、8 失败、无 skip。unsigned dtype view 的 dtype/shape/完整位图/共享指针/stop-grad 断言已经通过；其中 7 个失败发生在标量索引后的 copy_ 写回，另 1 个发生在自定义 Function 返回视图的 stride。原完整 JUnit、stdout/stderr 和失败 report 保留于 LAB/_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/new-core-math-prepare-002/run-view-storage。stage 自然 rc1、actions 为空；worker 在必执行节点断言处抛错，因此没有 READY 或终态握手，不能将此协议结果误写成数学根因。末审没有实际 exit_integrity，只引用原 initial site 和真实 runtime_after，不宣称末态 site 全量相等。

源码定位确认 copy_ 默认调用 _ip 时未请求保留物理存储，绕过已存在的原生存储写入；标量 getitem 本身已是实际共享 allocation。修复仅令 copy_ 请求 preserve_storage=True，继续由既有条件区分不可微存储写入与可微图更新，并保留真实 expanded-overlap 检查。TapeOp 还继承了 dense-only 输入转换且 infer_shape 未复制 storage_strides；现允许它消费真实 strided 输入并复制物理描述，保留原 contiguous replay fallback，不新增伪 stride 或 NumPy staging。

新增 Native Function 两参数测试覆盖输入或输出 transpose 的真实 stride、指针、完整值和非均匀权重下的完整梯度；显式局部开启 Native transpose_storage_view，避免把 Native 默认物化行为误当输入视图。原 17 个公开测试、全部数学 observer 与预算未改变。新 Python AST 3.7 和实际 g++ syntax-only 检查通过；这些修复尚未获得运行通过结论，因涉及 Tape C++，source3 的 68acf core 不能作为新源的运行资格。新源码冻结入口为 LAB/_state/ascend-compat/real-model/tensor-view-main-r251/source-004.json，后继首先新建 core 再重跑完整公开视图、Function 和 copy 相关回归，以及原 500/168/252 检查；真实模型、最终双卡和性能仍未完成。


### R252：恢复连接后的 source004 CPU 验证与历史 Qwen 对拍（2026-10-27）

服务器 SSH 已恢复。实际 HEAD 为 d1934d7de97eee196133100822ee4bb1f1684ee5；源码
source004 的 3353 文件 manifest SHA256 为
57688b2ba2b186c5a9fd31d0ec1b2bd158473eb75f6d57c43457e663f42856f3。
服务器 fetch 因超时仍失败，原记录在 LAB/_state/ascend-compat/handoff/resume-20261027-origin-check-001/；
另从本机实际 ls-remote 确认 origin/wip/cgq-day23-fix-20261021 为
305c2f0866f163b0549d47b2e4264c5d41a86385，与服务器已有远端跟踪引用一致且为 HEAD 祖先，
不将本机远端查询写成服务器 fetch 成功。2.0-refactor 后来前进的提交仅做只读审查，
没有再次合并。修改前源码、暂存及未暂存差异和 manifest 已保存在
LAB/_state/ascend-compat/real-model/tensor-view-main-r251/source004-preservation-20261027-001/。

source004 cold007 实际新 CPU core SHA256 为
b26739374bc233a95d305fa3a26cc645f276f5852c5215f020104e08238312a1，244/244 编译、
stage 自然 rc0、零回落，最终进程/缓存/锁审计见
LAB/_state/ascend-compat/distributed/autograd-edge-next-core-r239-007/final-terminal-audit.json。
同源公开 view/storage 两个完整文件 17/17P；随后两个邻接完整文件 9P/11F，无 error 或 skip，
自然 rc1，原 JUnit 与全部输出保留于
LAB/_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/new-core-math-prepare-003/run-storage-neighbors/。
独立执行 Native Function 整文件 48/48P，包含真实非连续输入和输出的两个新增测试；
实际进程、缓存与锁末审见同目录 native-function-only-prepare-001/final-terminal-audit.json。
这些结果只针对 source004 CPU，不替代原 500/168/252、最终主树双卡或真实模型验收。

历史单卡 Qwen3-0.6B BF16 LoRA 的 source123 Jittor 完整产物与 R251 原生 Torch/NPU 产物，
以原始跨 runtime 比较器、原预算及完整数据实际比较 2187 项全部通过，包含 logits、
112 个 LoRA 参数、优化器状态和 4 次更新边界的每个可训练梯度。
结果位于 LAB/_state/ascend-compat/real-model/qwen3-historical-cross-r252-002/result.json，
SHA256 94883915c693f8e8bad38316c7a4921e5802bc49037b5b64bbff87842074d3df。
原执行尝试因 isolated Python 缺少比较器目录在数学前失败，stderr 已保留；
第二次仅补充原脚本目录，没有更改预算。该历史 Jittor 的确定性开关不可用而 Native 为 True，
不构成同策略结论，亦不能外推至当前 source004。另借用 FSDP CPU 点对点阈值做独立敏感性诊断，
112 个最终参数数组中共有 34636/573440 元素超出该额外阈值，最大绝对差约 6.85e-5；
该阈值不是单卡验收预算，不改变原 2187 项结果。

进一步真实复现 no_grad 物理写入覆盖反向传播所需旧值：exp(x).sum() 删除中间 holder 后，
原生梯度为 exp(旧 x)，Jittor 得到全 1，原始完整输入与梯度保存在
LAB/_state/ascend-compat/distributed/zero-inplace-r252-r239-004/ 等 cohort。
原生 saved-input、自定义 Function 与 detach 别名版本错误也仍未解决；
上游 saved-storage 处理候选与 zero 候选必须在本分支重新运行验证。
原生 copy 20 个别名/广播例及原位 zero 标量参考仅提供合同，不能授予 Jittor 通过。
直接试用上游 zero 函数在本源的 CPU 实验仍有非叶或视图指针/别名失败，未将候选当作修复。
具体数据与完整报告位于 LAB/_state/ascend-compat/distributed/storage-neighbors-owner-review-r233-002/
及 zero-inplace-r252-r239-006/。三个旧视图断言改为真实 PyTorch 强 root 所有权和物理 stride
合同，隔离原生 Torch CPU 实测 3/3P；新增 Native Jittor 物化视图覆盖仍待实测。
当前工作树的标量常量、冻结视图 RHS 候选以及新测试均未得到运行通过资格。

在确认上述实际 exp 反例后，已按当前上下文移植上游提交
0dcd03a9b9e87b8666112e72af405a075c67221b 的四个 saved-output 生产改动，
并新增上游四项 Native 及本次非零输入 copy_/zero_ 回归；详细来源与候选字节见
LAB/_state/ascend-compat/distributed/fsdp2-pointwise-precision-r238/upstream-saved-output-review-001/。
该改动只作为待验候选：需要新 core 和实际原始失败复验，不能将 saved input 版本检查、
视图 zero_ 或捕获重放等能力视作已修复。

R252 后继 source008（manifest a1b565fb3f0d610c2092e0261511318e390782e96ab4f756baa9276ed519b098）
实际冷构建 244/244，CPU core 为
60cfa22779be8f7162bd74f9d54e41345e6085f6cd93243532faefb8d92af10f；
stage 自然 rc0，完整进程/缓存/锁末审见
LAB/_state/ascend-compat/distributed/autograd-edge-next-core-r239-008/final-terminal-audit.json。
该 C++ owner 在 Native 保存输出四组合 4/4P；在 Torch 非零 exp 实际反例中 copy_ 分支通过，
但 zero_ 分支因 source.grad 为 None 仍失败。完整六文件共 91 项、83P/8F/0E/0skip：
公开 view 17P，Native Function 48P，原位存储 9P/7F，梯度 view 5P/1F。
标量除法原完整组合通过；失败及各文件 JUnit、完整输出、终态在
LAB/_state/ascend-compat/distributed/r252-saved-output-gates-r239-001/。
其中冻结叶张量原位修改仍会丢失 detach alias 的真实存储关系。新加的 Native 物化视图
完整文件另 8/8P、自然 rc0、末审见 LAB/_state/ascend-compat/distributed/r252-view-owner-native-r239-001/。
不将局部通过写成整体通过，也不将 CPU 结果外推到 ACL。

按仓库要求执行 source008 的完整 tests/structure，实际 1676 项为 1669P/5F/2skip，
stage 自然 rc1，两个 skip 为 CUDA 不可用。原失败在
LAB/_state/ascend-compat/distributed/r252-structure-r239-001/run-structure/data/junit.xml；
末审见同目录 final-terminal-audit.json。两个并行子 pytest 未显式加载已安装的 xdist；
预算报告把配置 worker 数误写为 affinity 限制后的实际数；
两处用户输入边界计数各因真实新增检查增加一项。
针对这些证据已更新预算记录及相应测试、显式加载子进程 xdist 并逐项更新检查表；
这些改动尚待新源码实际复验。原 source008 工作树、暂存和未暂存补丁已完整保存在
LAB/_state/ascend-compat/real-model/tensor-view-main-r251/source008-preservation-20261027-001/。

单卡完整逐 microbatch logits、累计梯度与每步参数/优化器状态观察器已建立独立 LAB 候选。
原生 CPU 小模型真实比较带观察与不带观察的 2071 项严格相等，完整运行与 first+新进程
resume 的 5858 项严格相等，缺字段与数值扰动负控均拒绝；实际数组和终态见
LAB/_state/ascend-compat/real-model/qwen-single-observer-tiny-native-cpu-r238-003/。
此前两个夹具/调度失败的完整输出未覆盖。观察器可能改变物化和同步，不用于性能判断；
真实 Qwen、Jittor 和 NPU 的新观察流程仍待实际运行。


### R252 后继：真实单卡逐批对拍、CE 根因收窄与 FSDP2 CPU 候选（2026-10-28）

本节仅记录各冻结源码与实测范围，原始失败和完整数组保留在 LAB，不扩大验收。
结构修正后的 source0fb（3353 文件，manifest SHA256
0fb1c0a25856ba44948c42900ec05fcc4b1940c6619a461c4c46734049b81d21）
完整 tests/structure 实际 1674P/0F/2 CUDA skip，真实退出及锁审计见
LAB/_state/ascend-compat/distributed/r252-structure-r239-002/final-terminal-audit.json。
原1669P/5F/2skip、xdist及worker配置失败、错误边界数量差均保留在
r252-structure-r239-001/。source0fb 新私有 ACL 冷构建实际 core
35ec3be599cd77b7b94027279bb4647228d6623b74f263d4fa1a976dd635b693，
自然 rc0、设备正向见证与末审见
LAB/_state/ascend-compat/handoff/qwen-single-jittor-observer-r238-006/final-terminal-audit-cold-006.json。
同源码真实 NPU provider 2×2 输出与梯度完整严格相等、实际 ACL logical 0、fallback 绝对及增量为0；
该小张量通过不构成完整模型通过。此前空设备掩码207004、资源/入口守卫失败及所有原日志仍保留。

真实 Qwen3-0.6B BF16 LoRA：新 Native 与 source0fb JT 各自 full/first/新进程 resume
均自然完成。JT 自身 state 933、完整原梯度 1136、补充逐批轨迹 5870 均 strict0；
原跨 runtime 完整协议 2187/2187P。然而新增同一观察器的完整逐micro/逐step跨 runtime
为4932P/5F，原 forward atol=.005/rtol=.03、gradient atol=.01/rtol=.05未更改。
5F为step3/4的四份完整logits与step3 micro2一份LoRA梯度，前两步完整logits逐位相同。
原结果、实际NPU归属、fallback及终态入口：
LAB/_state/ascend-compat/real-model/qwen-single-jittor-observed-20261027-r7/；
Native独立参考为 real-model/qwen-single-native-observed-20261027-r1/。
Native torch_npu内部 fallback 计数不可用，原样保留为 null；不能冒称两端均已计数为0。
完整逐数组定位在前者 analysis-first-divergence-002/。

使用同样112个实际梯度、初始参数、0→3e-5学习率的两运行时四cell AdamW 重放，
每套tape原预算933/933P，但不宣称严格逐位相等。固定任一优化器仅切换真实step1梯度tape，
均重现约3.94e-5的step2参数差；step2全部112梯度两tape逐字节一致。
原始NPZ与因果对照见 LAB/_state/ascend-compat/real-model/qwen-adamw-identical-tape-r239-002/。
同源共享全部112参数与rotary buffer的初始/update2 × early/late八cell前向，
四组完整logits共27956224值和1236实际BF16权重数组全部逐字节相同；
JT加载Native update2后late输入与Native原训练对应输出严格相同。
实验是干预定位，原默认训练4932P/5F仍未修复；见
real-model/qwen-shared-fp32-2x2-r239-002/runtime-delivery.json。

首两micro逐层追踪中，双方各自350数组观察中性和各自原轨迹236数组均strict0；
实际BF16权重两批各309/309严格相同。loss seed均exact1.0，最早公开非exact在lm_head
输出BF16 incoming gradient，分别141和445元素不同。实际num_items_in_batch数值均92，
但JT int32、Native int64，沿模块传播的118项结构差异如实保留。详见
LAB/_state/ascend-compat/real-model/qwen-initial-backward-trace-r239-001/runtime-delivery.json。
后继真实NPU CE控制12场景/每端176成员完整运行，自然rc0、资源/零回落审计入口
real-model/qwen-count-sum-ce-owner-r238-001/ce-npu-prepare-001/final-terminal-audit.json；
已完成全数组host比较：原数值预算164P，额外raw BF16位图检查12F；
CE上游loss_sum/逐rowloss incoming两端逐位一致，而FP32 logits gradient分别有1536538和
1572601元素不同，转BF16后恰复现上述141/445个元素的原训练位差。两端各自count int32/64
在该fixture中除计数dtype外其余数组逐位相同，因此不能声称计数dtype导致这一数值分叉。
完整比较结果见 real-model/qwen-count-sum-ce-owner-r238-001/ce-analysis-001/result.json；
实际cross entropy backward内核/归约语义仍需隔离与修复，不放宽预算。

source0fb CPU原FSDP相关500/500和AMP168/168严格P（含完整metadata），
原混合252为208P/44F，FP32 84/84P、两个BF16场景分别23/84F与21/84F，
完整终态见 LAB/_state/ascend-compat/distributed/source0fb-math-replay-r233-001/。
仅CPU无bias BF16线性层的隔离候选已真实冷构建，31维护实例/298数组严格P，
原252改善至242P/10F（34旧F转P，0旧P退化）；剩余8项SiLU辅助输出和2项混合输入梯度F。
该候选仅在隔离树，未进入MAIN，不能外推NPU/双卡；原失败完整保留于
distributed/cpu-biasless-accumulator-isolated-r233-001/。
独立SiLU候选与Native36参考、整数sum默认/out dtype候选与Native参考、no_grad/frozen zero窄候选
均在LAB，尚未合MAIN或获得全部后端运行资格。zero隔离两完整文件由14P/8F改善为19P/3F，
余copy广播/部分重叠及可微非叶zero失败；Tensor.sigmoid、capture alias、saved-input版本
仍为独立待修边界。

最终MAIN双卡FSDP2、随机恢复、公开激活重计算、完整主树模型5F修复及对应NPU性能/显存
仍待完成；性能须在对应正确性通过后评估。此次暂停时没有推送，先前全部本地改动继续保留。
