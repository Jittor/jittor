# 新 910B3 多库综合适配记录

本记录对应[综合计划](../testing/2026-10-21-ascend-integrated-compat-plan.md)，唯一活动状态源为[整改看板](../architecture/refactor-board.md)的 ASC 任务。当前最终源码为 **MAIN-r28/123 文件**，manifest SHA256 `d3904da402f4a0c0b17d61102ec8bf704c97391f338415d7f56ac567e311510e`。下文按实际冻结源码区分结果，保留原始失败和未运行项。

| 本轮验收 | 实际结论与源码范围 |
| --- | --- |
| 用户环境 | cgq 私有 Conda/Python3.11.17、CANN8.5.0、原生 Torch2.7.1/torch_npu2.7.1.post2 实卡 smoke；项目 dev 与独立 oracle 分离，安装目标均在私有目录 |
| 最终主树回归 | source123 完整 structure 1430P/2 CUDA skips、layout P；1432 不同 case 与 source121 逐 node 相同；97.py+1.pyi AST3.7 P，既定 Ruff 相对 HEAD 新增0、减少12 |
| 八模型及综合 Trainer | 各自 source65 等冻结证据上，八模型完整输出/271 参数+2输入全梯度；Trainer4updates/8micro、61跨runtime检查；连续与2+2恢复216完整状态及轨迹 exact，dropout0顺序fixture |
| 共享真实权重推理 | 0.6B既有冻结源码BF16完整推理P；source121的1.7B与4B各3输入/15完整数组/58,343,424 logits bitexact、cached/uncached greedy8 exact、actual NPU/fallback0；不是123重跑 |
| 最终主树 AMP | source123 actual MAIN no-loader 10P/0skip/fallback0，低精度activation/FP32master两步Linear/LoRA目标84完整数组/2880元素bitexact；whole168 strict仍F，两份BF16+FP32input LoRA input_grad共45元素超原阈值 |
| 真实0.6B BF16 LoRA | source123 R6完整单次工作流rc0，4updates/8micro、112 FP32 master、448全梯度归档、评估生成/adapter保存重载通过；R7由实际step2 checkpoint补fresh2恢复通过：933完整状态与轨迹strict0、224梯度数组及224 dtype标签bitexact；Native同模型全流程交叉未运行 |
| FSDP2 | own88/r7六fresh worlds/12 rank-phase，FP32 full8及各自3+5精确恢复P；最终MAIN123 NPU0 worlds/0 reports，资源不足；旧120入口archived/source-bound |
| 速度/显存 | NPU未验收。1.7B配对流程仅18交替calls后停止，generation steady/decode未完成、rc-2/acceptedfalse；4Bperf未运行，CPU控制器资格不能替代NPU性能 |

尚未达到全部适配退出条件。默认 FP32 整模严格数值差异、FP32input BF16 LoRA 严格输入梯度、activation checkpoint 不重计算、deterministic setter 未实现、最终主树新双卡与完整性能均保留明确边界。raw CPU core 四文件另有46P/9S/1个原样 `test_concat_random` 的300秒超时，未删case、缩规模或放宽超时，不称全部core通过。模型和fixture、性能与功能、旧源与最终源均不混用。后文按时间保留各阶段事实，最新验收见文末。

## 源码与环境

初始同步目标为 `origin/wip/cgq-day23-fix-20261021`，HEAD与远端均为 `305c2f0866f163b0549d47b2e4264c5d41a86385`；新分支 `codex/ascend-integrated-compat-20261021`，开始时工作树干净，无该仓库既有运行任务。同步凭据在 `$JITTOR_LAB_ROOT/_state/source-sync-20261021.json`，未合并其他远端分支。

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

本轮未得到合格的主树双卡 Jittor FSDP2 完整训练、恢复交叉或 NPU 性能结果；共享卡资源变化期间未接触外部进程、驱动或公共环境。

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

R23 在5/6 fresh 空闲快照后启动双卡 Jittor full。约10.97秒时 live monitor 发现外部 PID 1476170 占物理卡5，driver 返回125并只终止本 cohort 的 launcher 进程组。rank0/rank1 日志分别有706和555字节：rank0 已开始 setup_hccl 的 MPI-free HCCL 算子编译，rank1 到达 Jittor import；没有进入模型训练或产生训练阶段报告。launcher PID 1476146 的进程组已核验为空，事后 fresh 快照显示卡5/6均无进程。该次暴露 HCCL 算子首次 JIT 也需要串行预热，R23 的短暂初始化不作为 FSDP 结果；记录在 distributed/r23/。

R24 随后复用相同的冻结源清单，在物理卡5上按 rank role 0、1 顺序运行 R5 的单卡 HCCL/DTensor prewarm worker，并分别使用 R19 的 rank0/rank1 独立 Jittor cache。两份报告均为 PASS：world_size=1 的 FP32/BF16 HCCL collective 与 DTensor factory 检查通过，实际物理设备为5，fallback_after=0，source_unchanged=true；rank0/rank1 返回码均为0。prewarm-complete.json 记录报告 SHA256（rank0 018f6abdc1197191e372b1acef28328516a41186c1dd53bed0ae51c41b8cffbd；rank1 9eacf3d773b7169ca7c2228aee0b1554e1fd69b7ac7ec04c108e852d763ae140）。这仍是两个顺序单 rank 的 warmup，不是双 rank HCCL 或 FSDP2 证明。

R24 双卡 driver 校验 current-source manifest 与 R5 Native baseline 后，在 runtime launcher 创建之前的 fresh admission 检查捕获物理卡6外部 PID 1493168，直接退出。没有创建 launcher、rank worker 或模型输出；admission 前后 npu-smi 快照与 driver traceback 保留于 distributed/r24/。不曾向外部进程发送信号。

本节扩大了串行预热范围到 core、HCCL collective 与 DTensor factory，但 MAIN-r28 Jittor 双卡 FSDP2 full、prefix/resume 交叉仍没有运行通过；R23/R24 也没有性能数据。

### R25–R29：多物理卡尝试与资源抢占复核

R25 使用 fresh 空闲的物理卡3/5启动双卡 Jittor full，约10.64秒时 live monitor 发现外部 PID 1524441 占卡5，driver 返回125。两份 worker 均写出 process-start.json，但没有训练终态；rank0 的 HCCL ops 已加载并开始 communicator 初始化，rank1 到达 Jittor import。自有 launcher PID 1524412 的组已退出；外部 PID 随后不在事后卡表中，没有对其发送信号。R25 日志和 before/during/after 卡表保存在 distributed/r25/。

R26 建立新根目录，fresh 快照发现卡3已有进程，卡5空闲；因此没有启动 Python driver、launcher 或 worker。随后 R27 用 fresh 空闲卡5/6再次进入 driver，但 driver 在创建 runtime launcher 前捕获卡6外部 PID 1532400 并退出，rank worker 数为0。两次快照分别保存在 distributed/r26/、distributed/r27/。

R28 再用 fresh 空闲卡3/5进入双卡 Jittor full，约11.61秒时发现外部 PID 1610224 占卡5并返回125。两 rank 都有 process-start.json，rank1 到达 HCCL communicator 初始化；无训练终态或完整参数报告。自有 launcher 进程组已空，事后卡表中3/5/6为空，外部进程未被触碰，证据在 distributed/r28/。

R29 用 fresh 空闲卡3/6启动后约2.14秒，live monitor 捕获外部 PID 1992911 占卡6，driver 返回125；rank日志均为0字节。自有 launcher PID 2013322 及进程组已核验退出；事后卡表仍显示卡6有外部进程，未向其发送信号。完整记录在 distributed/r29/。

这些 cohort 均复用 source manifest SHA256 aa56c083de1404dcfc744b31edc3dcb973f574d3dcb0d048c691d8ffa400d383 与 R24 串行单 rank prewarm 证据，但都没有得到 Jittor FSDP2 full 的训练终态；R25/R28 虽启动到 worker setup，也没有可接受的模型更新或恢复比较。MAIN-r28 双卡 Jittor FSDP2、prefix/resume 与配对性能仍未验收。
