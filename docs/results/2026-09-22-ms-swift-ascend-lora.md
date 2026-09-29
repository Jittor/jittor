# 2026-09-22：ms-swift LoRA 的 Ascend torch shim 验证

**状态：锁定 tiny 单卡公开训练已完成前向/梯度对拍、完整 checkpoint 新进程恢复和精确续训；tiny 公开推理 L0/L1/L4 已通过。真实尺寸推理 v4 严格数值对拍仍失败，逐层诊断定位中；分布式训练未完成，真实多机受资源阻塞，所有轨道 L5 均未通过。**

已有运行基线为 `69c3bdfd80e67cfbbacb5192f59ae859f86f4df0`，另有各运行 manifest 所记录的未提交修改。当前工作树已同步到 `61294cd14673ba60f4072542a2e73ea2c8f8c509`；后续公开入口与断点运行采用该同步基线及各自归档的 dirty source；没有把旧运行重新标记为新提交验证。维护者：Torch compatibility / ACL backend maintainers。核心初始化、梯度状态语义、依赖版本、后端/驱动或协议变化时重新验证。本任务不提交、推送或创建 PR。

## 验收范围与当前状态

本报告采用[统一验收合同](../../agent/skills/ms-swift-torch-compat/references/verification.md)的飞书唯一刻度：L0 导入/构造，L1 前向/推理，L2 训练/梯度，L3 完整 checkpoint 保存、全新进程恢复并续训，L4 真实公开 ms-swift CLI/launcher 端到端，L5 正确性通过后的稳态性能。后端身份、NPU 驻留、分布式 HCCL、严格零回退、有限值和精确键/形状是每层门禁，不独立占一个等级。

| 轨道 | 当前证据 | 未完成项或阻塞 |
| --- | --- | --- |
| 单 NPU 训练 | 锁定 tiny case 的公开三步对拍、完整 checkpoint fresh-process 精确续训通过，覆盖 L0–L4 适用门禁 | L5 真实尺寸稳态性能未完成；不推广其他模型或浮点输入 |
| 单机多 NPU 训练 | 两卡真实 HCCL 四类通信通过；完整训练验收仍未完成 | 按资源分别验证 2/4/8 NPU；实际 HCCL、每 rank 梯度/优化器状态、完整恢复与公开 launcher 待验证 |
| 真实多机多 NPU 训练 | `resource-blocked`：作业 720 当前仅一个实际主机 | 需要至少两个真实主机且每主机至少两张 NPU；不能用单机多进程或主机别名代替 |
| 单 NPU 推理 | tiny 公开 Swift 推理 L0/L1/L4 通过；约 1.1B 公开入口执行完成但 logits/KV 严格比较失败 | L5 稳态性能未运行；训练 L2/L3 不适用 |

四轨分别验收，不能用单卡 case 或通信微检替代分布式公开 launcher。CPU 模型、对拍、训练与性能不在本任务范围。没有新增 adapter，也没有修改 ms-swift 掩盖缺口。已有记录证明的是下列确实执行的部分，不再沿用“单卡 L0–L5 完成”的总体声明。

## 已有运行环境

验证 ms-swift 自身 `swift.tuners.LoRAConfig` 经 Jittor torch shim / ACL 执行，与独立原生 PyTorch/torch_npu NPU oracle 比较；已有数值和计时运行均使用单 Ascend 910B3。

- Python 3.11.15；oracle/shim 使用独立 package site。
- ms-swift 4.5.2、PEFT 0.17.1、Transformers 4.57.6，两侧一致。
- 原生 torch 2.10.0+cpu wheel 配 torch_npu 2.10.0，实际在 NPU 执行；wheel 标签不作为设备证据。
- CANN 9.0.0、driver 25.5.1；官方镜像 digest：`sha256:8742779115ff73944113f5826836c7aa538757c7401e1a636696d184dba8bc4e`。
- 同完整初始 parameters/buffers、同离线固定输入；已有运行键包含源码、版本、设备、协议、权重、输入、环境及包装器哈希。并行进程使用独立 JITTOR_HOME/可写缓存；串行可独占复用已完成的编译缓存，来源记入 manifest，训练及恢复仍使用新进程。相同键复用保存的结果。

原始文件均在 `$TASK_STATE`，不提交日志、NPZ、模型、容器文件或缓存。精确环境见 `session/host-environment-facts.json`；各运行 `manifest.json`、`source.patch`、`source-untracked/` 界定源码范围，不能仅凭共同 HEAD 把不同运行当作同一源码结果。新增四轨运行按环境合同补齐拓扑、每 rank 固定数据划分、全局 batch、优化器和 checkpoint 等运行键输入；不改写旧 manifest。

## 按统一刻度归属已有证据

| 证据归属 | 结果与实际范围 |
| --- | --- |
| L0/L1/L2 定位用例：`ms_swift_lora_llama` | Swift 构造、完整 NPU 前向/反向及独立 oracle 对拍通过。输出误差 0；8 个必需 LoRA 梯度最大归一化误差 `3.7604574904210894e-7`。21 个主干参数保持冻结；integer input_ids 输入梯度不适用。40 项 NPU inventory、线程数 2 和版本/输入/权重条件通过。 |
| L2 多步训练证据：`ms_swift_lora_llama_adamw3` | 固定输入、3 次真实 AdamW 更新通过；83 项初始参数、逐步 loss/梯度/参数/update-delta 轨迹齐全。梯度和 delta 独立按对应 oracle 幅度归一化，避免大初始参数掩盖错误更新。逐步 backward/update 驻留及冻结集合通过；允许首步 LoRA A 数学零梯度，要求整体确实更新。实际 harness 容差：loss/参数 0.002，梯度/delta 0.01。该证据不包含完整 checkpoint，不能归为 L3。 |
| L5 所需的历史测量材料：约 1.1B FP32 Swift LoRA | 3 次预热与 10 次完整训练步、177 项产物及真实 NPU 驻留比较通过。未完成 L3/L4 前置，尚不能通过 L5；实测数字保留下表。 |
| 所有已执行 case 的候选后端/回退证据 | 实际 ACL 身份、单 NPU 驻留、`fallback_policy=error` 和核心计数 0。它们是已执行阶段的硬门禁证据，不构成独立 L5，也不能推出未运行轨道通过。 |

原生端没有已验证的通用 fallback counter：较新的原始报告使用 `None` 表示未提供，绝不是零计数通过。早期单步 oracle 报告中的 0 是旧报告占位，不能视为原生实测，原始产物保留不改。原生 NPU tensor residency 与已知 CPU-fallback 告警检查是独立证据，不替代通用计数。候选的零计数来自真实核心接口和严格策略。

候选尚无证据证明每个分布式 rank 的 hostname、global/local rank、world size、NPU ID、HCCL、梯度、优化器与 checkpoint 状态符合新协议，也尚无完整训练状态的全新进程恢复与续训结果；原生参考的新增证据见下文。上述旧证据只按其已覆盖内容复用，不能补推这些缺项。

## 新协议公开入口与实际断点（01:20 更新）

以下为同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 上的新增证据。
原生通过只建立 oracle，不意味着候选相应等级通过。两侧锁定 tiny Llama（hidden 64、
2 层、2 heads、vocab 128）完整初始模型和 Swift q/v LoRA；FP32、HF32 关闭，
全局 batch 8、固定顺序、AdamW 三步。CPU 模型执行不在范围。

| 项目 | 实际结果与运行键 |
| --- | --- |
| 原生公开训练与完整恢复 | control `5863b9829ebcd8f8864f4af32e47c185384b48213a6dc132400d68b3c2685182`，fresh restore `d5f276cce0420015129475dd8bdaf5f988f9741f999dca3a01c2b4dc9be3ba9e`；比较 `2c07a6603be0ea89806fb7725d9ed4c0a29c53ddfc0984cf1763c448154f45dc` 全项严格一致。 |
| 原生公开推理 | `f99cfd9668f4ab1e36c76e51fb636976218ff9d161656f33fd4f7b0a33b01234`：实际模型及 adapter 加载，4 个请求/16 个 greedy tokens，记录 logits 与完整 KV cache；候选对拍未执行。 |
| 原生 RNG 恢复 | 汇总 `bda8ca45308e6f72f4629b113fc115fac69e0551d855d5b44acee1d811ff3047`：rand/randn/非零 dropout 的进程内、保存、独立进程恢复共 9 个阶段通过；不要求两运行时 RNG 字节格式相同。 |
| 候选公开训练首断点 | `7efb6e6d4fc373337faa4384cbf11aeb854ca97f37cd01eaa7b2eefb02742c4f`：核心编译完成，Transformers from_pretrained 的真实 meta Tensor 缺失；同时观察到 NPU 被发现为 cuda 的设备接口问题。 |
| 候选 RNG / launcher / 确定性断点 | `8e7848e5471eeadd752fa2165603c841dda6c5b5d4aa92fffc65057119da9fad` 缺 NPU RNG checkpoint API；`917b4148d36a288bb6ef894c592a30e638aae350696cd2883939e577b6d5e547` 缺公开 torch.distributed.run；`e289dddff5e921c6ef5cb1f533a44259d27f1bb30fd0a6b0013b30be6eb67bfd` 确定性 setter 是空操作，CANN 独立查询仍为 0。 |
| 两卡 HCCL 微检 | `13538b5180ab2cc6a9a4924a2f56859ddbaca88eae1a73187f58ad23646b9608`：原生 4 项通信通过；候选真实 HCCL 初始化/all-reduce/broadcast 通过，all-gather 因 compat 未分派至已有 HCCL 算子失败。候选已执行阶段 fallback=0，整体微检失败。 |

原生完整 checkpoint 包含 Swift adapter、验证未变的完整 base、optimizer/scheduler、RNG、
trainer step 与数据位置证据。为让 HF Trainer 正确识别 Swift adapter checkpoint，证据工具
生成标准 safetensors shard index，映射到实际已有 adapter 文件；没有修改下游库。
恢复由公开 `resume_from_checkpoint` 完成，插件只观察并核验初始恢复状态及续训轨迹。
未开启 `full_determinism` 的原生对照出现约 `9.54e-7` 的 loss-only 差异；严格比较如实失败，
没有放宽容差。双方公开训练协议已锁定 `--full_determinism true`，之后原生严格比较通过。

上述断点对应的核心 meta/RNG、ACL 确定性、compat NPU facade/launcher 与 HCCL dispatch
修改正在整合验证，不能据源码存在宣称支持。原始证据分别保存在 `$TASK_STATE/delivery/`
的 `runs/`、`inference-runs/`、`api-probes/`、`hccl-runs/` 和 RNG 汇总目录。
真实多机仍缺第二台 Ascend 主机；没有模拟第二主机或把元数据合同检查写成多机通过。

## 公开入口基础能力与当前断点（2026-09-22 核验）

同步基线 `61294cd14673ba60f4072542a2e73ea2c8f8c509` 加各 manifest 记录的 dirty source。
`delivery/foundation-runs/6ed8ba92ad53ecaa5dba198ecbfbfcb6bc85ce60ec3d3e2d545e1711fba767d7`
通过五项真实 ACL/meta 检查（5 executed / 5 passed），并通过真实 NPU 默认 SFRL pool 峰值生命周期、reset-live 及原样 Swift 内存查询。fallback_policy=error、fallback_count=0。
meta 覆盖无存储构造、默认梯度状态、Parameter/clone/detach、assign 与 to_empty；不包含 CPU 模型计算。
峰值由 NativeRuntime 的真实分配/释放事件维护，范围仅直接 raw backend→默认 SFRL pool，排除 workspace、驱动及 HCCL 内存；单可见设备执行，不声明多设备隔离已验证。

已保留的公开 Swift candidate 前置失败：
- `0ad826750685f89973b11e3b85b706641ce42d9fff18cd19d8ca71c2ba1667e0`：Transformers eager 导入仍要求可导入的 `torch_npu.npu_fusion_attention`。compat 仅发布明确 UNIMPLEMENTED 的失败入口，未增加融合注意力能力。
- `67e1933e026844841bf6ccc826fae9ce0d6f4ddb91cb5ada03ce629dd5e738e7`：base 模型加载成功，device_map 为 npu:0；数据预处理 Manager 因 shim 覆盖短 TMPDIR 而报 AF_UNIX path too long。使用已有 JITTOR_TORCH_KEEP_TMPDIR=1 的启动环境选项解决，不修改下游。
- `ffea416466d0f1411b8942cde8999e3847552d488017f7b490932e070cf3d609`：进一步完成数据预处理及模型/adapter 准备；打印模型时 native extra_repr 假定 compat LayerInitializer 有 __code__ 失败。此运行未进入有效训练步，不算 L4。

运行键独立保留；串行复用已完成的 67e193 JITTOR_HOME 仅复用编译产物，训练进程全新，原失败不会升级为通过。内容依赖校验未关闭。

维护回归 `06f9c8c5fb489b5995381d7e692ec6935c6f7f239909e7e2fb959ec1066bdd65` 实际执行 8 项，8 通过、0 skip，覆盖 meta、设备身份、确定性与默认 SFRL 峰值生命周期；严格 ACL 回退策略为 error、计数为 0。修复了仅构造、尚未使用的 SFRL 描述符导致峰值账本永久拒绝的问题。此前 `a13164f914a39b8bfa24aca09e508afb9ec268317b60e4aab9b917bcfd0105c5` 的 7 通过/1 失败记录原样保留，不重标为通过。

compat 默认 `extra_repr` 修复的元数据定向检查 `4feb1222a3f5d30cc4d56cf61a2802c6c0684105bc03d4a3d57d5d240710997b`：3执行/3通过/0 skip，不构造张量或执行模型。

| 新增候选基础能力证据 | 已验证范围 |
| --- | --- |
| RNG 汇总 `0169ea0215bcfc4a8a5e0546b4956af195df9df70d8ceec392b092c851e34978` | rand/randn/非零 dropout 各自的进程内恢复、保存、全新进程恢复，共 9 阶段全部通过；另一个坏 payload 检查通过，拒绝后状态及后续随机数逐位不变。仅单可见 NPU，不推出多设备原子恢复或完整训练 checkpoint 通过。 |
| truth reduce `91c31989aa3adefd8a1b3ba97a4368efd5fa0ae45d498d3853e93d8fadb42546` | 真实 NPU truth-reduction dispatch、keepdims 与 isin 定向检查 1 执行/1 通过。 |
| bool setitem `52a48eabd0fe9239dcfc2310f4b3adef9815f5e6412385e07aef633ab86b58f8` | 真实 ACL 布尔切片赋值及精确设备数据检查 1 执行/1 通过。 |

这些候选检查均保持真实 ACL 驻留、error 策略和 fallback_count=0。RNG 各阶段仍是独立新进程；串行独占借用已完成运行的原编译缓存不改变运行键、产物校验或失败结论的口径。

候选公开训练仍未通过 L3/L4。`eafd149a7d2b1d87c48ee283c132a69d07972808986ece68b9a06f807fe5c0b3` 的 optimizer.param_groups 泄漏内部 native Var、`119b8a297e14660cc11f25ced08d84362afd13a67ebc608107be09671d96938d` 的恢复后 step dtype/device 失败均保留。修复后的真实 NPU optimizer 检查 `0266c5b109b13a5430ab6ae11f22147211862936a344785b93645dda5515d76a` 通过；与原生 `8b135db2331a8087687de74f961f09575a95569ec99d1754d9eb907afa9069c3` 的比较 `4472300f7709b3c7f67f3777fcbdd1f38cd3c8f58f8ec360a395f68f8ead87f8` 通过，30 个浮点值最大绝对差 `1.49e-8`，跨运行时比较按原定容差，双方同进程新 optimizer 恢复后的继续更新各自精确一致。维护回归 `1dedc18ab4b82958aa923d665e6ec24441e6a6abcf47ae697cb8b0afb4cc30aa` 为 1 通过/0 skip；这些不替代全新进程训练恢复。

公开训练 `04ff0f3c96bae2adee8bddd7ede4560fdc131b7d1af6f0fe5e3c6d65fa20ed6c` 随后因 scheduler.state_dict 含函数对象失败。LambdaLR/MultiplicativeLR callable 状态保存恢复、initial_lr 与真实初始化/get_lr 上下文修复的 7 项纯元数据检查，原生 `ece21d60b3ac1600c62287ce05179a792d00f7c5a5990467fa154edefc6afc95` 与候选 `c758d788387e501fad0a600b9fb09cbb255ded9e14b73f800e6ec33ab9064754` 均 7 通过/0 skip。未构造模型或张量；公开训练仍需新运行验收，不能将该失败重标成功。

独立单 NPU 公开推理已通过 L0/L1/L4：原生 `01b7fedc778aea38328103c079416bbd1bcafdfb0d30041dcb7cb499f3ce7179`、候选 `7af2b00b150da0d59f07d0ce2ccf62d8f2b12011f568114c9b2a8072206ea532`，审计键 `862472f6603a4ec4e8df0e970701464d49525f2a3d5967034bb1772a7c6d8d8b`。v3 协议按实际运行时分类持久参数与派生非持久状态；锁定模型及 adapter 加载完整检查、4 个请求的输出结构与 41 项张量快照通过，logits/cache 最大绝对差 `1.1920928955078125e-7`，包含 EOS 的真实 greedy token ID 精确一致。候选真实 ACL、error 策略、fallback_count=0；原生无统一计数器，保持 unknown/null。L2/L3 训练项不适用于该推理轨道，L5 未运行。此前 `77481e34fdd42326b13a82307a672e0da77010693ca29f7e203192b0251b9f64` 的 BOOL 切片赋值失败及旧比较失败保留，不以新协议改写旧运行结论。

真实多机启动合同元数据检查 `d8bcd38e2bc5800ca8aec9a58516fa468aee0b3504af288b27a67d65596efb66`：6项通过，只使用标准库；实际单主机资源检查退出78且未启动worker。真实多机仍资源阻塞。

## 最新增量证据（2026-09-22 核验）

以下各项保留自己的 manifest、dirty source 和失败产物；同步基线仍为
`61294cd14673ba60f4072542a2e73ea2c8f8c509`。运行完成、定向回归通过和完整轨道验收
分别记录，不以局部修复结果推定公开训练或恢复已通过。

| 项目 | 已验证结果及边界 |
| --- | --- |
| 两卡真实 HCCL | `fbb22a3a15f2e2aebe38f8b2da34c4f7b05291a92dbed4e9c0cf1fc369b0722c` 的 `attempt/comparison.json` 通过：两 rank 实际执行 all_reduce、broadcast、all_gather、barrier；候选严格 error 策略、fallback_count=0，原生 counter unknown/null。它只证明单主机两卡通信，不证明分布式 Swift 训练、checkpoint 或公开 launcher 通过。此前 `13538b…` 的 all_gather 失败保留。 |
| 合并维护回归 | `3de8eff27405164f16960255b784fdf19de0665925a344e90bbf1346325406cc`：11 collected / 11 executed / 11 passed / 0 skip，error 策略、fallback_count=0；postflight 确认源码及包装器未变。覆盖 meta、设备身份、确定性、默认 SFRL 峰值、truth reduction、bool 赋值及 AdamW 公开状态/恢复。 |
| 布尔索引及梯度 | `9f47403bd65f5872e15f8549f4388bbfa63ca3c8dcc0820018a8c1737e32c553`：真实 ACL `test_npu_single_bool_index_with_full_slices` 1 执行/1 通过/0 skip，error 策略、fallback_count=0。该定向结果不代替完整训练回归。 |
| 公开训练的当前失败 | `4a789c193c61b32689e362f4a232b1ae01c9eaf3e749d8fa68db444c0df099c2` 已执行两次实际 optimizer 更新，step 2 保存后 `on_save` 观察器拒绝 `parameters/value2` 非 NPU 驻留，整体退出 1。不能把已经生成 checkpoint 文件视为完整保存/恢复通过。 |
| 前两步离线诊断 | `93cc0163388ddd787a57a29f89e429168e9f39d1074fca6cdd80363695b9b4dc` 只核对该失败运行已存在的两步轨迹：29 初始参数、8 可训练参数，键/形状及实际输入等精确结构检查未发现差异；loss、logits、梯度、参数、更新和 optimizer 有逐类误差统计。报告明确 `numeric_pass_fail=not evaluated; no predeclared cross-runtime tolerance`，没有数值通过判定，也不授予 L3/L4。 |
| 序列化驻留修复回归 | `d98d50adf6b11ebe82aa6cc1c758d91b2bfa8062ac8b8791b7525bea20cbf3c1`：真实 NPU `test_npu_serialization_preserves_live_state_dict_residency` 1 执行/1 通过/0 skip。在 error 策略和 forbid_backend_fallbacks 下执行，保存的计数证据为 `fallback_delta=0`；不把增量字段误写成未记录的绝对计数。仍须重新执行公开训练保存、全新进程恢复与续训。 |
| 真实尺寸原生公开推理 | `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26`：约 1.1B FP32 模型、Swift q/v LoRA，经真实公开 CLI 完成 4 请求/16 forwards/16 greedy tokens，各 prompt 521 tokens，退出 0，完成文件及快照哈希齐全、postflight 通过。只建立原生正确性执行证据；候选对拍和独立稳态性能尚未通过。观察器含同步/D2H，运行耗时不是 L5 结果。 |

原始证据分别位于 `$TASK_STATE/delivery/` 的 `hccl-runs/`、`maintenance-runs/`、
`api-probes/`、`runs/`、`partial-training-comparisons/`、`real-size-inference-runs/`。
真实尺寸 fixture `fc8a8deae25e7fcc7e97ae0cb6a390926a95c9c2a3c3f36a4064f2622ddedc30`
复用历史锁定初始参数，原生 NPU 构造并导出；独立计算的非持久 buffer 未注入。
它的 tokenizer 词表为 32000，不能复用 tiny case 的 128 词表解释输入。
这些新增结果不改变 tiny 公开推理已通过、公开训练 L3/L4 未完成、四轨 L5 未通过及
真实多机资源阻塞的总体边界；CPU 模型测试仍未运行，也不声明 CPU 支持。

## 原生单机多卡公开参考（02:10 更新）

2/4/8 NPU 的原生参考均经过真实 `swift.cli.main → torch.distributed.run`，每 rank
记录实际 HCCL、设备绑定、完整模型/梯度/优化器/RNG 和 checkpoint。全局 batch 8，
每 rank microbatch 1，分别累积 4/2/1 次；三步控制运行及 step 2 保存后的全新进程续训
均执行。严格恢复比较包括 raw microbatch loss/logits、全部可训练梯度、参数/update、
优化器/调度器、RNG 及实际下一批输入，未放宽容差。

| 原生规模 | 控制运行 | fresh restore | 严格恢复比较 |
| --- | --- | --- | --- |
| 2 NPU | `05f8013d9a5b0396940e15e8da1cf56b58eb77e5d63adf2fcdc5cc60d0d51058` | `6b799fa1a1357e2886309127f217d19245f4496b2276e08524676c4a0742184b` | `7c5d52f528a321c033515f6fd9d124e1fb86b5c368f78f1288c0cd5f0d703d22`：通过 |
| 4 NPU | `e30f43651fe2511c12d0c9f47f559edf777f9eee8b16e46d56a34885f86b3daa` | `daedc1022a4d40ece9c7f490aa98d8ed4ea19eb39a805e11fc47df6ab4ffa30b` | `0f18b4ff9fa535d9dec9b5e799eb54c653ad3a17deaedbdb56b36ddeb0a74117`：通过 |

八卡控制运行 `f384f5d90bf6815aacb2ebaaf7581c12d46eb6c6d5664cc3fcfb5dfbfae2e42f`、
恢复运行 `9befd9d3dee6ec45af64caa5d38d871a73e9935da2a48d466f7b114c18579f85` 均通过；
严格恢复比较 `86cebe5de67d423cf9edcb4ac2fc42e0be211465f4573909459d6f882bdf291c` 通过。
八卡前置 HCCL 微检 `6978c37ee93ea34ee38e21855d215a0876d851b5bb3daac39cca0b1650962d3d`
在所有 rank 完成四类通信。八卡窗口独占设备，候选计算在 worker 全退出后才恢复。

实际分区另行对照单卡的固定 token 输入：rank r 在第 s 个全局步骤、第 k 个本地
microbatch 消费记录 `8*s + world*k + r`（从 0 计），控制运行共同覆盖 0..23，
续训覆盖 16..23。各 rank 同步后的梯度和参数一致。这是输入分区与同规模恢复证据，
不是跨 world 的数值等价结论，也尚未覆盖同步前本地梯度。

启动时须使用 CANN 9 接受的 `HCCL_DETERMINISTIC=true`，并采用短的作业本地
`TMPDIR` 避免 AF_UNIX socket 路径上限。公开数据集 barrier 先初始化 communicator，
之后 Transformers 构造 Trainer 时将环境变量写为 `1`；起止记录保留实际变化，未修改
下游或截获写入。成功结论不推广到后写值之后新建的其他 process group。此前错误值
与过长临时路径的失败分别保留在 `012da5ed…`、`0cdb7f45…`，不记为通过。

详细证据见 `$TASK_STATE/delivery/native{2,4,8}-public-result.md` 与各 run 的归档。
候选单机多卡对拍仍未完成；真实多机继续受资源阻塞。

## 保留的真实尺寸测量

已有测量完成 1,100,611,584 参数的 FP32 Llama / Swift LoRA 训练，563,200 个可训练参数，batch 1/seq 512、eager attention、HF32 关闭。3 次预热后测量 10 次完整 zero_grad/forward/loss/backward/AdamW/device-sync；没有把 D2H 快照放进计时。177 项 loss/最终参数/梯度产物通过精确 key/shape/dtype 与数值比较，性能产物归一化误差容差为 **0.02**。时间统计已由原始样本重算。

| 指标 | 原生 torch_npu | Jittor ACL |
| --- | ---: | ---: |
| 中位训练步时间 (s) | 0.1469488514 | 0.6504897987 |
| 最小时间 (s) | 0.1463247468 | 0.6423129011 |
| P10 / P90 (s，线性分位) | 0.1467983386 / 0.1471413851 | 0.6438713219 / 0.6545085962 |
| 输入 token/s（512/中位时间） | 3484.206 | 787.099 |
| loss token/s（511/中位时间） | 3477.400 | 785.562 |
| 结束时 allocated/live (B) | 4,475,036,160 | 4,477,397,504 |
| 结束时 reserved (B) | 8,023,703,552 | 11,530,141,696 |
| 预热后峰值 allocated (B) | 7,316,457,472 | 未提供 (None) |
| 预热后峰值 reserved (B) | 8,023,703,552 | 未提供 (None) |

候选中位耗时为原生的 **4.42664 倍**。这组数字只证明所列 case 和测量协议的历史结果，不表示性能领先、达到额外速度阈值或四轨 L5 已验收。候选首个预热步约 **123.7077 秒**，冷启动成本单独保留，不混入稳态中位数。全部 10 个样本及 3 个预热时间保留在原始运行报告中。

原生内存来自 `torch.npu` allocator，峰值在预热后重置；候选来自 `jt.core.device_memory_used/reserved(0)` 同步边界样本，峰值未知，不能据此比较峰值节省。候选真实 ACL 身份、warmup/final 及每步 loss 驻留、error policy 和实际 fallback 计数 0 均通过。被比较的最终运行已在同步前创建并物化返回的 detached loss，避免 lazy 值尚未驻留造成错误证据。

## 已验证修复与维护边界

1. 核心两处 atomic 计数直接流输出在 GCC 10 下产生歧义，改显式 `.load()`；相同 translation-unit 检查及后续完整核心构建通过。
2. accelerator 静态初始化触发 sync_all，访问未构造的全局 fetch 队列。两条队列改由 NativeRuntime 首次访问构造，保留 pending/deferred 生命周期和 cleanup 顺序；结构检查及真实 NPU 回调次序、重复 sync 不重放、第三次 fetch 回归通过。不据此宣称任意回调重入/退出竞态安全。
3. 维护的 shim deploy 补齐缺失 Torch distribution 元数据；未私造兼容包。metadata API 2.11.0 与默认模块 Jittor 版本不同，属于现有发布策略。
4. compat `_ip` 改读取真实 requires_grad，修复 host-to-NPU copy_ 在 no_grad 中解冻参数。8 组合 Parameter/Linear、host/NPU、冻结/可训练及 connected inplace 梯度回归通过。没有泛化到所有 `.data` 或启用梯度时的冻结目标写入。
5. runner 显式将 from_numpy 输入迁至 NPU，保留核心对混合设备的拒绝。真实 Embedding 回归通过。
6. `torch.get_num_threads()` 改委托核心 OpenMP 查询，修复 CPU 数量 192 误报为线程数的问题。真实 OpenMP 2→3→5 及委托检查通过；setter/interop 未因此获得实现。

5 项真实 NPU 回归、17 项证据 metadata 及 2 项线程检查已有通过记录。已有门禁按白名单执行：结构目录包含 CPU 前向/梯度/对拍，不能整个运行并称纯结构。选中的 C++ ownership/OpenMP 检查是元数据/结构验证，不是 CPU 模型测试。

**已有定向门禁：** `checks/36933c7f2428b9a58b5a6a2dbf00fef7c9688b54d728b6de4b80a24e1a1c3ccb`：81 collected / 81 executed / 81 passed / 0 skipped，17.80 s（不含首次编译）；真实 NPU 回归期间 `fallback_policy=error`、核心计数 0。覆盖受影响的 fetch 生命周期、复制/原地梯度、设备迁移、线程查询、多步/性能证据合同、当时的文档及后端结构检查；当时布局检查和 `git diff --check` 通过。这是原基线记录，不代表当前同步提交或本次新验收合同已经重跑门禁。

## 可追溯证据

下列键分别位于 `$TASK_STATE/runs/` 或 `$TASK_STATE/comparisons/`。原始路径、脚本名及产物内历史等级标签不改写；下表按工作负载识别，验收等级以上文统一刻度为准。

| 角色 | 完整运行/比较键 |
| --- | --- |
| 单步 oracle | `169c603c50d5e964bd72ecc42b93add81d6f85219a51baad58025727c6d61050` |
| 单步 candidate | `7fe94c94b5c193995015dd68d4b090047378cf50dbefdd9d39a68ab73bcde52b` |
| 单步 comparison | `7cd7db6a32a6bbe9d64e3bcd004cc00a2c6df85e619a0f6dcbd4e0327dc2829e` |
| 三步训练 oracle | `26cd043896ff6ec1444c5326ebb3ca695a7e65798e06e77916bc4763e45e1c4e` |
| 三步训练 candidate | `6edc0ea84b3dd7a0002adbb6553817ec3ea8794ed3dcc6f88417157e75e5a024` |
| 三步训练 comparison | `9f611125ddda3f8cff0e8bbd73b4d97b727d6d6f710c7312d8a8f2520cb6f85a` |
| 真实尺寸测量 oracle | `b6b279d397337080bb94a6321a68d57ab0f5a1df4d348d0f369725c7f0c154db` |
| 真实尺寸测量 candidate | `8adf387f8708d531c0e2514c2fc216032551e083f327669a355a585bf8028e44` |
| 真实尺寸 measurement comparison（修正容差字段后最终结果） | `b8e312b7b1d7b4dc38342ee7fb689f7658bb2ba71e62c7ca8c361c28e09ecf9d` |

单步 oracle/source 指纹为 `e0ec299638c5c4126473c618dfa4cbc4934cf291632c03b7a0200c88f67a4d54`，候选为 `7d419fee6bce4c4665e601e8eaf73d76430c4dbaaf2d9eb116be45b40d108d78`；保留原生参考与修复后候选各自 manifest，不声称源码相同。三步训练双方 source 指纹均为 `f77b84977411919a22af8dcbebd7264a24df6a858c96aeed91da2ad29c688f1b`。真实尺寸测量双方 source 指纹均为 `e9eb4f23ba9728c7dbb8d32c87fba60f50b73c0e965bd844c29a029fe1acf215`。比较 manifest 还哈希实际比较器和全部输入产物。

## 历史用例重放接口

先检查运行键和产物完整性：没有相关变化就复用，不能为了重述等级重复模型计算。`TASK_STATE` 指向本次未版本化证据目录；`ORACLE_RUN`、`CANDIDATE_RUN` 取上表对应完整运行目录。所有 Python、编译和测试均通过 `srun --jobid=720 --overlap` 执行；登录节点仅做 Git、文本与编排。更改源码前停止同一工作树已有模型/编译。

```bash
# CASE 取本报告三个显式 case 之一；新键确需运行时先 oracle，再加载完整 weights 运行 candidate。
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" torch npu "" "$2"' \
  bash "$TASK_STATE" "$CASE"
srun --jobid=720 --overlap bash -c \
  'source "$1/host-env.sh"; bash "$1/run-case.sh" jittor npu "$2/result.weights.npz" "$3"' \
  bash "$TASK_STATE" "$ORACLE_RUN" "$CASE"

# 历史脚本名不代表当前等级；compare-l3.py 比较的是三步训练，现归 L2。
srun --jobid=720 --overlap "$TASK_STATE/host-oracle-python" \
  "$TASK_STATE/compare-l3.py" "$ORACLE_RUN" "$CANDIDATE_RUN"
srun --jobid=720 --overlap bash "$TASK_STATE/run-final-gate.sh"
```

单步、三步、真实尺寸对应的历史比较器分别为 `compare-l2.py`、`compare-l3.py`、`compare-l4.py`。包装器保存精确命令及 manifest；失败退出码、缺产物、非有限值、缺梯度、dtype/shape 差异及驻留不符都拒绝。相关源码或协议改变时生成新键并保留原证据。这些内部 case 的重放不代替四轨真实公开入口验收。

## 公开训练对拍及恢复断点（04:25 更新）

新 `public-training-parity-v2` 协议在运行前绑定分类容差、比较器及说明文件；
初始序列化参数、optimizer step 和组控制状态精确比较，Adam moments 按预声明容差比较。
候选 `f976d62b228464b1bfb5db14d048875eba67d2b43e64e68140a83fc3e392a332`
与原生 `9a6afcfa43e316f04b2bb1b8f505e123a4a25fe78adf52925d3f67610ab40395`
均完成真实公开 Swift 三步训练及 checkpoint-2/3 保存。离线运行键
`16f3b0b36ad4e1756300462c4866cc205d06e7cb3f74c9585a90236342d5fa5a`
对拍通过：24 个固定 microbatch、全部可训练参数梯度、更新及优化器状态；候选记录
ACL、error 策略、fallback_count=0。输入为整数 token，输入梯度不适用。
这补齐公开训练的 L1/L2 证据；因 L3 尚未通过，不宣布训练 L4 完整通过。

原生 fresh-process 恢复 `a3f7f0bfc7c5396dd6e6ceb83d123ed99f5cc6d3e42fb39956b48c6d3b907c36`
经离线键 `24f86a62135fc2cc0598a3f028d64f672a477b3a5e0bcf089844640263922969`
证明第 3 步与不中断参考精确一致。候选恢复
`474f9c6f6709ab4096738503bbf4db0d13ab2de772a7ca1924b27a382848b2d9`
在恢复 host RNG 控制状态时报 `RNG host engine state mismatch`，整体失败。
独立宿主控制解析复现 `ccfdf21578f79a10171c75b3919c4d10fd6ba71ef7399fed1685edf4b1b5e300`
确认原解析拒绝有效状态、显式消费分隔空白后接受；它不是 CPU 模型验证，也不代替修后 NPU 恢复。
此前 `0169ea…` 九项 RNG 结果覆盖 NPU counter replay，并未覆盖这条 host engine 恢复分支。

单卡 hook 微检查 `8f182ec2b3e2aafa1e2ddb31506af90fbfb5800b4d57efca2533d96564f48b95`
中无 hook 的两步训练通过，但注册 hook 使同一 Parameter 的真实 leaf 标志由 true 变为 false，
观察组失败；后续两卡 hook 阶段未运行。不得将 plain 阶段或 HCCL 通信通过改写为分布式训练通过。

## RNG、真实尺寸数值及 leaf hook 增量（05:00 核验）

以下结论限定各运行 manifest 所绑定的实际源码，不将已通过的单项检查推广为完整恢复
或分布式成功。CPU 模型计算仍不在范围；RNG host 控制状态解析不属于 CPU 模型验证。

| 项目 | 实际证据与结论 |
| --- | --- |
| host RNG 恢复修复 | `25d0c76911f74f826502852e94bf65cb4a80b1054a44eaabed3b00aaacf0f32a` 通过：真实 checkpoint 的 torch.load / torch.random.set_rng_state 控制路径、自身状态往返和三类坏 payload 拒绝原子性均检查；ACL RNG 状态不变，error 策略、fallback_count=0。维护回归 `7fdf95ddd29a44cc56bb238194ebbcd29bf1af480cdb8534e95001eb21bc74f2` 的 `test_cpu_state_restore_does_not_reseed_acl` 为 1 执行/1 通过/0 skip，初始计数 0 且 fallback_delta=0。完整 fresh-process 训练恢复仍须重跑，不能据此声明 L3。 |
| 真实尺寸公开推理执行与失败 | 原生 `5f47d3f5d51baca172f57c1a9c44777fa85d1b321cbf4f771f617c7cfe8fcd26` 与候选 `d47623e80206f235e482634be57b0937a845f9f855874acdc1a9aa074f5f22b7` 均完成约 1.1B 公开推理、4 请求/16 tokens。但严格比较 `d69c8471117551bf55972e2eee631354f3f23c67d45cda161de5ffaa7ed4b4a3` 退出 1：首个 logits 快照在固定 rtol=1e-4/atol=1e-5 下 65/32000 元素超差，最大绝对差 `2.849102e-5`。执行成功不等于 L1/L4 数值通过，L5 不得启动。 |
| 全量离线数值诊断 | `0cf881e2d2e1dfb73f87b56eb24448e8a3ac3f9a42437c07e72c44175af1f775` 只分析现存快照：1618 个张量中 1220 个未通过原严格阈值；序列化初始状态精确、public responses/tokens 精确，元数据未发现差异。非持久 inv_freq 有 18/32 个数不同、最大差 `5.960464477539063e-8`，自身仍在预定计算容差内；最早已观察到的 KV 差异为 layer0 keys，最大差 `9.083747863769531e-5`、218 元素超差。没有逐层 activation，因此不能把“最早可见 KV 层”写成根因层；未放宽容差。 |
| Rotary 原语定位 | `328cd35fc81e78cdd4a1f08f1e3564bd1c52aff3c2ac9c1ad00e8cada76996df` 的双方真实 NPU 角色均执行完成；离线比较 `7b88d9dcaba71c25d6ce0e89393e55e8c442d9b2482f5261e12dc4b1b87e2161` 对 arange/cast、指数、pow、倒数、position matmul、频率拼接、cos/sin 等全部 13 个阶段逐元素精确一致。候选 count0、原生 counter null。该结果仅排查锁定原语路径；不能据此宣称完整模型数值失败已解决，也不能在没有复现时修改 pow/rotary 算子。 |
| 原生 leaf hook 与优化器提交 | `11f793b313d9be9aa75b3146d24f7fc324182dc768e59343c9d693c40782cc00` 已完成核心构建，注册身份及前两次 backward 的 hook/local-gradient 检查通过，但 SGD step 使 Parameter 成为 `grad_fn=fused_sgd` 的非 leaf，整体失败。共享优化器提交边界修复后，`453a803e6b68df16f1b3d62f7e9f4cf4c6c7440a6bacc2343210462491211475` 的原定维护方法 1 执行/1 通过/0 skip，error 策略，初始 count0 且结束 delta0；覆盖 leaf 身份、顺序/替换/移除、局部与累积梯度、更新后继续 hook，以及错误 shape/dtype/device 替换拒绝。非 leaf hook 沿用旧机制，实际两卡 hook、完整公开训练及恢复未由此通过。 |

优化器矩阵的前置失败也原样保留：`6e9911e55790fbcf2e2d541a0613ebd917d88c13ca32c43c53c6d2f66a5b2296`
因对无 momentum 的无状态 SGD 索引不存在的 state 失败；测试改为允许该明确无状态类别。
`0044d0890b76e35635af134bacea67b9e2152208d3a02d08b55eeda27d60fb56`
在 SGD 构造参数 `fused=False` 处失败。现有 shim 构造器不接受该关键字，属于明确 API 缺口；
覆盖便携提交路径可使用现有 parameter-group `fused=False` 开关，不因此宣称构造器兼容。
修正后的真实 ACL 矩阵 `7793dd80adc83474a50f90a097f4ad55ad5edabb50d4632f22e473bc5ac1f96c`
已通过：1 个维护方法、7 个 factory 各 2 次实际更新、0 skip；error 策略、初始 count0 及
结束 delta0。明确断言每类完整 state 键集合，覆盖 SGD 自动/便携 momentum、Adam、
AdamW 便携/fused、RMSprop、Adan 的 leaf/hook/冻结参数/状态策略及移除后第三次 backward。
该矩阵不掩盖 SGD 构造关键字缺口，不代替真实分布式训练或完整恢复。

新增原始文件分别位于 `$TASK_STATE/delivery/api-probes/`、`real-size-inference-runs/`、
`real-size-drift/runs/`、`rotary-primitive-probe/` 以及 `$TASK_STATE/evidence-audit/`。
总体保持：tiny 推理已通过其适用正确性/公开入口；真实尺寸推理数值失败；公开单卡训练
已存在三步 L1/L2 对拍，但候选完整 L3 恢复未通过，故训练 L4 未完整验收；分布式训练未完成，
真实多机资源阻塞，所有轨道 L5 均未通过。

## 单卡公开训练完整恢复（05:12 核验）

共享优化器提交与 RNG 解析修复后的新控制运行，原生
`37ea72fd6cf1bbb2b9911a1d473f666967845935308f9f4262ffe8252e4b0477`
与候选 `7dee183aee0fb4117c0ec3365036e952c5e5ecb000ce5de7ad3048703ec459d1`
均完成公开 `swift sft` 三步训练。严格跨运行时比较
`393d90698d94c489d15a344e5989bd330d54ca469321e1a38e8ec35caf1de784`
通过，未改变既定容差。

候选以全新进程从 checkpoint-2 恢复的运行
`5c9a9d5b7073b04c6005ab0560669a380f21feeb7ad3473280ebbc2d2cb12e4b`
完成 step3；精确续训比较
`3f33178fb0239e408463b362858926a9a198efa49ff19d2916d2b03e0bfc939e`
通过，比较覆盖 8 个 loss、8 个 logits、16 个梯度快照、96 个优化器数组、120 个参数数组、
62 个状态数组及 29 个参数更新，并核验完整 checkpoint 文件哈希、恢复状态与 RNG。
候选实际 ACL，fallback_policy=error、fallback_count=0，实际处理 8 个续训 microbatch。
原生 fresh-process 恢复 `c81dc2e430d5b4c61222d492088ed90789a015603663089e62db61e100a5ae66`
也由 `981578a35a71f30d868547c2aab970e881e2241b78c479eb47b1a1b231a34ffd`
通过同运行时精确续训比较。

这补齐锁定 tiny、整数 token 输入、普通单卡训练 case 的完整 checkpoint 恢复与公开入口证据。
输入 token/label 为整数，不把其不可微梯度虚构成通过；不推广到浮点输入或其他模型配置。
单卡训练 L5 仍未运行，分布式公共训练及真实尺寸推理的剩余门禁分别验收。

## 维护回归、结构范围与多机准备（05:26 核验）

维护的真实 ACL 组合回归
`abffd41cb89d077885ce7ae2f15ead5845c93e299da66b86488ac8acf27d9219`
为 15 执行、15 通过、45 个 pytest 阶段均通过、0 skip，error 策略且初始/结束 fallback_count=0。
覆盖 meta、实际 NPU 身份、确定性、显存计数、布尔归约/写入/索引、优化器公开状态与恢复、
序列化不改变现场驻留、leaf hook 及七优化器提交路径。原生 full structure 命令不是纯元数据：
`3fa2c39b4b6f9d271d9e75c3a4e70cd66c61d0a97fd1eafadc145d2d017a1220`
布局检查通过，但 structure pytest 在约 15% 时已有七项未分类失败，之后进入 native collect-only
编译；审查发现后续含范围外 CPU 数值用例，故仅中断该 pytest。未取消作业 720，没有将中断
或 skip 记为通过。其原因尚未逐项定位。

审查后的结构子集与布局检查
`ed81a9b938a7fa6f0afa7e37c0362325f94823a0fb819e81e65f873bce47d59d`
通过：38 执行/38 通过/0 skip，另有 1 个显式不选择的独立 C++ 编译项，不计为通过。
子集限定 holder 源码归属、fallback 策略、ACL 结构边界和文档结构；不声明全仓结构通过。

双卡原生公开 v3 证据已加入每 microbatch 的八个 LoRA 本地梯度以及完整输入可微性清单。
控制 `040da30a6c84e9a5f7adfcc5fa0b937dbd14ffc21224ef8c57a9dc67955fff98`
与新进程恢复 `e355fd1965719d7d51e04f864c506210aad3be82f53a59bfb75eb92b489c976c`
由 `c3304c115c84bec9b1490e0cf63b4599947c3c4ec824812e7aebfc71a8a3ebb9`
通过各 rank 精确续训比较；候选公共双卡尚未通过。v3 比较器的 20 项纯 NPZ/元数据正反例
`63b74f983ce11b558ad9276c0704cfcdfbf25b592f4a1ab502d4096e7615c787`
通过，不能替代真实分布式梯度次序验证。

真实多机独立实现已准备在 `$TASK_STATE/delivery/multihost-proposal/after-v2`，重基当前
v3 观察协议，包含真实 Slurm 主机核验、全局 rank 私有缓存、跨主机整池租约、HCCL 微检
及真实公共 launcher、完整 checkpoint 共享路径和逐节点退出核验。元数据检查
`de2c9f4b228cad66a8e7eceeeffbdbe0f9301c6e66eb8d04148ad13833d690a3`
通过，实际 allocation 仍只有一台主机，runtime_execution=false；没有模拟第二主机，
未声明多机 HCCL、训练或恢复通过。解除阻塞需要两个实际 Ascend 主机，每主机至少两张设备，
并实测驱动/CANN/网络和共享文件系统锁一致性。

## v3 完整训练证据与 v4 推理断点（05:50 核验）

以下是各 manifest 所绑定源码/协议的结果，更新前文对应轨道的当前状态；历史失败仍保留。

**单卡公开训练 v3。** 新协议逐 microbatch 记录八个 LoRA 参数 hook 的实际梯度贡献，
并列举全部位置/关键字 tensor 输入的 dtype、shape、requires_grad 和适用性。
原生 `e016f2f04b70328477872ccc53dee1fa4ac7884707a10a93b5850a0bb26b8798`
与候选 `5a590c596d9fb3dba46a573fd54757ddaeb805a079267e3dc7914467bf0a8c05`
由 `7c43d97d69cff0cb25aab81666d1b824a4e5f1f6cb818f70ce23f5f5e5fda8df` 严格对拍通过。
候选 fresh-process 续训 `11ce99d47af1386ac5c4cc9358d92adb6c4795782e9359f2bad292d740234622`
经 `417badbe996de264b994a43dcf47acdf0299f37b64f95699e86a2ad4dca56653` 精确通过；
原生续训 `6025abcb44db8ca606160c16ab8fa37ed3999a98f6125ede65875fda533a70fd`
经 `988b9465c2ab6a861329f58a0c98999f9183d011da9ee6035d80ed03136e8876` 精确通过。
输入是不可微整数/bool，明确 N/A；协议拒绝未覆盖的浮点或 requires_grad 输入，不能声明浮点输入梯度支持。
该结果补强锁定单卡 case 的 L0–L4，未运行 L5。

**原生多卡与候选双卡必须分开。** 原生四卡公开控制
`d147e1bbfccd8a26fff48de2d2a949f66d88bfedeca54e536970b829b269f71a`、恢复
`23a371f2f63ee8fd4c247e2b9cc575fe8dd9b6ebcb3bd2ae0fc436c1317f7d59`
由 `273675978e0423805b86f2df4c9b607cca99c40806b8603b43a915bf01df1d9f` 通过逐 rank 精确续训。
原生 2/4 卡控制及恢复的固定 global-batch 分片审计四项均通过，依次为：
`298dc9f900042898b05bf2be54333545bd1800964679a4d77d4c46e7a597d9d4`、
`34389fd62a1cba88f8bc35542e1b5f96278d4179f783ae8ebdf1a3d2c3171ff6`、
`ca4b33148c9c81c5e04e7e39efc92f4027c6888a7f379f8f19663f9ac81dd694`、
`9fa8efbb844a13ae9eaedc3c309ca3770034b0a84ffda6b44b6ff6faed79b520`。
这些是 oracle 证据，不代表候选多卡验收。

候选双卡 `938ecba806bf8d01c9a7e6de21692617786de9f0d2b4d4147adbc8e9a669aa8b`
完成各角色私有缓存预热及实际 HCCL 双 rank 通信检查，但真实公开 Swift/Accelerate 入口
在导入 `torch.distributed.algorithms.ddp_comm_hooks` 时失败。兼容命名空间修复已应用，
新键 `54683642cdd439e0b5341703117fbbc5900dcd7957dd83c87226663e35354e5e`
已越过导入断点，但在公共 Trainer 的 token 计数 gather 中因裸 `scalar[None]` 被 ACL 拒绝而失败。最小索引入口修复已应用，待新运行验证；尚无候选多卡训练/梯度/恢复通过结论。

**真实尺寸推理仍严格失败。** 显式共同 NPU 初始化 v4 的原生
`146b00bb6140fd8076c3bb16f651d97cf94cc51d40a00eb6b77c76d25c8c1f72`
与候选 `796e39e26587c85945fd4e0e493a59cba3a07173a07606c5b6547455b0cae1b1`
完成执行，但严格比较 `782de967bf7d8a903f9425b8ff8d225612732117f799ae299aa6c1460f4cd284` 失败。
全量离线诊断 `7f63b78b4f92bb96b1c795ea8f2eca0f5b1ba74430384fe9083453f7b37ba827`
记录 1618 个 tensor 中 34 个超出原定 rtol=1e-4/atol=1e-5；初始参数/buffer 精确，
公开响应与 tokens 精确，候选 fallback0，均不能抵消 KV/logits 数值门禁失败。
逐层 activation 诊断 `ccf303a5175cba78e1f4e6cddd1c906c67db6dedaf8022cd957197cefca66059`
确认双方观察前后各自 prefill logits 逐位一致；首个非零差异在 layer0 RMSNorm 输出，
最大绝对差 `9.5367431640625e-7`，该点本身尚无超差；首个已观察严格超差在 layer3 residual 输出。
这是定位证据，不是 Rsqrt 或其他算子根因定论，不放宽验收阈值，不启动 L5 性能。

**默认设备 API。** `fe7a1203de38cbd308b3c60b7b207b384d262f0f13d293b5e0d9eef92a973c72`
真实复现 `set_default_device('npu:0')` 后 getter 错报 `cuda:0`。修复位于 compat 的默认设备
报告路径，复用 ACL 类型解析并遵守实际 factory 上下文，不修改模型或 observer 掩盖错误。
定向真实 ACL 维护回归 `3650bcf86c25136e038a93482eb96394b4168afbb017c5db902e0e02b4a55376`
为 1 执行/1 通过/0 skip、初始 fallback0 且 delta0；CPU/meta 分支只查元数据，不执行 CPU 模型。

截止本节：候选双卡新运行、候选四/八卡完整 public 路径均未完成；真实多机仍缺至少两台主机。
所有轨道 L5 未通过。原生 torch_npu 无通用 fallback 计数器，继续记录 null；
候选 error+0 与真实 NPU 驻留证据仅对各自已经实际验证的运行生效。
