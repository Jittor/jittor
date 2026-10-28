# TRELLIS.2 dense 模型训练验证

- 状态：实际独立 CPU 与 CUDA L3 验证完成，六个场景中候选 fallback 均为零。这是限定的模型门禁，不是预训练 3D 流水线或性能验收。
- 日期：2026-10-15。
- 基线：`6d39728fffc37d4618c1cb91a9caa1d67beec311`（干净已提交门禁/runner；正式验证全程源码与 HEAD 哈希未变）。初始 CPU `ae1a9bf1f045b11cf49f1665ad4d4e5d9892f787` 与 CUDA 预热 `862e6395d614d70763fc4a6f5ef050200fbb2510` 作为更早的独立证据保留。
- 上游：Microsoft TRELLIS.2 `75fbf0183001ed9876c8dbb35de6b68552ee08bd`。
- 维护者：下游兼容层维护者。
- 复查条件：上游声明/模型源码、Torch shim 的 attention/Conv3d/normalization/autograd/optimizer 语义、后端精度/设备检查或本门禁变化。

## 范围与协议

门禁从干净固定的上游 checkout 构造真实导出的 `SparseStructureFlowModel`、`SparseStructureEncoder` 与 `SparseStructureDecoder`。其 tiny 架构保留实际的 attention、卷积、归一化与残差路径。dense attention 后端采用上游支持的 `sdpa` 选项，两个 runtime 均使用 FP32 参数/计算并关闭 TF32/cuDNN benchmarking。

Flow 模型的上游零初始化输出层会产生恒为零的初始输出。仅导入它或检查该输出无法确立数值支持。本门禁改为从独立二进制 Torch 进程传入确定性的非零 FQN 绑定权重与全部 buffer。它检查三步中每个命名参数梯度、每个被请求输入梯度、输出/loss、更新后参数与 SGD 动量 buffer。随后每个被请求输入被独立扰动，并重复完整训练比较；输出必须改变，其他输入保持不变。没有 adapter、monkeypatch 的上游实现、合成扩展或复制的模型。

oracle 导入真实 Torch C 扩展，并断言未导入 Jittor。候选在导入同一上游模型之前激活 shim。两个进程都记录完整源码身份/哈希、确切权重/probe 哈希、精度与线程/affinity 设置。CPU 在 import 前显式禁用 CUDA。accelerator 检查会同步每个被观测张量，核实其实际设备位置与逻辑 CUDA ordinal，并要求后端 fallback 策略为 `error` 且计数为零。

数值预算从现有生态比较原样继承：forward/loss/parameter/momentum 的 scaled relative 误差至多 `2e-3`；参数/输入梯度至多 `1e-2`。每个字段使用仅基于参考的尺度下限，等于最大参考量级的 `1e-3`，并共享 `1e-6` 绝对分母下限。它们并非根据 TRELLIS 候选误差选取。所有被比较数组必须有限，并具有完全一致的 schema/shape。

## 自动推导的覆盖

纯源码 inventory 枚举全部字面 lazy export map 与 sampler 导出，保留缺失定义与可选依赖路径。各独立轴不会相加成整库百分比。

| 轴 | 声明成员 | 本报告实际验证成员 |
| --- | ---: | ---: |
| models | 9 | 3 |
| pipelines | 2 | 0 |
| modules.sparse | 33 | 0 |
| samplers | 4 | 0 |
| datasets | 11 | 0 |
| trainers | 15 | 0 |
| renderers | 4 | 0 |
| representations | 4 | 0 |
| exported_modules | 4 | 0 |
| package_api | 2 | 0 |

在该源码 pin 上，有八个声明导出没有匹配定义：`datasets.TextConditionedSparseStructureLatent`；`modules.sparse.SerializeMode`、`sparse_serialized_scaled_dot_product_self_attention`、`SparseSubdivide`、`sparse_nearest_interpolate`、`sparse_trilinear_interpolate`、`encode_seq` 与 `decode_seq`。它们仍保留在各自分母中。抽象 `Sampler` 导出也仍保留在 sampler 轴。trace 调用不额外授予组件覆盖。

该 checkout 没有已发布的 package 版本字段。其 `source@75fbf0183001ed9876c8dbb35de6b68552ee08bd` 身份仅表示确切的干净源码提交，不代表发布版本，也不代表验证未来每个 TRELLIS 版本。

## 结果与环境

独立 Torch runtime 为 Python 3.11.16 / Torch `2.11.0+cu128`，带其真实 `_C` 共享库。私有候选解释器使用只读依赖与所述工作树中的 Jittor。NumPy 为 `1.26.4`，einops `0.8.2`，easydict `1.13`，两个进程中的实际上游源码 pin 相同。上游 setup 推荐 Torch 2.6.0；本报告验证已安装的 2.11.0 runtime，而不是单独的 2.6.0 组合。

正式 CPU：三个 case 通过，无 skip，44.57 秒。正式 CUDA：三个 case 通过，无 skip，61.65 秒。更早的 CPU 运行耗时 92.79 秒，单独保留。CUDA 单卡为 RTX 4090，驱动 580.178.04，NVCC 12.2.140 与 g++ 11.5.0；oracle 构建为 CUDA 12.8。这些是进程隔离的 runtime，并不表示工具链相同。每个基线分别比较 294（Flow）、300（Encoder）或 300（Decoder）个数组。独立输入变体使每设备总量达到 2,376 个数组对。墙钟时间为门禁时长，不是速度比较。

默认 Flow 路径最初在独立 Torch 环境中因缺少 `flash_attn` 而失败。选择上游支持的 `sdpa` 后可使用 dense 模型路径。实际上游 `SLatFlowModel.forward` 与 `SparseConv3d` 构造在独立参考环境中分别因缺少 `flash_attn` 与 `flex_gemm` 失败。另作的真实 import probe 发现 `FlexiDualGridVaeDecoder` 缺 `cumesh`，完整流水线 import 边界缺 `trimesh`。`SLatFlowModel` 与 `SparseUnetVaeEncoder` 的导入成功并不确立 forward/backward 支持。这些是首批依赖断点，不证明每个 CPU 路径都不支持，也不证明 CUDA kernel 可用。

Cu122 wheel 目录报告的 cuDNN 发行物元数据为 `9.26.0.51`，而实际选中的 `libcudnn.so.9` API 返回 `92400`（9.24.0）。CUDA runtime API 返回 `12020`。声明版本与实际版本都记录在平台证据中，并附所选库文件的哈希；未修改任何共享依赖。上述实际 CUDA Conv3d forward/backward 场景就是针对这一所选二进制组合通过的。替换这些二进制或解决元数据冲突需要重新验证。初次私有环境 import 失败，原因是 Cu128 元数据遮蔽了 Cu122 栈且找不到 `cublas.h`；隔离的候选路径已修正为在严格 wheel 检查下选择既有只读 Cu122 组件。失败日志仍是原始证据的一部分。

| 设备 | 模型 | 含独立扰动的被比较原始数组对 | 最大绝对差 | 最大 scaled relative 差 |
| --- | --- | ---: | ---: | ---: |
| CPU | SparseStructureFlowModel | 1,176 | 3.42727e-7 | 2.68249e-6 |
| CPU | SparseStructureEncoder | 600 | 8.82149e-6 | 2.48022e-5 |
| CPU | SparseStructureDecoder | 600 | 1.88276e-5 | 2.56591e-4 |
| CUDA | SparseStructureFlowModel | 1,176 | 2.30968e-7 | 1.99612e-6 |
| CUDA | SparseStructureEncoder | 600 | 6.43730e-6 | 1.35543e-5 |
| CUDA | SparseStructureDecoder | 600 | 3.94583e-5 | 3.93669e-4 |

最大列可能指向不同张量。两者都使用上述未改动的字段特定预算。每个 case 检查 30 或 32 个参数 FQN；Flow 模型请求 `x`、`t` 与 `cond` 的梯度，每个 VAE 请求 `x`。每个基线/扰动规格都检查全部三个训练步。

更早一次干净 `7493641ba8bc9ecab3d93f45141efaedde0aef3d` 运行也通过了三个 CPU 与三个 CUDA 数值场景，但其 Git pathspec 在源码元数据中遗漏了 `trellis2/__init__.py`（92 个文件）。collector 拒绝了该不完整源码绑定。那些原始 JSON/NPZ 文件与拒绝日志原样保留；上述被接受的矩阵是在源码枚举修复后重跑的，而不是修补旧元数据。

## 复现

使用上述完整 SHA 的干净官方上游源码 checkout、记录了依赖的私有 Python 环境与独立二进制 Torch 解释器。将所有状态/缓存/模型/日志路径设在仓库之外。每设备使用独立 `JITTOR_HOME`，首次编译串行执行。

```bash
export JITTOR_TRELLIS_CHECKOUT="$JITTOR_LAB_ROOT/trellis-validation/TRELLIS.2"
export REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_TRELLIS=1
export JITTOR_TORCH_SHIM=1 PYTHONPATH=python
export JITTOR_TEST_DEVICES=cpu
export JT_BACKEND=cpu JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' nvcc_path=''
export JITTOR_TRELLIS_EVIDENCE_DIR="$JITTOR_LAB_ROOT/_state/trellis-validation/fresh-cpu"
<jittor-python> -m pytest -q -s compat/tests/torch/test_ecosystem_trellis.py
```

CUDA 需选择恰好一张已分配的可见卡、隔离的 CUDA 缓存与完整受支持的 CUDA 组件栈；设置 `JITTOR_TEST_DEVICES=cuda JT_BACKEND=cuda JT_USE_CUDA=1 use_cuda=1` 与新的证据目录。两个 runtime 进程执行同一个受版本化 runner。原始目录以独占方式创建；复用已有目录会报错，而不是静默接受陈旧文件。

## 证据与限制

原始日志、二进制数组、元数据、源码/API 快照与哈希未版本化，位于 `$JITTOR_LAB_ROOT/_state/trellis-validation/`。`frozen-formal-source-complete.json` 记录干净的 test/runner/shared-harness 哈希；`accepted-formal-source-complete.json` 验证它们在执行过程中与 HEAD 均未变化。测量基线仍为历史 `6d39728fffc37d4618c1cb91a9caa1d67beec311`；collector 与报告随后针对 origin `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` 整合（本地合并 `f2d4bc30f53c950e8939eeff6daf6999c78ea04e`）。这不会把更早的数值执行重标为更新源码的运行。每个 case 目录包含基线与逐输入扰动的参考/候选 NPZ 与 JSON、比较摘要与 SHA256 产物清单。`scorecard-source-complete-final.json` 含六条被接受的 L3 记录与零条被拒记录，源自全部 93 个受跟踪上游 Python 文件（含 package 根）。通用下游 scorecard 验证平台/精度/runtime 身份、原始哈希与源码分母；TRELLIS collector 只把这些具体模型执行转成 L3 记录。

未确立任何已训练 checkpoint、预训练权重、图像/文本条件栈、CuMesh/FlexGEMM/nvdiffrast/o-voxel/FlashAttention 扩展执行、4B 流水线、mesh 几何、BF16/FP16、NPU/ROCm、多卡、整进程峰值显存或速度验收。历史报告推导的完整流水线比值 `1.0926x` 未在此复现，仍不在本验收范围内。L3 加观测到的零 fallback 不意味着 L4 性能验收或整库 L5 结果。

## 维护检查

同步 origin `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` 之后，core runner 通过 322 项测试，含 42 项 declared skip 与一个既有预期失败。其 native 与 Torch session 使用独立 CPU 编译缓存；首次编译包含在原始日志中，不是性能结果。源码 inventory、比较与 collector 契约测试覆盖陈旧/缺失源码身份、缺失梯度、非有限值、非二进制 oracle/fallback、重复场景、原始文件变化与不完整产物清单。最终 host 契约套件通过 38 项测试；Ruff 与仓库布局通过。要求的完整结构运行以 1,476 passed、six skipped、两项失败结束：本新报告尚未进入中央结果 toctree，生成的根 manifest 也需要同一份新报告。两个中央登记文件属于集成方负责，特意留给该整合处理；没有其他结构失败。原始命令/结果位于外部证据根下的 `logs/final-host-complete.log`、`logs/final-core.log`、`logs/final-structure.log`、`logs/final-ruff.log` 与 `logs/final-layout.log`。


## 新的已安装来源证明与维护重放（2026-10-16）

状态：同样三个 tiny dense 模型再次通过完整独立 CPU 与 CUDA
训练，并对每个 runtime 与输入变体提供已安装来源证明。
测量源码为干净 `b6947541b5bddfa87098d40c6f4e09ad0e704c54`，基于
已发布 `ffbb9087610dd737edb7ab758449c1f8f645437e`；通用持久目录
helper 的字节与 `cc86b8674f713b1924affb954aecfc6fad6bd57a` 完全一致。
更早的测量保留其原始身份与字节。特别地，
当前通用审计器拒绝全部六行历史 6d397 记录，因为它们缺少
实际已安装来源绑定；仅有完整旧源码哈希列表不足以
满足该更新的证明要求。本次重放不回填这些记录。

官方 Microsoft checkout 仍为干净
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`。它没有根打包声明、
已发布 package 版本或 wheel。实际安装即其在 import 路径上的真实 checkout，
显式记录为 `installation_kind=source-checkout` 与
`source@75fbf0183001ed9876c8dbb35de6b68552ee08bd`。没有虚构的 PyPI 发布、
METADATA、wheel 或被修改的 `__version__`。两个进程都捕获实际解析出的
`trellis2.__file__`，独立对每个已安装 package 文件取哈希，将
完整 **93 Python / 93 payload** 集合与重新生成的干净 inventory 比较，
并在训练后再次验证。多余、缺失、变化或越界的文件
fail closed；生成的 Python 字节码被排除。collector 独立
检查每个 base 与逐输入扰动元数据包，独立的
`perturbN-comparison` role 使每个变体对通用审计器可见。

全部原始计算与预算保持不变：30 个 Flow 参数与
三个可微输入，32 个 Encoder 参数与一个输入，32 个 Decoder
参数与一个输入；确定性非零命名初始权重、seed 0、
三步 SGD（`lr=.003, momentum=.2, weight_decay=.01`）、输出/loss、全部参数
与输入梯度、更新后参数与动量。每个输入分别
加上 `+.125`，重复完整三步轨迹。这些是输入变体，
不是额外的随机 seed。所有数组具有精确 schema/shape、有限 float32
值、原始 forward `2e-3` 与梯度 `1e-2` 字段判据及参考
尺度下限。未引入新 adapter、强制的替代 attention oracle 或数学
实现。上游支持的 dense `sdpa` 选择保持不变。

实际 opt-in `ecosystem_trellis` nox 执行在 CPU 上通过 **3 / 0 skipped**
（pytest 197.76 秒，含冷启动准备共八分钟），在 CUDA 上通过 **3 / 0 skipped**
（pytest 469.06 秒，含冷启动准备共十三分钟）。通过要求实际
数值比较，而不仅是 worker 成功。新 host 负向用例保留
成功 worker 的输出，同时拒绝失败的比较，而不写入
passed 完成包。显式持久目录 helper 拒绝已存在
或仓库本地输出；全部 136 个最终原始文件正常写入，未轮询或复制
JIT 产物。CUDA session 使全部 68 个 CPU 文件哈希保持
不变。全部 191 个跨设备初始状态与被保存输入字段精确匹配。

| 设备 | 模型 | 独立重算的数组对 | 最大绝对误差 | 最大 scaled 误差 |
| --- | --- | ---: | ---: | ---: |
| CPU | SparseStructureFlowModel | 1176 | 3.42726707e-7 | 2.68249228e-6 |
| CPU | SparseStructureEncoder | 600 | 8.82148743e-6 | 2.48021997e-5 |
| CPU | SparseStructureDecoder | 600 | 1.88276172e-5 | 2.56591094e-4 |
| CUDA | SparseStructureFlowModel | 1176 | 2.30967999e-7 | 1.99611972e-6 |
| CUDA | SparseStructureEncoder | 600 | 6.67572021e-6 | 1.80022135e-5 |
| CUDA | SparseStructureDecoder | 600 | 3.79979610e-5 | 3.85377480e-4 |

独立 host 重放重算全部 **4752 个数组对 / 32 个 runtime 元数据
包**，包括每个被请求输入变体；所有已保存比较指标
与重算精确一致。完整当前通用准入接受 **六条
L3 记录 / 零 rejected**：models 在 CPU 与 CUDA 上各 **3 / 9**，其他所有
轴为零。trainer 轴仍为 **0 / 15**；普通真实模型 SGD 轨迹
不确立上游 Trainer 或预训练流水线。仅在副本上，全部
192 个实际逐阶段缺失/变化源码、不完整/多余 payload、版本与
来源改动都被拒绝。三个诚实重算哈希的通用审计器攻击分别针对
CUDA Flow base、第三个扰动与训练参考元数据；每个只拒绝
该受影响行，其余五行仍被接受。

两侧使用全新私有 Python 3.11 环境、NumPy 1.26.4、einops 0.8.2
与 easydict 1.13。独立 Torch 2.11.0+cu128 prefix 从
完整官方 wheel SHA256
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad` 新解压；
全部 **11842 / 11842** 原始文件（含 RECORD）在执行前后均被验证。
实际 `_C` SHA256 为
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
未修复或替换任何共享包。CPU 在 import 前遮蔽 CUDA。CUDA
使用一张显式分配的 RTX 4090、驱动 580.178.04 与真实 NVCC 12.2.140；
oracle 构建为 CUDA 12.8。两者都使用严格 native kernel math、FP32、TF32 OFF、
matmul `highest`、cuDNN benchmark OFF、两个线程与一致的实际 affinity。
实际进程环境证明 CPU/CUDA unit cache 独立、结构缓存
独立且 benchmark 根未使用。显式候选库 probe 再次返回 cuDNN
9.24.0 / CUDA runtime 12.2；cuDNN 发行物元数据差异仍然可见。
上述任何计时都不授予速度或 L4 结论。

聚焦 host 契约通过 **83 / 0 skipped**，包括无框架的 `python -S`
证明执行。完整源码结构检查通过 **1624 / 4 skipped**；两项 native
检查由该私有 editable core 安装启用。Ruff 与布局通过。
全部 2882 个受跟踪文件与链接字节在完成的
测量与独立审计中保持不变。初次仅 host 的 manifest 枚举
在写入前拒绝了 Git 加引号的非 ASCII 名称与一个受跟踪目录符号链接；
改用 NUL 分隔名称与 Git 链接字节解决了该诊断，未修改
源码或 runtime 证据。64 条测试 warning 是未改动的上游非法
转义序列；它们不隐藏 skip 或失败节点。

使用现有 skill 与 `ecosystem_trellis` session、全新私有
解释器、同一干净上游 pin 与新的绝对外部
`JITTOR_TRELLIS_EVIDENCE_DIR` 复现。确切命令/环境、实际平台文件、
原始数值数组与全部证明/负向结果均未版本化，位于
`$JITTOR_LAB_ROOT/_state/trellis-installed-source-fresh/20261016/`。
`scorecard-installed-source-v1.json` 绑定六条被接受的行；
`independent-{cpu,cuda}-raw-audit-v1.json` 与
`accepted-frozen-source-and-cross-device-v1.json` 保留独立结果。
经两次重读的 `sealed-source-bound-raw-manifest-v1.json` 绑定
**179 个文件 / 842528632 字节**，包括 136 个原始 raw 文件、完整 provider
证明与官方 Torch wheel。其 SHA256 为
`afe80a51c9c9df8846d9bc7a0c7413bbdfb3a2222286174aa62ec2e086f50436`。
后续文档改动不重标这些 b694 测量。
上述全部完整流水线、sparse 扩展、预训练、额外精度/后端、
整进程显存与性能边界仍未关闭。

## 中央整合复核（2026-10-16）

维护者：Jittor 核心维护者。整合源码为干净
`e05eaa4c0`；测量源码仍为上述原始完整 `b6947541b5bddfa87098d40c6f4e09ad0e704c54`。
复查条件：生产者、collector、源码证明、上游 pin 或共享
准入规则变化。整合未新增数值测量。

集成方使用原始完整 CPU/CUDA 产物与原始判据独立重算
全部 **4752 个数组对与 32 个元数据包**，未导入 Jittor。
整合后的 collector 与通用审计器再次
接受 **六条 L3 记录、零 rejected**，CPU/CUDA models 各 **3 / 9**，
trainer 各 **0 / 15**。全部 **2882 个原始 Git blob**（含受跟踪符号链接
目标文本）与冻结测量快照匹配。全部 **11842 个实际 Torch
provider 文件**与其原始证明匹配。原始 179 文件 raw 封存
与最终 **184 个文件 / 842586487 字节**封存都经独立重新哈希，无
不匹配；后者 SHA256 为
`1451445897974e5e4339dcb159f271aae1216c5231902b30c1aa3dee9edffe47`。

整合后的主树通过 68 项 TRELLIS host 契约加 15 项持久输出
契约，core **322 passed / 42 skipped / one existing expected failure**，
`tests/structure` **1508 passed / 6 skipped**，`compat/tests/structure`
**114 passed / zero skipped**。布局、生成 manifest 检查、改动 Python
Ruff 与差异检查通过。六项结构 skip 保留其实际原因，
包括两项解释器安装解析检查；import 路径执行
不被重标为私有 editable 安装。根验证输出、
终端日志与新的通用 scorecard 未版本化，位于
`$JITTOR_LAB_ROOT/_state/trellis-main-source-integration-dfe/20261016/`。
未改动任何旧元数据或原始测量，也未授予更广的流水线、
预训练、trainer、后端或 L4 结论。
