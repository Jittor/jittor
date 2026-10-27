# 安装来源的 OpenMMLab train_step 在 CPU 与 CUDA 上

- 状态：两个原始 tiny train_step 场景在安装来源下通过完整的 CPU/CUDA 梯度、扰动与三步训练；仅限组件级 L3。
- 日期：2026-10-15。
- 测量基线：干净 `9345f99aad2a67ec7c043427a1dfd661c53c4369`，执行前同步到已发布的 `593249c5f0e82febb931234aa35003d0d41c8b33`。后续维护门禁/postprocessor 源码：`99120d4a5c1f1814833e15e16e90bc993cf5f1b8`。
- Schema 前置：`fc8bbc41dcce2986de4276e2c7b10b154ea0fcf3` 与 `8018b73c85f3f46966da2a7751aca9890626fca9`。
- 维护者：Jittor Torch 兼容与下游验证维护者。
- 复查条件：上游 pin/registry/setup 声明、安装 payload、provider、精度、组件 builder、训练 owner、证明 schema 或维护门禁变化。

本次新测量保留完整的[原始 OpenMMLab 协议](2026-10-15-openmmlab-training-oracles.md)。
原始四行及其原始文件保持 `2aa27ffe...` 身份；未回填任何来源。
新记录独立绑定实际安装包。未使用 adapter、复制的训练算法、上游补丁、
模型下载或伪造的编译扩展。

## 上游与实际安装

| 组件 | 干净上游 / 模块版本 | 实际 wheel METADATA Name / Version | Python 文件 / 完整安装 payload |
| --- | --- | --- | --- |
| MMEngine | `390ba2fbb272816adfd2883642326d0fd0ca6049` / 0.10.7 | `mmengine` / 0.10.7 | 153 / 157 |
| MMCV | `3a5fb4b96ab1b241e7379adf7c326ef6edef6515` / 2.2.0 | `mmcv-lite` / 2.2.0 | 128 / 531 |

两者都是从未修改的干净源码归档自建的 wheel，**不是官方 PyPI wheel**。
MMEngine wheel SHA256 为 `6e4c14ecdb9030e2ce1a0ee2b4169200ed961c93b8193b6364372f0a105113c4`；
MMCV-lite 为 `0da8de72a749ef224f5ab7772e516ec262f868a3c82dbc2de760dcdb6d98e764`。
MMCV 有 534 个受跟踪包文件；三个未随包发布的非 Python 文件保留在清单中，
但不表示为已安装。所有受跟踪 Python 文件与其干净源码字节一致。

有限的分发名证明由实际受跟踪的 `setup.py` 推导，绑定其哈希、
完整支持的条件表达式与每个声明分支。MMEngine 通过 `MMENGINE_LITE` 声明 `mmengine`
或 `mmengine-lite`；MMCV 通过 `MMCV_WITH_OPS` 声明 `mmcv` 或 `mmcv-lite`。
未解析的声明或丢弃的分支按失败关闭处理。实际 wheel 的唯一 METADATA Name、
Version 与全部包字节都独立检查，每个 runtime 阶段显式记录
实际 `distribution_name` 与 `distribution_version`。不修改任何版本字符串。
legacy 产物角色后缀 `-official-wheel` 是 schema 标签，不是对这些 wheel 来源的声明。

MMCV 记录还绑定其实际 MMEngine 依赖、完整源码/payload 与 registry
引用。只绑定 MMCV 自己的文件会漏掉其 `train_step` 与 optimizer wrapper 的
owner。不存在 `_ext` stub，也不从 lite 包推断编译版 `mmcv.ops` 的验收。

## 完整的原始训练范围与预算

| 用例 | 实际模型与输入 | 请求完整梯度的范围 | 三步数组 |
| --- | --- | --- | --- |
| `mmengine_train_step` | LayerNorm(6)、Linear(6,8)、SiLU、Linear(8,2)；样本 4×6 | 全部 6 个命名参数与样本 | 75 |
| `mmcv_convmodule_train_step` | registry 构建的 ConvModule BN/ReLU + GN/LeakyReLU、Linear head；样本 2×3×8×8 | 全部 8 个命名参数与样本 | 102 |

原始的真实 `BaseModel.train_step` 负责预处理、loss 前向、标量/列表日志解析、
反向、通过 registry 构建的 `OptimWrapper` 做范数裁剪、真实 momentum SGD 步与
清零。`RecordingSGD` 只在调用真实 SGD owner 前观测裁剪后的梯度。
三步中的每一步必须恰好调用一次 loss 前向与优化器更新。每个命名参数
都可训练且发生变化；目标是固定的 float32 输入，不请求梯度。

基础 seed 0 与扰动 7/19 使用完全相同的初始参数与 buffer。CUDA 复用
CPU 参考初始状态。两个 runtime 对每个扰动都必须改变输出与至少一个被请求的梯度；
全部 16 个此类 runtime/用例/设备/扰动观测均已独立核实。训练保持输入 seed 1/2/3、目标 seed 101/102/103、学习率 0.01、momentum 0.9
与 clip norm 1。每一步比较预测、全部解析出的 loss、输入、目标、输入梯度、
每份裁剪后参数梯度、更新后参数、momentum buffer 与命名 buffer。MMCV 的 BN
counter 是真正的标量 int64，必须恰为 1/2/3，包括在实际 CUDA 设备上。

每个阶段保留原始归一化度量：
`max_abs_error / max(reference_field_max, gradient_block_max * 0.001, 1e-6)`，非梯度字段严格低于 0.002，
梯度严格低于 0.01。梯度块限于当前步；
非梯度使用自己的字段尺度。输入、目标与 counter 还必须精确一致。
对称的完整键、精确的 shape/dtype 与有限值是强制的。不放宽容差、
不允许 broadcast 折算、不做聚合参数尺度稀释，也不以下游自有 `train_step` 之外的
普通 SGD 训练替代。

正式测量之后的 `99120d4...` 改动把已满足的扰动梯度条件变成
显式的维护门禁与 collector 守卫。它只改四个后处理/测试文件；所有
测量 producer、builder 与计算都与测量源码逐字节一致。
原始 9345 身份保留。后续 nox 执行有其自己的实际 99120 源码身份。

## 实际平台与数值结果

两个 runtime 都使用 Python 3.11、NumPy 1.26.4 与同一套实际安装的 MM 包。
oracle 是独立的官方 binary PyTorch 2.11.0+cu128；候选是 Jittor 的 Torch 兼容层。
CPU 显式屏蔽 CUDA 并禁用其 compiler/backend。CUDA 在分配的 NVIDIA RTX
4090 上使用真实 NVCC 12.2.140 运行。精度为 FP32：CPU TF32 OFF / matmul `highest`，CUDA matmul/cuDNN
TF32 ON / matmul `high`，cuDNN benchmark OFF。两者都使用两个 runtime 线程，与记录的 CPU
affinity 一致，严格原生 kernel 数学与离线控制。Platform JSON 保留 provider metadata
与实际卡占用；这是正确性证据，不含性能或 L4 结论。

| 组件 / 设备 | 全阶段数组对 | 最大绝对误差 | 最大归一化误差 |
| --- | --- | --- | --- |
| MMEngine / CPU | 99 | 2.38418579e-7 | 5.21069171e-7 |
| MMCV / CPU | 132 | 2.38418579e-7 | 1.54802998e-6 |
| MMEngine / CUDA | 99 | 4.69446182e-4 | 4.32249954e-4 |
| MMCV / CUDA | 132 | 9.93609428e-5 | 6.24255403e-4 |

全部 462 个数组对都在不变的原始预算下用 NumPy 独立重算。
共有 32 个 runtime 阶段 metadata 文件与 956 个实际张量观测。候选观测
在同步后检查物理 CPU/设备常驻与 dispatch，不移动张量来
制造常驻。候选后端 fallback 策略为 `error`，实际 fallback 为零。
CPU/CUDA benchmark 缓存与维护 unit 缓存分开。

只读 oracle provider 仍是完整的官方 wheel，SHA256 为
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`；实际 `_C` SHA256 为
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
正式测量后的重哈希验证全部 11,841 个原始安装 wheel 文件（1,624,262,090 字节），按原 provider 证明
排除 pip 重写的 RECORD，并验证精确的 2,227 个 Python 源码集合。
既有私有 CUDA 12.8 provider 布局以只读方式复用。未修复任何共享包。

## 维护门禁、证明负例与限定覆盖

实际干净的 `99120d4...` nox 执行在 CPU 上完成 **2 passed / 0 skipped**（pytest 179.02 s），
在 CUDA 上 **2 passed / 0 skipped**（pytest 423.47 s）；冷会话分别约八分钟与
十三分钟。独立隔离的规范 CPU core 完成 **322 passed / 42 skipped /
1 个既有 xfail**，native 与 Torch 模式均 exit 0。

新的 opt-in `ecosystem_openmmlab_installed` 会话执行两个完整的原始协议、
三个输入 seed 与全部真实训练步。它要求两个实际 wheel/checkout、独立
Torch、已安装依赖版本、显式 CPU/CUDA 分配与零 fallback。它拒绝
位置过滤并清除 `PYTEST_ADDOPTS`；数值失败会越过 worker exit
status 传播。实际类方法为 `compat/tests/torch/test_ecosystem_openmmlab_installed.py` 中的
`EcosystemOpenMMLabInstalled::test_mmengine_train_step` 与
`::test_mmcv_convmodule_train_step`。仅这些 opt-in 执行并不确立 nightly 调度。

全部 32 个实际 metadata 包通过完整证明。仅在拷贝上，384 个缺失/错误的 Name、
METADATA 版本、源码、非 Python payload 或 wheel 哈希变异被拒绝，适用时
对两个被引用库分别进行。原始 raw 数组与 metadata 保持不变。
Host 契约覆盖不完整张量、shape/dtype/非有限值、原始阶段预算、
BN 标量/counter 轨迹、未变的扰动梯度、observer 物理常驻与
数值失败传播。全部 71 个定向 host 契约通过；完整 `tests/structure` 为 **1510 passed / 4 skipped**，
Ruff/布局检查通过。Host/structure 在 9345 上运行，后处理补丁随后
提交为 99120；core 与 nox 在干净的 99120 上运行。
最终仅文档的检查为 **14 项通过，另有本新报告的一项中央 toctree 登记待完成**。
该索引与 MANIFEST 归整合方负责；在登记完成前不声称文档套件
全绿。

common auditor 接受 **4 条 L3 记录**，拒绝 0 条，检查 **102 个产物绑定**。
它独立拒绝两份如实重哈希、但把 CUDA platform 拷贝标为 TF32 OFF/highest
而实际测量为 ON/high 的副本；未改动的 CPU 行仍被接受。另外三种
common-auditor 攻击各自拒绝受影响的 MMCV 行：base oracle 漏掉其 MMEngine
分发名，seed-19 候选漏掉自己的名字，或训练候选漏掉其
MMEngine 依赖名。所有原始 scorecard/raw 字节保持不变。

| 自动源码声明轴 | CPU L3 成员 / 分母 | CUDA L3 成员 / 分母 |
| --- | --- | --- |
| MMEngine model_component | 1 / 7, BaseDataPreprocessor | 1 / 7 |
| MMEngine optim_wrapper | 1 / 5, OptimWrapper | 1 / 5 |
| MMEngine loop / weight_initializer | 0 / 4, 0 / 8 | 0 / 4, 0 / 8 |
| MMCV model_component | 1 / 70, ConvModule | 1 / 70 |
| MMCV hook / transform | 0 / 1, 0 / 19 | 0 / 1, 0 / 19 |

这些限定源码轴保留条件声明与依赖扩展的条目；它们
不是整库 runtime registry 或 API 分母。MMCV 对 MMEngine 的两处引用使用
同一批已计成员，不构成额外广度。每个已验证场景保留精确
用例、全部 registry 引用、tiny shape 种类与完整可训练/输入梯度范围。不授予完整 Runner、
dataset/hook/loop、分布式训练、编译版 mmcv.ops、更大模型、其他加速器、存储
复用、速度或整库覆盖。

## 复现与不可变证据

先在独立解释器中准备实际包。wheel 与源码路径是必需的；
维护会话不下载也不安装下游包。

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_ECOSYSTEM_PACKAGE_SITE="$MM_SITE" JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$MM_SITE" \
  JITTOR_MMENGINE_CHECKOUT="$MMENGINE_CHECKOUT" JITTOR_MMENGINE_WHEEL="$MMENGINE_WHEEL" \
  JITTOR_MMCV_CHECKOUT="$MMCV_CHECKOUT" JITTOR_MMCV_WHEEL="$MMCV_WHEEL" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_openmmlab_installed
# CUDA: allocate CUDA_VISIBLE_DEVICES explicitly, use a separate unit cache,
# set JITTOR_TEST_DEVICES=cuda and provide real nvcc_path/JT_BUILD_NVCC_PATH.
```

有源码支撑的原始数据与精确命令/环境产物未版本化，位于
`$JITTOR_LAB_ROOT/_state/openmmlab-source-bound/20261015/`，测量在
`formal-9345f99aad2a-v1/{cpu,cuda}`。`frozen-source-9345f99aad2a-v1.json` 与不可变 Git
ref 绑定原始源码；`post-formal-guard-source-v1.json` 区分后续守卫验证。
`formal-independent-array-audit-v1.json`、`formal-installation-independent-negatives-v1.json` 与
`formal-gradient-perturbation-independent-v1.json` 保存独立检查。

`scorecard-v1.json` 是最终的四记录 scorecard。成功重读的 raw seal 为
`mm-sealed-raw-manifest-v2.json`：**1,021 个文件 / 927,817,307 字节**，SHA256
为 `bdc6ff7a996db0d2e93df636bb77f9f2a14fecf52696fa3c323d7dafadc4790a`。
它绑定原始测量、比较、provider、完整安装的 MM payload、
命令、诊断与两份源码快照。最终证据 manifest 另外绑定
scorecard 与文档提交。已封存文件不被改写。第一次 seal 尝试
包含其自身仍未关闭的日志，保留为失败的诊断；v2 排除 writer 日志
并独立重哈希每个条目。

初始环境/setup 诊断单独保留：缺失的私有依赖
安装到本任务自己的 prefix，候选 core 的可编辑安装按
既有 `editable_mode=compat` 契约修正。第一次 structure 尝试有一次 namespace
解析失败；定向检查与随后的完整 structure 运行都通过。一次额外的
core-v1 启动漏掉任务本地 lab root，在部分原生结果后停止；不计为
完整 core。core-v2 显式隔离 lab root，两种模式都通过。
这些 setup/验证诊断不重标正式测量，也不修复共享包。

## 主整合验证

整合维护者同步已发布的
`cd7d45ab99e086f2b027be35bffa66d3b33efdd7`，并保留上述原始测量
源码身份。干净整合源码
`c4935e9061964f17fa7d8ac4fb9930e1b84f5bfd` 增加维护门禁、源码
契约、discovery 与中央 results/MANIFEST 登记。全部 73 个既有
nox 函数 AST 不变；只新增 `ecosystem_openmmlab_installed`。

在该整合源码上，provisioning、Swift、分发名、安装
来源/版本证明、OpenMMLab、shared coverage、VERL、PEFT 与
Transformers 的 host 契约通过 **228 / 0 skipped**。规范 core 通过 **322 / 42 skipped /
1 个既有 xfail**，native 与 Torch 模式 exit 0。完整
`tests/structure` 加 `compat/tests/structure` 在 235.97 秒内通过 **1622 / 6 skipped**。
六个 skip 保留其加速器/声明/未安装原生分发的边界；不推断任何安装或硬件结论。
布局、生成的 manifest、Ruff 与空白检查通过。

维护者独立重哈希全部 313 个 nightly 证据文件与全部
1033 个 OpenMMLab 证据文件：无不一致。原始失败尝试与
测量/源码身份均保留。隔离的维护数值运行及其独立保留的输出
在下文记录；整合检查不重标原始数值测量。未版本化整合记录位于
`$JITTOR_LAB_ROOT/_state/maintained-gates-main-integration-cd7/`。


## 持久化维护输出重放

2026-10-16，干净的 `0ccc36c3c6075268f1ec7e22f57879436b42f55b` 在 CPU
（**2 passed / 0 skipped**，103.62 s）与 CUDA（**2 passed / 0 skipped**，115.75 s）上通过
两个完整维护节点。该源码把已发布的
`0f2f9481cd59072cc02a1edff4bc16bc12d1b071` 与 `cc86b8674f713b1924affb954aecfc6fad6bd57a` 和
`a486108543eda4f0985f3a69f80077df220a88de` 中的精确 helper 与 OpenMMLab 生产字节合并。这些 runtime 是测试耗时，
不是性能证据。上游 pin、完整训练范围、原始
预算、独立 Torch provider、精度与零 fallback 检查与上文相同。
不授予额外覆盖或 L 级别。

可选 `JITTOR_OPENMMLAB_EVIDENCE_DIR` 把正常的最终 producer 输出保留在
`<root>/<device>/<case>` 下。通用 `artifact_directory` helper 要求
绝对路径、位于所有 Git checkout 之外，并独占创建每个叶子；
已存在的叶子被拒绝。未设置该变量时，临时清理仍是默认。
持久化输出还必须位于 nox 会话 scratch 之外，因为
下一次会话会清理该 scratch。失败的比较保留其原始文件，
但不写成功完成记录。只有在全部 seed-0/7/19 与训练
比较、扰动梯度与初始权重检查通过后，调用者才写含
四个实际最终比较的 `maintained-gate-result.json`。Host 负例验证这些边界，包括 worker
成功退出但数值比较失败。

每台设备直接保留 36 个最终文件，无需轮询：16 个 metadata JSON、
18 个 NPZ（含两个初始状态归档）与两个完成记录。独立
host 审计通过不变的 comparator 重读全部最终文件：
**462 个数组对、32 个完整 runtime metadata 包与 956 个物理张量
观测**通过。每个完成记录与其独立重算的四个比较精确一致。
全部 41 个初始状态与训练输入/目标字段在 CPU 与 CUDA 之间
精确相等。在随后的 CUDA nox 会话之后，全部 36 个 CPU 文件
的哈希保持不变。运行后的 provider 检查还验证
**11,842 / 11,842** 个原始官方 wheel 文件（含 RECORD）；
这一新的私有解包及其完整证明与上文较旧的 11,841 文件安装不同。未修改共享包
安装，unit/benchmark 缓存保持分开。

先前未改动的 c4935 维护执行同样通过：初始 CPU
2/0（178.61 s）、预热 CPU 2/0（103.44 s）与 CUDA 2/0（422.53 s）。其尝试的
额外临时文件归档保留为诊断证据：CPU
归档漏掉最终文件或捕获了绑定前的 metadata，因此
不声称其为完整的 CPU 来源保留证明。CUDA 最终归档通过
其独立的 231 对审计。上文的显式持久化输出重放
取代了为保留证据而做的轮询；那些更早的原始文件或
源码身份均未改写。

helper/OpenMMLab 改动通过 25 个定向 host 测试、Ruff 与布局检查。
完整源码结构验证通过 **1624 / 4 skipped**；全新的
私有 core 安装使整合维护者早先 1622 / 6 结果中被跳过的两项原生分发检查得以
运行。该改动增加目录所有权与证据保留，不改训练数学或 fixture。

未版本化产物位于
`$JITTOR_LAB_ROOT/_state/openmmlab-integration-c4935/20261016-persistent/`：
`mm-nox-{cpu,cuda}-v2.command.json`、`maintained-output/`、
`independent-{cpu,cuda}-audit-v1.json`、
`cross-device-and-retention-proof-v1.json` 与 `frozen-source-v1.json`（493 个文件）。
两次重读的最终 seal `persistent-final-manifest-v1.json` 绑定
**88 个文件 / 14,503,356 字节**，含全部 72 个最终 raw 文件：
SHA256 为 `2ddfd200da7c74e578b87fd2bd32bca7726a9a392b30449ea727b65b807b2821`。
它还绑定完整 provider 证明与上一阶段不可变的
630 文件 manifest，后者的 SHA256 仍为
`0f437f9bfe496efc4ee092257cd1fb760a5aac7e45b1d756d6827c473f5de174`。
全部 630 个上一阶段文件、全部 493 个测量源码文件与全部 provider
文件在封存前都已成功重哈希。原始四行 scorecard
保持其测量所得 9345 身份。


### 中央持久化输出整合验证

整合维护者独立重读全部 88 个封存文件（14,503,356
字节），重算两个不变的完整 CPU/CUDA 协议（每设备 231 个数组对、
16 个 metadata 包与 478 个观测），并重哈希完整的
11,842 文件真实 Torch provider 及其原始 wheel：零不一致。
此 host 重算使用实际干净的测量源码 `0ccc36c3c`；
它不把这些测量重标为整合运行。

在整合后的 `9bee3b603` 上，25 个定向 host 测试通过，规范 core 报告
322 passed / 42 skipped / 1 个既有 xfail，完整 structure 在
209.99 秒内报告 1622 passed / 6 skipped。布局、生成的 MANIFEST、Ruff
0.15.22 与空白检查通过。这些结果与上文更早的
1624/4 源码树检查分开。未版本化中央证据位于
`$JITTOR_LAB_ROOT/_state/openmmlab-persistent-root-integration-ffbb/`。
