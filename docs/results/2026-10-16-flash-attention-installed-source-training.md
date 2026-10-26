# 具有不同 installed-source 身份的 FlashAttention 训练

- 状态：完整全新的十二 case 维护 CUDA 门禁、独立原始数组审计与独立源码/共同复核均通过。全部十二条独立验证的 CUDA L3 记录仍因缺少 CPU 对应项而阻塞；共享 CUDA 广度不计入贡献。
- 测量源码：干净 `c33627faaf99f1cc9f422421f90c3896d4cbc326`。
- 维护者：原生 CUDA 与 Torch 兼容维护者。复查日期：2026-10-16。
- 复查条件：上游/CUTLASS 固定版本、Torch ABI、facade/bridge、原生 autograd、provider 选择、比较预算或 source-proof 工具变化。

## 结果与保留范围

维护的 `ecosystem_flash_attention` nox 入口执行全部原有十二个 CUDA case，即六个公共 dense/packed/varlen API × FP16/BF16。每个 case 在独立的 Torch 与 Jittor 进程中运行 base 与受控噪声扰动 `0.125`，固定种子 `20261015`，三次学习率 `.01` 的 SGD 更新，FP32 参数与参数/输入梯度，非 causal attention，head 维度 32，dropout 为零。dense shape 为 `(2,11,2,32)`；varlen shape 为 `(11,2,32)`，序列长度 `[4,7]`。全部 48 个 worker 与所有请求的梯度都在范围内。

24 个运行时配对保留 720 次实际数组比较与 216 次独立更新比较。初始参数要求字节相等。未改动的归一化是逐张量最大绝对参考值，下限 `1e-8`：FP16 前向 `.003`，梯度/更新 `.006`；BF16 `.03`。输入哈希校验固定的 Q/K/V/target 与已改变的噪声；输出必须在两个运行时上都实际改变。十二个节点在 1063.92 s 内无跳过通过。独立的直接 NPZ 审计通过全部 936 项检查与 24 项实际输入扰动检查；其最大归一化误差为 FP16 `0.002433830240340736`、BF16 `3.90727286094882e-7`。FP16 参数/输入梯度在 140/144 个数组上字节相同，BF16 为 144/144；其余全部满足其各自不变的预算。包含自动源码/数值/共同校验的完整维护 nox 在 27 分钟内成功。这些是正确性门禁耗时，不是速度测量。

这是 tiny 组件的训练正确性。KV cache、其他维度/causal/dropout 设置、生产模型规模、其他硬件与速度都在本全新协议之外。官方七 API 清单单独保留；该协议 `axes=[]`，不虚构注册表分母。

## 两个实际实现

固定官方上游为 FlashAttention **2.7.4.post1**，commit `5231d95fe13733fb534c01895f7ea88c6a6c7793`，实际官方 CUTLASS gitlink 为 `c506e16788cb08416a4a57e11a9067beeee29420`。oracle 与 candidate 是不同的实现：

| 身份 | 独立参考 | Candidate |
| --- | --- | --- |
| Python | 全部 58 个受跟踪的上游包 Python 文件，未改动 | 五个递归校验的 Jittor 自有 facade 文件 |
| 安装 | 本地 wheel，完整 77 成员 RECORD/载荷 | 真实 `jittor-torch==2.0.0` carrier METADATA/RECORD/载荷，私有部署 facade |
| API metadata | 真实本地 wheel distribution | 显式自有的兼容 metadata；不是上游 wheel 安装 |
| 原生代码 | 先前真正源码构建的官方扩展 | 实际官方源码的原生 bridge 与 packed 扩展、生成的 C++/CUDA 与已加载二进制 |
| Torch 身份 | 完整官方二进制 Torch `2.11.0+cu128`，11842 个已安装 wheel 成员 | 来自测量源码的 Jittor Torch 兼容前端 |

本地参考 wheel **不是已发布的 PyPI 二进制**，本任务也**没有重建其原生扩展**。在合法 Git 源码归档中运行的未改动上游 `setup.py build_py` 打包了实际固定的 Python 与原始已验证原生二进制。所有 wheel Python 成员都与受跟踪上游源码匹配。被未改动上游 build_py 排除的两个受跟踪非 Python 包文件被显式记录。原始原生构建 manifest 对照全部 85 个官方 CUDAExtension 源码声明、上游 CUDA flag、原始 Torch/CUDA ABI 与二进制字节进行校验。

参考 wheel SHA256：`c0685ff498af6cde5376f9b24353241957f1ecc534fe8e8976cd8f49a5d65a74`。
原始源码构建的参考扩展 SHA256：`3c83031779c640c6179800acf4828609df61794b768ff938a42fe49b6eeeaa8a`。
官方二进制 Torch wheel SHA256：`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`。

candidate 扩展在私有任务缓存中串行冷编译，包括原始 head-32/64 × FP16/BF16 构建配置。第一次真实 bridge 构建属于干净 `cf4560f90`；后续同步的源码与目录修正以 `c33627faa` 为测量点，并捕获了实际加载的原生 core/bridge 字节。既有私有缓存条目由常规 builder 处理；没有向新运行时复制 JIT 缓存。被取代的可变 preflight 字节作为证据另行归档，从不作为编译缓存使用。

每个张量进程重新校验完整导入的 wheel/carrier 字节、源码描述符、全部 1654 个测量框架/工具文件、实际生成的 bridge 源码与已加载二进制。实际编译器配置还绑定 2343 个 Python/CUDA/pybind 头文件与真实 `pybind11==2.13.6` METADATA/RECORD/载荷。对所有操作数、输出、loss、参数与梯度都观测实际 CUDA 指针驻留，并记录运行时进程 ID、物理设备 UUID、选定 NVIDIA metadata 与已加载库哈希。

平台：Linux，CPython 3.11.16，NVIDIA RTX 4090 D / sm89，驱动 580.178.04。candidate NVCC 为 12.2.140 / GCC 11.5，实际运行时 API 12020；参考使用真实 Torch Cu128，实际运行时 API 12080。这并不主张 candidate 的每个依赖都来自统一的 Cu122 栈。每个实际库/provider 路径都被保留。部分历史只读 NVIDIA 依赖 metadata 缺少 RECORD；该缺失被显式记录，而不是修复或称为完整的 NVIDIA 安装证明。完整 Torch、参考 Flash wheel 与 candidate carrier 的 RECORD/载荷检查仍然强制。

## CPU 边界与共同准入

一个全新的独立真实 Torch 进程，使用实际安装的本地 wheel 并隐藏 CUDA，分配了 CPU FP16 Q/K/V 并调用公共 dense 函数。上游在 CPU 上抛出了 `flash_attn::_flash_attn_forward` 的 `NotImplementedError`。固定源码也为 dense/varlen 的前向与反向自定义算子声明了 `device_types="cuda"`。这是不受支持的 CPU 边界，不是数值 CPU 对应项。

新行使用精确的 `flash-attention-training` 协议与不同的实现证明。普通的注册表/installed-source 规则不变。维护的共同审计器返回 **0 接受、12 阻塞、0 拒绝**，全部十二行 `independent_verified_level=3`，并带显式原因 “Missing verified CPU counterpart for this exact protocol”。独立验证的 CUDA L3 在没有匹配的真实 CPU 对应项时仍然阻塞；不伪造 CPU fallback，CUDA 广度不计入贡献。历史上仅 checkout 的原始结论与旧 schema 结论既不打补丁也不重标。其原始数值证据保持独立；当前 installed-source 审计因缺少安装/源码绑定拒绝了全部十二条旧记录。

## 维护入口与不可变证据

使用 [Flash skill 配置](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/flash-attention-torch-compat/SKILL.md)，包括显式的 `JITTOR_FLASH_ATTN_HEAD_DIMS=32,64` 与 `JITTOR_FLASH_ATTN_DTYPES=fp16,bf16`。使用私有预置的 candidate 与独立二进制 Torch 解释器、显式分配的 GPU、实际固定的上游 checkout、本地参考 wheel/原生构建 manifest 与完整官方 Torch wheel 证明。把 `JITTOR_FLASH_ATTN_SOURCE_PROOF` 设为包含四个真实路径（`upstream`、`wheel`、`oracle_build`、`torch_wheel_proof`）的 JSON，然后：

```bash
JITTOR_TEST_DEVICES=cuda JITTOR_FLASH_ATTN_EVIDENCE_DIR=<new-external-case-root> \
  <controller-python> -m nox -s ecosystem_flash_attention
```

该 session 检查真实 oracle CUDA，只安装私有 candidate 兼容包，通过公共 `shim.deploy --target/--check` 部署/校验 facade，运行全部十二个既有 ecosystem pytest 节点，然后自动运行完整数值 collector/共同 scorecard。既有 case 目录会被拒绝；失败的显式输出保留。这个 opt-in 入口与“新增 nightly CI 接线”的主张是分开的。

独立纯 host 复核使用精确的测量源码与新输出：

```bash
<oracle-python> agent/skills/flash-attention-torch-compat/scripts/collect_scorecard.py \
  --root <measured-worktree> --upstream <pinned-official-checkout> \
  --source-proof <external-source-configuration.json> \
  --evidence-dir <immutable-formal-case-root> --output-dir <new-external-recheck> \
  --baseline c33627faaf99f1cc9f422421f90c3896d4cbc326
```

单独的原始数组审计独立重算原有 936 项检查与输入扰动。完整 Git/源码、实际外部安装/原生/编译器字节、setup 失败、全部 worker 日志/JSON/NPZ 与复核都在 Git 之外封存。未版本化的原始证据位于 `$JITTOR_LAB_ROOT/_state/flash-source-bound/20261016/`：`formal-v4/cuda`、自动生成的 `formal-v4/cuda-scorecard`、`independent-recheck-v4` 与 `formal-v4-independent-array-audit.json`。完整 `evidence-manifest-v4.json` 绑定 2888 个 Git 源码条目与 17666 个外部/原始文件，SHA256 **`41434968792cf06967514d8b9aaba9932f39498285795d74e7b0fac48d514304`**。运行后的 Git 快照与初始快照字节相同（SHA256 `12574dc22a3682d5f191c4e85a8dcc1c64486bd4e5ac48a75019d98f4906d13b`）。两个 collector 在全部语义、数值与平台字段上一致：零接受、十二阻塞、零拒绝，`level`/`verified_level` 为 null，`independent_verified_level=3`。原始冻结 worktree 与证据保持只读。

setup 失败被保留：第一次仅源码的 Git 打包尝试撞上上游无条件的 ROCm 子模块网络步骤；合法的归档布局打包在无改动的情况下成功。candidate preflight 先在数学之前因 facade 未部署而失败，随后因缺少真实 pybind 头文件再次失败；两者都只有 controller-only 启动身份，没有回填运行时证明。在实际修复之后，独立的 39 项最小检查对通过。随后 formal-v3 在启动张量 worker 之前就让全部十二个节点失败，原因是 nox 尚未创建其 oracle state 目录；一个定向 host 回归复现了该问题，目录初始化被修正，formal-v4 使用全新输出与新的干净源码。更早的单测环境打包/xdist 装配失败也被保留。

最终测量前的验证：77 项受限 provenance/比较/nox host 检查；必需的 `tests/structure` 1511 passed / 4 skipped，200.12 s；Ruff、布局与 diff 检查通过。完整 Git 快照把 2888 个受跟踪文件对照其实际 Git blob 校验，符号链接按其目标文本处理。原始失败与被取代的源码身份与最终测量队列保持分离。

## 单独的构件准入后续

在封存原始队列之后，一个单独的干净 worktree 在 merge `9d6fb0a703cd7b4df0d00c965494d7ef4f9993bf` 上同步了远端 `4cefbacdca771f7d25512beccc541d902836b894`。一个真实二进制 Torch 的 CPU 纯 host 探针构造了一个自洽的反事实 wheel/前缀/RECORD：它加入了一个未跟踪的非 Python JSON 资源，并正确地重新哈希了每个 RECORD 成员。旧的描述符与实际导入包捕获接受了全部 78 个成员。这作为一次失败的准入测试保留，而不是张量或 GPU 证据。

窄范围的后续在这个受限协议中修复了**实际本地源码构建构件的 SHA256** `c0685ff498af6cde5376f9b24353241957f1ecc534fe8e8976cd8f49a5d65a74`。这不是 PyPI Flash wheel 主张。完整的自洽增/删载荷 fixture 现在即使其 RECORD 与安装完全匹配也会失败。一次真实正例检查重建原始构件描述符，并证明它与已封存的 `c33627faa` worker 捕获的描述符完全相同；真实的自洽变体在固定构件守卫处被拒绝。任何不同的合法本地构建都需要经复查的构件固定与全新的源绑定证据。candidate facade 包闭包已经要求精确递归校验的五个自有文件载荷。普通注册表/安装规则与 CPU 前置条件不变。

后处理器提交：**`be4827ef3e80a8f2c0c48ed67f2f6298c39ddee7`**。其 58 项纯 host 工具检查通过；必需的 `tests/structure` 在其自身缓存中单独串行 CPU bootstrap 之后以 501.82 s 通过 1516 / 4 skipped。Ruff、布局与 diff 检查通过。未版本化的后续证据位于 `$JITTOR_LAB_ROOT/_state/flash-artifact-closure/20261016/`：`coherent-variant-red-v1.json`、干净处理器的 `coherent-variant-green-v2.json` 与 `postprocessor-manifest-v1.json`。后续 manifest 绑定 2895 个后处理器 Git 条目与 96 个证据文件，SHA256 **`42d02a6b12d59fe3f16571bf4c9b586de8e87854e1a7b98a6e45ff386faa9da7`**。该守卫仅用于准入：原始 48 个 worker、936 项数值检查、测量源码与原始 manifest 既未重跑也未重标。

中央集成独立重新哈希了全部 17666 个测量/外部文件
（9467317936 字节）、全部 96 个后处理器产物（606105132 字节），以及两棵
完整 Git 树对其原始 blob。它独立重放了全部
720 次数组比较、216 项更新检查与 24 项扰动检查，并使用
原预算。这些检查保留测量 `c33627faaf99f1cc9f422421f90c3896d4cbc326`
与单独的构件守卫基线；它们没有把任何一方重标为集成提交。

集成还复现了一个 collector 缺陷：以指向冻结测量 checkout 的 `--root`
执行当前 collector 时，错误地检查了 collector 自身的框架树并导入了历史校验器。
修复把测量根穿透进共同协议校验，同时从审计器树加载
当前 collector、comparator 与固定构件守卫。
`--root` 必须指向处于精确完整 `--baseline` SHA 的干净受跟踪 checkout；
其原始探针哈希来自该 Git commit。其他协议校验器保持
既有调用契约。历史 checkout 无法选择更旧的守卫。
根归属的审计脚本、失败尝试与复核未版本化保留在
`$JITTOR_LAB_ROOT/_state/flash-main-integration-4ce/20261016/`。

最终集成验证：182 项定向 host 检查，CPU core 322 passed /
42 skipped / 1 既有 xfail，扩展结构 1628 passed / 6 skipped。
布局、manifest 与 Ruff 检查通过；72 个无关的 nox 函数体
未改动。当前守卫的 collector 完成了完整十二 case 复核：
0 接受、12 阻塞、0 拒绝，每个数值/协议/平台字段
都与冻结正例对照匹配。自洽的 78 成员伪造 wheel
与其合法安装 RECORD 被当前固定构件守卫拒绝。
后处理器在集成提交之前运行；其精确归属方字节与
复核记录与原始干净测量基线分开记录。
