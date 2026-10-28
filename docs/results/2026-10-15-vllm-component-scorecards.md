# vLLM 0.29.0 原语 CPU 与 CUDA scorecard

- 状态：下列当前已安装来源 L2 重放已验证；历史 72e 行未通过当前严格的源码准入。无引擎验收。
- 日期：2026-10-16；原始 72e 测量章节保留其 2026-10-15 身份。
- 原始测量源码（新重放之前的章节）：`72e12a9c662b46d5a5389f7e031c3997520d4562`，采集全程干净。
- 维护者：Jittor 兼容层维护者。
- 复查条件：固定的上游、builder、runtime、adapter、精度策略或门禁变化。

## 身份与覆盖

独立解释器执行同一个 vLLM 0.29.0 包与
Transformers 5.13.0。oracle 导入真实 PyTorch 2.13.0+cu130 二进制；
候选导入 Jittor Torch 前端。干净上游 checkout 为
`98dff2a81d747d1dba01a47f939f48c3526d4206`，即确切的官方 `v0.29.0`
tag。已安装的生成版本报告 `0.29.0` 与提交标记
`g98dff2a81`。原始 vLLM wheel SHA256 为
`09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`。
从该 checkout 枚举的全部 2312 个 Python 包源码文件与
已安装包逐字节一致；inventory 还绑定 589 个 registry/目标
源码文件。仅安装包本身不是数值证据。

自动源码 registry 有 **376 个模型族与 40 个 CustomOp
登记**，彼此为独立的限定轴。每台设备的一个场景验证
`silu_and_mul`、`rms_norm` 与 `rotary_embedding` 的
原生路径：**3/40 CustomOps，0/376 模型族**。Runtime 插件仍是开放集合。更早的
`51a99565c398c8320de8131e07731c75c52eb87c` inventory 分母不同，
此处不使用。

| 设备 | 场景 | 深度 | CustomOp 分子 | 模型分子 | 梯度范围 |
| --- | --- | --- | --- | --- | --- |
| CPU | `vllm_decoder_primitives`，tiny FP32 | L2 | 3/40 | 0/376 | 全部四个输入梯度；零可训练参数 |
| CUDA | 同样的固定权重与 shape | L2 | 3/40 | 0/376 | 全部四个输入梯度；零可训练参数 |

## 场景与数值协议

维护中的 builder 在真实 vLLM 配置下、关闭 compilation
与自定义 kernel，构造真实公开的 `SiluAndMul`、`RMSNorm` 与
`RotaryEmbedding` 对象。`RMSNorm(16, has_weight=False)` 有一个故意
未使用的 weight。输入为 gates `[5,32]`，hidden states/query/key `[5,16]`，以及
五个最大位置为 32 的整数位置。Rotary head 维为
8。四个输出拼接为 320 元素向量。这验证上游
原生组件执行；不验证 vLLM `_C` kernel。

两个解释器收到相同的序列化初始状态与 NumPy 生成的
输入。base seed 为 0。seed 7 与 19 在保留该
初始状态的前提下改变输入，并检查每个 oracle 输出与 base 不同。
每次运行精确比较五个数组：输出以及 `gates`、
`hidden_states`、`query` 与 `key` 的梯度。要求键集合对称、shape 精确、值有限。
因此每台设备比较 15 个数组，无缺失
梯度。没有参数更新或训练声明。

冻结的逐张量归一化最大误差使用 oracle 的最大
绝对量级，下限 `1e-8`。限值为输出 `0.002`、输入梯度 `0.01`，
在各设备与扰动间不变。

| 设备 | Seed 0 最大值 | Seed 7 最大值 | Seed 19 最大值 | 全部数组的最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1.0609394e-7 | 9.7602168e-8 | 1.9150882e-7 | 4.7683716e-7 |
| CUDA | 6.7617129e-8 | 9.7602168e-8 | 1.1146150e-7 | 4.7683716e-7 |

两个候选设备都报告 fallback 策略 `error`、计数零与
所请求设备。CPU 显式遮蔽 CUDA 并读回 `use_cuda=0`。
CUDA 读回 `use_cuda=1`、`use_acl=0`。两份 runtime 元数据记录在
TF32 关闭、cuDNN benchmark 关闭、最高 matmul 精度与一个
BLAS/OpenMP 线程上一致。Jittor kernel math 为严格。

机器使用 AMD EPYC 9554 CPU；CUDA 进程看到一张 NVIDIA GeForce
RTX 4090。实际 runtime 采集记录驱动 580.178.04、oracle CUDA 13.0
与候选 nvcc 12.2.140，以及包来源与 GPU 身份。这些是
正确性结果。不从这个 tiny fixture 推断性能预算、进程显存峰值、独占
机器使用或 L4 声明。

## 维护门禁与复现

确切门禁节点为
`compat/tests/torch/test_ecosystem_vllm.py::EcosystemVLLM::test_vllm_decoder_primitives`。
专用 `ecosystem_vllm` nox session 要求预先准备的实包
与独立 oracle。其必需标志拒绝缺失依赖、
错误解释器身份与错误设备。CPU 与 CUDA 正式 nox 运行在源码封存前各
通过一项测试、零 skip；这些日志保留其
原始基线。八项 host 契约覆盖该 session 接线。
对应的共享 owner 修复及其更广泛的回归结果见
[RMSNorm 契约与 Triton 指针别名](2026-10-15-rmsnorm-triton-alias.md)。

```bash
# Provision real oracle/candidate dependencies and set verified source/wheel inputs.
export JITTOR_VLLM_CHECKOUT="$VLLM_CHECKOUT"
export JITTOR_VLLM_WHEEL="$OFFICIAL_VLLM_WHEEL"
export JITTOR_VLLM_EVIDENCE_DIR="$FRESH_EXTERNAL_EVIDENCE_DIR"
REAL_TORCH_PYTHON=<oracle-python> JITTOR_CI_PYTHON=<candidate-python> \
  JITTOR_TEST_DEVICES=cpu nox -s ecosystem_vllm
# Select the allocated GPU externally; keep an independent cache for each device.
REAL_TORCH_PYTHON=<oracle-python> JITTOR_CI_PYTHON=<candidate-python> \
  JITTOR_TEST_DEVICES=cuda nox -s ecosystem_vllm

# Historical generic diagnostic; strict admission uses the maintained source-bound gate.
<host-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo vllm --cases vllm_decoder_primitives --device <cpu-or-cuda> \
  --oracle-python <oracle-python> --shim-python <candidate-python> \
  --repeats 1 --seed 0 --no-memory --out <outside-checkout>
```

原始产物位于
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`：

- `vllm-clean-72e12a9c6/manifest.json` 绑定采集命令、干净测量
  源码、两台设备的 NPZ/metadata/日志与源码哈希。
- `vllm-central-inventory-72e.json` 与
  `vllm-installed-source-proof-72e.json` 绑定 registry 与已安装源码。
- `platform-vllm-{cpu,cuda}.json` 记录实际平台与精度值。
- `vllm-scorecard-evidence-72e.json` 将两条数值记录绑定到其
  确切门禁、完整 registry 引用与原始产物哈希；
  `vllm-scorecard-72e.json` 是重新生成的中央审计器输出。
- `replay_vllm_primitives_v2.py` 保留固定权重扰动命令；
  `assemble_vllm_scorecards.py` 组装而非杜撰数值证据。

中央审计接受两条记录。负向检查拒绝未知
registry 成员、缺失输入梯度，以及声称 TF32 启用的平台精度块——
即使其产物哈希有效，但实际测量
元数据表明为关闭。全部 44 个采集产物经独立重新哈希。

scorecard 显式使用历史 `--baseline
72e12a9c662b46d5a5389f7e031c3997520d4562`。之后的整合与文档
提交不改变该测量身份。初次 collector 在成功的 CPU base 运行后使用了
过期的报告文件名；其失败日志保留，修正后的 collector 复用了未变的
base 产物。

## 引擎与 Omni 边界

一个真实 vLLM 0.29.0 CUDA oracle 执行了保存的 tiny Qwen3 引擎，两个
prompt、各三个 token，带 log probabilities。候选引擎停在
真实的 `torch.fx._graph_pickler.GraphPickler`/`Options` import。没有添加空的
GraphPickler 实现。oracle 日志为
`logs/oracle029-engine-cuda-v5.log`；候选失败为
`logs/shim029-engine-cuda-initial.log`。组件成功不消除该
引擎阻塞，也不确立模型覆盖、服务性能或训练。

源码 `446c2b5474dabffccf9a0034836eeab77c62b4b8` 的 vLLM-Omni 是另一项
待验证工作。实际 SCM 版本为
`0.29.0rc2.dev58+g446c2b547`。本次封存采集时，其 tiny MiniMaxH3DiTModel CUDA 推理
存在已复现的 RMSNorm 精度契约差异，而真实 CPU
路径拒绝只支持 CUDA 的 Omni 平台。两条路径都不包含在这些
vLLM 记录中。原始失败与数值产物保留；
此处不授予任何 Omni 模型、流水线、训练或 CPU 覆盖。

## 整合审计

整合同步了 `824cb354d5cced0743b3207d50d518111ba59e76`，并
保留干净的 `72e12a9c662b46d5a5389f7e031c3997520d4562` 测量
身份。独立复核重新哈希了全部 44 个原始采集产物与九个
最终交付绑定，将该 Git 提交与八个测量源码文件匹配，
并从干净官方发布 checkout 重新推导完整 registry。
它还针对两个 runtime 使用的实际已安装包重新哈希全部 2312 个受跟踪包 Python 文件，
重算 30 个数值数组比较，并验证两次扰动都改变了输出。
当前共享审计器接受两条记录，零拒绝。

封存 scorecard 的 SHA256 为
`24d170691d2ea4efbb772644d146b33f6c0dd5a45692413f2d23d9a5bcc4fea1`；
最终交付 manifest 的 SHA256 为
`42dab72c4bd2ea9a0d07e884cf19e8b176b67385f6fefdd5d9316b30bda8ed03`。
中央结果导航与生成的包 manifest 由该整合登记。
独立检查与日志未版本化，位于
`$JITTOR_LAB_ROOT/_state/vllm-scorecard-integration-824/`。
之后的 Omni 精度与存储契约调查仍是独立工作；
本组件报告不隐含任何后续模型验收。

## 新的完整已安装来源重放（2026-10-16）

测量源码为干净 `c9978992512024a4951a139e4c3d221aaa2584b1`。
维护者：Jittor 兼容层维护者。复查条件：源码/wheel pin、
生产者、adapter carrier、nox session、collector、精度或准入规则
变化。原始 72e 记录未变：当前通用审计器
因缺少实际发行物/源码绑定而独立拒绝两者。
下面的新结果来自完整的新 CPU 与 CUDA 执行。

干净官方上游仍为 `98dff2a81d747d1dba01a47f939f48c3526d4206`，
原始官方 vLLM wheel SHA256 见上。每个 worker 将实际
`vllm.__file__`、真实发行物 METADATA/版本与合法已安装
RECORD 绑定到完整原始 wheel 与 **2312 Python / 4904 个包 payload 文件**。
实际 Transformers 5.13.0 payload 也必须在两个 runtime 之间匹配。
现有候选 `jittor-torch-adapters` 2.0.0 carrier 正常安装
并绑定到其实际 bootstrap、入口点、METADATA、RECORD 与 editable
源码 URL。未新增 adapter 或数学实现。probe 在执行前后对全部 **1604** 个相关框架、adapter 与维护门禁源码文件
取哈希，要求测量 HEAD 干净未变。

原始场景与预算保持不变：FP32、固定 `[32,8]` rotary
cache、输出 `[320]`、全部四个被请求输入梯度、seed 0/7/19、输出
归一化最大值 `<= .002` 与梯度 `<= .01`、oracle 尺度下限 `1e-8`。
每个扰动在两侧都改变输出与至少一个梯度，初始权重
不变，worker PID 各不相同。CPU/CUDA oracle 初始 cache
数组也精确相等。可训练参数或更新为零。

实际公开 nox 执行在每台设备上通过 **一项 test / zero skipped**：
CPU pytest **246.02 秒**，CUDA **353.91 秒**。未导入 Jittor 的独立 host 重放
重新检查全部 **30 个数组对**、**12 个不同 worker**、**60
个物理张量观测**与每个 runtime 源码/安装包。
CPU 显式遮蔽 CUDA 并报告 CUDA 不可用；CUDA 实际指针
属性解析到其显式分配的设备。候选 fallback 仍为
`error`，计数零。要匹配实际精度与 runtime 条件，要求
TF32 与 cuDNN benchmark OFF、最高 matmul 精度与一个线程。

| 设备 | Seed 0 最大归一化误差 | Seed 7 | Seed 19 | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| CPU | `1.0609394e-7` | `9.7602168e-8` | `1.9150882e-7` | `4.7683716e-7` |
| CUDA | `6.7617129e-8` | `9.7602168e-8` | `1.1146150e-7` | `4.7683716e-7` |

修正后的 collector 与普通通用审计器接受 **两条 L2 记录，
零 rejected、零 blocked**。每台设备仍为 **CustomOp 3/40，模型 0/376**。
后处理器的零参数计数/列表字段起初被对调；
新的正例 schema 断言先复现了该失败，随后在 `ffff94ee2` 只修正
这些输出字段。其最终 77 项聚焦契约通过。
该后处理改动保留原始 c997 源码采集，并使用
冻结的测量 checkout 做验证；它不重写 worker 元数据、
不重标测量、不改变判据，也不授予训练等级。

实际平台为 EPYC 9554 与一张 RTX 4090，驱动 580.178.04，
GCC 11.5.0，候选 NVCC 12.2.140 与真实 Torch 2.13.0+cu130。私有全新
解释器使用既有只读真实 provider。完整官方 Torch
wheel SHA256 为 `60fcdcb2f3876e21146cb4524ef06397d727ca9ad5f020818547e25075fe3cb7`；
全部 **12117 个原始已安装文件**、合法的 pip 重写 RECORD 与
原始 wheel 在执行后重新哈希。未修复任何共享包。
候选 NVIDIA 元数据单独记录：cuDNN 元数据为 9.26.0.51，
而显式真实 provider API probe 报告 **9.24.0**；CUDA runtime 报告
**12.2**。十个继承的 NVIDIA provider 元数据目录缺少 RECORD；
该事实保留，不杜撰 RECORD，也不作完整 NVIDIA wheel 完整性
声明。这些显式 provider probe 并不表示每个库都被
每个组件 worker 加载。完整平台文件保留原始 oracle
采集，并在新文件中加入这些候选事实。

初次新 CPU 尝试在进行数值之前停止，因为现有 adapter
入口点缺失。随后两次 CUDA 启动在数值之前停止，
原因是旧 nox session 强制 CUDA wheels OFF 且系统工具链缺少
`cublas.h`。维护 session 现在按显式选项执行，默认
wheel 发现 ON。所有失败日志保留。私有 unit 与逐设备 nox
缓存独立；任务自有缓存正常复用，不复制 JIT
二进制，也不把后续运行称为另一次冷构建。CUDA wheel 指纹的
首次编译串行执行。完整引擎依赖的 `pip check` 对未使用的引擎 extras/pin 仍
不满足；本组件结果不证明存在可安装的完整引擎依赖闭包。

全部 **2890 个原始 Git blob 与运行后工作树文件**（含符号链接
目标文本）与冻结源码快照匹配。后续 CUDA 运行使
20 个原始 CPU raw 文件保持不变。四十个原始 c997 raw 文件与八个
新派生比较文件被保留。完整封存覆盖 **199 个文件 /
1035348925 字节**，SHA256
`4b67d5ebc7a1fdd1fd871fa5271b46a31535429babb94495ac41eddb8a95cc7a`。
未版本化命令、实际平台文件、raw 数组、独立重检、
失败尝试、官方 wheel 输入与 `scorecard-installed-source-v4.json`
位于 `$JITTOR_LAB_ROOT/_state/vllm-installed-source/20261015/`。
通过更新后的 [vLLM skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/vllm-torch-compat/SKILL.md)
以真实源码/wheel 输入、全新独立解释器与新的外部
输出路径复现。本结果不授予整模型、引擎、Omni、训练、
预训练、额外后端或性能支持。

`2c9b2176d` 处的主整合精确保留其余 **73 个 nox 函数**，
并将其余 11 个交付任务文件与 c997 匹配。在 `ffff94ee2` 的
后处理修复之后，聚焦 host 检查通过 **77 / zero skipped**，core 通过
**322 / 42 skipped / one existing expected failure**，完整结构加
compat 结构通过 **1627 / six skipped**（206.34 秒）。两项结构 skip
是解释器安装解析检查；不从源码路径 import 推断
editable 安装。中央终端日志与整合源码
检查未版本化，位于
`$JITTOR_LAB_ROOT/_state/vllm-main-integration-5def/20261016/`。
