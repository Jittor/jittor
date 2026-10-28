# SiLU 精度与 MiniMaxH3 推理边界

- 状态：SiLU/SwiGLU 契约已验证；Omni 模型准入被阻断
- 日期：2026-10-15
- 基线：公共 SiLU `3fe6a787a2a58ca8f7da4edd9a0fae0e91ea5be9`；最终干净合并源码 `ac4d4cf28f7d2c68c0d4d7256fb1e6f5a3f92b1f`
- 维护者：Jittor 核心、CUDA 与 vLLM adapter 维护者
- 复查时机：SiLU/SwiGLU 精度、adapter 分派、SDPA 后端选择、H3 上游源码或其 autograd 契约发生变化时

公共低精度 SiLU 路径现在以 FP32 计算其操作数，并把结果舍入到请求的 dtype。
CUDA 推理 SwiGLU 在乘以第二个操作数之前保留了这一 SiLU 舍入边界。vLLM adapter
保留了上游的 `forward_native` 方法以及私有 CUDA 方法实际的 no-autograd 契约。
这些修复通过了各自的公共与真实库 CPU/CUDA 检查。它们**并不**确立整模型 MiniMaxH3
的正确性或训练能力：原始默认 attention 比较仍然失败，真实 Torch 和候选方都缺少
大部分模型梯度。

## 固定源码与环境

独立 oracle 使用真实的 Torch `2.13.0+cu130` 及其实际编译的 vLLM 扩展，而非 shim。
两侧都执行 vLLM `0.29.0`，确切源码
`98dff2a81d747d1dba01a47f939f48c3526d4206`，以及 Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8`，安装的 SCM 版本为
`0.29.0rc2.dev58+g446c2b547`。旧 runbook 的 cu129 环境和更早的源码 SHA 不为此组合
提供任何数值证据。候选编译使用 CUDA 12.2.140；设备进程使用独立缓存、单宿主线程、
即时执行和 strict CUDA kernel 数学。TF32 被禁用，Torch matmul 精度为
`highest`。候选方快照在宿主观测之前断言实际物理设备；后端后备被禁止，其计数保持
为零。不报告任何性能、L4、多 rank 或引擎结果。

H3 fixture 是真实的 `MiniMaxH3DiTModel`，包含一个 main block、一个 token refiner、
hidden=32、四个 head、head 维度=8、FFN=64、33268 个参数。全部 36 个命名参数和一个
buffer 由 NumPy seed 3 以相同方式初始化。输入使用 seed 7 和 19：两个 packed 文本、
两个视频和两个音频 token。这是 tiny 模型配置，不是预训练 checkpoint，也不是 Omni
pipeline。原始前向协议冻结参数，同时其四个浮点输入需要梯度；它不是
`torch.no_grad()` 测量。

## 已验证的修复

保存下来的 H3 SiLU 输入首先区分了过早的窄 sigmoid 舍入与合法的 SDPA 算法差异。
修复之前，BF16 公共 SiLU 已有 74/256 个输出值不同。修复之后，在保存的输入上，
FP16/BF16 公共 SiLU 以及 SiLU-times-up 输出在 CPU 和 CUDA 上都与真实 Torch 逐位
一致。一阶和二阶导数通过冻结的 1% 缩放预算；它们并未被描述为精确值（最大缩放误差：
CPU 0.719424%，CUDA 0.675676%）。FP32/FP64 行为和解析二阶导数仍被覆盖。

真实的 vLLM `SiluAndMul` 矩阵使用三种 dtype 和两种输入梯度状态。
`custom_ops=['none']` 走原始的上游 native 方法；默认 CUDA 走真实的编译 `_C` oracle。
低精度 native 输出与 CUDA native 输入梯度逐位一致；CPU native 梯度保留最大缩放误差
0.359712%。默认 CUDA 输出与全部六个数组逐位一致，且两侧都没有暴露 autograd 连接，
即使输入需要梯度也是如此。adapter 不会静默地把 native 路径变成那个仅推理方法。
在已安装的 CUDA wheel 上，CPU 默认 `_C` 因 kernel 不可用而失败，而真实的 native
CPU 路径可以执行。

| 源码修复批次上的检查 | 结果 |
| --- | --- |
| 原始所有者批次上的公共 SiLU 测试 | CPU 6 通过；声称的 CUDA 六节点运行在测试内部选中的是 CPU，故不提供 CUDA 证据 |
| 原始 CUDA 环境的邻居运行 | 15 通过；每节点的设备执行记录不足，故该计数不提供额外 CUDA 证明 |
| 合并的 adapter/kernel/公共定向测试 | CPU 12 通过；原 CUDA 进程报告 27 通过，但其六个公共 SiLU 节点在 CPU 上执行，不能计为 CUDA |
| 既有 `nox -s ecosystem_vllm` | CPU 1 / CUDA 1 通过，零跳过 |
| 原生与 Torch CPU core | 322 通过、42 项已声明跳过、1 项既有 xfail |
| kernel/adapter 修复后的完整结构 | 1507 通过、4 项已声明跳过、零失败 |

既有的组件门禁仍覆盖同样的三个 CustomOp 成员；增加精度配置并不会增加模型族覆盖。
相关的共享修复保留各自的源码与证据标识：
[RMS/别名契约](2026-10-15-rmsnorm-triton-alias.md)、
[like 工厂](2026-10-15-like-factory-leaves.md) 以及
[SDPA opmath](2026-10-15-sdpa-opmath.md)。

## 原始 H3 默认比较仍被阻断

全部 37 个初始状态数组精确一致。原始 0.2% 前向预算未变。相对真实默认 attention 的
最大缩放输出误差为：

| 输入 seed | 视频 | 音频 | Loss |
| --- | ---: | ---: | ---: |
| 7 | 0.120073% | **0.583711%** | **1.701884%** |
| 19 | 0.181590% | **0.449398%** | **0.265659%** |

六个字段中有四个失败。一项独立的真实 Torch MATH 校准（实际观察到使用数学算子，
且融合替代路径不存在）在全部六个最终字段上与候选方逐位一致。在 194 个中间快照中，
184 个逐位一致；十个 FP32 time-embedding/projection 别名最多相差
1.4901161e-8。每一个被观测的 BF16 中间值相对于该 MATH 校准都是精确的。
这说明该 fixture 上的剩余差异在**真实默认实现与真实 MATH 实现之间**同样存在。
它不替换原始参考，不改变其预算，也不把失败的默认门禁变成通过。

全梯度审计另行让全部 36 个参数和四个浮点输入都可训练。在真实 Torch 和候选方中、
在两个 seed 上，都只有四个最终的视频/音频 head 参数有梯度。
**32/36 个参数梯度和 4/4 个输入梯度缺失**。诊断写出实际数组与缺失名称清单，
然后以非零状态退出。任何编造的 backward、填零梯度或仅 head 的 SGD 轨迹都不被接受
为完整模型 L3。

## CPU 与覆盖边界

最先尝试 CPU，隐藏 GPU 并显式禁用候选 CUDA。
已安装的真实 CUDA wheel 的 H3 构造器请求 CUDA 能力并失败；候选方拒绝不可用的 Gloo。
仅凭这些 wheel 失败不能证明所有 CPU 实现都不存在。固定版本的上游另行支持 CPU
打包，但其内置 Omni diffusion 平台插件为 CUDA、ROCm、NPU、XPU 和 MUSA。
`UnspecifiedOmniPlatform` 声明设备 `cpu`，但不实现所需的 diffusion-attention
选择器。该源码快照没有声明任何内置的 H3 CPU diffusion-attention 运行时；
外部插件与未来的 CPU 构建不在这一源码有界结论之内。

源码重新生成的 Omni scorecard 中**没有任何被准入的数值记录**：
模型族 **0/92**、diffusion 模型 **0/68**、pipeline 模型类型键 **0/64**，
各自绑定完整的 `446c2b...` SHA。这些是相互独立的维度，不是可相加的覆盖率。
前向诊断不是准入收集器，也没有被登记为通过的共享生态模型用例。R27/R28 仍然开启；
没有新的 Omni L2/L3、CPU 运行时、引擎或性能主张。

## 复现与原始证据

独立的既有工具源码 `5216ac20da92d67fbf374fdd5c0898ff2ec84fbb`
包含 `agent/skills/vllm-omni-torch-compat/scripts/probe_h3.py`，
支持独立的运行时/设备、固定的干净上游 pin、覆盖保护、源码标识、物理设备检查，
以及在任何梯度缺失时以 `--mode grad-audit` 失败。仅针对真实实现的
`--torch-math` 会生成独立的校准结果。请使用
[Omni skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/vllm-omni-torch-compat/SKILL.md)
中的命令与环境前置条件。

使用显式设备选择和所选运行时运行公共回归：

```bash
JITTOR_TEST_DEVICE=cuda PYTHONPATH=python \
  <jittor-python> -m pytest -q tests/nn/test_silu_opmath.py \
  tests/backends/cuda/test_swiglu_dtype_boundaries.py
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda PYTHONPATH=python:adapters \
  <jittor-python> -m pytest -q adapters/tests/vllm/test_layers.py
```

使用隐藏 GPU 且设置 `JT_USE_CUDA=0` 的独立 CPU 进程；仅 CUDA 的测试在目标设备上
检查，不计为 CPU 执行。既有真实库门禁要求独立的 `REAL_TORCH_PYTHON`、已安装的匹配
vLLM，先 `JITTOR_TEST_DEVICES=cpu` 再 `cuda`，并运行 `nox -s ecosystem_vllm`。

未纳入版本控制的产物位于
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`。最终生产证据清单
`silu-h3-final-evidence-manifest-ac4-v1.json` 绑定 173 个文件，4,403,832 字节，
SHA256 为 `afcaf5b5b7c2225b7849f2e6ea5cbb7d1e691507ab93ebc3961114d26a8c965e`。
它包含修复前/后的数组、实际库调用路径、固定输入 H3 的原始记录与缺失梯度清单、
源码快照和门禁日志。原始元数据仍具权威性：更早的 `86073` 和 `cea664` 以及显式的
自有源码哈希不会被重新标注为干净的 `ac4`。原始 CPU 失败、默认失败以及此前两处中央
文档登记失败都予以保留；后者已由随后发布的中央登记清除。


随后从干净检出 `5216ac20da92d67fbf374fdd5c0898ff2ec84fbb` 运行了该工具，
它保留了生产修复并集成了已发布的 CPU MPI 源码 `01351fe61`。先重试 CPU，
仍保留真实的平台/Gloo 错误。在 CUDA 上，每个运行时都逐位复现了其原有的干净 ac4
六个前向字段和 37 个初始状态。每个运行时独立的梯度审计也逐位复现了全部 14 个可用
数组和 37 个状态，记录了相同的缺失名称，并以非零状态退出。两个源码标识保持独立。

该工具的八项宿主契约通过。完整结构最初通过 1506 项测试、四项跳过，并拒绝新宿主
测试中收集期的动态导入；把该导入移入 `setUpClass` 修复了该测试。随后全部 42 项受
影响的 pytest 与子进程结构契约通过。没有放宽任何结构规则。布局与 Ruff 通过。
这些仅涉及工具的结果不能取代上文生产批次的核心或真实库门禁。

额外的 `maintained-h3-tool-final-manifest-5216-v1.json` 绑定 25 个原始文件以及三个
工具/测试/skill 源码哈希，SHA256 为
`49d0d6a0688b17517514abb5685305c1cc15938e8658ba954774ad7f013db8d9`。

## 主树集成与修正后的设备证据

集成自同步后的 `4ce19215cba8fb15d7b347f53fda012ced9b4143` 开始。
原始 runner 设置了进程 CUDA 标志，但漏掉了 `JITTOR_TEST_DEVICE=cuda`；
因此公共 SiLU 测试的本地作用域选中了 CPU。其原始的六节点 CUDA 主张以及记录不足的
邻居主张已在上文撤回。原始日志保持不变。独立的保存输入探针和真实库探针具有显式的
设备参数和物理驻留断言，保留各自的有界证据。

首次带真实 CUDA 选择的集成运行暴露了 float64 SiLU 误差
`1.55156931e-7`：共享 CUDA sigmoid 展开对 double 输入也调用了 `expf`。
它现在对 float64 选择 double `exp`，保留既有的 float32 路径。公共 SiLU 原有的
`2e-13` double 容差未变，其测试现在还直接检查 sigmoid 以及前向、梯度和二阶导数。
这是在 `ac4` 之外的额外核心修复，而不是对更早源码测量的重新标注。

在该修复之后，真实 CUDA 上的公共 SiLU/SwiGLU/unary 回归通过全部 29 个节点，
零跳过；adapter 契约在 Torch 模式下单独通过六个节点。
CPU SiLU/unary 回归通过 11 个节点，另有 15 项已声明 CUDA 跳过，宿主
probe/adapter 契约通过 14 个节点。一次初始的 CPU erfinv 参考检查失败，
因为环境中只有部分 SciPy 命名空间；改用已安装的真实 SciPy 1.13.1 重试后通过。
最初的混合原生/adapter 命令被进程模式策略拒绝；最终运行使用独立的原生与 Torch 进程。

Core 通过 322 个节点，42 项跳过和一项既有 xfail；完整结构通过 1,505 项，六项跳过。
一项独立集成审计重新哈希全部 173 个生产证据文件、25 个工具证据文件以及三个确切的
工具源码，并在原始预算内重算了 144 组保存输入/组件数组对。
六个 H3 输出和 37 个权重被独立复核：原始默认参考仍然在四个字段上失败，
而全部六个 MATH 校准字段字节一致。不产生任何新的完整模型准入。

未纳入版本控制的集成证据位于
`$JITTOR_LAB_ROOT/_state/silu-integration-4ce/`；独立的 H3 前向与生产哈希审计位于
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-013/next-silu-*.json`。
