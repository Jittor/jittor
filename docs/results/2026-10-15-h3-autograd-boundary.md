# 真实上游 CUDA 路径中的 MiniMaxH3 autograd 边界

- 状态：根因已验证；完整模型的梯度/训练准入仍被阻断。
- 日期：2026-10-15。
- 初始诊断源码：干净检出 `2857a614b30d5315da74621c662a759e3f187ea4`。
- 既有诊断源码：`521314943b480517b6fb422c09359fc521081a8f`。
- 维护者：下游验证与 vLLM adapter 维护者。
- 复查时机：固定版本的 Omni 操作分派、Triton/CUDA autograd 注册、受支持平台
  或 H3 训练 API 发生变化时。

本次后续工作保留 [SiLU/H3 报告](2026-10-15-h3-silu-contracts.md)、
其原始的默认 oracle 失败以及全部封存原始文件。它调查为何两个运行时都缺少
32/36 个参数梯度和全部四个浮点输入梯度；它不替换原始按缩放相对值的 0.002（0.2%）
前向预算，不选择 MATH oracle，不把组件计为模型，也不只训练最后的 head。

## 实际标识与协议

干净上游仍为 vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`
（确切 release 0.29.0）和 Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8`（实际安装 SCM
`0.29.0rc2.dev58+g446c2b547`）。独立解释器为二进制 Torch
`2.13.0+cu130`，实际扩展路径/SHA256 在新的既有运行中捕获。这不是 Diffusers 用的
Torch 2.11+cu128 解释器。没有任何新的追踪改变旧测量的实际版本或源码标识。

tiny 版真实 `MiniMaxH3DiTModel` 保留一个 main block、一个 refiner、hidden32、
36 个命名参数共 33268 个标量、固定初始 seed3 以及输入 seeds7/19。
全部 36 个参数和四个声明的可微输入（`x`、音频、文本和时间）都显式为完整梯度审计
启用；mask 和整型索引保持为固定控制数据。构造器初始的可训练性单独记录；
它从不缩小审计的期望列表。求值和公共 `model.train(True)` 路径是两次独立的真实执行，
使用默认 attention。

| 请求 | 真实结果 |
| --- | --- |
| 真实 CPU | H3 构造器请求 CUDA 能力；隐藏 GPU 时报告 `No CUDA GPUs are available` |
| 候选 CPU | 真实上游初始化拒绝缺失的 Gloo：`Fallback Gloo backend is not available` |
| 真实 CUDA eval / train | 两次运行中两个 seed 都缺少 32 个参数梯度和全部四个输入梯度 |
| 候选 CUDA eval / train | 两次运行中缺失的名称与计数相同；有物理 CUDA 快照且零后备 |

全部六个进程都保留非零退出状态。四次 CUDA 运行在被拒绝之前保存了实际的缺失名称
清单；每次只有 14 个可用数组，而不是完整的 86 字段输出/loss/梯度请求。
构造器在两个运行时上都有 19 个为 true、17 个为 false 的参数 requires-grad 标志。
审计把全部 36 个参数和四个输入请求记录为已启用，全局
`grad_enabled=True`；构造器的部分冻结不能解释观察到的计算图断裂。真实扩展的 SHA256
为 `819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`。

全部四次运行都保留相同的 37 数组序列化初始状态。每个运行时的 eval/train 对都逐位
保存相同的 14 个数组。真实观测包装器以及随后的既有探针在每种模式下也都逐位匹配这
14 个数组。这验证了观测/标志/元数据的改动保持了被记录的计算；
**它并没有把完整的候选梯度集合与独立 oracle 比较，也没有确立完整模型的数值正确性**。
候选方使用 CUDA 12.2.140 工具链和 strict CUDA kernel 数学，TF32 为 OFF/highest。
图捕获/重放与性能未单独验证。

## 最早的图断裂

真实模型上的观测钩子保持原始操作，并记录每次双输入执行中的 92 个模块/函数边界。
最早观察到的 true-to-false 梯度边界是 token refiner 的 RMSNorm，其实际选中的方法为
Omni `RMSNorm.forward_cuda`。它分配一个输出，把 `weight.data` 传给编译后的 `_C`
RMS kernel，并返回一个没有 autograd 边的输出。Omni `SiluAndMul.forward_cuda`
同样用真实的 `_C` kernel 写入一个已分配的输出。

`VllmConfig(custom_ops=['none'])` 控制的是 vLLM 的 CustomOp 注册表。Omni 独立的
`diffusion.layers.custom_op.CustomOp.dispatch_forward` 不读取该设置；它从实际平台
选择 `forward_cuda`。把该配置理解为禁用每一个 Omni 融合 kernel 是错误的。
既有诊断现在分别记录请求和每个实际选中的方法；它不会改道它们。

真实 CUDA 调制函数独立确立了更多断裂：
`rms_norm_indexed_scale_shift`、`indexed_gate` 和
`indexed_gate_rms_norm_scale_shift` 分配 `empty_like` 输出，并用裸 Triton launch
填充它们，没有 autograd 注册。在真实 Torch 中，三者尽管输入可训练，却都返回
`requires_grad=False`。最后的 `indexed_scale_shift_` 在其目标上保留了既有的计算图，
但不会为 shift/scale 增加边；仅凭非 None 的输出梯度不足以证明该函数的完整导数契约。

这些是上游推理操作的契约。它们不能通过设置额外的 requires-grad 标志、调用
`train()`、修改候选方的核心数学、发明一个 backward，或只把四个 head 参数报告为
完整训练来修复。

## CPU 与官方替代方案

每个设备八个真实源码函数用例覆盖 FP32/BF16。三个 out-of-place 的 CPU 分支
对其全部输入都有真实梯度。
最后的 CPU in-place `indexed_scale_shift_` 反而在反向时抛出真实的 Torch
版本计数器错误。对应的 CUDA 用例没有暴露新的 shift/scale 图边。
这些孤立函数结果不构成可执行的完整模型 CPU 路线。

固定版本的 Omni 内置平台注册表声明了 CUDA、ROCm、NPU、XPU 和 MUSA；
其回退为 `UnspecifiedOmniPlatform`，后者声明了一个 CPU 设备但不实现所需的
diffusion-attention 选择器。此处没有编造任何 CPU 平台，CUDA wheel 的失败也不能
证明每一个可能的树外 CPU 插件都不受支持。

实际 H3 pipeline 的 `forward` 被 `torch.no_grad()` 装饰。源码的官方区域编译辅助函数
只编译 H3 声明的 `MiniMaxH3DiTBlock` 区域，把最终归一化留在这些区域之外。
它是推理编译特性，不是完整训练契约的证据。在被审计的固定版本 H3
模型/pipeline 和编译所有者中，没有找到替代的完整训练 API。实验性的整模型编译和
树外插件既未被本有界审计授予，也未被排除。

## 既有诊断的改动与验证

只有既有的探针及其宿主契约发生变化。`--model-mode eval|train` 默认沿用先前的 eval
行为，并调用真实的 Module 标志 API。元数据现在区分 vLLM 请求与实际操作方法，
记录构造器与请求中的可训练性、全局 grad 状态、可执行文件以及真实扩展 SHA256，
并检查额外执行的上游 activation/dispatch 源码。缺失梯度仍会保存其真实清单并导致
非零退出。此失败之后不运行任何优化器轨迹。

十项探针宿主契约通过，包括在不调用或替换选中方法的情况下进行观测，以及拒绝非法的
可调用路由。完整结构套件连同这十项测试通过 1,518 个用例，含四项已声明/加速器跳过，
零失败。Ruff、diff 检查和仓库布局通过。随后的文档契约通过 13 个用例，其中两处中央
索引/MANIFEST 登记为集成方取消选择。仅有一个诊断工具及其宿主测试发生变化；
没有新的原生计算、compat 行为或 adapter 修复。

首次宿主尝试使用了自动 shim 缓存，并在完成前被终止；成功的宿主/结构运行使用显式的
隔离单元缓存。全部新的模型诊断使用独立的 CPU/CUDA benchmark 缓存，源码全程保持干净，
首次编译为串行。

在 Omni `446c2b5474dabffccf9a0034836eeab77c62b4b8` 上，既有的源码重新生成清单在
CPU 和 CUDA 上仍为 `model_family 0/92`、`diffusion_model 0/68` 和
`pipeline_model_type 0/64`。没有新的完整模型或 pipeline L2/L3 记录被准入，
也不得出任何性能/L4 结论。另一个进程占用了已分配的加速器；此处只尝试了正确性诊断。
未完成的 R27/R28 目标没有被关闭。

## 证据

原始文件未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/omni-h3-autograd/20261015/`：确切的命令/环境、
源码观测、CPU/CUDA 孤立函数记录、原始完整模型数组/权重与缺失清单、图观测以及验证
日志。初始钩子使用上述干净的同步源码；随后既有探针运行各有自己干净的源码 SHA。
既有的 ac4/5216 和 Diffusers 1a92 封存文件既未被覆盖也未被重新标注。

`sealed-evidence-manifest-v1.json` 绑定 51 个原始文件，1,377,904 字节，
SHA256 为 `2c02826b8defd601421c94fc44cbc86f52b7755c08183b696ed058c045fcbdd8`。
它包含两次 CPU 环境搭建失败、全部四次被拒绝的 CUDA 梯度审计、原始数组/权重、
独立的图/源码观测、确切的命令与环境、实际原因审计以及验证日志。缓存/venv 目录树
和此后的文档被排除。封存之后再次检查了完整原始清单哈希与自有诊断源码哈希。
