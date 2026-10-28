# 固定的 H3 上游推理 kernel 与训练边界

- 状态：原有私有 CUDA 路径缺少完整 AD；训练 opt-in 仅为设计。
- 日期：2026-10-16。
- 测量基线：干净 `aa9e669f514432a3aae77840d1604458c1b9f3ff`。
- 文档基线：同步后的 `00ceb26fd395707fb33649cc43709af045110a9a`。
- 维护者：下游验证与 vLLM adapter 维护者。
- 复查条件：固定的 Omni dispatch/kernel、拟议的训练契约，或 compat 公共 `owned_runtime_hook`、`transaction`、`module_patcher` API 变化。

固定的上游**推理实现不提供整模型训练 AD**。调用 `train()` 并启用全部参数/输入梯度，并不会为其输出缓冲区 CUDA/Triton kernel 提供导数。这延续[原始图边界诊断](2026-10-15-h3-autograd-boundary.md)；它既不改变那些原始文件，也不把匹配的缺失梯度当作 L2/L3。
单独的 aa9 attention 工作改进了原始默认前向与原语梯度精度；模型缺失的 32 个参数梯度与四个输入梯度仍是一项独立、未解决的边界。

## 全新的实际诊断

上游未改动：Omni `446c2b5474dabffccf9a0034836eeab77c62b4b8`
（实际 SCM distribution `0.29.0rc2.dev58+g446c2b547`）与 vLLM
`98dff2a81d747d1dba01a47f939f48c3526d4206` / `0.29.0`。
真正的独立 oracle 是二进制 Torch `2.13.0+cu130`，扩展 SHA256
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`。
已安装源码、完整 wheel 载荷、METADATA/RECORD 与源码身份
在执行前后都被校验。Omni 的固定 C6 wheel 是经复查的
本地源码构建构件，**不是 PyPI 官方 wheel**。不修改任何包、版本、
上游实现、candidate core 或 adapter。

| 实际请求 | 结果 |
| --- | --- |
| CPU，16 个组件 case | 14 组完整梯度；两个最终 in-place case 在反向时因版本计数器错误失败；worker 退出码 1 |
| CUDA，同样 16 个 case | 零组完整梯度；原有私有 kernel 实际执行；worker 退出码 1 |
| CPU / CUDA tiny Q/K 对照，各两种 dtype | 两侧实际 eager 调用都产生全部四个请求的 q/k/norm-weight 梯度；每个 worker 退出码 0 |

Case 覆盖 FP32/BF16、RMSNorm 与 SiluAndMul（各带真实 eval/train 标志），以及全部四个 H3 modulation 函数。全局 grad 已启用，inference mode 已禁用。实际 CUDA 输入/输出指针属于已分配卡；RMSNorm/Silu kernel 出现在实际算子 trace 中，modulation Triton launch 完成。这些是真实上游组件诊断，不是 candidate 比较或整模型 CPU 支持主张。CPU 与 CUDA 使用分离的进程与私有缓存；其精确命令与环境已保存。本最小流程未采集全新的进程库映射快照。

RMSNorm 把 `weight.data` 传给真实 `_C::rms_norm`，填充已分配输出，返回 `requires_grad=False, grad_fn=None`；x 与注册的 weight 都请求了梯度。SiluAndMul 与三个 out-of-place modulation 函数同样返回没有 AD 边的输出。最终的 in-place modulation 保留其目标原有的 `MulBackward0`，但不给出 shift 或 scale 梯度。因此，非 None 的输出 `grad_fn` 并不证明完整 AD。在 CPU 上，原始 RMS kernel 尝试失败，转而运行上游自身的 native 分支；这与已完成的 CUDA kernel 分开记录。

## 完整的受限边清单

以下归属方穷尽了 **one-main-block / one-refiner / hidden32 / 33268-scalar / 36-parameter** recipe 所触及的私有 kernel 接口。外部清单记录文件哈希、精确函数行范围与完整的其余 32+4 个名称。它不枚举这些图断点修复之后公共算子中可能出现的每一个未来缺陷。

| 相对 `vllm_omni/diffusion/` 的归属方 | 实际固定路径边 / 剩余义务 |
| --- | --- |
| `layers/custom_op.py`：`CustomOp.dispatch_forward` | 独立于 vLLM `custom_ops=['none']` 选择平台 CUDA 方法；train 模式不会切换它 |
| `layers/norm.py`：`RMSNorm.forward_cuda`、`_forward_fused` | refiner 前置 norm/最终 norm、refiner Q/K norm 与最后一层 norm 丢失激活与权重边 |
| `layers/activation.py`：`SiluAndMul.forward_cuda` | 两个 MLP gate 都需要对 fused fc1 输出两半的导数 |
| `attention/ops/minimax_h3_modulation.py`：`rms_norm_indexed_scale_shift` | 主 pre-attention 路径需要 x、norm 权重、索引 shift 与 scale 的导数 |
| 同文件：`indexed_gate_rms_norm_scale_shift` | 残差加 gate 分支与归一化第二输出需要全部六个输入导数，包括索引归约 |
| 同文件：`indexed_gate` | 主残差/门控 MLP 合并需要残差、gate 与分支导数 |
| 同文件：`indexed_scale_shift_` | 最终 AdaLN 需要 x/shift/scale 导数**以及**正确的 in-place 别名/版本语义；CPU 当前报错，CUDA 静默缺少 shift/scale 边 |
| `layers/fused_qk_norm_rope.py` | tiny head_dim8/rotary6 走实际 eager `F.rms_norm` 加 RoPE；四个边都被观测到。128/96 fused Triton 分支在本 recipe 之外，不是已展示的 tiny 模型断点 |

剩余公共算子链在最终完整 AD 中同样是必需的：
video/audio patch 与 text 投影；time sin/cos 与两个投影；
QKV split/view、默认 SDPA 与输出投影；MLP 投影；AdaLN
SiLU/投影/chunk 与重复索引归约；BF16/FP32 cast；
`index_select`/`index_add_` 多模态打包；最终输出选择/mask；
以及 world1 SP prepare/gather 恒等。默认非量化 vLLM Linear 路由
到公共 `F.linear`；SDPA 保留原默认后端。这些源码
事实不能替代完整的 candidate/oracle 梯度比较。
量化 GEMM、更大的 fused RoPE、TP/SP/ring >1、多请求打包、
compile/capture/replay 与性能都在本受限诊断之外。

原始整模型 aa9 总账在全部八个
runtime/mode/seed 行中具有相同的缺失名称：10 个 embedding/time 参数、九个 refiner 参数、
10 个 main-block 参数与三个最终 norm/AdaLN 参数。缺失的浮点
输入为 x、audio、text 与 time。只有四个最终 video/audio head
参数有梯度。它仍然是一次失败的完整梯度审计。本次全新
组件运行既不重跑也不重标更早的整模型运行。

## 显式训练 opt-in：提案，而非实现

按照[适配 skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/downstream-library-adaptation/SKILL.md)，
这些库私有 kernel 属于第三行 adapter/上游修复
类别。它们不建立缺失的公共 Torch/core 能力。
优先采用上游显式的可微训练 API；否则提议一个
单独的版本化、**默认关闭**的 opt-in，位于
`adapters/jittor_adapters/vllm/omni_training.py`，并在同处放置
`adapters/jittor_adapters/vllm/tests/test_omni_training.py`。
注册/bootstrap 改动将另行划定范围。本报告不创建
其中任何文件，也不授予任何训练支持。

拟议实现只拥有上述六个私有算子契约。它必须要么
组合既有公共可微算子并保持已验证的原始舍入，要么
在真实前向周围使用显式指定的公共 autograd
包装器，并配以独立验证的导数。
仅调用 `forward_native` 不够：要保留 BF16/FP32 舍入与
原前向预算，为反向保存稳定值，校验重复
索引与两个输出 cotangent，并为下划线算子定义别名/版本行为。
在没有显式新 API 的情况下用 out-of-place 函数替换它
不是等价修复。不接受为裸
`empty`/Triton 伪造反向、detached 权重、框架私有状态、编译器状态伪装、
CPU 冒充、强制 MATH oracle 或全局 no-grad 变通。

在准入该未来 opt-in 之前：

1. 固定两个完整上游 SHA、源码构建构件与已安装载荷；
   使用 `_common.require_version` / `SUPPORTED_VERSIONS` 并拒绝未知
   版本/源码布局。保持原推理导入与默认值惰性。
2. 继承可重定位/公共框架入口契约与负例测试；
   安装、双重激活、回滚/卸载、导入顺序与部分
   失败都必须保留归属。移除 `adapters/` 后必须让 core 门禁
   保持完好。不要编辑 `torch`/Jittor 版本或全局框架属性。
3. 在同一个**新的**显式 opt-in 下定义可独立执行的
   真实 Torch 训练参考。其原上游没有完整梯度
   oracle，因此新的训练证据不能标记为原始默认 AD。
   在 BF16 真实 CUDA 比较之前，用 FP64/FP32 对照独立公共数学
   或有限差分验证每个导数；仅存在是不够的。
   保留反向输入/输出 dtype 与 in-place 别名规则。
4. 先尝试真实 CPU，保留任何上游平台拒绝，然后做真实 CUDA
   并用独立解释器、物理驻留与零 candidate fallback。
   CPU 组件成功不会虚构整模型 CPU 平台或
   配对 scorecard。缺少必需库在常规情况必须表现为可见 skip，
   在验收机上则必须硬失败。
5. 保持完整 tiny 架构、混合 dtype、原权重 seed3、
   输入 7/19 与默认 attention。按原缩放
   相对 0.002 比较全部输出，两种模式下全部 36 个参数梯度与
   四个输入梯度，梯度预算在执行前固定。既有原语
   0.01 契约不构成省略模型梯度或改用 0.02 的许可。
6. 只有完整 L2 成功之后，才为 L3 校验多步 loss、每个更新过的参数、
   优化器/调度器状态与包括全部 RNG
   流的全新进程保存/恢复。仅 head 更新不合格。任何新的测量训练
   协议都需要单独的源码/原始绑定与常规 CPU/CUDA 配对；
   原有阻塞记录保持不变。

## 证据与独立复核

未版本化的全新证据位于
`$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary/20261016/`。
`independent-component-audit-v1.json` 检查全部 32 个组件行、166 个实际
组件数组与四个额外 tiny-QK 对照 / 40 个数组。它检查
有限值、精确的已保存梯度存在性、真实预期失败与
安装/源码身份一致性；不授予任何数值训练等级。
`bounded-ad-edge-inventory-v1.json` 绑定精确上游归属方行范围与
先前整模型缺失名称总账。

`evidence-manifest-v1.json` SHA256
`e0bef985feb0f83bf2ed64d339e1b160f552256d224c7ff72049bada277e1e24`
绑定 12,010 个产物路径 / 1,198,351,359 字节与未改动测量树的全部
2,912 个 Git mode/blob/content 行。它包含不可变的先前整模型
封条/总账引用；这些显式属于历史，不是全新模型运行。

只读的 `recheck-evidence-v1.py` 及其结果位于同级目录
`$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary-review/20261016/`。运行：

```bash
<NumPy-python> "$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary-review/20261016/recheck-evidence-v1.py" \
  "$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary/20261016/evidence-manifest-v1.json"
```

复核器 SHA256 为
`9361714feeba3598fd567f7aaba85e3b9ac0cd9be16a70169423ece4149c8006`。
它独立通过了全部字节、Git blob 与实际数组，只写
stdout，且不导入任何运行时。十四项纯 host 文档契约
通过；只有集成方归属的索引可达性登记被排除。布局
与 diff 检查通过；本报告未改动任何数值生产归属方。producer 命令被保留以便全新、
独占输出的复现，而不是覆盖这些结果。第一次
封存尝试把仓库的目录符号链接当作常规文件而拒绝；
该保留的失败通过哈希符号链接目标文本来修正，未
改变任何测量源码或数组。本报告不包含任何新的生产计算、compat
行为或 adapter 实现。
