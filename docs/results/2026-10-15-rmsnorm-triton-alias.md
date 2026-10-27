# RMSNorm 可选契约与 Triton 指针别名

- 状态：已验证 CPU/CUDA 算子契约与一个维护 vLLM 组件门禁。
- 复查日期：2026-10-15；owner：Torch 兼容与 Triton bridge 维护者。
- 整合基线：`c14bdb2f49653a7422f8bcf726a55d089b46f4b1` 加本次改动。
- 源码改动：`b48e000d3525a3b85dbfd398c0b6ce4ffa09676f` 与
  `72e12a9c662b46d5a5389f7e031c3997520d4562`。
- 复查条件：module-call dispatch、RMSNorm 契约、Var storage/strides、
  Triton 参数指针或受守卫拷回、上游 vLLM 或共享
  ecosystem runner 变化。

## 已确认的失败与修正

通用 RMSNorm 快捷路径在模块的 `has_weight` 或 `pass_weight` 禁用
affine tensor 时仍会消费它，并把部分方差当作
完整 reduction。一个刻意未使用的 CPU 权重还导致 CUDA 设备不匹配。
Python 与原生重复推理 dispatch 现在把这些契约交回
模块自身的 forward。两个守卫都按常规解析可选属性。缺失
属性只清除 AttributeError；其他 descriptor 错误仍会传播。
带权重的标准邻接保留其既有原生实现。

测试对照 NumPy float64 检查前向值
与解析的输入/权重导数，然后在 no_grad 下重复推理两次。
覆盖非单位权重、未使用的 host 权重、部分方差、
缺失可选标志与标准带权重邻接。独立 binary Torch
解释器在候选验收前已在各实际设备上通过全部五个测试。

另一个真实 `@triton.jit` 自增 kernel 暴露了静默写丢失。
bridge 把一个输入 bounce 到其受守卫 buffer，传入
别名输出的真实指针，随后又把旧输入拷回，
覆盖了正确的原位结果。修复前独立 Torch 通过六个
FP32/BF16 用例；候选只通过两个不重叠输出用例。
修复后，六个保存的候选数组
与其独立参考逐位相等。

bridge 现在在 bounce 参数之前检测物理寻址字节区间之间的
重叠。重叠参数保持其真实指针关系，与参数名无关。
区间分析包含 stride 空洞与偏移；未知 layout
保守地禁用独立 bounce。测试覆盖两个输出名、两种 dtype、
分离输出、同一对象、同 storage view 与偏移 view。
负 stride 的范围检查是 host 契约，不声称执行负 stride CUDA view。

## 整合验证

| 检查 | 结果 |
| --- | --- |
| CPU RMSNorm 与指针范围契约 | 8 passed，0 skipped |
| CUDA RMSNorm/真实 Triton 写入与三个 host 范围契约 | 9 passed，0 skipped |
| 两个定向会话，全会话 fallback 守卫 | 计数 0 → 0 |
| Core，native 与 Torch 模式 | 322 passed，42 skipped，1 个既有 xfail |
| Structure 与 tools | 1716 passed，6 个声明/环境 skip |

CUDA 会话为 RMSNorm 测试显式选择 CUDA，并使用真实
安装的 Triton 包。CPU 隐藏 CUDA。每个进程有自己的编译缓存；
首次编译串行。实际源码为本 checkout，CUDA 原生
kernel 数学为 strict。这些是正确性结果，不是性能比值。

整合使用 Linux x86_64、AMD EPYC 9554、Python 3.11.16 与 NumPy 1.26.4。
CUDA 使用一张分配的 RTX 4090/SM89、driver 580.178.04、Jittor 的 pip-cu122
provider（`pipcu122_a89fa4fed08c`）、nvcc 12.2.140 与 GCC 11.5.0。物理分配
与原始编译路径保存在外部日志中。RMSNorm fixture 是 FP32
2×4 张量；别名 kernel 使用 16 个 FP32/BF16 元素。这些 fixture
不确立 TF32 matmul、真实规模模型、graph regime 比较或内存/吞吐测量。独立子进程
oracle 是 Torch 2.13.0+cu130；该二进制的 CUDA 构建与
Jittor 的工具链不同。

仅源码审计最初未能发现新的 vLLM 测试。其负向
回归与修正现在保留实际的 class-qualified selector，同时
不设置审计的已验证级别。公共 `ecosystem_vllm` nox 会话
固定独立 Torch 2.13.0、vLLM 0.29.0 与 Transformers 5.13.0，要求全部四个
浮点输入梯度，并在子任务的 precommit 验证中通过了一个真实 CPU 与一个真实 CUDA
组件测试。其无参数组合通过公共原生
路径执行 SiluAndMul、无权重 RMSNorm 与 RotaryEmbedding。它不执行
任何模型族或 engine，因此不能确立 L3。
随后干净、registry 绑定的 CPU/CUDA 组件证据记录在
[vLLM 组件 scorecard](2026-10-15-vllm-component-scorecards.md) 中；它保留
自己的测量基线，不授予 engine 覆盖。

使用所选设备的实际工具链、独立缓存与已安装 Triton
复现定向测试：

```bash
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda backend_fallback=error \
python -m pytest -q compat/tests/torch/test_rms_norm_optional_weight.py \
  compat/tests/triton/test_triton_torch_compat.py::TestTritonTorchCompat::test_pointer_alias_preserves_inplace_writes \
  compat/tests/triton/test_triton_backend.py::TestPointerRangeAliasing
```

CPU 端隐藏 CUDA，设置 `JT_USE_CUDA=0` 与 `JITTOR_TEST_DEVICE=cpu`，并省略
CUDA Triton 启动节点。vLLM 门禁为 `python -m nox -s ecosystem_vllm`，
需要两个已准备的解释器；其会话拒绝不完整配置。

未版本化子任务证据位于 `$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`。
其 `b48-validation-manifest.json` 封存 23 个原始文件；SHA256 为
`41bbf487346f9af992aacc682d6039e1f79569e9f314a4d6caf42a3179f15e7c`。
这些 precommit 结果保留基线 `6f2e479f63b3686b0e63c41d324148a769bacddd`
加已提交改动；它们不会被重标为后续干净测量。
整合独立重哈希全部 23 个产物，比较六个别名
数组，并把全部六个 owned 修复文件与其提交匹配。新的整合
日志、设备/fallback metadata 与初始审计失败位于
`$JITTOR_LAB_ROOT/_state/vllm-owner-integration-c14/`。

本批不产生 engine、Omni 数值验收、任意 RMSNorm dispatch 契约
等价的结论，也不产生 NPU 或 ROCm 支持。Omni 中单独的 BF16 乘法
顺序契约仍在调查；这些改动保持
既有的原生 RMSNorm 数学。

## 后续：自有 forward 与私有 RMS 输出存储

原始数值探针保留基线 `963566b462e645f81994e48aaf81a5cda8266899`
加逐项哈希的 owned 改动。在冻结测试并保留
这些改动后，远端整合产生 `1ae4a257511633eeeb745784d99c1792a5bbe49c`；
最终 core/structure 回归使用该基线加同一份 owned 源码
字节。这保持了前一批的原始文件与源码身份。独立参考是真实的 vLLM 0.29.0 CUDA 扩展
（tag 提交 `98dff2a81d747d1dba01a47f939f48c3526d4206`）配 Torch
2.13.0+cu130；候选 CUDA 使用上述同一套 cu122/strict 工具链。
未重标任何 release 版本或旧数值记录。

带可调用 `forward_native` 的模块拥有其 dispatch 与数值
契约。Python 与重复推理 C++ 按类名的 RMS 快捷路径现在都
让该模块的真实 forward 负责。三个 dtype 回归使用
非单位 affine 权重，检查其输出、输入与 affine 梯度，以及
随后两次 no_grad 调用。只有在观测 Python
调用计数时禁用 replay；graph capture 本身不变。独立 Torch
解释器在 CPU 与 CUDA 上通过全部八个公共契约测试。保存的
最终候选数组也与独立 oracle 一致：FP16/BF16
输出、输入/权重梯度与重复输出逐位相等。FP32
最大绝对差在 CPU 上为 7.1525574e-7，在 CUDA 上为 4.7683716e-7；
最大相对 L2 误差低于 8.7e-8。

私有编译的 vLLM RMS 算子在最终低精度
cast 之前独立地以 FP32 乘以 affine 权重。其 adapter 现在向
既有公共 `jt.nn.rms_norm` 提供 FP32 操作数，然后 cast 结果，不
改变原生/Hugging Face RMS 数学。对保存的 BF16 7×32 fixture，
旧的先 cast 后加权结果有 58/224 个元素不同；修正后全部重复
输出都与真实 CUDA 扩展精确一致。这是算子
fixture，不是对每个归一化变体的断言。

同一个真实扩展暴露了独立的输出 buffer 契约：
`out.assign(...)` 改变了输出的物理指针，尽管其 Python
身份保持稳定。adapter 现在在 `jt.no_grad()` 内调用公共 `out.copy_`。
FP32/FP16/BF16 oracle 与候选探针验证返回值、
Python 身份、物理指针与既有 view 值。FP16/BF16
输出与别名数组逐位相等；FP32 最大绝对差
为 2.3841858e-7。回归还检查输入/权重不可变、无新的 autograd 历史、所选设备上的真实分配，
以及不变的零 fallback 计数。候选 CPU 测试使用
独立 NumPy 公式；不声称有编译版 vLLM CPU kernel。
adapter 不会强制清除输出上既有的 requires-grad 标志。

| 后续检查 | 结果 |
| --- | --- |
| 独立 Torch 公共 RMS 契约 | CPU 8 / CUDA 8 passed |
| 候选 CPU RMS/norm 邻接 | 33 passed，6 个声明硬件 skip |
| 候选 CUDA RMS/norm 邻接 | 36 passed，3 个既有 xfail |
| 原生与 Torch CPU core | 322 passed，42 skipped，1 个既有 xfail |
| 完整 structure | 1501 passed，4 个声明/硬件 skip |

显式配置所选设备以复现候选邻接：

```bash
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda PYTHONPATH=python:adapters \
python -m pytest -q compat/tests/torch/test_rms_norm_optional_weight.py \
  compat/tests/torch/test_torch_compat_norm.py \
  adapters/tests/vllm/test_rms_norm_precision.py
```

在独立的真实 Torch 解释器中直接运行公共测试文件
（禁用 shim）即可使用同一 fixture。私有编译 oracle
需要真实匹配的 vLLM CUDA 扩展。原始源码快照、探针、
前后数组与定向日志由上述未版本化证据目录中的 `rms-own-forward-validation-manifest-v1.json` 绑定，
SHA256 为 `fc794439d203e39e98ae398f5ae0ca512bdd11df1f53f00388a0736538d3d333`。

本次修复不接纳 MiniMaxH3DiT 或 Omni engine。真实冻结权重
H3 fixture 仍不满足既有数值标准：RMS 修复后 seed-7 loss 相对
误差为 1.7086%。保存的中间 Q/K/V 在首个后续
SDPA 不匹配处完全相同；attention 精度仍是另一个
共享 owner 的调查。上游 CPU 平台选择也拒绝
这条 Omni 路径。本后续不授予任何新的模型族分子、L3、性能或 CPU Omni
支持。
