# 自动混合精度请求了混合 dtype 卷积

- 状态：已修复，在真实 CUDA 设备上验证
- 日期：2026-09-14
- 基线提交：`4c3ab0e4`（两个修复提交 `63fc1485`、`4c3ab0e4` 中的后一个，基于 `5a084737`）
- 验证范围：真实 CUDA 设备（MiniMax-H3 实验环境，卡型原文未记）上的
  `tests/backends/cuda/test_cudnn_conv_amp_dtype.py`、`test_cublas_matmul_amp_dtype.py`、
  shim 的 `TestAutocast` 与下文的回归选择，以及 MiniMax-H3 tiny 32x32 t2va 端到端。
  未覆盖：ROCm、NPU
- 维护者：Torch 兼容层维护者
- 复查条件：amp 寄存器的操作数处理变化，或 shim 的 `torch.autocast` 不再只记录
  `jt.flags.amp_reg`

## 问题

`torch.autocast(float16)` 作用于 float32 权重是已验证的 MiniMax-H3 视频解码配方。在
shim 上它构建了一个 float32 输入 / float16 输出的卷积——没有任何 cuDNN 算法能提供
这种请求——于是 `cudnn_conv3d` 死于 `best_algo_idx == -1`，中止了解码。根因是什么，
以及相比 `--vae-dtype float16` 绕过，真正的修复是什么？

## 根因

`compat/torch/grad.py` 的 `_AutocastContext` 把 `torch.autocast` 映射到
`jt.flags.amp_reg`。该寄存器影响算子*为其结果推断*的 dtype（`src/type/nano_string.h`：
`amp_prefer16` 下的 `float_dtype`），但不转换算子的操作数。Torch 的 autocast 正好
相反：它把算子的输入转换到计算 dtype，结果精度低是因为计算本身精度低。

cuDNN 卷积算子以 `create_output(nullptr, dtype_infer(x->ns, w->ns))` 创建输出
（`backends/cuda/kernels/cudnn/cudnn_conv{,3d}_op.cc`）。因此在 `amp_prefer16` 下，
一对 float32 输入被要求产出 float16 结果。逐元素内核能吸收这一点（其生成代码在写回
时转换），但 cuDNN 要求输入、滤波器与输出使用同一 dtype，没有这样的算法；于是该
算子未能满足自身的 `best_algo_idx != -1` 不变式。

## 修复

- `src/core/var.{h,cc}`：`cast_operand_to_compute_dtype(value, dtype)` 返回把浮点
  操作数转换到算子计算 dtype 的 cast，无需转换时返回空。调用者在它所构建的算子把该
  cast 作为输入接收之前，保持其 owner 存活。
- 六个 cuDNN 卷积构造函数（2-D 与 3-D、前向与两个反向方向）计算一次
  `dtype_infer`，转换不同的那个浮点操作数，并通过 `forward` 委托给一个新算子——这
  正是 `BinaryOp` 已用于其整数提升情形的模式。

## 验证

- 最小复现 `jt.nn.conv3d(float32, float32)` 在 `amp_reg=prefer16` 下：修复前在
  `cudnn_conv3d_op.cc:296 [check failed: best_algo_idx!=-1]` 抛出，输入为
  `in: float32, float32 / out: float16`；修复后正常完成并返回 float16。
- `tests/backends/cuda/test_cudnn_conv_amp_dtype.py`（真实 CUDA）：`amp_prefer16`
  下的 2-D 与 3-D 前向（结果等于显式 float16 卷积）、`amp_prefer16` 下的 3-D 反向，
  以及不带 amp、会提升到 float32 的混合操作数对。4 passed。
- 正常路径：`tests/backends/cuda/test_cudnn_conv3d_algo_cache.py` 与
  `tests/backends/cuda/test_cudnn_op.py` 的 `conv3d` 用例原样通过。
- 端到端，MiniMax-H3 tiny 32x32 t2va，VAE 在 autocast 下为 float32
  （`run-tiny-parity.sh`）：`grep -c best_algo_idx` 为 0（此前每次运行都是 9），
  `video decode failed` 为 0（此前为 1），运行写出由 VAE 解码的真实 124 帧视频，
  而不是占位内容。

## matmul 与 linear 中的同一根因（在 `4c3ab0e4` 中修复）

卷积修复后，tiny 运行越过了卷积，随后在视频 VAE 的 attention 中失败，报
`query, key and value must have the same dtype`。这是同一根因——amp 寄存器在各算子
之间应用不一致——又出现在两个地方：

- **CUDA cuBLAS matmul 忽略了该寄存器。** `cublas_matmul` 与
  `cublas_batched_matmul` 根据操作数设置输出 dtype，因此 `amp_prefer16` 下的
  float32 乘积仍是 float32，而 CPU 与混合 dtype 用例所用的通用路径返回 float16。
  同一个模型，两种精度，取决于后端。两者现在都从 `dtype_infer` 取 dtype 并转换其
  操作数，与卷积算子保持一致。
- **分解后的 `linear` 让 bias 抵消了 dtype。** `linear` 是
  `matmul_transpose(x, weight) + bias`。matmul 返回计算 dtype，但当 bias 是 float32
  Parameter 时，shim 的 torch 对齐提升把和抬回 float32（在 shim 中是
  `f16 + f32 -> f32`，而同一寄存器下的原生 jittor 给出 float16）。torch 的 autocast
  会把 bias 与 linear 算子的其余部分一起转换。`linear`（以及过去照搬它的
  `nn.Linear.execute`）现在会在寄存器激活时把更宽的 bias 转换到乘积的 dtype。随后
  的 RoPE 乘法因此让 q、k、v 保持同一 dtype，attention 检查得以通过。

验证：

- `tests/backends/cuda/test_cublas_matmul_amp_dtype.py`（真实 CUDA）：
  `amp_prefer16` 下的 float32 matmul 与 bmm 为 float16，不带 amp 的 float32 仍为
  float32。3 passed。
- 在 `torch.autocast(float16)` 下，输入为 float16、权重与 bias 为 float32 的 shim
  `torch.nn.functional.linear` 与 `torch.nn.Linear` 返回 float16。该用例已加入
  `compat/tests/torch/test_torch_compat_unimplemented.py` 的 `TestAutocast`。
- 回归：`test_cublas_matmul_grad`、`test_fp16`、`test_bf16` 与
  `test_cudnn_conv_amp_dtype`——74 passed，2 xfailed。
- tiny 运行的视频解码现在能完成（`video decode failed` 消失），这是第一个提交单独
  未能达到的。

## 遗留、无关问题

tiny 运行在写完所有产物后仍以 `corrupted double-linked list` 退出 134。那是单独
跟踪的退出时堆破坏问题，不是 autocast 缺陷。

原始日志与 tiny 运行位于 `$JITTOR_LAB_ROOT/_state/h3/`，不入库。
