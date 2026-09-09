# 2026-07-05 复数类型 CUDA 可用性与 torch 对齐审计

状态：✅ 已完成审计；🟡 有新增待修风险。

## 结论摘要

- 当前 `2.0` 分支的用户面复数主路径是原生 `complex64`，不是 `nn.ComplexNumber`。`complex128` 未注册、未实现。
- CUDA 上原生 `complex64` 的创建、numpy 往返、`real/imag/angle`、`view_as_real/view_as_complex`、`torch.complex`、基础四则、`conj/abs/sum/mean`、`matmul/bmm`、`exp/log/sin/cos/sqrt`、一阶反向在官方测试和真 PyTorch CUDA 对拍中基本可用。
- `nn.ComplexNumber` 已是 deprecated internal bridge，聚焦验证中 CUDA 值和桥接正确；不应再作为用户面返回类型。
- 明确缺口：CUDA `complex64.prod()` 编译失败；CUDA general `linalg.eig(complex64)` 缺 `cupy.linalg.eig`；native complex JVP/二阶 autograd 未支持；`complex128` 缺失。
- 新发现风险：`torch.fft.rfft` 在 CUDA 上存在序列相关错误。隔离运行可过，但在先执行一批 complex forward/grad 后，`fft(real)` 仍正确，`rfft(real)` 会返回虚部接近 0，导致 `rfft->irfft` 不对齐真 PyTorch。
- 本机未验证 NPU：`npu-smi` 不可用。

## 环境

- 分支：`2.0`
- Jittor Python：`/home/zy/miniconda3/envs/jt311/bin/python`
- 真 PyTorch CUDA oracle：`/home/zy/rt_venv/bin/python`，`torch 2.12.1+cu130`，RTX 4090
- Jittor CUDA：`/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`
- 本轮 Jittor 测试统一使用：

```bash
REAL_HOME=/home/zy \
HOME=/home/zy/projects/jittor-lab/_state/jittor-torch-diff/complex_cuda/home \
JTCUDA=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux \
CUDA_HOME=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux \
nvcc_path=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc \
LD_LIBRARY_PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/lib64:$LD_LIBRARY_PATH \
PYTHONPATH=/home/zy/projects/jittor/python \
cache_name=complex_audit_native \
use_parallel_op_compiler=0 \
CUDA_VISIBLE_DEVICES=0
```

## 源码审计

- dtype 注册：`python/jittor/src/misc/nano_string.h` 仅注册 `complex64`；dtype 推导中复数统一提升到 `complex64`。
- 存储与算术：`python/jittor/src/type/complex_compute.h` 定义 `complex64 { float real, imag; }`，支持 `+ - * / == !=`、`jt_conj`、`jt_cabs`、`exp/log/sin/cos/sqrt`，并实现 CUDA `atomicAdd(complex64*)`。
- op 派发：`python/jittor/src/type/complex_op_type.cc` 覆盖 add/sub/mul/div/negative/abs/conj/exp/log/sin/cos/sqrt/cast/equal/not_equal/mean/init_add/init_multiply/init_mean。未支持 op 会响亮报错。
- 桥接层：`python/jittor/nn.py` 中 `ComplexNumber` 标为 deprecated；`view_as_real/view_as_complex` 是 native complex64 与 `float32[...,2]` 的桥。
- torch 面：`python/jittor/torch_compat.py` 中 `torch.complex/view_as_complex/polar/fft` 主路径返回 native `complex64`。
- linalg：`python/jittor/linalg.py` 的 native complex public API 通过 `ComplexNumber` 内部桥实现 `inv/svd/svdvals/qr/eig/eigh/pinv`。
- gradfunctional：native complex `vjp` 支持；native complex `jvp` 明确 `NotImplementedError`。

## 已跑验证

```bash
python -m jittor.test.test_complex64_native
# Ran 15 tests in 268.321s, OK

python -m jittor.test.test_complex64_linalg
# Ran 22 tests in 27.796s, OK (skipped=1)
# skip: CUDA general eig，因为 cupy.linalg 没有 eig

python -m jittor.test.test_complex64_gradfunctional
# Ran 7 tests in 149.045s, OK

python -m jittor.test.test_torch_compat_fft_einsum
# Ran 42 tests in 105.289s, OK
```

新增辅助工具：

```bash
REAL_HOME=/home/zy HOME=/home/zy/projects/jittor-lab/_state/jittor-torch-diff/complex_cuda/home \
JTCUDA=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux \
CUDA_HOME=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux \
nvcc_path=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc \
LD_LIBRARY_PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/lib64:$LD_LIBRARY_PATH \
PYTHONPATH=/home/zy/projects/jittor/python cache_name=complex_audit_probe \
use_parallel_op_compiler=0 CUDA_VISIBLE_DEVICES=0 \
/home/zy/miniconda3/envs/jt311/bin/python \
agent/skills/jittor-torch-diff/complex_cuda_parity.py jt /home/zy/projects/jittor-lab/jittor-torch-diff/complex_cuda_parity_20260705_out

CUDA_VISIBLE_DEVICES=0 /home/zy/rt_venv/bin/python \
agent/skills/jittor-torch-diff/complex_cuda_parity.py rt /home/zy/projects/jittor-lab/jittor-torch-diff/complex_cuda_parity_20260705_out

/home/zy/miniconda3/envs/jt311/bin/python \
agent/skills/jittor-torch-diff/complex_cuda_parity.py cmp /home/zy/projects/jittor-lab/jittor-torch-diff/complex_cuda_parity_20260705_out
```

对拍摘要：稳定项 32 项 PASS；预期 Jittor 缺口 3 项（`prod`、`linalg_eig_*`）；聚合探针中 11 项标为 sequence-sensitive，其中 `rfft/irfft_rfft` 是本轮确认的新风险，ComplexNumber/linalg 聚合异常由聚焦探针排除为结论项。

## CUDA 可用性矩阵

| 类别 | 结论 | 说明 |
| --- | --- | --- |
| dtype / 创建 / numpy 往返 | ✅ 可用 | `jt.array(np.complex64)`、roundtrip、zeros、dtype 判定通过 |
| `real/imag/angle` | ✅ 可用 | CUDA 与 PyTorch/Numpy 对齐 |
| `view_as_real/view_as_complex` | ✅ 可用 | 官方单测与聚焦探针通过；聚合场景需继续盯序列相关风险 |
| `torch.complex/polar` | ✅ 可用 | 返回 native `complex64` |
| 四则 / neg / 标量 | ✅ 可用 | 真 PyTorch CUDA 对拍约 `1e-7~1e-6` |
| `conj/abs` | ✅ 可用 | `abs(complex64)->float32` |
| `sum/mean` | ✅ 可用 | CUDA `sum` 依赖 `atomicAdd(complex64*)`，已可用 |
| `prod` | 🔴 CUDA 不可用 | 编译到 `cuda_atomic_mul(complex64*)` 时 `atomicCAS(complex64*)` 不存在 |
| `matmul/bmm` | ✅ 可用 | CUDA 对拍通过 |
| `exp/log/sin/cos/sqrt` | ✅ 可用 | CUDA 对拍通过 |
| 一阶 autograd | ✅ 主路径可用 | add/mul/div/conj/abs/matmul/exp/log/sin/cos/sqrt 等已覆盖 |
| native JVP / 二阶 | 🔴 未支持 | `gradfunctional.jvp` 对 native complex 明确抛 `NotImplementedError` |
| FFT `fft/ifft` | ✅ 可用 | 官方测试和对拍通过 |
| FFT `rfft/irfft` | 🟡 有序列相关风险 | 隔离测试可过；复数 forward/grad prelude 后 `rfft` 虚部会变近 0 |
| linalg `inv/svd/svdvals/qr/eigh/pinv` | ✅ 前向可用 | 官方 CUDA linalg 测试通过；按重构/残差验证 |
| linalg general `eig` | 🔴 CUDA 不可用 | `cupy.linalg` 无 `eig`；CPU 可用 |
| `ComplexNumber` | 🟡 内部桥可用，不建议用户使用 | 聚焦 CUDA 值/桥接通过；继续保留给 linalg 内部 |
| `complex128` | 🔴 未实现 | 未注册 dtype |

## 与 PyTorch 对齐性

- 已实现的 native `complex64` 基础算子与 PyTorch CUDA 数值基本对齐，常见误差 `1e-7~1e-6`，最宽容限按 `3e-3` 放在辅助探针里。
- linalg 不直接比较特征向量/SVD 向量的相位或符号，而比较重构残差、奇异值/eigenvalue；这是正确的对齐方式。
- PyTorch CUDA 支持 `complex64.prod()` 和 general `torch.linalg.eig()`；Jittor CUDA 这两项目前落后。
- PyTorch CUDA `torch.fft.rfft()` 在任意执行历史下应稳定返回半谱复数；Jittor 当前发现序列相关偏差，应作为后续修复优先项。

## 后续建议

1. 修 CUDA `complex64.prod()`：给 complex multiply reduce 增加可用实现，避免走 `atomicCAS(complex64*)`。
2. 修 `torch.fft.rfft` 序列相关 bug：优先缩成单测，锁定 lazy graph / 函数封装 / native complex slice 复用的根因。
3. CUDA general complex eig：不要依赖 `cupy.linalg.eig`，需要替代实现或清晰 fallback。
4. native complex JVP/二阶 autograd：目前是响亮缺口，不是静默错。
5. `complex128` 是否需要进入路线图，取决于 torch 兼容目标。
6. 真 NPU 复验：本机无 NPU，需要在 910B 环境跑同一套官方测试与新增 rFFT 序列探针。
