# float32 accumulation precision

Matrix multiplication and convolution can accumulate at lower precision than
float32 in exchange for speed. This page explains which tiers Jittor offers, what
the default is, and when they take effect under the lazy graph.

## Tiers

The same tiers apply to cuBLAS's compute type and to cuDNN's math type, each
domain with its own switch (see below):

| Tier | cuBLAS compute type | cuDNN math type | Meaning |
| --- | --- | --- | --- |
| `highest` | `CUBLAS_COMPUTE_32F` | `CUDNN_FMA_MATH` | true float32 accumulation |
| `high` | `CUBLAS_COMPUTE_32F_FAST_TF32` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | tf32 |
| `medium` | `CUBLAS_COMPUTE_32F_FAST_16BF` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | bfloat16 |

**A tier only applies to float32 operands.** float16 and bfloat16 always
accumulate in float32 (`CUBLAS_COMPUTE_32F` / `CUDNN_DATA_FLOAT`), and float64
uses float64. This matches torch's rules — low-precision inputs already need
high-precision accumulation, and a "faster" setting should not spend extra
precision here.

cuBLAS algorithm hints follow the compute type: tensor ops are used only when a
low-precision compute type was requested, and the two are no longer selected
separately.

## How to set it

The native interface also has **two separate switches**:
`float32_matmul_precision` only governs matmul, and `float32_cudnn_precision`
only governs cuDNN (convolution and RNN).

```python
jt.flags.float32_matmul_precision = "high"     # matmul only
jt.flags.float32_cudnn_precision = "highest"   # cuDNN only, back to strict float32
```

The Torch frontend likewise has two separate switches, consistent with torch
semantics:

```python
torch.set_float32_matmul_precision("high")   # 只影响 matmul
torch.backends.cudnn.allow_tf32 = True       # 只影响 cuDNN（卷积与 RNN）
torch.backends.cuda.matmul.allow_tf32 = True # matmul 的另一种拼写，同一个开关
```

Both frontends share torch's defaults: matmul is `highest` and cuDNN is `high`
(that is, `torch.backends.cudnn.allow_tf32 = True`). On Ampere and later, a
float32 convolution therefore runs in tf32 by default: each multiplication keeps
only a 10-bit mantissa of its inputs while accumulation stays float32, which puts
the relative error of a convolution's output around 1e-4 to 1e-3. Code that needs
strict float32 convolution (parity checks, gradient checks) should set
`float32_cudnn_precision = "highest"` explicitly. The setters of the two
frontends do not affect each other's state.

The native frontend's cuDNN default used to be `highest`, set together with
`float32_matmul_precision`. That was stricter than both torch and Jittor 1.3.10 —
1.3.10 does not set a math type and gets cuDNN's default, which allows tf32 on
Ampere and later — and it cost ResNet-50 training (batch 64) 128 ms per step
against 95 ms on 1.3.10, the whole gap in the convolution backward.

The multiple spellings of one semantic (`allow_tf32`, `fp32_precision`,
`get/set_float32_matmul_precision`) are all **views** of the same flag and do not
each store a separate copy of state, so three spellings never give three
different answers.

## When it takes effect under the lazy graph

Jittor executes lazily, so "when this setting is read" must be explicit:

- **Each operator captures the current tier pair** (matmul tier + cuDNN tier) at
  construction;
- fusion requires the two operators' tier pairs to be **equal**;
- graph rewriting, parallel compilation, execution and backward construction all
  restore the tier pair captured at construction.

In other words, **changing the switch after the lazy graph is built does not
quietly change the library precision that graph already requested**. These scopes
do not modify global flags and do not synchronize not-yet-completed tensors.

## Implementation locations

| Content | Location |
| --- | --- |
| Policy and scopes | `src/runtime/float32_precision.{h,cc}` |
| cuBLAS selection (shared by the three gemm operators) | `backends/cuda/libraries/cublas/include/cublas_compute_type.h` |
| cuDNN selection (shared by the six convolution operators) | `backends/cuda/libraries/cudnn/include/cudnn_wrapper.h` |

## The numerical contract beyond this page

Subnormals flushed to zero on CUDA, convolution algorithm auto-selection changing
gradients, IEEE special-value arithmetic, the NaN semantics of
`maximum`/`minimum`, and CPU reduction precision — entries once scattered at the
end of this page — now live on the [numerics contract](numerics-contract.en.md)
page, each with measured evidence and the test file that pins it. This page only
covers the **accumulation precision tiers** for float32.
