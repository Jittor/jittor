# float32 accumulation precision

Matrix multiplication and convolution can accumulate at lower precision than
float32 in exchange for speed. This page explains which tiers Jittor offers, what
the default is, and when they take effect under the lazy graph.

## Tiers

A single tier simultaneously determines cuBLAS's compute type and cuDNN's math
type:

| Tier | cuBLAS compute type | cuDNN math type | Meaning |
| --- | --- | --- | --- |
| `highest` (default) | `CUBLAS_COMPUTE_32F` | `CUDNN_FMA_MATH` | true float32 accumulation |
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

Native interface: the Runtime's `float32_matmul_precision` writes both the
matmul and cuDNN layers.

The Torch frontend has **two separate switches**, consistent with torch
semantics:

```python
torch.set_float32_matmul_precision("high")   # 只影响 matmul
torch.backends.cudnn.allow_tf32 = True       # 只影响 cuDNN（卷积与 RNN）
torch.backends.cuda.matmul.allow_tf32 = True # matmul 的另一种拼写，同一个开关
```

The defaults differ: in the **Torch frontend** matmul is `highest` and cuDNN is
`high`; in the **native frontend** both are `highest`. The setters of the two
frontends do not affect each other's state.

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
