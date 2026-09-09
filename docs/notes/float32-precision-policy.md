# Float32 Accumulate Precision

## What it was

Four knobs answered "how is this product accumulated", in four encodings,
and they disagreed with each other.

| knob | scale | reaches |
| --- | --- | --- |
| `use_tensorcore` | 0 / 1 / 2 / 3 | matmul *and* convolution; also changed float16 and bfloat16 accumulation |
| `cuda_allow_tf32` | 0 / 1 | matmul only |
| `cuda_allow_cudnn_tf32` | 0 / 1 | convolution only |
| nothing | — | `cublas_acc_matmul`, which hard-coded float16 accumulation |

`use_tensorcore==1` and `cuda_allow_tf32==1` both mean "tf32 is acceptable
for a float32 matmul", but only the first also switched float16 products to
a float16 accumulator; `use_tensorcore>=3` asked for
`CUBLAS_COMPUTE_32F_FAST_16F`, for which torch has no name at all.

Three cuBLAS ops carried three copies of the selection.
`cublas_matmul_op.cc` and `cublas_batched_matmul_op.cc` were
character-for-character identical. `cublas_acc_matmul_op.cc` differed in one
line: `CUBLAS_COMPUTE_16F` unconditionally where the other two wrote
`use_tensorcore ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F`. So the
accumulate precision of one float16 matmul was a property of which op the
graph happened to pick, which no API exposes. Measured against a float64
reference over k=8192: `cublas_acc_matmul` was off by 0.63 where
`cublas_matmul` on identical inputs was off by 0.14.

The convolution ops had the same shape of defect twice over.
`cudnn_conv` and the three `cudnn_conv3d` ops set the convolution
descriptor's compute type to `CUDNN_DATA_FLOAT` for reduced-precision
operands and asked for tensor-op math; `cudnn_conv_backward_x` and
`cudnn_conv_backward_w` passed `getDataType<Ty>()` — float16 accumulate —
and left the math type at `CUDNN_DEFAULT_MATH`. A float16 convolution
therefore declared one accumulate precision going forward and another coming
back, and its own backend-API fast path (`cudnn_conv_plan.h`, which always
sets `CUDNN_DATA_FLOAT`) declared a third.

## What it is

The native `float32_matmul_precision` Runtime setting writes both native
matmul and cuDNN tiers. The independent Torch frontend owns a separate pair:
`torch.set_float32_matmul_precision()` changes only matmul, while
`torch.backends.cudnn.allow_tf32` changes only cuDNN (convolution and RNN).
Torch defaults to `highest` matmul and `high` cuDNN; native defaults to
`highest` for both. Neither frontend's setters mutate the other's state.

| tier | cuBLAS compute type | cuDNN math type | meaning |
| --- | --- | --- | --- |
| `highest` (default) | `CUBLAS_COMPUTE_32F` | `CUDNN_FMA_MATH` | true float32 accumulate |
| `high` | `CUBLAS_COMPUTE_32F_FAST_TF32` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | tf32 |
| `medium` | `CUBLAS_COMPUTE_32F_FAST_16BF` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | bfloat16 |

The tier governs **float32 operands only**. float16 and bfloat16 always
accumulate in float32 (`CUBLAS_COMPUTE_32F` / `CUDNN_DATA_FLOAT`), and
float64 in float64. That is torch's rule, it was already the default for two
of the three cuBLAS ops and for the convolution forward, and it is the one
place where a "faster" setting used to spend accuracy nobody asked to spend.

The cuBLAS algorithm hint mirrors the compute type — tensor-op exactly when
a reduced-precision compute type was requested — rather than being selected
separately, which is how the two came to be chosen with opposite senses
(6.B05).

Where it lives: `src/runtime/float32_precision.{h,cc}` (policy and scopes),
`backends/cuda/libraries/cublas/include/cublas_compute_type.h` (one `cublas_gemm_mode` for
all three gemm ops), `backends/cuda/libraries/cudnn/include/cudnn_wrapper.h`
(`cudnn_conv_compute_type` and `cudnn_conv_math_type` for all six conv ops).

## Delayed graphs and frontend boundaries

The native Runtime stores separate matmul and cuDNN tier fields. A Torch
binding enters a thread-local pair resolved from its install context; native
calls retain the existing Runtime-following policy. Each Op captures the
pair at construction. Fusion requires equal pairs; graph rewriting,
parallel compilation, execution and native backward construction restore
the captured pair. Thus changing a Torch switch after constructing a lazy
graph cannot silently change that graph's requested library precision.
These scopes do not mutate global flags or synchronize pending tensors.

The cuDNN RNN descriptor uses the same cuDNN tier as convolution. Its reserve
space key includes the selected math type, and Python's flattened-weight
offset cache includes the cuDNN tier rather than the unrelated matmul tier.
Float32 native RNN selects FMA by default; Torch's default cuDNN setting
permits TF32. Half/bfloat16 accumulation rules remain unchanged.

## The default is unchanged, deliberately

`highest` is exactly what `use_tensorcore=0, cuda_allow_tf32=0,
cuda_allow_cudnn_tf32=0` selected before — `CUBLAS_COMPUTE_32F` with
`CUBLAS_GEMM_DEFAULT`, and `CUDNN_FMA_MATH`. Nothing about this change asks
a user to accept different numerics by default, which is why it needed no
end-to-end ecosystem evidence to land: the two substantive changes are both
*towards* the value the majority of the code already used.

1. `cublas_acc_matmul` accumulates float16 in float32, like the other two
   gemm ops and like torch. One call site in the repository.
2. `cudnn_conv_backward_x` / `_w` declare float32 accumulate and tensor-op
   math for reduced-precision operands, like the forward and like their own
   backend-plan fast path.

## The deprecated knobs

`use_tensorcore`, `cuda_allow_tf32` and `cuda_allow_cudnn_tf32` remain, as
overrides that can only *raise* the tier for the domain they name:

    matmul tier = max(policy, use_tensorcore tier, cuda_allow_tf32 ? high : highest)
    conv   tier = max(policy, use_tensorcore tier, cuda_allow_cudnn_tf32 ? high : highest)

so every value they had keeps meaning what it meant, and leaving them alone
makes the policy the whole answer. `use_tensorcore=3` folds into `medium`:
`CUBLAS_COMPUTE_32F_FAST_16F` and `CUBLAS_COMPUTE_32F_FAST_16BF` cost the
same on every tensor-core generation and bfloat16 keeps float32's exponent
range, so the float16 variant was strictly the worse of the two.

These deprecated overrides apply only to native Runtime-following calls.
Torch's explicit context pair bypasses them, so native legacy flags cannot
raise the precision tier of an independent Torch operation. Torch's
high/medium boolean-toggle roundtrip remains the shim's documented
approximation; it does not claim to reproduce PyTorch 2.12's rejection of
mixed old/new precision-control APIs.

## Reading the choice back

Each op logs its decision immediately before calling the library, at `vvv`:

    cublas_matmul algo select: precision=highest computeType=CUBLAS_COMPUTE_32F algo=CUBLAS_GEMM_DEFAULT
    cudnn_conv precision select: precision=highest computeType=CUDNN_DATA_FLOAT mathType=CUDNN_FMA_MATH

That line is the observable: several of these choices are invisible in the
output values. On sm_89 with cuDNN 8.9, setting the convolution descriptor's
compute type to `CUDNN_DATA_HALF` produced bit-identical gradients to
`CUDNN_DATA_FLOAT` (max error 0.05796 either way over a 4096-deep reduction),
because the kernels cuDNN picks accumulate in float32 regardless. The
convolution half of this change is therefore a latent defect closed, not a
measured accuracy gain — unlike the cuBLAS half, which moves a real 0.63 to a
real 0.14.
