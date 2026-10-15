# CUDA BF16 Linear bias accumulation

- Status: verified on CUDA SM89; CPU behavior is unchanged.
- Reviewed: 2026-10-15.
- Owner: Jittor core maintainers.
- Synchronized baseline: `3890a4ae8c79767fe5251a1afae818308db4d447`.
- Implementation: `e64d4a27f282740ad9e496bbf0d1330ed4717a27` and
  `e07a2e0469594b56680716516277f27a1f984635`.
- Recheck when: Linear dispatch, cuBLAS dtype inference, AMP input casting, or
  native matmul differentiation changes.

## Problem and implementation

The portable BF16 Linear path rounded its GEMM output before adding bias.
For BF16 input and weight `1.0078125`, and bias `0.00390625`, this returned
`1.015625`; independent PyTorch Linear returned `1.0234375`. The latter adds
bias to the FP32 accumulator before the single BF16 output rounding. Three
CUDA shapes reproduced this difference with identical BF16 operands.

The registered CUDA `linear_bias_accumulator` provider reuses native
`CublasMatmulOp` with BF16 operands and FP32 accumulator output, adds the
rounded BF16 bias in FP32, and casts the result once. The new `fp32_output`
option defaults to false and explicitly rejects non-BF16 compute dtypes.
Ordinary FP32/FP16 matmul behavior remains covered by its existing tests.
The native matmul backward remains differentiable, including `create_graph`;
there is no custom backward formula or extra adapter. Autocast rounds inputs
and bias to BF16 before the operation, and converts gradients back to each
original parameter's dtype.

The Torch-mode native Module fast call for a biasless Linear passes the new
constructor argument explicitly as false. Its FP32, FP16 and BF16 inference
and training cases preserve the original result and gradient dtypes.

## Verification

The target used an RTX 4090, SM89, driver 580.178.04, and the installed
`pipcu122_a89fa4fed08c` provider with nvcc 12.2.140. The separate oracle was
binary PyTorch `2.11.0+cu128`; its `_C` extension was loaded and the Jittor
install-context marker was absent. These are different CUDA library stacks,
not a claim about every CUDA version or architecture.

| Check | Result |
| --- | --- |
| BF16 CUDA `(rows, in, out)` = `(2,8,16)`, `(2,16,8)`, `(2,8,4)` | Outputs and all input/weight/bias gradients bitwise equal to the independent oracle |
| Rounding midpoint, rank 1/2/3, module/autocast, malformed shapes, second derivative, biasless Module inference/training | 6 CUDA tests passed; independent binary Torch CPU/CUDA each passed the same six numerical contracts |
| Native accumulator option, FP32/FP16 defaults, existing Lt paths | 20 passed; provider observation verifies actual BF16 operands, FP32-output request, CUDA residency and zero fallback |
| AMP CPU / CUDA, including the six CUDA contracts | 76 / 82 passed |
| CPU core | 322 passed, 42 skipped, 1 existing xfail |
| Full structure before report insertion / repository layout | 1460 passed, 6 skipped / passed |

The final constructor follow-up repeated CPU core successfully. With this new
report present, full structure reports 1458 passed, 6 skipped and exactly two
documentation-integration failures: the report needs the central results-index
entry and generated MANIFEST registration. Those shared files are maintained
by the integrating owner; computation and other structural checks passed.

The six structure skips comprise two accelerator-dependent checks, two
declared cases, and two source-install resolution checks. The source-checkout
interpreter has no installed Jittor distribution metadata for the latter.

Reproduce from the repository root with isolated caches and the configured
CUDA toolchain:

```bash
JITTOR_TORCH_SHIM=0 PYTHONPATH=python <target-python> -m pytest -q \
  tests/backends/cuda/test_cublas_matmul_amp_dtype.py tests/nn/test_lt_linear_cuda.py
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cuda PYTHONPATH=python \
  <target-python> -m pytest -q compat/tests/torch/test_bfloat16_linear_rounding.py \
  compat/tests/torch/test_torch_amp_fidelity.py \
  compat/tests/torch/test_torch_amp_boundaries.py \
  compat/tests/torch/test_torch_amp_training_loop.py
```

Raw logs and observations are unversioned under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/`: `bf16-linear-*-final-v2.log`,
`bf16-linear-ctor-*.log`,
`bf16-linear-oracle.npz`, and the rank-specific
`multirank/native/rank*/bf16-linear-accumulator-v1.npz`.

## DeepSpeed and remaining boundaries

The unchanged residual model in `_ecosystem_deepspeed_cases.py`, real
DeepSpeed `0.19.7+f5af15c`, and two independent CUDA ranks also exercised
ZeRO1 BF16 training with FP32 master partitions. Six microsteps, gradient
accumulation of two, three updates, save, a fresh Engine load, and resumed
training produced 241 observations per rank. Every input/target gradient,
incoming parameter gradient and reconstructed full parameter gradient matched
binary Torch bitwise. Outputs and BF16 parameters also matched bitwise;
FP32 master/moment differences were at most `1.49e-8`. Native strict
same-runtime resume and zero-fallback checks passed. The existing `2e-3`
forward/state and `1e-2` gradient gates were unchanged. The final registered
provider repeated this result; logs use `bf16-registered-accumulator-v2`.

This supplementary ZeRO1 result does not complete DeepSpeed R14. The same
two-rank ZeRO2 protocol executes and resumes with zero fallback, but full
gradients diverge after reduction while the initial local incoming gradients
match bitwise. Those failing observations remain preserved under
`bf16-accumulator-v2`; shared buffer mutation is a separate investigation.

CPU retains the original separate product/bias rounding. A trial FP32 CPU
composition improved the midpoint forward and first gradient but changed the
second derivative from `4.0625` to `4.09375`, so it was excluded. Independent
Torch's explicit FP32 composition confirms that latter quantization boundary;
it is not equivalent to native BF16 Linear's backward. No CPU BF16 precision
fix or ROCm/NPU validation is claimed here.

## Parent integration verification

The parent integrated all four Linear commits, including the native module
constructor correction, on `71c5be74ef6991b75fa79d73fe52eb04c0319d59` together
with the delivered TP AdamW and compatibility checkpoint changes. All 36
delivered files matched their final source references byte for byte before
validation. Results and generated source-manifest registration were added.

On that frozen combined source, CPU core passed **322 tests** with 42 declared
skips and one existing xfail. Native CPU owners passed **56**, with 15 CUDA
skips; compatibility CPU owners passed **18**, with the CUDA-only BF16 class
excluded. Actual CUDA native Linear/cuBLAS owners passed **20**, and actual
CUDA compatibility owners passed **24**, both with zero skips. Full structure
passed **1470**, with six skips. This is correctness verification, without a
speed claim. The older independent two-rank numerical evidence retains its
original source labels. Raw integration logs, source equivalence and artifact
audits are unversioned under
`$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`.
