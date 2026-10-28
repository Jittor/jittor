# CPU BF16 accumulators and LayerNorm

Status: verified for the bounded operator and Engine checks below; the maintained
CPU/CUDA two-rank protocol scorecard remains pending. Owner: Jittor maintainers.
Reviewed: 2026-10-15. Recheck after changes to CPU contraction, oneDNN matmul,
LayerNorm, autograd snapshots or the pinned downstream package.

Code: `a2ca1b364c79c853623c4f65bf58b9232d7f4171`. Clean post-sync measurement:
`5f5ab7ecfee0a896c3565392cc9282559c9f8953`, incorporating remote
`0751e425b0f197b8e7b388f218613378c71f701a`. Existing MPI prerequisites retain
their independent [report and source baseline](2026-10-15-mpi-bfloat16-process-groups.en.md).

## Changes and references

CPU BF16 Linear previously rounded its matrix product before adding bias.
For `1.0078125 * 1.0078125 + 0.00390625`, that produced `1.015625`; independent
binary Torch produces `1.0234375`. The new path retains BF16 operands and adds
bias to the FP32 accumulator before the single BF16 output conversion.

The existing oneDNN matmul owner now exposes an optional FP32 output for BF16
operands. Its default output and FP32 route remain unchanged. Descriptor and
JIT caches distinguish input/output storage widths, including empty contracted
dimensions. Backward retains BF16 gradients and native higher derivatives.
With `JT_BUILD_USE_MKL=0`, the same existing contraction uses native FP32 opmath.
Its custom gradient reuses that contraction recursively, rounding the completed
GEMM gradients rather than each broadcast contribution. Neither path substitutes
FP16 or a GEMM with FP32 input tensors.

The generic Function saves independent raw-reshape holders of the original
autograd graph before first-order taping. Actual input/weight holder rebinding,
repeated calls through one Function instance and all three second derivatives
match 28 independent Torch observations bitwise. The scalar second-derivative
contract remains `4.0625`; an experimental implementation yielding `2.03125`
was rejected and is not production code. This snapshot contract covers graph
rebinding, not a deep copy against arbitrary physical storage writes.

For CPU non-mixed BF16 LayerNorm, backward now consumes BF16 saved mean/rstd
and BF16 affine partials. Forward keeps unrounded FP32 statistics. Mixed FP32
affine parameters and other dtype/device paths retain their existing behavior.
This follows the separately observed statistics and accumulation boundaries in
[Torch's CPU LayerNorm owner](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp).
Selection uses native operand placement, not a global CUDA flag.

## Real execution

Platform: Linux x86_64, AMD EPYC 9554, GCC 11.5, four CPU threads per rank;
oneDNN 3.9.1 with an observed `brg_matmul:avx512_core_bf16` BF16-to-FP32 primitive.
The independent CPU reference is binary Torch `2.11.0+cu128`, with an actual `_C`
extension and no shim marker. CUDA regressions execute on RTX4090/SM89 with
NVIDIA driver 580.178.04 and genuine pip-cu122/nvcc 12.2.140, under a forbidden
fallback policy. These checks make no timing or performance claim.

On clean `5f5ab7ec`, CPU ZeRO1/2 × oneDNN/native-opmath executes real DeepSpeed
`0.19.7+f5af15c`, two ranks, BF16 model parameters, AdamW, six microsteps and
three updates. Every rank updates all 166 FP32 master-partition elements;
fallback count is zero. Save/load into a new Engine and continued training are
bitwise equal within each runtime: 64 strict fields per rank for ZeRO1 and 56
for ZeRO2. Candidate process groups are actual MPI; reference groups are Gloo.
This does not establish native Gloo support.

Across those four diagnostic cases, all 1,792 numerical fields satisfy the
unchanged state/forward `0.002` and gradient `0.01` budgets. All 1,024 requested
gradient fields are checked; 1,012 are bitwise equal. ZeRO1's gradients are all
bitwise equal; maximum ZeRO2 gradient error is `1.2822511817226889e-4`.
Maximum state error is `2.1931527140484152e-7`. Before the affine-partial repair,
ZeRO2 had two moment fields at `0.0025165948`, exceeding the original budget;
that failure remains in the evidence, without changing tolerances.

These are API diagnostics, not maintained-gate admission. Their independent CPU
reference retains its original `f945b223cd4316feaa733556ef1413344f98e11f`
measurement identity. The full paired protocol still needs fresh, matching
CPU/CUDA maintained measurements, source/installation bindings and both input
offsets. No common scorecard numerator is granted here.

Clean post-sync gates:

| Check | Result |
| --- | --- |
| CPU new accumulator/LayerNorm/public Linear contracts | 12 passed, zero skipped |
| Actual CUDA Linear/LayerNorm contracts | 14 passed, 3 existing xfails, zero fallback |
| Core, native and Torch modes | 322 passed, 42 skipped, 1 existing xfail |
| Runtime device-selection structure contract | 7 passed |
| Repository layout | passed |

Additional owner checks before the final synchronization passed native CUDA
53 cases and compat CUDA 91 cases with zero fallback; the latter retains three
existing xfails. Five independent CPU LayerNorm shapes have bitwise forward
agreement; maximum per-field scaled gradient error is `0.00502513`. BF16 affine
reductions depend on row/thread grouping and are not claimed bitwise for arbitrary
shapes. The tested LayerNorm higher derivative retains a one-ULP difference
within the existing gradient budget. No full-library DeepSpeed, ZeRO3, offload,
other optimizer, hardware or performance coverage follows from these cases.

The earlier full structure run had 1,502 passes, six skips and three failures:
MPI report index/manifest registration, plus the newly introduced direct device
flag read. The latter is fixed and its full seven-case contract rerun passes;
central documentation registration remains an integration responsibility.

## Evidence and reproduction

Unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`.
`cpu-bfloat16-math-v1-evidence.json` binds 1,574 runtime-source hashes and 105
raw/script/checkpoint hashes, including clean case directories, both independent
reference packets, source manifests, the rejected numerical result and gate logs.
Its SHA256 is
`80aa5e3b0a77699f954dba1f75b749a025c94ae62e3b97ea80b3244b6f75f81a`.
`seal-math-evidence-v1.py` recomputes every diagnostic comparison and strict/F0
metadata check. It does not promote those diagnostics into protocol admission.

With separate native/reference interpreters and the explicit optional oneDNN
paths configured, the reusable operator regressions are:

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_bfloat16_cpu_accumulators.py \
  compat/tests/torch/test_bfloat16_layer_norm_cpu.py \
  compat/tests/torch/test_bfloat16_linear_rounding.py

JITTOR_TEST_DEVICES=cuda CUDA_VISIBLE_DEVICES=<allocated-device> \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_bfloat16_linear_rounding.py \
  compat/tests/torch/test_torch_compat_norm.py::TestLayerNorm
```

Keep CPU/CUDA caches separate, serialize first builds and retain independent
rank homes/temp directories. The archived `run-math-clean-proof-v1.py` captures
the exact task-local launch environment and immutable source checks. Its machine
paths are evidence, not repository defaults or a replacement for the maintained
Engine launcher.

## Main-tree integration verification

Integrated onto synchronized `01351fe6174a44b688bb17ad3f5f7f0fd1cf340d`.
All ten code/test files match the reviewed `a2ca1b364` source byte for byte;
the runtime source snapshot remained unchanged throughout verification.
An independent integration audit rehashed the original 1,574 source and 105
artifact bindings and recomputed all 1,792 numerical fields, including the
1,024 gradients. Original tolerances, failed results and measurement identities
remain intact.

The integrated tree passes CPU operator contracts with actual oneDNN enabled
(12 passed, zero skipped), the disabled-oneDNN Linear route (6 passed, zero
skipped), actual CUDA Linear/LayerNorm regression (14 passed, three existing
xfails), core (322 passed, 42 skipped, one existing xfail), and full structure
(1,505 passed, six skipped). Ruff for all changed Python files and repository
layout pass. The report and source manifest are registered centrally.

Integration logs and source checks are unversioned under
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-013/`; the independently
recomputed diagnostic audit is under
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-075/independent-audit.json`.
These results continue to establish operator correctness and the bounded Engine
diagnostic, without granting maintained DeepSpeed protocol admission.
