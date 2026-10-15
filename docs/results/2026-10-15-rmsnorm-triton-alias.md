# RMSNorm optional contracts and Triton pointer aliases

- Status: verified CPU/CUDA operator contracts and a maintained vLLM component gate.
- Reviewed: 2026-10-15; owner: Torch compatibility and Triton bridge maintainers.
- Integration baseline: `c14bdb2f49653a7422f8bcf726a55d089b46f4b1` plus this change.
- Source changes: `b48e000d3525a3b85dbfd398c0b6ce4ffa09676f` and
  `72e12a9c662b46d5a5389f7e031c3997520d4562`.
- Review when: module-call dispatch, RMSNorm contracts, Var storage/strides,
  Triton argument pointers or guarded copy-back, upstream vLLM, or the shared
  ecosystem runner changes.

## Confirmed failures and corrections

The general RMSNorm shortcut consumed an affine tensor even when a module's
`has_weight` or `pass_weight` disabled it, and treated partial variance as a full
reduction. An intentionally unused CPU weight also caused a CUDA device mismatch.
Python and native repeated-inference dispatch now leave those contracts with the
module's own forward. Both guards resolve optional attributes normally. A missing
attribute clears only AttributeError; other descriptor errors still propagate.
Weighted standard neighbors retain their existing native implementation.

Tests check forward values and analytic input/weight derivatives against NumPy
float64, then repeat inference twice under no_grad. They include nonunit weights,
unused host weights, partial variance, absent optional flags and the standard
weighted neighbor. The independent binary Torch interpreter passed all five
tests on each actual device before candidate acceptance.

A separate real `@triton.jit` increment kernel exposed silent lost writes. The
bridge bounced an input into its guarded buffer, passed an aliased output's real
pointer, then copied the old input back over the correct in-place result.
Independent Torch passed six FP32/BF16 cases; the candidate passed only the two
disjoint-output cases before repair. After repair, all six saved candidate arrays
are bitwise equal to their independent reference.

The bridge now detects overlap between physical addressed byte ranges before
bouncing arguments. Overlapping arguments keep their real pointer relationship,
regardless of argument names. Range analysis includes stride holes and offsets;
unknown layouts conservatively disable independent bounces. Tests cover two
output names, both dtypes, separate outputs, the same object, same-storage views
and offset views. Negative-stride extent checks are host contracts, not a claim
of executing negative-stride CUDA views.

## Integration verification

| Check | Result |
| --- | --- |
| CPU RMSNorm and pointer-range contracts | 8 passed, 0 skipped |
| CUDA RMSNorm/real Triton writes and three host range contracts | 9 passed, 0 skipped |
| Both directed sessions, whole-session fallback guard | counters 0 to 0 |
| Core, native and Torch modes | 322 passed, 42 skipped, 1 existing xfail |
| Structure and tools | 1716 passed, 6 declared/environment skips |

The CUDA session explicitly selected CUDA for the RMSNorm tests and used a genuine
installed Triton package. CPU hid CUDA. Each process had its own compile cache;
first compilation was serial. The actual source was this checkout, and CUDA native
kernel math was strict. These are correctness results, not performance ratios.

Integration used Linux x86_64, AMD EPYC 9554, Python 3.11.16 and NumPy 1.26.4.
CUDA used one allocated RTX 4090/SM89, driver 580.178.04, Jittor's pip-cu122
provider (`pipcu122_a89fa4fed08c`), nvcc 12.2.140 and GCC 11.5.0. Physical allocation
and original compile paths stay in external logs. RMSNorm fixtures are FP32
2-by-4 tensors; the alias kernel uses 16 FP32/BF16 elements. No TF32 matmul,
real-size model, graph-regime comparison or memory/throughput measurement is
established by those fixtures. The independent child oracle is Torch
2.13.0+cu130; that binary's CUDA build differs from Jittor's toolchain.

The source-only audit initially failed to discover the new vLLM test. Its negative
regression and correction now retain the actual class-qualified selector while
leaving the audit's verified level unset. The public `ecosystem_vllm` nox session
pins independent Torch 2.13.0, vLLM 0.29.0 and Transformers 5.13.0, requires all four
floating input gradients, and passed one real CPU and one real CUDA component
test during the child task's precommit validation. Its parameter-free composition
executes SiluAndMul, weightless RMSNorm and RotaryEmbedding through public native
paths. It does not exercise a model family or an engine and cannot establish L3.
Fresh registry-bound component scorecard collection is separate work.

Reproduce the directed tests using the selected device's actual toolchain,
independent cache and installed Triton:

```bash
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda backend_fallback=error \
python -m pytest -q compat/tests/torch/test_rms_norm_optional_weight.py \
  compat/tests/triton/test_triton_torch_compat.py::TestTritonTorchCompat::test_pointer_alias_preserves_inplace_writes \
  compat/tests/triton/test_triton_backend.py::TestPointerRangeAliasing
```

For CPU, hide CUDA, set `JT_USE_CUDA=0` and `JITTOR_TEST_DEVICE=cpu`, and omit the
CUDA Triton launch node. The vLLM gate is `python -m nox -s ecosystem_vllm` with
two provisioned interpreters; its session rejects incomplete configuration.

Unversioned child evidence is under `$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`.
Its `b48-validation-manifest.json` seals 23 raw files; SHA256 is
`41bbf487346f9af992aacc682d6039e1f79569e9f314a4d6caf42a3179f15e7c`.
Those precommit results retain baseline `6f2e479f63b3686b0e63c41d324148a769bacddd`
plus the committed changes; they are not relabeled as clean later measurements.
Integration independently rehashed all 23 artifacts, compared the six alias
arrays, and matched all six owned repair files to their commit. New integration
logs, device/fallback metadata and the initial audit failure are under
`$JITTOR_LAB_ROOT/_state/vllm-owner-integration-c14/`.

No engine, Omni numerical acceptance, arbitrary RMSNorm dispatch-contract parity,
NPU or ROCm support follows from this batch. The separate BF16 multiplication
ordering contract in Omni remains under investigation; these changes preserve
the existing native RMSNorm mathematics.
