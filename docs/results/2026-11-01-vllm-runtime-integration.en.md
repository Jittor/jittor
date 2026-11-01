# Main-repository integration of runtime prerequisites used by vLLM

[中文](2026-11-01-vllm-runtime-integration.md)

- Status: targeted runtime and installed-entry acceptance complete; broad smoke is not fully green. Remaining baseline AMP and lazy-import limitations are explicit below.
- Date: 2026-11-01–2026-11-02.
- Remote baseline: `origin/2.0-refactor`, `88b48a1f18b2964f05090f65561e07ab64343715`.
- Integrated code: four fixes replayed through `5bd26b7859c9589a6d641bbe78ee46ef987593fc`, preserving later remote implementations.
- Scope: cuFFT/CUB build compatibility, accelerator memory queries, dtype-view storage/writeback, and Triton constexpr packing/cache identity.
- Owners: Jittor core and Torch compatibility maintainers.
- Review when: the affected core/frontend interfaces, CUDA toolchain, Triton, upstream vLLM or claimed devices change.

## Ownership and source baselines

These framework and general Torch capabilities belong in normal builds of the main repository.
The independent adapter continues to own vLLM-specific protocols. This report neither moves core
fixes into the adapter nor treats older wheels with the same version number as containing them.

| Capability | Integrated commit | Behavior |
|---|---|---|
| cuFFT/CUB | `0467f3d991a9ae6b16fb3faf81041be4ab97212e` | Admit cuFFT 12 enum removals and the CUB 3 reduction interface while preserving older branches |
| Memory queries | `379db21f22f6b69b9e5c730d0fcc9134180cd49d` | Expose `torch.accelerator.get_memory_info` through the existing CUDA query |
| Dtype views | `267888bded3f2f68ea38ba6c42e3d108b8503b70` | Preserve byte strides, offsets, storage and bidirectional writes for valid noncontiguous views; dtype views remain nondifferentiable |
| Triton arguments | `5bd26b7859c9589a6d641bbe78ee46ef987593fc` | Specialize `None` as constexpr, retain full signatures for cache identity, and omit constexpr slots from runtime packing |

Review also separated the modern Triton native-specialization helper assertion from the general
ASTSource cache test. Older releases without that helper skip only the helper assertion; cache
identity remains tested.

## Historical evidence: 2026-10-30, code `40d4414f4`

These results belong to the independent development baseline
`40d4414f45b89b7cb7f5efdc06ca622b674d772a`, not the current integration with remote changes.
The environment used Python 3.11, RTX 4090, the CUDA 13.0.88 toolchain, cuFFT 12/CUB 3 and Triton 3.7.1.

| Scope | Historical result |
|---|---|
| CPU core tier | 327 passed, 45 environment/opt-in skips, one expected failure |
| CPU dtype-view and neighboring contracts | 27 passed; nine shared-semantics cases additionally checked against independent PyTorch |
| CUDA dtype views | 11 passed |
| CUDA accelerator memory queries | Three passed |
| CUDA cuFFT | Four passed |
| CUDA softmax | Five passed, including scheduling, gradients and independent numerical references |
| CUDA Triton backend file | 23 passed; three multidevice cases skipped with only one allocated device |

Historical raw logs and build artifacts remain unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-product/20261030/`. The historical vLLM pin was 0.29.0 at
`98dff2a81d747d1dba01a47f939f48c3526d4206`. These framework regressions do not establish whole-library support.

## Acceptance on the integrated baseline

Source behavior regressions ran at `9391a815dd6cc67b2d2fbe61fd9b4a1fad1d345a`.
Commit `ee116312ed3448a51c77ab1c9e2abeb45677f7f7` subsequently changed only core/frontend
release identity to `2.0.1`, matching dependencies and version contracts, without changing these
computations. The aggregate adapter carrier remains `2.0.0`. Final structure, shim aliases and
installed acceptance ran at `ee116312`; these source identities are not presented as one run.

Environment: Linux x86_64, Python 3.11, one RTX 4090, the complete CUDA 13.0.88 toolchain,
cuFFT 12/CUB 3 and Triton 3.7.1. CPU/CUDA caches were separate and initial JIT was serialized.
Source gates asserted core/compat import paths. Final version gates used another isolated
environment/cache and confirmed core `2.0.1`.

| Scope | Baseline | Current result |
|---|---|---|
| CPU core tier | `9391a815` | 327 passed, 45 skipped, one expected failure |
| CPU dtype-view / memory-query contracts | `9391a815` | 13 passed, one skipped |
| CPU native views | `9391a815` | 56 passed, four skipped |
| CUDA cuFFT / softmax | `9391a815` | Nine passed |
| CUDA dtype views / memory queries | `9391a815` | 14 passed |
| CUDA Triton backend | `9391a815` | 24 passed, three multidevice cases skipped |
| Complete `tests/structure` | `ee116312` | 1577 passed, four skipped in 267.84 seconds |
| Final full structure after gate cleanup | `0121c8a91` | 1577 passed, four skipped in 213.73 seconds |
| `test_torch_shim_aliases.py` | `ee116312` | 11 passed in 15.49 seconds |
| CPU smoke, native session | `9391a815` | 4353 passed, 2561 skipped, 332 deselected, five expected failures, 14 failures; targeted clearance below |
| CPU smoke, Torch session | `9391a815` | 5207 passed, 1882 skipped, 1593 deselected, 30 failed in 2702.54 seconds; not a passing whole-suite run |

The first structure run had six failures because the CPU environment incorrectly inherited
`JITTOR_CUDA_WHEEL_DISABLE=1`, preventing the expected environment contracts. After setting it
to `0` for CPU, 22 targeted cases passed, followed by the complete passing structure run on
`ee116312`. Tests were not changed to suppress these failures.

The native smoke failures were diagnosed without changing the running source tree. Nine lacked
SciPy; two MKL mocks were overridden by the canonical environment flag; one pretrained-weight
fetch used a broken Python CA path. With declared NumPy 1.26.4 / SciPy 1.13.1 and the system CA,
31 cases passed and 22 device cases skipped, including real Res2Net inference. The two MKL cases
passed after removing the conflicting environment override.

Two gate defects required test-only repairs after minimal reproductions: explicitly discover BLAS
and probe its two products directly against NumPy (39 matrix cases passed, 17 skipped; 73 related
structure cases passed), and measure argmax's own live-operation delta instead of assuming the
whole process starts at zero (the previously failing two-test sequence passed; the complete misc
file had eight passes and six skips). Scalar-mask retention remains recorded in
[KI-MEM-004](../../agent/manuals/known-issues.md#ki-mem-004-scalar-boolean-assignment-retains-one-live-operation).
These reruns clear the native failures; they do not relabel the original smoke invocation as green.
Commit `0121c8a91d070dcee15d1cca7188bcc8267874ba` records the gate cleanup and known limits.
No core/frontend computational implementation changed for this cleanup.

The 22 host/device metadata tests were incorrectly entering the CPU lane and used an older
vLLM constructor. They now require CUDA, supply the 0.29 block-capacity argument, and compare
host proxies using NumPy dtypes. The adapter explicitly preserves the upstream empty-request
ValueError. All 22 cases passed on real CUDA; the CPU lane records 22 explained accelerator skips.
This test/empty-input cleanup does not change the core/frontend archive identity above.

The isolated bf16 training case fails identically on the actual remote baseline `88b48a1` and
candidate `ee116312`: final losses 0.158203125 versus float32 0.1223832667, relative difference
0.2926859144 against the unchanged 0.25 threshold. Both source identities were checked in separate
worktrees/caches. Explicit float32 loss arithmetic is narrowed inside CPU autocast, while the
previous LayerNorm dtype fix remains active. This pre-existing issue is
[KI-COMPAT-025](../../agent/manuals/known-issues.md#ki-compat-025-cpu-bf16-autocast-narrows-an-explicitly-float32-loss).
The six lazy-import/order failures are tracked as KI-COMPAT-026. These limits are not fixes in this
integration and are not represented as passing smoke results. Acceptance here covers the four
runtime prerequisites, normal package upgrade, and the stated single-device inference recipe.

Source entry points are `python tools/run_test_suite.py --tier core` / `--tier smoke`.
In Torch mode, set `JITTOR_TORCH_SHIM=1` and `PYTHONPATH=python:tests`, then use
`python -m pytest -q` with `tests/structure`, `compat/tests/torch/test_torch_shim_aliases.py`,
`compat/tests/torch/test_dtype_view_strides.py`, `compat/tests/torch/test_accelerator_memory_info.py`
and `compat/tests/triton/test_triton_backend.py` (the last on actual CUDA).
Native CUDA runs separately against `tests/backends/cuda/test_cufft_errors.py` and
`tests/backends/cuda/test_softmax_cuda_grad.py`, without mixing native and Torch modes in one process.

### Attribution of Torch smoke namespace and environment failures

The complete Torch smoke run must not be counted as passing. For its seven activation/alias/FSDP
structure failures, independently rerunning four complete files on original `9391a815` first
produced 44 passes and one failure. The remaining failure came from the validation environment's
`000-frontend-stubs.pth`, which added `shim/resources/stubs` and violated the environment-only
activation test's no-resource-path premise. After moving that file intact into the external state
directory as a backup, all 45 cases passed in 18.76 seconds; see `logs/shim-clean-env-files.log`.

Importing only `torch.distributed.run` and `jittor.compat.fsdp2.activation_checkpoint` after
collection reproduces the other six failures (six failed, 35 passed; `logs/shim-order-repro.log`).
The former adds an unregistered Torch module key, so strict ownership rejects reinstallation;
the latter is a legitimate lazy submodule import that adds a package attribute absent from the
initial FSDP reflection snapshot. Relevant tests, ownership checks and lazy-import implementations
are unchanged from remote `88b48a1` and were not modified by these four fixes. Strict ownership
checks remain intact; these assertions/tests were not relaxed or edited. Test isolation and runtime
ownership of lazy modules need separate follow-up. Passing files do not establish whole-smoke success.

## Normal installation and upstream entry-point revalidation

An isolated installation environment started with old `2.0.0` fixtures. Normal pip installation
of the candidate adapter and its dependencies automatically upgraded core/frontend to `2.0.1`,
without manual patches or a `--no-deps` bypass. Both direct dependencies use immutable official
repository source archives at `ee116312`, with `#subdirectory=compat` for the frontend. This does
not claim a PyPI publication of `2.0.1`. PEP 610 provenance and comparison of 1348 installed files
confirmed the source identity. Runtime imports came from site-packages without source-worktree paths.

With vLLM `0.29.0` and adapter `2.0.0.dev3`, one official Python API regression and seven official
`vllm serve` HTTP regressions passed. Qwen3-0.6B ran on one CUDA device with float32, eager,
TP1 and TRITON_ATTN. All 596049920 parameters belonged to Jittor CUDA before and after generation,
with zero fallback. This is execution evidence for the current installed identity.
Historical tiny-model independent-oracle and token comparisons retain their original baseline;
no complete independent oracle was rerun here, so historical numerical errors are not new
measurements of this source revision.

Current raw logs, environments and artifacts remain unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-core-merge/20261101/`. `logs/` holds source and installed acceptance;
`version-gate/` holds final source identity, structure and alias logs. `merged-provenance.json`
records installation origin and file comparisons; `merged-runtime-provenance.json` records actual
imports and CUDA identity. Integration is tracked in [PR #18059](https://github.com/Jittor/jittor/pull/18059); no whole-suite green claim is made.

## Remaining scope limits

This does not add ROCm, Corex, NPU or multidevice acceptance. Storage façade pointers, general
gap-containing `as_strided` and historically observed split Python writeback paths have independent
limits, recorded in [KI-COMPAT-024](../../agent/manuals/known-issues.md#ki-compat-024-storage-façades-and-some-constructed-views-retain-alias-gaps).
Physical sharing for this dtype-view path must be checked through Tensor.data_ptr, strides, offsets
and write-through behavior. Correct read values or foreign-kernel writes do not establish general
Python alias semantics.
