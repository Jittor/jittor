# Main-repository integration of runtime prerequisites used by vLLM

[中文](2026-11-01-vllm-runtime-integration.md)

- Status: validation of the integrated baseline is in progress; historical results below do not admit the current baseline.
- Date: 2026-11-01.
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

Current results are pending. Recheck entry points include layout, `tests/structure`, CPU core/smoke,
`compat/tests/torch/test_dtype_view_strides.py`, `test_accelerator_memory_info.py`,
`tests/backends/cuda/test_cufft_errors.py`, CUDA softmax tests and
`compat/tests/triton/test_triton_backend.py`. Installed acceptance must use normal core/frontend
builds of this integration, then invoke official vLLM API/server entry points in an independent
environment. Previously patched wheels cannot establish acceptance of current source.

## Remaining scope limits

This does not add ROCm, Corex, NPU or multidevice acceptance. Storage façade pointers, general
gap-containing `as_strided` and historically observed split Python writeback paths have independent
limits, recorded in [KI-COMPAT-024](../../agent/manuals/known-issues.md#ki-compat-024-storage-façades-and-some-constructed-views-retain-alias-gaps).
Physical sharing for this dtype-view path must be checked through Tensor.data_ptr, strides, offsets
and write-through behavior. Correct read values or foreign-kernel writes do not establish general
Python alias semantics.
