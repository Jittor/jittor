# bf16 autocast and GradScaler boundary verification

- Status: Fixed within this report's scope; compatibility-layer fidelity stays `APPROXIMATE`.
- Date: 2026-10-14.
- Baseline: `cded3102037edc3663b8f2054a6ebc06da6ebfd2`; the fix commit is determined by the commit containing this file.
- Maintainer: Torch compatibility layer and AMP maintainers.
- Scope: real CPU, NVIDIA RTX 4090 CUDA; ROCm, ACL/NPU not verified.
- Review when: AMP registers, node flag layout, convolution dispatch, GradScaler state machine, or the reference Torch version change.

## Confirmed problems and changes

Before the fix, all `torch.autocast(dtype=torch.bfloat16)` matrix multiplications with FP32 inputs produced FP16, giving NaN with input `1e5`. Added a separate bf16 preference bit and widened the AMP field saved and restored on Op to seven bits, so the backward executed after exiting autocast also retains the request. The old `prefer16` rule is kept; an explicit bf16 request applies to both all-FP32 and mixed FP32/FP16 operands.

The CPU oneDNN convolution entry can only return FP32, so the low-precision region uses the existing portable convolution implementation. CUDA transposed convolution's bias is also converted to the compute dtype, avoiding bf16 results being promoted by FP32 bias. Measured matmul, linear, conv2d, conv_transpose2d all output bf16, with gradients for input, weight and bias executing on the declared device. New regressions use `forbid_backend_fallbacks()` to forbid backend fallback and check actual residency of outputs and each gradient after synchronization.

An independent real PyTorch 2.11.0+cu128 CPU/CUDA process confirmed the following GradScaler behaviors, which this implementation matches:

| Boundary | Behavior after alignment |
| --- | --- |
| scale=0, input gradient=0 | IEEE reciprocal produces NaN; no Python divide-by-zero exception; the finiteness check reads the gradient before division |
| input gradient finite, overflows after dividing by a small scale | this step still executes; newly produced Inf is not re-judged as input overflow |
| scale=0.25, input gradient=Inf | skip the step, scale falls back to 0.125; Torch mode keeps the gradient for the caller to inspect |
| CPU float64 gradient=1e300 | keeps the float64 finiteness-check domain, not misjudged by demoting to float32 |
| CUDA float64 gradient=1e300 | consistent with Torch CUDA's float-parameter finiteness check, judged non-finite and skipped; this difference was actually checked side by side |
| growth causes float32 scale overflow | keep the original scale, reset the growth tracker |
| initial scale is ±Inf, NaN | keep IEEE numeric semantics; no extra finite-value clamping |
| manual update | accept 0, below 1, Inf, NaN and same-device scalar tensors without gradients; reject vector shapes |
| repeated unscale/step, uninitialized, no-gradient check | error per Torch's stage and initialization constraints; each optimizer records independently |
| FP16 gradient | reject before unscale, keep the gradient's original value |

Native and Torch modes share the numeric algorithm; the native optimizer's existing behavior of clearing consumed gradients after skipping a step is kept. In Torch mode the caller is responsible for `zero_grad()`. Device-state paths also cover scale=0, fallback below 1, division overflow and growth overflow; existing CUDA capture/replay regressions pass.

## Verification method and results

Source and independent Torch oracle run in separate processes. A private Python 3.11 venv uses pytest 7.4.4, pytest-timeout 2.3.1; the Jittor compiler is GCC 11.5, CUDA 12.2.140. The Torch oracle confirmed `_C` is an independent binary extension and actually created and computed CPU/CUDA tensors. Python dependencies missing from the original environment were added to the lab-private venv; the shared environment was not modified. Each run's HOME, JITTOR_HOME, TMPDIR and raw logs are in `$JITTOR_LAB_ROOT/_state/refactor-goal/amp/`, first JIT serial.

Targeted selection as follows, with `JITTOR_TEST_DEVICES=cpu,cuda`; source mode activates the compatibility frontend through the repository Torch bootstrap:

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_torch_amp_boundaries.py \
  compat/tests/torch/test_torch_amp_fidelity.py \
  compat/tests/torch/test_torch_amp_training_loop.py \
  compat/tests/torch/test_torch_compile_replay.py::TestCompileTrainingCuda::test_a_gradient_scaler_skips_overflows_on_the_device

JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q \
  tests/nn/test_grad_scaler.py tests/runtime/test_amp_reg_bits.py \
  tests/core/test_node_flag_layout.py
```

- Compatibility layer: 103 passed, 0 skipped; includes the new numeric and API boundaries and real CUDA capture/replay.
- Native layer: 29 passed, 1 pre-existing xfailed, 0 skipped. The pre-existing expected failure is weak scalar promotion of float16 loss multiplied by a Python float (`KI-DTYPE-004`), not part of this fix.
- Additionally verified with real CPU/CUDA that 1×1, 1×2, 2×1 matrices return bf16 with all-FP32 inputs.
- Node layout and AMP bit targeted checks passed; import layering keeps 165 in-cycle modules, no growth.
- `bash tools/check_repo_layout.sh` and `git diff --check` passed.
- Full structure selection: 1427 passed, 4 skipped, 3 failed (177.06 seconds). One is the baseline `DropoutACL` gradient registration being stale; the other two require wiring this new report into the results toctree and updating `MANIFEST.in`, shared files the main task maintains centrally and re-verifies. Two skips are package-resolution checks because this private venv lacks the core distribution; the other two are declared skips; this is not used to claim the complete structure gate passed. New AMP and node layout checks passed.
- The main task runs core and related regressions uniformly after integration; this subtask did not rerun the full suite.

## Retained precision and behavior boundaries

### Main-tree integration recheck (2026-10-15)

The main tree's synchronized baseline is `ad47b58143f5c54e83114f923e23c0c69322130a`. After AMP and complete RNG restoration were integrated together, the following rechecks were done with separate CPU/Torch caches and serial first compilation:

- `tools/run_test_suite.py --tier core --serial-compile`: 322 passed, 42 skipped, 1 pre-existing xfailed; both native and Torch subprocesses returned 0.
- Combined real-CUDA AMP, capture/replay, RNG and checkpoint regression: 127 passed, 0 skipped; declared-CUDA computation executed on CUDA.
- Native AMP, node flags and new ACL acceptance case selection: 29 passed, 4 skipped, 1 pre-existing xfailed. All four skips are due to no ACL hardware and cannot be taken as NPU gradient passing.
- Full `tests/structure`: 1432 passed, 6 skipped, 0 failed. After completing the report index and manifest, and making the gradient inventory recognize CodeOp backward inside helpers, all previous structure failures were eliminated. Two skips are resolution checks because this source interpreter lacks the core distribution; the rest are declared or accelerator-capability skips.
- Repository layout and `git diff --check` passed.

Unversioned logs are in `$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`, namely `amp-rng-integration-core.log`, `amp-rng-integration-cuda.log`, `amp-rng-integration-native.log` and `amp-rng-integration-structure.log`. CUDA warmup initially failed using the removed `setup_cublas` interface; after switching to the existing `probe_library(..., load=True)`, actual library loading, CUDA computation and the combined regressions above passed.

### Precision and behavior limits

The bf16 here is a real dtype and exponent range, not approximated as FP16. Computation still does not guarantee bit-identical results with Torch: the CPU portable convolution backward has an extra bf16 intermediate rounding; the example weight gradient is 892928 versus 897024 for independent Torch, an error of about 0.46%, and the regression uses a 1% tolerance; matmul and linear examples' output, input and weight gradients use exact assertions, and the CUDA cuDNN example also gets 897024.

Autocast is still based on a global AMP register, lacking Torch's complete per-op allow/fallthrough policy, per-device context, and weight conversion cache; e.g. some FP32 add may also be lowered. GradScaler still uses the optimizer bridge and one host-visible reduction for its regular checks, and does not implement Torch's private `_check_inf_per_device` and multi-device gradient-copy protocol. This report does not upgrade these boundaries to `EXACT`, nor claim that uncovered ops, shapes or devices are compatible.
