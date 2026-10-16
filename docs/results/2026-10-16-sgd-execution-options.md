# SGD foreach and fused execution options

- Status: KI-COMPAT-008 closed for the verified CPU/CUDA option contract.
- Reviewed: 2026-10-16.
- Baseline: `ae1df1983391d0696e72a0dec98696fe2236a1e3` plus this report's three production changes.
- Owner: Torch compatibility and native optimizer maintainers.
- Review when: SGD dispatch, parameter groups, state export or checkpoint loading changes.

`torch.optim.SGD` now accepts keyword-only `foreach` and `fused`, preserving
constructor defaults and group overrides in its state dictionary. The native
SGD owner selects its existing batched kernel according to these preferences:
`fused=False` disables that kernel; `foreach=False` disables it unless
`fused=True` explicitly requests it. Defaults and `foreach=True` allow the
backend's eligible implementation. Asking for both options to be `True` raises
the same `RuntimeError` as Torch.

Eligible CUDA FP32 tensors use the existing batched SGD kernel. Other supported
tensors retain native JIT updates on their own backend. No new optimizer
mathematics, host update or backend fallback is introduced. Native constructor
defaults remain unchanged. State export also retains inherited execution options
for newly added groups, and a restored optimizer uses its saved group options.

Before this change, all eight legal option combinations raised `TypeError` in
the frontend initializer. The maintained CPU regression had **27 failures /
27 CUDA skips**; independent Torch accepted the legal combinations and rejected
the conflicting pair. After the change, the new test passes unchanged in
genuine Torch `2.13.0+cu130`: CPU **27 / 27 CUDA skips**, CUDA-visible **54 passed**.
Candidate tests also include existing precision, checkpoint and SGD neighbors:
CPU **47 / 34 CUDA skips**, CUDA-visible **80 / 1 CPU-domain skip**. Whole-call
candidate guards record zero backend fallback.

The contract covers plain SGD, momentum with weight decay and dampening, and
Nesterov, using two parameters and three backward/update steps. All parameter
and momentum arrays match 30 independently generated genuine Torch FP32 goldens
exactly. Inputs are dyadic, so the same exact references exercise each execution
choice on CPU and CUDA. A serialized state round trip restores both options
and exact subsequent parameter/momentum updates. Group overrides and inherited
options on newly added groups are checked separately.

CUDA observations check actual driver pointer attributes. Dispatch instrumentation
counts calls and delegates to the original native kernel: disabled choices make
no batched calls, enabled eligible choices write both parameters, and the
restored optimizer follows the saved selection. The mixed-group case launches
only the eligible group. Tests retain the original published gradient objects
and their complete values after each update.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_sgd_execution_options.py \
  compat/tests/torch/test_sgd_checkpoint_continuation.py \
  compat/tests/torch/test_torch_compat_sgd_precision.py \
  compat/tests/torch/test_torch_compat_optim.py::TestSGD
```

CPU low-precision cross-runtime differences from the separate
[checkpoint report](2026-10-16-sgd-checkpoint-continuation.md) remain open.
This result does not establish complete optimizer ABI, distributed SGD, other
accelerators or full-model support. Raw commands, failures, exact source hashes,
independent goldens and loaded core snapshots are unversioned under
`$JITTOR_LAB_ROOT/_state/sgd-execution-options/20261016/`.

A separate CPU probe exposes **KI-COMPAT-015**: portable SGD rebinds the parameter
allocation, leaving a retained detached alias stale. All four observed option
combinations differ from genuine Torch in that storage contract. Option parsing,
selection and numeric tests above pass, but storage-preserving portable updates
remain open; CUDA aliases and other optimizers have not yet been checked.

Adjacent native optimizer contracts pass CPU **41 / 9 CUDA skips** and
CUDA-visible **50 / 0 skips**, including CPU controls. Core passes
**322 / 42 skips / 1 existing xfail** and combined structure gates
**1628 / 6 skips**. New-test Ruff is clean; the changed legacy API file retains
eight existing Ruff diagnostics, with zero new diagnostics. Layout and manifest
checks pass.
