# SGD checkpoint continuation and precision-policy state

- Status: bounded optimizer continuation verified; CPU low-precision parity remains open.
- Reviewed: 2026-10-16.
- Owner: native optimizers and Torch compatibility maintainers.
- Original fresh-process measurement: `72aebf9264f1360178ac876c88fedff4602bb6d5`.
- Maintained test source: `2715b895ee76d3f9d0971640bdfd62e83f9139f1`.
- Review when: SGD updates, momentum initialization, precision policy, optimizer state or serialization changes.

Two maintained tests add checkpoint coverage without changing optimizer mathematics.
The portable Torch contract covers FP16/BF16/FP32, plain SGD, momentum with weight
decay and dampening, and Nesterov. Each uses two parameter groups with different
learning rates, saves after three real backward/update steps, and compares two
resumed steps with uninterrupted execution. Restored constructor defaults and
parameter values deliberately differ, so the test must restore the checkpoint's
group fields, full parameters and momentum. Published gradients remain unchanged
by each update.

A separate native test checks real `jt.save`/`jt.load` for FP32 on CPU and CUDA,
with `torch_math=False` and `True`. Loading into an optimizer constructed with the
opposite policy restores the saved policy and exact continuation. A legacy state
without that field preserves the constructor's policy. This native test is a
same-process serialization test; the separate portable evidence below uses new
producer/consumer processes.

Eight actual fresh workers cover candidate and genuine Torch on CPU and CUDA.
Each producer writes nine real checkpoint files, then continues training. Its new
consumer reads those same files and continues with restored state. Independent
review rehashed the complete arrays, checkpoint files, original seven source
receipts against Git blobs, worker script, provider receipts and loaded framework core binaries.
It recomputed all **504 same-runtime restore/continuation arrays** exactly and
another **504 published gradient arrays** directly from the loss coefficients,
including FP16/BF16 rounding. All eight PIDs differ. CUDA observations include
actual driver pointer attributes; candidate whole-call fallback counts remain zero.

| Cross-runtime scope | Original-budget result |
| --- | --- |
| CUDA, nine configurations in producer and consumer phases | 18 matched, 0 failed; 336 update/gradient arrays checked |
| CPU, same configurations/phases | 7 matched, 11 with low-precision differences; 336 update/gradient arrays checked |

Including initial and saved snapshots, independent review also compares all 444
raw arrays per device: CUDA matches 444/444; CPU matches 302/444, with 142
low-precision differences retained.

CUDA FP16/BF16 comparisons are exact; FP32 uses `atol=2e-9, rtol=2e-6`.
CPU mismatches affect the six low-precision configurations, with eleven of their
twelve phase records failing the original budget. They remain open and are not
removed from the tests or raw results. Same-runtime exact resume passes for all
configurations on both devices. Complete raw parameter groups are retained:
candidate groups additionally expose `n_step`, so this is not complete optimizer
ABI equivalence. The genuine reference is independent Torch `2.13.0+cu130`, whose
actual `_C` SHA256 is
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`.

```bash
SGD_CHECKPOINT_DEVICE=cpu JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_sgd_checkpoint_continuation.py

# In a real CUDA-enabled environment:
SGD_CHECKPOINT_DEVICE=cuda JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_sgd_checkpoint_continuation.py
JITTOR_TORCH_SHIM=0 PYTHONPATH=python \
  python -m pytest -q tests/optim/test_sgd_policy_checkpoint.py
```

The original fresh-process data stays bound to `72aebf926`, before the later
explicit device labels and native test were added. Unchanged portable helpers
are used by that harness; subsequent test commits do not relabel its measurement.
Raw commands, failures, checkpoints and arrays are unversioned under
`$JITTOR_LAB_ROOT/_state/sgd-checkpoint-contract/20261016/`; independent review is
under `$JITTOR_LAB_ROOT/_state/sgd-checkpoint-independent-review/20261016/`.
Main integration records are under
`$JITTOR_LAB_ROOT/_state/sgd-checkpoint-main-integration/20261016/`.

This result does not grant low-precision Torch CPU parity, distributed SGD,
full-model support, other accelerators or a unified ecosystem L3 grade.

Main integration on the maintained source passes the portable test plus existing
SGD precision/behavior neighbors: CPU **20 / 7 CUDA skips**, CUDA-visible
**26 / 1 CPU-domain skip**. The native checkpoint test plus optimizer-core and
RMSprop neighbors pass CPU **41 / 9 CUDA skips**, CUDA-visible **50 / 0 skips**.
CUDA-visible counts include the native test's CPU controls; explicit CUDA cases
check their real device. All four whole-call guards report zero fallback.
Core passes **322 / 42 skips / 1 existing xfail** and combined structure gates
**1628 / 6 skips**. New-test Ruff, repository layout and manifest checks pass.
The final portable test also passes nine cases per device unchanged in genuine
Torch, and its standalone candidate CUDA run passes the strict execution gate.
