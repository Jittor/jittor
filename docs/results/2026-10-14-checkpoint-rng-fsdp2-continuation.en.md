# Native RNG restoration and FSDP2 FULL checkpoint continuation

Status: CPU and CUDA single-process Accelerate RNG and numerical continuation
verified; two-rank CUDA FSDP2 FULL model/optimizer continuation verified against
independent PyTorch. Sharded distributed-checkpoint I/O and ACL remain unverified.
This report supersedes the RNG and multi-rank limitations in the earlier
[optimizer checkpoint report](2026-10-14-optimizer-checkpoint-continuation.en.md).

Baseline: `7602f9040c1d63a61209bc50a575cb440c787d57`, integrating the first checkpoint
implementation `1da40ecc0` and fetched `origin/2.0-refactor` commit `7ed9b3b52`.
Implementation: the follow-up commit containing this report.
Maintainer: Jittor compatibility maintainers.
Review when: native RNG snapshot formats, Torch RNG owners, checkpoint FQN or
rank-zero transforms, FSDP shard replacement, or downstream versions change.

## Verified failures and changes

A real `Accelerator.save_state`/`load_state` probe initially restored Python and
NumPy streams but rewound the Torch CPU stream to its seed. The frontend now
encodes the existing native CPU generator snapshot as a CPU `uint8` tensor and
restores that complete snapshot through the native owner. This preserves the
advanced stream position and does not reseed CUDA. Native CUDA already provides
cuRAND seed/offset snapshots and required no change. Invalid byte payloads and
legacy seed-only integer tensors fail explicitly; an invalid restore leaves the
current stream position intact. Snapshot bytes are opaque and are not portable
between native Jittor and PyTorch generators.

The existing advanced-prefix `torch.random.fork_rng` regression now passes
without an expected-failure marker. CPU RNG tests explicitly select the host
backend even in a CUDA process. The actual Accelerate regression checks Python,
NumPy, Torch CPU, and the selected device's next random values after restoration,
then checks the resumed model, moments, steps, gradients, scheduler and loss.
This is a fixed-input training case with separate stream-continuation assertions;
it does not cover shuffled dataloader or arbitrary custom-object restoration.

The actual two-rank FULL save/load case exposed three additional boundaries:

1. Accelerate imports `checkpoint.default_planner` even for FULL files. Both
   planner names are now importable, registered as `UNIMPLEMENTED`, and their
   constructors unconditionally raise `NotImplementedError`. The FULL path uses
   ordinary `torch.save`/`torch.load`; this does not implement sharded planning.
2. A nonzero rank discarded lazy FULL gathers before rank-zero serialization,
   while advancing to the optimizer metadata collective. Both ranks hung at
   different NCCL operations. Host-offloaded tensors now materialize before any
   rank discards its FULL result. The retained timeout traces identify rank zero
   inside serialization's `sync_all` and rank one inside metadata all-gather.
3. FULL model loading refreshes flat shard views. Optimizer loading now uses the
   existing FSDP parameter-refresh owner before resolving FQNs, preserving live
   optimizer ownership after shard replacement. No update mathematics changed.

## Runtime and numerical evidence

Both runtimes used Accelerate 1.10.1, offline deterministic weights and inputs.
The independent oracle was binary PyTorch 2.11.0+cu128, with shim identity absent
and Jittor unloaded. Missing environment packages were supplied in external
isolated overlays; shared environments were not modified. CPU and CUDA used
separate caches, with initial native/NCCL compilation completed before two-rank
execution. Checkpoint numerical acceptance cases forbade backend fallback and
observed no count change.

| Verification | Result |
| --- | --- |
| Identical single-process checkpoint cases, CPU | PyTorch 7/7; shim 7/7 |
| Identical single-process checkpoint cases, CUDA | PyTorch 7/7; shim 7/7 |
| Combined CPU serialization/FSDP ownership/RNG/checkpoint regressions | 79 passed, 2 accelerator skips |
| CUDA RNG/checkpoint/planner regressions | 24 passed, no skips |
| Torch compatibility files from the `core` whitelist | 224 passed, 1 accelerator skip |
| Two-rank FSDP2 FULL Accelerate save/load | Both runtimes, both ranks passed |
| Maximum absolute single-process error, CPU / CUDA | 1.49e-8 / 3.73e-9 |
| Maximum absolute FULL checkpoint error, either rank | 1.87e-9 |

Cross-runtime comparisons include every saved/restored/resumed model tensor,
optimizer moment and step, learning rate and resumed loss. Single-process
comparisons additionally include all parameter gradients and input gradients.
Tolerance was `atol=2e-6, rtol=2e-5`. PyTorch's additional AdamW group field
`decoupled_weight_decay=True` was verified explicitly and excluded from shared
metadata equality; all remaining metadata keys were compared. Generator algorithms
and opaque RNG bytes were not compared between frameworks: each runtime must
exactly reproduce its own next values after restoring its own checkpoint.

The two-rank case wrote actual `pytorch_model_fsdp.bin`, `optimizer.bin`, and two
per-rank `random_states_*.pkl` files. Reports confirm two NCCL ranks, real CUDA
parameter devices, nonzero storage pointers and shim/oracle identity. The FSDP2
case checks fixed-input numerical continuation; the detailed RNG assertions run
in the single-process CPU and CUDA cases.

The required repository layout check passed. The full `tests/structure` run
completed with 1,426 passed, four skipped and four failed: unavailable ASV in the
isolated environment, the existing ACL Dropout gradient-manifest mismatch,
unregistered report toctree entries, and the generated packaging manifest. The
last two require the integration owner's shared index/manifest update; those
files were explicitly outside this worktree's edit ownership. Raw output is
`logs/followup-structure.log`. This is not an all-green structure claim.

## Reproduction and evidence boundaries

Integration recheck on 2026-10-15 started from synchronized main-tree baseline
`ad47b58143f5c54e83114f923e23c0c69322130a`. The complete native-plus-Torch CPU
`core` runner passed with 322 passed, 42 skipped and one existing expected failure.
The combined real-CUDA AMP/RNG/checkpoint selection passed all 127 cases without
skips. The complete structure selection passed with 1,432 passed, six skipped
and no failures after updating the shared report index, packaging manifest and
CodeOp-gradient inventory. Two structure skips concern distribution resolution
in this source-only interpreter; the others are declared or accelerator
capability skips. Repository layout and whitespace checks passed. This recheck
supersedes the worktree structure failures above and does not add ACL or sharded
checkpoint acceptance. Unversioned integration logs are under
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`, named
`amp-rng-integration-core.log`, `amp-rng-integration-cuda.log` and
`amp-rng-integration-structure.log`.

Use independent, version-matched environments with the shim deployed only in its
own interpreter. Run these selections on CPU, then CUDA, setting
`CHECKPOINT_DEVICE` for the checkpoint test and explicitly selecting the matching
native backend. Set `JITTOR_REQUIRE_ACCELERATE=1` to refuse a missing dependency:

```bash
python compat/tests/torch/test_optimizer_checkpoint_state.py -v
JITTOR_TORCH_SHIM=1 python -m pytest -q --rootdir=. \
  compat/tests/torch/test_torch_fork_rng.py \
  compat/tests/torch/test_core_misc_owner.py \
  compat/tests/torch/test_checkpoint_planner_boundary.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py
```

For two-rank shim execution, enable the installed NCCL backend and use the native
launcher with two allocated devices. A first compilation must finish serially.
Set `JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED=1`, `ACCELERATE_USE_FSDP=true`, and
`FSDP_VERSION=2`, then execute:

```bash
python python/jittor/distributed/launch.py -n 2 --backend nccl --standalone \
  --logdir "$JITTOR_LAB_ROOT/_state/checkpoint-optimizer/fsdp-shim-logs" \
  -m pytest -q -s --rootdir=. --timeout=90 --timeout-method=thread \
  compat/tests/torch/test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume
```

Run that identical case with independent PyTorch's two-process launcher. The
oracle run used a byte-identical external copy of the test, without repository
conftest installation hooks, and asserted `torch_compat_active=False` and
`jittor_loaded=False` in each report. This avoids converting the oracle into a
shim through test collection.

Raw logs, JSON reports, the failed pre-fix RNG probe, both-rank timeout traces,
and strict numerical-comparison script remain outside the repository under
`$JITTOR_LAB_ROOT/_state/checkpoint-optimizer/`. Relevant logs are
`logs/rng-cpu-targeted.log`, `logs/rng-cuda-targeted.log`,
`logs/followup-targeted.log`, `logs/core-torch.log`, `logs/fsdp-reference.log`,
`logs/fsdp-shim-refresh.log`, `logs/numeric-comparison.log`, and the corresponding
per-rank log directories. The `core` result is the Torch compatibility whitelist
selection, rather than a claim that the complete native-plus-Torch runner ran.
