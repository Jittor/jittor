# Integrated regression and NCCL defect closure

- Status: exit conditions for KI-DIST-002 and KI-DIST-003 passed; CPU/CUDA targeted integration verification.
- Date: 2026-10-15.
- Integration baseline: `f02e7aec0b43daf1b275a97eab340de7c4e917d3`; local changes were kept after synchronizing `origin/2.0-refactor`, and this document is committed together with the final integration changes.
- Maintainers: Jittor core, Torch compatibility and distributed maintainers.
- Recheck when: collective scheduling, default tensor placement, FSDP sharding/optimizer, NCCL version or gate process mode changes.

## Real NCCL 2.18.3 verification

Two RTX 4090 ranks used Python 3.11.16, nvcc 12.2.140 and driver
580.178.04. The official `nvidia_nccl_cu12-2.18.3` wheel has SHA256
`89cf220910c43e53df992f54f7aa2a35c6d606b077c9ae95b1f17c544d3ad3f3`.
During serial warmup on each rank, the actual `ncclGetVersion` returned `21803`, and
the process mapping was checked to load only that version's real library; Python
package metadata was not used as a substitute for dynamic-library evidence.

| Targeted scope | Result per rank |
| --- | --- |
| External pure-native fast collective sequence | 75 exact checks, Torch shim not loaded, fallback delta 0 |
| Computation checks of the PR #18031 native smoke | 20 checks, both ranks exited 0, including bf16, DDP and a training step |
| New maintained test `tests/backends/comm/nccl/test_native_collectives.py` | 1 passed, 0 skipped, 75 exact checks, fallback delta 0 |
| Reordered DeviceMesh + Torch Adam subclass | 1 passed, 0 skipped, three-step NumPy reference and old-view lifetime passed, fallback delta 0 |

The fast sequence rotates reduce, bf16 reduce, broadcast, gather and scatter into the
second position without adding device-level synchronization between collectives to
mask the old symptom. The native smoke only adjusts the old script's local-rank
assertion: the launcher's logical device index differs from `LOCAL_RANK`, so the
assertion uses `LOCAL_RANK` from the environment and its computation checks remain
unchanged.

KI-DIST-002 did not reproduce in the current tree; its exit conditions are verified,
but the original cause was not isolated and cannot be attributed to any single fix
commit. KI-DIST-003, together with the existing tensor-placement owner correction
`0745b1bab5` and explicit CUDA inputs in the test, verified the complete three-step
computation on the real device. Both open entries were removed from the issue ledger;
the acceptance scope and recheck conditions are retained here.

`nox -s nccl` now warms up serially on each rank, actually loads NCCL through the
canonical `probe_library`, and then launches native and Torch modes separately, each
using its own log directory. New host contracts verify the process mode, independent
rank caches, warmup preceding launch, and node selection arguments. The actual
dual-GPU execution covered the two maintained tests in the table above; no claim is
made that the entire FSDP2 file or a complete nox session was accepted. Single-node
evidence also does not represent multi-node, HCCL or ACL acceptance.

After allocating two devices and isolating compilation caches per the environment
manual, the targeted scope can be re-verified:

```bash
python -m nox -s nccl -- tests/backends/comm/nccl/test_native_collectives.py
python -m nox -s nccl -- compat/tests/fsdp2/test_fsdp2_nccl.py::TestFSDP2Nccl::test_mesh_reordered_shards_custom_adam_and_lifetime
```

## Integration scope in the same batch

This batch integrated [SDPA masking](2026-10-14-cuda-sdpa-mask-coverage.en.md),
[graph replay budget](2026-10-14-graph-replay-retain-budget.en.md),
[CUDA 13 toolchain](2026-10-15-cuda13-toolchain-packaging.en.md),
[downstream training scorecards](2026-10-14-downstream-oracle-scorecards.en.md),
[TorchMetrics/TensorDict](2026-10-15-metric-state-and-tensordict-oracles.en.md),
[bf16 FULL continuation](2026-10-15-bf16-fsdp2-checkpoint-continuation.en.md) and
[checkpoint options](2026-10-15-checkpoint-state-options.en.md). The device,
component-level scope, rounding differences and unaccepted items in each report
continue to apply.

Before integrating checkpoint options, main-tree CPU core was 322 passed, 42 skipped,
1 existing xfailed; CUDA native SDPA/CUB/graph-retention targeted regressions were
37 passed, 0 skipped; compat and tools CPU regressions were 161 passed, 0 skipped.
The complete structure gate at that time was 1441 passed, 6 skipped, 0 failed.
Final re-verification after adding checkpoint options:

| Final main-tree scope | Result |
| --- | --- |
| CPU core, both process modes | 322 passed, 42 skipped, 1 existing xfailed |
| CPU checkpoint, including real Accelerate save/continuation | 30 passed, 0 skipped |
| CUDA checkpoint and custom Module state | 37 passed, 0 skipped |
| NCCL 2.18.3 dual-GPU bf16 FULL checkpoint | 1 passed per rank, 0 skipped, exact model/optimizer restore, two-step continuation and all four RNG kinds, fallback 0 |
| Complete `tests/structure` | 1441 passed, 6 skipped, 0 failed, 167.27 s |
| Layout and whitespace | passed |

The first checkpoint re-verification found an incomplete `safetensors` installation
in the shared environment, so real Accelerate failed during import. After switching
to an external directory with a complete dependency installation, the CPU/CUDA
re-verification above passed, and the import failure was not turned into a skip. The
first core launch inherited CUDA environment variables, which was inconsistent with
the CPU build; after explicitly setting `JT_USE_CUDA=0`, both modes passed. An
additional CUDA nodeid counting check did not recognize the naming form of these
tests; although the 37 cases passed, that session exited 1. The final targeted runs
retained the real CUDA capability and execution checks and verified them through the
tests' own physical-location, dispatch and fallback assertions; the session with the
failing count is not used as passing evidence. All of the first logs are retained.

Structure failures caused by a report entry or MANIFEST not yet being registered in
each subtask worktree retain their original baseline records; the integrated tree
registers the results index uniformly and regenerates MANIFEST, with the final gate
as authoritative. First JIT runs were warmed up serially, and CPU, CUDA and each
distributed rank used isolated caches; no compilation cache was shared with a
benchmark.

Raw logs, official wheel metadata, external scripts and rank output are unversioned
and live under `$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/`. Integration logs
include `logs/sdpa-graph-cuda13-native-cuda-recheck.log`,
`logs/integrated-compat-tools-cpu.log`,
`logs/integrated-legacy-nccl218-maintained-native.log` and
`logs/integrated-legacy-nccl218-reordered-mesh.log`; per-rank logs are in
`legacy-nccl/maintained-native-logs/` and
`legacy-nccl/maintained-reordered-mesh-logs/` respectively. The final integration
logs are `logs/final-integrated-core-cpu-recheck.log`,
`logs/final-checkpoint-options-cpu-deps.log`,
`logs/final-checkpoint-module-cuda-physical-recheck.log`,
`logs/final-integrated-structure.log` and
`logs/final-legacy-nccl218-full-checkpoint.log`.
