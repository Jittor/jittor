# Native two-node acceptance entry: maintained runner and single-host real CUDA re-verification

- Status: the maintained entry, the host contract and the single-host two-agent CUDA protocol pass; real multi-host acceptance is not complete.
- Date: 2026-10-15; Owner: Native distributed maintainers.
- Remote synchronization baseline: `7a1e66b3858279b40eaec0c4895c6059ce8eb8db`; after local work was preserved, merged as `5e4ca616584885a5553be025e3bb9df5b81a4f7f`.
- Final measurement source: `0c0e711c71bf05f2e7ff54f51b50fbbff562813f`; the worktree was clean at measurement time.
- Recheck condition: the launcher, runner, source/environment selection, NCCL, the maintained FSDP fixture, the independent oracle or the actual host topology changes.

## Gap and entry point

Previously the skill provided the maintained fixture and manual environment setup, while the actual orchestration script lived outside the repository and hard-coded local paths/devices.
It launched two local agents on its own and could not serve as the maintained command that each of two real hosts runs.

Added [run_native_multinode.py](../../agent/skills/jittor-distributed-verification/run_native_multinode.py)
and a worker, plus a thin `nox -s native_multinode -- <runner arguments>` entry point.
Complete commands are in the [distributed verification skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md#two-host-maintained-entry).
Explicit arguments bind the source path/SHA, the candidate and independent reference interpreters, node rank, master address/port, job id,
locally allocated devices, private caches and the shared results directory. The current protocol is limited to two nodes with one CUDA card per node.

Both hosts each run the same entry, executing in stages `warm → collectives → parity → baseline → resume`,
waiting for both sides to finish each stage; node 0 additionally runs `oracle` before parity and runs `seal` last.
No SSH is sent, no remote is configured, and checkpoints are not transferred automatically; a shared filesystem is an explicit prerequisite.
Warm on each side uses the public jtrun's real rank 0/world 1, and a cache index does not impersonate a communication rank.

Each of the three training stages creates a new public launcher/worker and invokes the existing maintained pytest nodes.
FSDP, optimizer, RNG, checkpoint mathematics and all numerical budgets are unchanged.
The source snapshot contains every regular Git file plus the target text of symlinks; real host identity, PIDs,
interpreter/prefix, the allowed runtime-environment fields, argv, return codes and log hashes are recorded in the actual process.
A strict environment allowlist avoids copying the full process environment. On failure the claim and partial logs are kept; overwriting and re-running over them is forbidden.

Seal requires every stage on both sides and all four independent references to have actually succeeded, then binds the raw data and invokes the NumPy rechecker.
`verification.json` is written only if the recomputation succeeds. Different hosts may use the same local paths and device ordinals;
isolation is judged by `(actual host identity, path/device)`; historical records without a host identity keep the strict isolation rule.
Host identity is not proof of external physical topology, and the tool does not grant real multi-host acceptance on that basis.

## Actual verification

Real CUDA used two RTX 4090 cards on the same physical host, Cu122/NVCC 12.2.140, with the candidate in native mode.
The independent interpreter loaded binary Torch `2.11.0+cu128`, with TF32 disabled and highest precision in use.
Each node used its own HOME/JITTOR_HOME/TMPDIR; the first core, extension and model JIT completed serially in order.
The final re-verification reused each node's private unit cache for this task; production/math sources were unchanged, and the cache was not shared with the benchmark
or another concurrent session. Elapsed time is not reported as a performance or peak-memory conclusion.

| Check | Result | Actual scope |
| --- | --- | --- |
| Host / tools / pytest / child / nox adjacency | 78 passed, 0 skipped | Includes the actual plan, child-process records, interpreter aliases, and negative cases for source and cache isolation |
| Full structure | 1508 passed, 6 skipped | clean `0c0e711c7`; pre-existing skips do not count as CUDA execution |
| Public singleton warm | both sides succeeded, fallback delta 0 | real rank 0/world 1 on each side; separate caches |
| Public two-agent collectives | 3 passed per rank, 0 skipped | reverse-order/independent/singleton subgroup, WORLD and mixed sets |
| Parity | 4 passed per rank, 0 skipped | SGD/AdamW × FP32/BF16, five-step independent dense oracle, all gradients/parameters/moments |
| Baseline | 4 passed per rank, 0 skipped | dropout 0.25, continue two steps after saving at step three |
| Fresh resume | 4 passed per rank, 0 skipped | new launcher/PID/model/optimizer, all arrays and four subsequent RNG draws exact over two steps |
| Seal / independent NumPy recomputation | 680 / 368 / 160 / 160 | oracle arrays / exact resume arrays / update budget / zero-gradient rejection |

The final full-suite run deliberately inherited a wrong source path and CPU test routing, and the runner still bound the explicit source and CUDA correctly.
A separate source audit matched the 2,818 Git blobs of the two snapshots one by one, including the target text of one directory symlink.
The 299 files of the historical `bc07df0c` were not rewritten; the new rechecker's historical recomputation JSON is field-for-field identical to the old result.

The first host check in fact found that a test handed the complete environment to `child_env` without `inherit=False`; it passed after the fix.
The first actual warm also reproduced `Path.resolve()` replacing the virtual-environment interpreter with the base Python, reporting a tqdm import failure;
this was changed to keep the explicit executable path, a symlink regression was added, and the worker now reads the real interpreter/prefix.
The full success matrix of `796c0903` is kept separately; explicit source/device routing pins were then added, and the whole protocol was finally re-run at `0c0e711c7`
in a new results directory. The original failures, the original measurement SHA and the original arrays were neither relabelled nor overwritten.

## Evidence and boundaries

Unversioned root directory: `$JITTOR_LAB_ROOT/_state/native-multihost-runner/20261015/`.
Final results are under `attempt3/shared/`; the maintained entry produces `commands/`, per-node/per-stage logs, the source snapshot,
complete NPZ/checkpoints, worker identities, `raw-bindings.json` and `verification.json`.
The binding SHA256 of the 282 raw files is `6ece2b422165093e0a93a4523fd3b40f777b188c41b72cd986da879352a8cf5e`.
`independent-source-audit.json` holds the independent Git/blob/source and device pin audit;
`structure-final.log`, `source-pin-fixed.log` and the final verification manifest hold the gate and additional evidence hashes.

This proves the maintained command can actually execute the complete protocol on two local agents; it does not prove the network,
firewall, filesystem, network-partition handling or multi-host training/recovery of two physical hosts. The entry also does not extend to multi-node HCCL,
arbitrary world size, elastic membership or checkpoint transfer. R17's real multi-host acceptance stays retained.
