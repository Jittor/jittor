# ARM CPU 全量门禁与并发稳定性修复

- Status: verified for the maintained native and Torch CPU sessions on AArch64
- Date: 2026-08-30
- Baseline: `07bdc5b9` plus this change
- Owner: Jittor core and test-infrastructure maintainers
- Review when: RingBuffer/MWSR synchronization, MPI initialization, concat graph
  construction, CPU topology detection, or complete-suite ownership changes

## Scope

This pass reran the complete maintained CPU inventory on AArch64 with Python
3.11.15, GCC 10.3.1, and NumPy 2.2.6. Native and Torch sessions used separate
`JITTOR_HOME` and `TMPDIR` trees below `$JITTOR_LAB_ROOT/_state/`; first JIT
builds were serialized and CUDA/ACL were disabled.

## Findings and fixes

- MPI initialization allocated its `MPI_Allgather` receive buffer by rank rather
  than world size. Rank 0 therefore had a zero-length VLA and multi-rank import
  could terminate with `SIGBUS`. The receive buffer now has one entry per rank,
  and the MPI regression checks local-rank metadata.
- RingBuffer used `volatile` indices and one shared wait flag/condition variable.
  This had data races and could lose producer/consumer wakeups. The SPSC indices
  and stop state are atomic, producer and consumer waits are independent, and
  cached peer indices keep the uncontended path bounded. The old implementation
  was not a valid speed baseline: five isolated runs produced two corrupt
  sequences and one hang. The corrected benchmark verifies every item across
  eight rounds and retains a topology-tolerant catastrophic regression limit.
- The multi-writer/single-reader log list concurrently traversed and mutated
  `std::list`. Each writer now owns a stable locked list; the reader splices
  batches before invoking callbacks and uses an atomic pending count for
  flush/stop ordering.
- Concatenating 30,000 values built one unbounded setitem graph and exhausted
  memory. Concatenation now forms a bounded 64-way tree while preserving dtype,
  shape, forward, and backward behavior.
- PIL conversion no longer requests an impossible zero-copy array under NumPy
  2, and negative hue shifts use signed intermediate arithmetic before wrapping.
- CPU topology detection now respects affinity and sysfs SMT sibling groups;
  the `/proc/cpuinfo` parser remains a fallback. AArch64 assembly parsing accepts
  both `.global` and `.globl`.
- `jtune` rewrites compiler arguments as parsed shell tokens, so removing `-g`
  cannot corrupt paths such as `merge-gate` or `linux-gnu`.
- Architecture-sensitive timing tests now compare against local baselines or
  the tuner measurements they actually observe. Offline distributed tests use
  a synthetic dataset, notebook execution has a realistic per-test timeout, and
  the aggregate Torch check allows serialized cold JIT compilation to finish.
- CUDA wheel fixture paths now follow the implementation's absolute-path
  contract without resolving equivalent mount aliases such as `/home` and
  `/beegfs/home` to different spellings.

## Verification

| Gate | Result |
| --- | --- |
| RingBuffer repeated sequence stress | 20 calls x 8 rounds, zero wrong items |
| Changed-module CPU regressions | `95 passed, 16 skipped` |
| Complete native CPU session | `743 passed, 722 skipped` in `24m37s` |
| Aggregate Torch compatibility subprocess | `1 passed` in `92.68s` |
| Complete Torch functional inventory | `1313 passed, 518 skipped` in `5m23s` |
| Clean-worktree structure inventory | `219 passed, 2 skipped` in `2m26s` |
| Repository layout | passed |

The first Torch run started from an empty cache on a heavily loaded host and
compiled about 3,500 kernels. Its aggregate compatibility subprocess exceeded
the old 600-second local timeout while still producing new kernels. After
raising that bounded cold-build timeout, the subprocess and complete functional
inventory passed from the retained cache.

The main checkout contains old ignored `.claude/worktrees` created outside this
task. They correctly make repository-boundary checks fail, so structure and
layout were rerun against the exact tracked diff in a clean worktree under
`$JITTOR_LAB_ROOT/worktrees/`. Those user-owned directories were not removed or
accepted as repository content.

## Limits

This report proves the maintained CPU sessions on the stated AArch64
environment. It does not replace the existing real CUDA or Ascend gates and
does not establish ROCm support. The RingBuffer latency check is a broad unit
regression guard, not a reproducible cross-framework performance benchmark.
