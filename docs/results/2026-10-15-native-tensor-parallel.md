# Native TP linear training and checkpoint continuation

Status: bounded native implementation; FP32 CPU singleton and single-host,
two-rank CUDA/NCCL validation. Maintainer: Codex native distributed owner.
Recheck after native collective adjoints, optimizer gradient routing, Module
parameter ownership, checkpoint serialization or CUDA backend changes.

Initial synchronized baseline: `0cddc76749e4f881d587a1f9ee865b9129464883`.
The initial implementation and tests were preserved as `02864239b`.
After all test processes stopped, the worktree fetched origin, created a new
branch at `aec3b683bffdd85fae42b3b94d8609c0eed7cf2f` and reapplied the change
as `e7cd1a467`. No stash, discarded local work or older owner overwrite was
used. The incoming checkpoint optimizer preflight and trainability fixes were
retained. The final production source is `416d1915e` (`e7cd1a467`, common-layout
validation `17275bed3`, and lazy public exports `416d1915e`). Test-only
follow-up `dc63a31a4` routes the fresh-process import probe through the
repository child-process helper; all production file hashes remain
identical. CPU/host checks were repeated after that test-only change. The
final device observer also asserts actual FP32 dtype.

## Implemented scope

The canonical native owner is
`python/jittor/distributed/tensor_parallel.py`, exported through
`jittor.nn.parallel` as `ColumnParallelLinear` and `RowParallelLinear`.
It imports native Module, native linear math and native collectives, without
importing Torch or the compatibility frontend. See the
[mechanism and calling contract](../notes/native-tensor-parallel.md).

Column weight/output features and bias are genuinely local tensors; row
weight/input features are genuinely local, while row bias is replicated.
Native Function supplies the communication adjoints: copy backward sums,
row reduce backward is identity, gather backward slices, scatter backward
gathers. Native optimizer update mathematics is reused. TP-marked parameters
skip the legacy data-parallel gradient reduction and periodic parameter
broadcast, whose semantics would mix unrelated local shards. DDP wrapping
the same parameters and optimizer-registered extra input gradients are
explicitly refused.

The high-level native training checkpoint stores each genuinely sharded
model tensor and optimizer buffer as a rank-local leaf. It preserves
replicated checks on common partition metadata, row bias, its buffers, optimizer metadata and ordinary
model parameters. It records ordered FQNs, rank/world, orientation and
local/global shapes, and validates ownership before target mutation. It
reuses the existing native model/optimizer/RNG checkpoint owner.

## Independent reference and acceptance

The committed fixture computes the entire unsharded MLP analytically in
NumPy float64: Linear(3,6), tanh, Linear(6,4), mean squared error and native
SGD semantics with learning rate .05, momentum .6, dampening .1 and weight
decay .01. Inputs and initial weights are fixed, with both `(4,3)` and
`(2,2,3)` inputs and both column gather/row scatter choices.

An independently installed **real PyTorch 2.11.0+cu128**, verified by package
path and actual `torch.Tensor` owner, ran this same unsharded model and SGD
on **CPU float64**, for five steps and both input shapes. The reference
script imports no Jittor runtime. Across output, input gradients, all four
parameter gradients, all updated parameters, all momentum buffers and loss,
the maximum absolute discrepancy from the analytical reference was
`1.11e-16`. This establishes the independent reference; it does not claim
that Torch CUDA ran this task.

Jittor FP32 results compare every field against this unsharded reference
with `atol=2e-6, rtol=2e-5` (loss absolute error `2e-6`). Checkpoint acceptance
requires **array equality** with the uninterrupted same-runtime continuation,
without relaxing the cross-framework tolerance or replacing it with a
configuration-only assertion.

Each of the four training rows runs three steps, saves, continues two more,
loads fresh model/optimizer objects and repeats those two steps. Every rank
checks the full replicated output/input gradient, its entire local parameter
gradient/update/momentum, and replicated row bias. Two additional independent
column/row tests check weighted-loss adjoints and non-divisible feature
rejection. Negative checkpoint tests intentionally drift replicated row bias or
construct inconsistent global partitions across ranks. Both must fail on
both ranks with no complete manifest. Only the layout rank field is
rank-local; the common equal-partition schema remains replicated.

The small MLP stores **28 parameter elements per rank**, versus 52 in the
unsharded reference: column `(3,3)` and `(3,)`, row `(4,3)` and `(4,)`.
Both layer instances have exactly their two local parameter Vars. CUDA
observers synchronize actual outputs, input gradients, all parameter
gradients, updated parameters and momentum buffers, assert FP32, inspect
native CUDA dispatch and device location **before** pointer materialization,
and query CUDA driver pointer ownership. Both ranks use separate visible
CUDA devices. These are storage and execution assertions, not an inferred
memory saving or a whole-process peak-memory measurement.

## Results and retained failures

| Run | Source / result |
| --- | --- |
| Missing public owner repro | Initial baseline: native import raises `ModuleNotFoundError` |
| Initial CPU TP + checkpoint | 29 passed, no skips |
| Initial CUDA minimal column adjoint | Each rank 1 passed, actual NCCL |
| Initial CUDA full | Each rank 6 calculation/training cases passed; negative check correctly refused drift but failed only on the test's `replicated` versus actual `replicas` regex |
| Corrected negative test | Each rank 1 passed; no complete manifest |
| Integrated CPU TP + checkpoint + ZeRO | 53 passed, no skips |
| Final CPU owners, including fresh-process lazy-import regression | 54 passed, no skips, 4.59 s |
| Integrated CUDA full | Each rank 8 passed, zero skipped, 80.42 s; launcher zero |
| Final CUDA after lazy exports | Each rank 8 passed, zero skipped, 4.26 s; launcher zero |
| Final core CPU | 322 passed, 42 skipped, one existing xfail; both process modes exited zero |
| Initial full structure | 1457 passed, 6 skipped, 4 failed: two real import-cycle regressions plus unregistered documentation and generated MANIFEST |
| Import/child-process regression recheck | Lint: all three contracts pass; targeted structure: 24 passed, zero skipped |
| Layout | Passed. Central documentation toctree and generated MANIFEST remain integration-owned gates |

All successful CUDA calculations run under `forbid_backend_fallbacks()` and
assert the process-wide fallback count remains exactly zero. The four
training cases also spy on the actual native optimizer legacy collective
owner: **zero gradient-reduction and zero parameter-broadcast calls**,
including `param_sync_iter=1`. Actual TP communication is required by
nonzero off-rank input/parameter-gradient contributions and the unsharded
reference comparisons. No physical NCCL traffic volume or latency is
claimed.

The full structure run caught an actual eager-reexport regression: importing
`nn.parallel` brought `distributed` into the core initialization cycle,
increasing cyclic modules from 165 to 168. Public TP exports now use module
`__getattr__`, retaining the canonical owner without loading it during core
import. A fresh-process behavior test guards that initialization order. Its launch
uses the repository helper to pin this checkout and apply the standard
subprocess timeout. The
import contract again reports 165 cyclic modules; no allowlist changed. The
final CPU, CUDA and core reruns above include this fix. The original four
structure failures are retained; the two central documentation/manifest
registrations were absent in that run; the integration checks below cover their completion. The six
structure skips include two package-resolution checks in the uninstalled
source interpreter, two declared skips and two accelerator skips.

The explicit checkpoint-replica probe first demonstrated that wrapping the
whole model/optimizer as rank-local wrongly accepted bias drift. The final
implementation marks only truly sharded leaves and retains replica
integrity checks. A second real two-rank probe also accepted incompatible
global column shapes when the whole layout was rank-local; the fix keeps
only its rank field local and validates all common layout metadata. That original accepted checkpoint and log are retained
outside Git. A first CPU invocation used a nonexistent test path and
collected zero tests; the corrected invocation is the 29/53-pass evidence.
A first fixture attempted the unsupported SGD constructor argument
`param_sync_iter`; it now sets the existing optimizer attribute after
construction. An early collective warmup omitted the explicit NCCL backend loader; a
post-sync singleton warmup omitted its required rendezvous environment.
Both failed before testing, were retained, and were corrected; the serial
warmups then loaded NCCL successfully (`ncclGetVersion=21803`). None of these failed attempts is counted as acceptance.

## Reproduction and raw evidence

Raw files, logs, reference identity, environment scripts and isolated JIT
caches are **unversioned** under
`$JITTOR_LAB_ROOT/_state/native-tensor-parallel/`. The worktree is pinned
through `PYTHONPATH=<worktree>/python`; native mode uses
`JITTOR_TORCH_SHIM=0`. Python 3.11.16, NumPy 1.26.4, pytest 7.4.4, CUDA nvcc
12.2.140, GCC 11.5 and NCCL 2.18.3 were used. Shared environments were not
modified. Each CUDA rank has its own `HOME`, `TMPDIR`, `JITTOR_HOME` and
`cache_name=native-tp`; launcher uses the warmed rank-zero cache. CPU,
CUDA-rank-zero and CUDA-rank-one first compilation is serialized. CPU test
and core-runner caches are also separate. The latter were copied from this
owner's inactive earlier test cache into the new state root; no lock or
mutable cache inode is shared, and the runner rebuilt changed source.

Targeted CPU command, with CUDA disabled in both canonical and legacy flags:

```bash
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 PYTHONPATH=python \
  <native-python> -m pytest -q tests/nn/test_native_tensor_parallel.py \
  tests/serialization/test_sharded_checkpoint.py tests/optim/test_native_zero.py
```

The actual CUDA command uses the native launcher after serial per-rank
warmup. The unversioned `rank_entry.py` sets rank-specific cache/home/temp
paths before importing Jittor, then invokes pytest without replacing the
native runtime. `JITTOR_TP_TEST_DIR` names a fresh shared checkpoint root.

```bash
<native-python> python/jittor/distributed/launch.py --backend nccl -n 2 \
  --device-ids=<first>,<second> --timeout 180 --log-dir <logs> -- \
  <native-python> <state>/rank_entry.py -x \
  tests/backends/comm/nccl/test_native_tensor_parallel_nccl.py
```

Primary log index below the state directory:

- `logs/missing.log`, `logs/cpu-first.log`, `logs/cpu-second.log`,
  `logs/cpu-checkpoint.log`, `logs/cpu-leaf-fix-corrected.log`;
- `logs/real-torch-identity-overlay.log`, `logs/real-torch-reference.json`;
- `logs/warm-rank0.log`, `logs/warm-collectives-rank0-loaded.log`,
  `logs/warm-rank1.log` (separate rank caches);
- `logs/two-rank-minimal/rank{0,1}.log`,
  `logs/replica-drift-before/rank{0,1}.log`,
  `logs/two-rank-first-full/rank{0,1}.log`,
  `logs/two-rank-negative-corrected/rank{0,1}.log`,
  `logs/layout-drift-before/rank{0,1}.log`;
- `logs/integrated-cpu-owners.log`, `logs/integrated-cpu-final.log`,
  `logs/two-rank-integrated-final-launch.log`,
  `logs/two-rank-integrated-final/rank{0,1}.log`, `final-source.json`,
  `logs/nccl-library-identity-corrected.json`, `evidence-index.json`;
- `logs/core-cpu.log`, `logs/structure.log`, `logs/layout-final.log`;
- `logs/lazy-cpu-owners.log`, `logs/lazy-cpu-owners-helper.log`,
  `logs/lazy-core-cpu.log`,
  `logs/import-layering-fixed.log`, `logs/lazy-structure-import.log`,
  `logs/lazy-structure-import-child.log`,
  `logs/two-rank-lazy-final-launch.log`,
  `logs/two-rank-lazy-final/rank{0,1}.log`, `lazy-final-source.json`,
  `lazy-evidence-index.json`, `accepted-source.json`,
  `logs/layout-lazy-final.log`.

## Main-branch integration verification

The integrated tree uses remote `d62e0deba6b6dc74e4e4d9aaa12bb6baefec2d95`,
including the CUDA RNG and tensor factory fixes. All 1113 native runtime files
and seven TP source/test files match the independently validated
`642c60b1bc2007315b4c717722c962223d0dafb2` tree byte-for-byte. After separate
serial cache warmups, both actual CUDA/NCCL ranks again passed all eight cases
with zero skips. Each rank's four training rows retained 28/52 owned parameter
elements, zero fallback, zero legacy reductions/broadcasts and exact continuation.
Root verification rehashed both rank logs and the source snapshots.

On the integrated main tree, CPU core passed 322 tests with 42 skips and one
existing xfail; TP/checkpoint/ZeRO CPU owners passed 54 tests with no skips.
The complete Torch-mode structure suite passed 1461 tests with six skips after
documentation indices and the generated manifest were updated. Layout and diff
checks passed. No earlier structure failure is counted as a pass.

The unversioned CPU integration logs are `logs/main-core.log`,
`logs/main-cpu-owners.log` and `logs/main-structure.log` under the original state
root. New-baseline CUDA evidence is under
`$JITTOR_LAB_ROOT/_state/native-tensor-parallel-bf16/`: `frozen-fp32-source.json`,
`frozen-fp32-acceptance.json`, `main-source-audit.json` and
`logs/frozen-fp32-two-rank/rank{0,1}.log`. Despite that state directory's name,
this acceptance is FP32 only; BF16/AdamW remain separate work.

## Limits

This completes a real native **column/row linear TP subset**, not R17's whole
FSDP/TP/ZeRO/multi-node target. It verifies WORLD-only equal feature partitions
on CPU singleton and single-host two-rank CUDA, FP32, replicated inputs/loss,
and fixed-world checkpoint continuation with momentum SGD. General TP
operators, attention/vocabulary/sequence parallelism, uneven or empty
feature partitions, mixed precision, larger worlds, hybrid DP/TP/FSDP/ZeRO,
checkpoint resharding and multi-host rendezvous/training remain outside this
acceptance. Adam-family math remains in its existing owner, without a new
TP-specific numerical acceptance claim. No throughput or allocator peak
improvement is claimed. R17 multi-host acceptance still requires distinct
actual hosts and cannot be satisfied by these two local ranks.
