# Native ZeRO1 training and checkpoint verification

Status: native optimizer-state sharding implemented and verified on CPU and
single-host two-rank CUDA/NCCL. Maintainer: Codex native distributed owner.
Recheck after optimizer arithmetic, collective scheduling, Store, checkpoint
or trainability semantics change. Native FSDP/TP/ZeRO2/3 and multi-host
acceptance remain outside this result.

## Source and missing capability

The initial synchronized remote was `b930461562e73c67332581fe2384eaf8cfd7d838`;
the existing native checkpoint implementation was preserved as
`82fb93150fc51e0608db7f95e08f8d48bb7f78ec` and reapplied as
`fefcee969b96e61f1d592eee0f74c11ed4f95bc4`. After stopping the CPU reproduction,
the remote checkpoint integration `82dda1ff5` was merged, retaining its report
update. The final tested source baseline is
`b8c150f07affd8b9651fba772f98816e61174be3` plus this change. Original branch
references and pre-synchronization evidence were retained.

Three CPU native Adam steps matched an independent NumPy trajectory, followed
by an actual `ImportError` for `jittor.distributed.ZeroRedundancyOptimizer`.
The new facade uses the existing full base Optimizer for backward/DDP and
instantiates the existing native algorithm with only owned parameters. It
imports no compatible Torch implementation and creates no duplicate optimizer
math. Native checkpoint preflight delegates rank-owned state validation to the
optimizer before model/RNG mutation.

## Results

The private Python 3.11.16 environment used NumPy 1.26.4 and pytest 7.4.4;
CUDA used nvcc 12.2.140, strict kernel math and the isolated NCCL 2.18.3
runtime. Source imports and every rank's cache were pinned to this worktree.
Each rank's initial compilation, including BF16 warmup, ran serially.

| Check | Result | Evidence under `$JITTOR_LAB_ROOT/_state/native-zero-training/` |
| --- | --- | --- |
| Missing public entry, with native Adam/NumPy baseline | Expected import failure | `logs/cpu-missing.log` |
| Native ZeRO CPU plus native checkpoint regression | 45 passed, zero skipped | `logs/cpu-zero-final.log` |
| Real CUDA/NCCL workers | 24 passed per rank, zero skipped, launcher zero | `logs/two-rank-validated-launch.log`, `logs/two-rank-validated/rank{0,1}.log` |
| CPU core, both process modes | 322 passed, 42 skipped, one known xfail; both zero exit | `logs/core-cpu.log` |
| Full structure | 1451 passed, six skipped, three failures; test basename fixed and its two checks passed, leaving two central documentation-registration failures | `logs/structure.log`, `logs/structure-registration-recheck.log` |
| Repository layout and diff whitespace | Passed | `logs/layout.log` |

The CUDA test was renamed to `test_native_zero_nccl.py` after the full structure
gate identified a collision with the CPU test basename. Its content hash is
unchanged from the actual two-rank run. The remaining structure failures require
the central documentation toctree and generated MANIFEST registration during
integration; they are not counted as passing here. `final-source.json` records
the tested code hashes, including this path-only rename.

CPU covers three algorithms, multiple groups, two-microbatch accumulation,
frozen parameters, late unfreezing and tied parameters. Ordinary trajectories
are elementwise identical to unsharded native owners and agree with an
independent FP64 NumPy oracle. NumPy implements native Adam's historical
epsilon-before-bias-scaling convention explicitly; it is not presented as
PyTorch Adam bitwise parity.

CUDA covers 15 algorithm/configuration rows, three late-unfreezing rows,
three deleted-group-override rows, BF16 SGD, tied/no-sync and rank mismatch.
Each checkpoint row trains three steps, saves, executes two further steps,
then restores into both existing and fresh owners and exactly matches the
full continuation, including owned moments, gradients, model values, steps
and loss. Deleted group LR overrides resume through the saved current default.
All device work runs under fallback-error policy with count zero and without
importing Torch. Active outputs, gradients, updated parameters and moments
are checked on device before readback, including CUDA driver pointer ordinal.
Untouched frozen zero tensors are separately materialized and checked as
storage; host snapshots are not described as GPU calculations.

Across the 15 primary CUDA rows, maximum absolute independent-reference error
was below `1e-7`; checks retain `atol=2e-6`, `rtol=2e-5` and loss absolute
error below `2e-6`. Exact continuation uses array equality, not those tolerances.
BF16 SGD uses exactly representable dyadic inputs and updates, checks raw
parameter bits, retains BF16 momentum, and matches independent NumPy SGD.

For an eight-element two-parameter model, Adam/AdamW construct 12 state elements
on rank zero and four on rank one, rather than 16 on each rank. SGD constructs
six and two rather than eight on each. Constructor instrumentation proves
only owned parameter identities reach state-buffer creation; driver checks
prove the active owned buffers are actual CUDA allocations. Rank-local
checkpoint payload counts agree. With one parameter, the non-owner constructs
zero moments and still trains/restores correctly. Combined optimizer state
elements are half the replicated two-rank total; this is not a whole-process
memory or performance claim.

NCCL's actual TRACE records contain 463 AllReduce calls per rank: 68 metadata
reductions and exactly 395 gradient reductions required by the 24 workloads.
For the three-step two-parameter DDP accumulation window, Python owner spies
also report six gradient reductions, versus 12 without DDP no-sync accumulation.
The physical calls exclude a hidden second reduction during owned updates.

## Reproductions retained and fixed

The first CPU matrix found that restoring trainable state into an initially
frozen target left it frozen. The native checkpoint helper now reapplies saved
flags on existing target objects after loading, and verifies both directions.
Bad partition/algorithm/moment/gradient metadata and unexpected replicated
moments fail before model mutation. Editing exported ownership metadata no
longer changes active ownership. Group override deletion is forwarded and
restored without retaining a fresh object's stale override.

An unobserved tied constant-gradient update initially hung after no-sync
rejection. Independent probes preserved the raw process stacks and located
the boundary between collective gradients and asymmetric owned updates.
Explicit exact-root gradient materialization on all ranks plus readiness
resolved it without NumPy readback or math changes. An additional run exposed
a cleanup RPC/collective GIL race; using the existing native Store `arrive`
contract for peers and root-owned cleanup resolved it. Eager CUDA module
loading and redundant leaf restarting did not resolve the original hang and
are not used as fixes. Raw failed runs remain in `logs/two-rank-first/`,
`logs/two-rank-computed-storage/`, `logs/two-rank-gradient-boundary/` and
the `logs/tied-*` probes; final complete evidence is `two-rank-validated`.

## Reproduction and limits

Run CPU native checks with `JITTOR_TORCH_SHIM=0`, `JT_BACKEND=cpu`, both
`JT_USE_CUDA=0`/`use_cuda=0`, both NVCC-path variables empty, own `PYTHONPATH`
and private `JITTOR_HOME`. Run pytest on `tests/optim/test_native_zero.py`
and `tests/serialization/test_sharded_checkpoint.py`. For the two-rank suite,
use `python/jittor/distributed/launch.py --backend nccl -n 2 --device-ids=<two>`
and pytest on `tests/backends/comm/nccl/test_native_zero_nccl.py`, with separate
rank homes/caches, `JITTOR_ZERO_TEST_DIR` pointing at shared checkpoint storage,
`JT_CUDA_KERNEL_MATH=strict`, fallback-error and no shim. The experiment's
`cuda-env.sh` and `rank_entry.py` preserve the exact commands and environment.
When pytest plugin autoload is disabled, load `pytest_timeout,xdist.plugin`
explicitly. Use repository-pinned pytest rather than an unverified newer major.

This is native optimizer-state training sharding, not merely file partitioning.
Parameters/gradients remain replicated and partitioning uses whole parameters.
Groups and parameter order stay fixed; checkpoint world/ownership must match.
Native DDP's trainability remains fixed after construction. Large model peak
memory, throughput, other algorithms/dtypes/backends, TP, parameter sharding,
ZeRO stages 2/3 and two real hosts were not accepted by this run. Two local
workers are not multi-host evidence. No unsupported capability is registered
as complete by this report.
