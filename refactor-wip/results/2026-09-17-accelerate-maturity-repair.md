# Accelerate training capability repair

- Status: verified within the stated CPU/CUDA and local two-rank scopes;
  complete structure gate retains four existing worktree/baseline failures
- Date: 2026-09-17
- Baseline commit: `7c8b309625f5b711181cf174a7e259446cec9f7a`
- Repair commit: `da820bf343d7adebe3ffce42bc817e4ac808147d`
- Owner: Jittor core maintainers
- Review when: AMP policy, scalar promotion, optimizer state loading, RNG state,
  distributed synchronization, checkpoint formats, or Accelerate wrapping changes

This followup addresses Accelerate's own training contracts after the
[wrapper performance repair](2026-09-16-accelerate-wrapper-performance-repair.md).
It does not use failures in other optional ecosystem libraries as evidence of
an Accelerate defect. Both frontends run Accelerate 1.10.1. The independent
reference is PyTorch 2.6; the shim distribution and declared Torch API version
are 2.11, while its native Jittor version remains 1.3.11. These are separate
version fields, not matching Torch versions or a claim of complete Torch 2.11
conformance. Test artifacts record frontend identity, distribution and API
versions, actual device and backend fallback count.

## AMP, BF16 and scaling

The minimal CPU baseline confirms that BF16 autocast selected FP16, losing
BF16's range; the scaler also lacked the complete per-optimizer protocol and
five-field state. Native AMP now has an explicit BF16 preference bit, preserves
the legacy preference bits and FP64 priority, and stores all seven AMP bits in
the node flags used by backward. The convolution entry casts temporary floating
computation operands to BF16 without converting model parameters. Compatibility
backward publishes gradients in the leaf's dtype. Linear, convolution and
matmul regressions require finite BF16 computation and FP32 master parameters,
gradients and optimizer moments.

The scaler uses a true zero-dimensional FP32 scale at the reference tensor's
full placement, with controlled host scale metadata and invalidated placement
caches after update/load. It distinguishes each optimizer's ready, unscaled and
stepped stages; rejects repeated or invalid calls; checks scaled gradients for
nonfinite values before unscale; and saves/restores scale, growth factor,
backoff factor, growth interval and growth tracker. Regressions cover multiple
optimizers with independent overflow, clipping, disabled mode, positional
signatures, growth/backoff below one, finite growth overflow, manual scale
updates and backend-specific FP64 detection. FP64 unscale values and storage
remain FP64: the independent CUDA foreach AMP check uses an FP32 detection
domain and skips a finite `1e100` gradient, whereas CPU updates it. Both
update a finite `1e20` gradient. Routine scale reads do not add a scale
tensor `.item()` synchronization to every optimizer's unscale.

Python weak scalars and typed zero-dimensional tensors follow separate
promotion paths. In particular, half values multiplied by Python `65536.0`
remain finite through operators, methods, `torch.mul` and `torch.multiply`.
Half multiplied by a typed FP32 zero-dimensional scale matches the independent
CPU/CUDA opmath behavior, including genuine CUDA overflow. The functional
bindings retain the original canonical native math call and do not dispatch
through application subclasses' Python arithmetic overrides. Public FP32
sqrt preserves FP32 under compatibility autocast; raw native AMP is unchanged.
Half `torch.add(..., alpha=65536)` also differs by backend: CPU raises a
representability error, while CUDA computes finite half output in wider math.

The final same-source 24-case compatibility suite passes 22 on actual CPU with
two CUDA-only skips, and 24/24 on actual CUDA (19.03 s). The independent PyTorch
CUDA reference passes 24/24 (2.32 s). Four targeted shared regressions pass on
CUDA, covering functional native/subclass protocols, reflected arithmetic and
integer true division. All compatibility selections report zero fallback.
Native CUDA AMP and node-layout regressions pass 10/10, including cuBLAS,
cuDNN and BF16 backward; the task-process peak is 886 MiB. Native CPU AMP and
flag-layout regressions previously pass 20 cases against the same production
core. Baseline counterexamples and intermediate runs remain in the lab rather
than being combined into a final acceptance count.

## Accumulation and training control

The historical accumulation/scheduler concern does not reproduce in the
minimal baseline. No accumulation algorithm change is needed. Exact maintained
regressions replace a weak learning-rate-decreased assertion: two microbatches
with accumulation two produce one update and StepLR `0.1 -> 0.05`; a third
tail microbatch produces a second update and `0.025`. Tests assert synchronization
flags, update count, exact parameter trajectory and scheduler epoch. They also
cover two prepared optimizers/schedulers with combined clipping and FP16
overflow skipping the optimizer and scheduler before a finite accumulated
update. These are specific tested combinations, not every multi-optimizer AMP
ordering permitted by upstream Accelerate.

## Checkpoint Placement and RNG

Two real fresh-process resume failures identify a canonical optimizer loader
bug: empty state rebuilt moments from old buffers on the old device, and CPU
serialized moments were assigned directly to CUDA parameters. The loader now
validates the complete load plan before applying it, initializes missing state
at the live parameter placement, and migrates loaded moments to that placement
while preserving their source dtype. Step metadata is handled separately.
Named FULL checkpoint loading delegates this policy to the canonical owner.

CPU RNG captures the native engine's complete state and full uint64 seed;
restore continues the stream instead of restarting from a seed. Torch seed
operations preserve Python and NumPy RNG streams. The legacy native int seed
ABI is retained, but legacy `get_seed()` callers explicitly reject wide seeds
they cannot represent.

**Complete CUDA RNG continuation remains unsupported (C5).** The current Host
cuRAND ABI cannot snapshot all interleaved subsequence progress after arbitrary
normal and float64 draws. The safe `JITTOR_CURAND_XORWOW_U32_V1` format permits
capture only after FP32 uniform generation since the last seed or safe restore.
Normal initialization, Gaussian training noise or float64 sampling therefore
can make ordinary `Accelerator.save_state()` fail explicitly, even when model
and optimizer serialization works. Sampling itself still computes normally,
and an earlier safe snapshot remains restorable. Uniform-only continuation
evidence must not be presented as general CUDA resume support. Exit requires
a runtime-owned complete state representation or a reviewed canonical RNG
transition, with mixed, odd, large and fresh-process continuation plus
throughput regressions. Raw large-counterexample and strict-subset experiments
remain in `${JITTOR_LAB_ROOT}/Accelerate/maturity_repair/`.

The final RNG/sampler selection passes 14 cases on actual CPU with 11 CUDA-only
skips, and 25/25 on two actual CUDA devices (30.17 s), with zero fallback.
Native CPU RNG tests pass 3/3. The CUDA selection covers device isolation,
large draws, unsafe-history rejection, restoration of an earlier safe snapshot,
mixed future draws and actual CPU/CUDA fresh-process Accelerate continuation.
Its two task-process peaks are 874/872 MiB. Mixed future draws after a restored
safe snapshot do not imply that a new snapshot is supported after those draws.

The named checkpoint/unsupported-boundary CPU suite passes nine cases on actual
CPU (9.85 s), with fallback count unchanged at zero. It covers named optimizer
groups, missing state, malformed-input atomicity, actual non-strict model keys,
Join entry refusal and SHARDED storage refusal. This CPU result predates the
last import-compatible default-planner additions and FULL lazy-collective fix;
the final required CPU integration now passes 53 cases with 17 CUDA-only skips
against the final source, as recorded under Continuous Gates and Performance.

## Samplers and Tail Metrics

The old sampler CPU baseline fails all nine seed/default-state/explicit-state
continuation checks across RandomSampler, SubsetRandomSampler and DataLoader.
Nonreplacement sampling delegates to canonical generator-aware CPU randperm;
default replacement uses CPU randint. The final sampler cases pass within the
RNG selection above. With a CUDA default backend, spies observe 33 randperm
and four randint calls executing on native CPU, preserving the sampler RNG
owner and continuation semantics. Explicit-generator replacement sampling is a
separate native integer sampling owner gap: it rejects without advancing the
generator. Exit requires canonical generator-aware integer sampling and
replacement continuation/isolation tests.

The actual two-rank prepared-loader case consumes 17 IDs with batch size four
and accumulation two. Gathered metrics contain each ID exactly once; flags are
`False, True, True`, including the tail update. Every persisted loss, gradient,
parameter, optimizer state and metric/flag record matches the independent
reference on both ranks. This result is distinct from sampler-generator support.

## DDP and No Sync

Actual CUDA NCCL DDP validates common initialization and closing-backward
gradient synchronization. The first `no_sync` backward keeps rank-dependent
gradients different; the second closing backward makes them equal before an
optimizer reads them. Both ranks' complete recorded gradients, parameters and
optimizer state match the independent workflow. Enabled uneven-input Join with
multiple ranks remains explicitly unsupported (C5): the reducer lacks notify
and shadow-collective hooks. Balanced update counts or `even_batches=True`
are the supported alternative.

## FSDP2 FULL Checkpoints

Named model/optimizer transforms now reuse the canonical loader and registered
FSDP gather/shard owner. The real two-rank, FP32, no-wrap FSDP2 FULL case runs
`Accelerator.save_state/load_state`, restores every model/moment/step value,
and matches the next loss and update. Optimizer step metadata is exported as
a public Tensor. Accelerate imports DefaultSavePlanner/DefaultLoadPlanner even
on the FULL path, so their names are import-compatible while actual planner
construction precisely rejects the unsupported SHARDED protocol. Before FULL
CPU-offload export drops a nonzero rank's result, it now consumes pending lazy
gather collectives; otherwise that rank could start the next collective out of
sequence.

The six persisted rank reports from the three scenarios compare successfully
against independent Torch numerical and behavioral records. Version, pointer
and device metadata are recorded separately, not compared as equal:

| Actual CUDA Workflow | Rank Reports | Maximum Absolute Numeric Difference |
| --- | --- | --- |
| DDP tail/metrics/accumulation | 2/2 pass | `3.73e-9` |
| DDP no_sync/closing backward | 2/2 pass | `9.31e-10` |
| FSDP2 FULL save/load/next update | 2/2 pass | `1.49e-8` |

Evidence is `_state/Accelerate/maturity_repair/distributed_oracle_comparison.json`
under `${JITTOR_LAB_ROOT}`. The inspected Accelerate FSDP2/version guards select
the same tested FP32/no-wrap/no-RAM-efficient workflow for metadata 2.6 and
2.11. The reference FULL loader requires its public dual CPU:Gloo/CUDA:NCCL
backend for CPU scalar broadcasts; the Jittor owner places transport values at
the target CUDA device. A separate rank-zero broadcast auxiliary combination
timed out in the independent reference and is not included in acceptance.
Neither this result nor successful FULL tensor transforms accepts SHARDED DCP
planner/chunk/storage, FSDP1, CPU-RAM-efficient loading or multinode resharding.
Those C5 limits remain in the active ledger and existing refactor board.

**Resource-budget incident:** the earlier tiny FULL v2 attempt issued
collectives out of sequence and one rank reached 12,182 MiB, exceeding the
user's per-device target. The 8 GiB polling guard detected it and terminated
only this task's process group; it did not prevent the allocation spike. This
was a real budget violation, so not every experiment stayed below 10 GiB.
The failed guard artifact is preserved as
`_state/Accelerate/maturity_repair/full_checkpoint_budget_v2_guard_failure.json`.
After correcting lazy-collective ordering, the v4 successful FULL run peaks at
1,076/640 MiB. Its lab-only NumPy factory checks reject allocation requests
above 64 MiB before allocation; neither that check nor `device_mem_limit`
constitutes a general hard CUDA allocator cap. The final guard did not trigger.

## Meta and Offload

Six actual CUDA functional cases pass for empty-weight construction, CPU/disk
offload, mixed dispatch, tied parameters, plain-bin checkpoint loading and hook
removal. Outputs match the NumPy reference, two forwards succeed and a third
forward after hook removal succeeds; the independent Torch workflow also passes
these six cases. The task-process peak is 436 MiB.

**Storage-free meta and capacity-releasing offload remain unsupported (C5).**
The probes observe `is_meta=True` together with a nonzero native pointer and
actual CUDA storage, both after empty-weight construction and after offloaded
forwards. Mixed dispatch and checkpoint loading retain the same storage on the
offloaded weight. The six functional passes therefore do not establish that a
model larger than available VRAM can be loaded or that post-forward weights
are released. The active ledger records the native meta/no-storage and storage
lifecycle owner. The temporary workaround is a resident model that fits VRAM;
exit requires genuinely allocation-free meta and post-forward weight release
while preserving numeric, tied-weight, checkpoint and hook-removal contracts.
The metadata 2.11 versus reference 2.6
guards also differ for a physical-device-plus-meta model loaded without an
explicit device map; matching every offload branch is not inferred from the
ordinary FULL case.

## Continuous Gates and Performance

The required correctness tool executes the seven maintained Accelerate/RNG
test modules with the optional library required: final CPU acceptance is
53 passed and 17 explicitly CUDA-only skipped (10.32 s). The dedicated
`nox` entry is connected to the CPU nightly workflow. This is locally executed
evidence and configured continuous coverage, not a claim that remote CI ran.

The maintained `benchmarks/accelerate_training.py` class runs through actual
ASV against the working tree in the dedicated `asv-accelerate` session. Other
sessions skip it before importing Torch/Jittor, so the generic benchmark gate
does not gain a mandatory Accelerate dependency. Final CUDA sanity passes all
four direct/full and SGD/AdamW combinations, with zero fallback, matched input
and initial-state audits, four warmup steps, eight timed steps and three repeats:

| Workflow | SGD, Eight Steps | AdamW, Eight Steps |
| --- | --- | --- |
| Direct | 60.8 +/- 1 ms | 63.4 +/- 0.02 ms |
| Accelerate full | 67.9 +/- 1 ms | 69.4 +/- 1 ms |

The uncertainty notation above is ASV's displayed estimate, not a separately
computed standard deviation. The measured task-process peak is 590 MiB. This
shared-device short sanity does not show the earlier several-fold wrapping
slowdown. It does not reaccept the historical ResNet epoch result and is not
a new full epoch or an uncontended wrapper-overhead estimate. Final
benchmark source SHA-256 is
`5021127762b80502e562aa3518d3d016121119ea0c05901fd1f8b5436bb01467`;
other source hashes and four actual-device audit records
are in `${JITTOR_LAB_ROOT}/_state/Accelerate/maturity_repair/asv-sanity/`.

## Reproduction and Final Gates

Maintained single-process cases are in
`compat/tests/torch/test_accelerate_scaler_protocol.py` and
`compat/tests/torch/test_accelerate_training_control.py`. They skip an absent
optional Accelerate installation in generic test collection; the dedicated
gate requires it and fails on absence. Raw baseline, oracle, actual-device,
budget and JUnit artifacts stay in
`${JITTOR_LAB_ROOT}/Accelerate/maturity_repair/` and the corresponding `_state`
tree. First JIT is serialized, unit and benchmark caches are separate, and
memory monitoring measures only this task's processes.

The complete structure selection executes once: **1,325 passed, two skipped,
nine failed in 257.19 s**. This is not a passing full structure gate. The five
additional failures have the following resolution; only affected cases are
rerun rather than repeating the complete selection:

| Failed Contract | Attribution and Final Evidence |
| --- | --- |
| Interpreter naming in tests | New RNG and generic ASV probes now use the existing child-process helper; affected subset passes |
| Child launch pins source tree | Generic ASV probe now uses the helper; independent Torch resume keeps its interpreter and avoids Jittor source pinning |
| Collection-time backend effects | Optional Accelerate absence now uses a skip marker; required mode still raises instead of skipping |
| Documentation index size | The added context paragraph exceeded 180 lines; it is shortened and final layout/related contract pass |
| Default build configuration | The CPU test invocation explicitly sets empty NVCC paths; the pure configuration case passes after removing those invocation overrides |

The repaired related selection passes **56/56** (25.66 s), including actual CPU
fresh-process resume. Actual CUDA resume passes separately, with a normal guard
exit and peak 874 MiB; the independent PyTorch CPU resume passes (6.60 s).
Production source is unchanged by these test-infrastructure repairs.

Four existing failures remain and are reproduced in a final four-case selection
(4.13 s): documentation reachability has three already tracked unlinked result
pages plus an existing user onboarding page; manifest comparison sees only that
untracked onboarding addition; the miscellaneous public-surface assertion lacks
the existing `sinc`; and the existing conv1d test assigns `use_cuda` without
restoration. Frozen `4d424432` controls in
`${JITTOR_LAB_ROOT}/Accelerate/perf_repair/structure_head_control.json` prove the
tracked baseline issues; the onboarding file is explicitly absent from that
snapshot. No user file or unrelated API is changed to clear these failures.

Layout passes after keeping the project context within its index-size contract.
The root and compat
manifests match canonical generation over tracked sources plus this task's
new files: root adds the required gate tool and five previously tracked docs;
compat adds one previously tracked runtime resource. Existing untracked user
onboarding remains excluded from the release manifest. Controlled canonical
generation and per-project current/HEAD/user-input diffs are recorded in
`${JITTOR_LAB_ROOT}/Accelerate/maturity_repair/manifest_audit/`. Complete and
targeted JUnit results are in the corresponding `_state/.../structure_final/`
directory. The four remaining failures must not be conflated with a passing
complete gate.
