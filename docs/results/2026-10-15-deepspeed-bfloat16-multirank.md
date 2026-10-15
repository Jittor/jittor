# DeepSpeed two-rank BF16 Engine verification

- Status: bounded CUDA correctness verification; no performance claim.
- Reviewed: 2026-10-15; owner: Jittor core maintainers.
- Synchronized numerical baseline: `5ddb1f1399dfb9976ba73500e2764e4c27638049`.
- Storage/scalar implementation: `15ec2a32e0bae713cb79723df9b865583e71a5cf`.
- Subsequent synchronization: origin `344dba6cd259259050c64b41541fdff281788b22`,
  merged as `d4a4f1100a36d1e6b34bfda49db1e37c85f5242e`.
- Recheck when: native view/alias storage, scalar division, autograd publication,
  collective output copies, optimizer state, or the pinned DeepSpeed source changes.

## Scope and independent reference

The later [maintained gate acceptance](2026-10-15-deepspeed-maintained-gate.md)
adds the two-input-perturbation matrix, exclusive FileStore bootstrap and CPU
Adam strict-resume correction. This earlier report keeps its measured source pin.

The real upstream `DeepSpeedEngine` executed backward, optimizer steps, save,
fresh-Engine load, and continued training on two CUDA devices. Both ranks used
BF16 model/input tensors, FP32 communication and optimizer master partitions,
GAS 2, six microsteps and three updates. Checkpointing followed microstep four;
the fresh Engine repeated the final two microsteps. The deterministic residual
Linear/GELU/Linear/LayerNorm/Linear model has eight parameters with 332 elements.
Each optimizer rank actually owned 166 FP32 master elements. Rank inputs differ
by a fixed offset. This evidence covers AdamW, input perturbation zero, and
ZeRO stages 1 and 2; the input offset still differs between ranks.

Both runtimes used official DeepSpeed source
`f5af15c2b4da10411e340850c00ba0cb5b14fd87` (`0.19.7+f5af15c`). The built package
SHA256 was `467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`.
The previously admitted source-pinned adapter remained in use; this batch adds
no adapter or alternate training algorithm.

The target was Linux x86-64, RTX 4090/SM89, driver 580.178.04, nvcc 12.2.140,
g++ 11.5 and `pipcu122_a89fa4fed08c`. The independent reference loaded binary
PyTorch `2.11.0+cu128`, its real `_C` extension, and no Jittor install-context
marker. These are different CUDA library stacks. The two runtimes recorded
matching thread/precision settings. BF16 operands, parameters, outputs and
requested gradients were materialized and checked on CUDA. Both ranks ran
with backend fallback forbidden and observed a zero counter delta.

## Shared defects and correction

Rebinding a dense slice during `copy_`, `div_`, `mul_` or `zero_` could leave
detached aliases reading the previous allocation. The existing native view
record now distinguishes a physical storage view from a materialized logical
transpose, including after the weak base holder expires. The proof uses an
actual pending share link or live allocation identity, never equality of two
unallocated pointers. Detached/no-grad dense writes reuse native `_copy_into`
after resolving pending readers. Materialized transpose writeback retains its
existing owner. Public copy broadcasting is validated, and partial physical
overlap is rejected before writing. Contiguous physical views also work in
the existing bounded `_foreach_zero_` implementation.

Separately, a frozen tensor divided by a Python float incorrectly acquired
`requires_grad=True`. The Python precision-promotion path created a trainable
internal array constant. The C++ fast path also constructed its internal cast
outside the existing Torch frontend scope. Both paths now preserve the public
gradient domain: only internal scalar constants stop gradients, while genuine
tensor input gradients remain connected. Dtype/opmath rules are retained.
The correction covers forward/reflected division, integer/Python/NumPy scalar
divisors, FP32/FP64/FP16/BF16, and NaN/Inf behavior. Non-leaf `zero_` uses native
selection to produce exact zero values and a zero derivative, including for
non-finite inputs.

The original unobserved ZeRO2 run reproduced a full-gradient error of about
31.6%. Merely materializing diagnostic observations could hide it. Acceptance
therefore uses the ordinary repository runner without those diagnostic wrappers.

## Results

The comparison reuses `_comparison_floor` and `_divergence` from the common
ecosystem harness: category floor `1e-3` times the largest reference magnitude,
forward/state tolerance `2e-3`, gradient tolerance `1e-2`. Metadata, shapes,
finite values and the actual optimizer class are checked independently.
No threshold was relaxed.

| Case, per rank | Numerical fields | Gradient fields | Forward/state result | Strict fresh resume |
| --- | --- | --- | --- | --- |
| ZeRO1, real `BF16_Optimizer` | 240 | All 144 bitwise equal to Torch | 84/96 fields bitwise equal; maximum normalized error `1.56e-7` | 64 fields bitwise equal |
| ZeRO2, real `DeepSpeedZeroOptimizer` | 208 | All 112 bitwise equal to Torch | 84/96 fields bitwise equal; maximum normalized error `1.56e-7` | 56 fields bitwise equal |

The field counts include resumed microsteps. Every incoming parameter gradient
is observed through a real Tensor hook. ZeRO2's public full-gradient getter is
called at GAS boundaries, where upstream actually exposes it; omitted
non-boundary full gradients are not counted as successful observations.

Strict resume compares saved/restored and final/resumed optimizer moments and
FP32 masters, plus both resumed microsteps' outputs, loss, model parameters,
input gradients, incoming parameter gradients and available full gradients.
Both ranks updated all 166 elements of their FP32 partition during the third
step. For ZeRO2 the update L2 norms were `0.00906536` and `0.02911891`.

| Shared regression on the numerical source | Result |
| --- | --- |
| CPU core, native and Torch modes | 322 passed, 42 skipped, one existing xfail |
| CPU inplace/scalar/dtype/division | 60 passed; nine CUDA-only skips |
| CUDA inplace/scalar/dtype/division | 69 passed, zero skips, fallback delta zero |
| Autograd/optimizer/storage/detach/load/to owners, CPU and CUDA | Each 108 passed, one ACL-only skip; CUDA fallback delta zero |
| Independent binary Torch public contracts, CPU and CUDA | Each 12 passed |
| Repository layout | Passed |
| Full structure before synchronization | 1459 passed, six skipped; two central registration failures for the prior Linear report |

The structure failures concern the prior report's results-index and generated
MANIFEST registration. Upstream `344dba6cd` contains those integration updates;
the table retains its measured original source label.

## Main-tree integration verification

The main checkout integrated the eight storage/scalar files byte for byte
onto synchronized `344dba6cd259259050c64b41541fdff281788b22`. CPU core passed
322 tests with 42 declared skips and one existing xfail. CPU integration owners
passed 49 tests with one CUDA-only BF16 class skipped. Actual CUDA inplace,
foreach-zero and autograd owners passed 51 tests without skips. The integration
audit rehashed all 31 files in the two-rank v2 evidence manifest and directly
checked all four rank archives for finite values and bitwise-equal gradients.
These owner checks do not relabel the historical two-rank numerical baseline.

Integration logs and the hash audit are unversioned under
`$JITTOR_LAB_ROOT/_state/inplace-scalar-integration/`.

## Reproduction and artifacts

Use the dedicated runner `compat/tests/torch/_ecosystem_deepspeed_runner.py`.
First warm each rank's distinct cache serially, including real NCCL setup.
Launch both ranks concurrently with separate device visibility, HOME,
JITTOR_HOME, TMPDIR and cache names, a common rendezvous, and a shared checkpoint
directory. The actual unobserved runner arguments were:

```bash
<jittor-python> compat/tests/torch/_ecosystem_deepspeed_runner.py \
  --runtime jittor --device cuda --dtype bfloat16 --world-size 2 \
  --rank <rank> --rendezvous tcp://<address>:<port> \
  --checkpoint <shared-checkpoint> --optimizer AdamW --zero-stage 2 \
  --output <fresh-case-directory>/rank-<rank>.npz
```

Set the matching standard RANK/WORLD_SIZE/LOCAL_RANK and MASTER_ADDR/MASTER_PORT
variables, `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1`, `JT_BUILD_USE_MPI=0`,
`JT_BUILD_USE_NCCL=1`, and a common `JITTOR_DIST_RENDEZVOUS_DIR`. The reference
uses the same arguments with `--runtime torch` in its independent environment.
Stage 1 changes only the stage-specific case/checkpoint/output locations and
`--zero-stage 1`. Every case must have a fresh output directory; stage,
optimizer and precision belong in its path. A manual output tag alone is not
an adequate collision guard.

Raw artifacts are unversioned under the evidence lab's
`_state/refactor-goal/deepspeed/`. The complete source/artifact manifest is
`multirank/deepspeed-bf16-stage1-stage2-v2-evidence.json`: 1208 source hashes and
31 original/result/comparison log hashes. The manifest records a historical
Stage2 output-name collision explicitly; the fresh stage-specific Stage2-v2
artifacts supersede the overwritten originals. Its current native artifacts are
`native/rank*/formal-stage1-bf16-scalar-scope-v1.npz` and
`native/rank*/formal-stage2-bf16-scalar-scope-v2.npz`. Comparison logs are
`formal-stage{1,2}-scalar-scope-comparison-v2.log`; strict-resume logs use
`formal-stage1-scalar-scope-strict-resume-v1.log` and
`formal-stage2-scalar-scope-strict-resume-v2.log` in the same directory.

## Leaf/view follow-up

The leaf-view gap observed during this work is corrected separately at
`489417b368640e724372d9b22b509d2ef42b00f3`, synchronized with origin
`3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6`. This does not relabel the earlier
matrix's source. Illegal grad-enabled writes through live or orphan views of
trainable leaves fail before mutation. Legal non-leaf writes retain the real
input and RHS gradients, including the zero input edge of a slice copy.
Frozen views can acquire a trainable RHS edge, both for initially frozen
leaves and reversible `requires_grad_(False)`. No-grad leaf-view writes and
frozen physical aliases retain their supported storage behavior.

The correction uses the existing native view provenance, writeback and graph
refresh owners. Native assignment keeps its default policy; only the existing
Torch `_update` path acquiring an RHS edge bypasses inheritance of the frozen
destination's old gradient flags. There is no replacement derivative algorithm
or blanket rejection of differentiable views.

| Check | Result |
| --- | --- |
| New view contracts plus inplace/foreach neighbors, CPU / CUDA-selected run | 26 / 26 passed |
| Same six public contracts, independent binary Torch CPU / CUDA | 6 / 6 passed |
| Native view/storage/setitem/step capture, CPU | 31 passed, 27 CUDA-only skipped |
| Same native files, CUDA-enabled run with explicit per-test device scopes | 58 passed, no skip, fallback delta 0; unscoped nodes still use CPU |
| Autograd/optimizer/storage/dtype owner neighbors, CPU / CUDA-selected run | 108 / 108 passed, one ACL-only skip each |
| CPU core | 322 passed, 42 skipped, one existing xfail |
| Structure and layout | 1478 passed, six skipped; layout passed |

The CUDA-selected 26 includes one private host provenance check; the generated
device cases execute on CUDA. CUDA owner checks also require fallback delta 0.
The historical native 58-node run did not globally enable CUDA: nodes with
explicit CUDA scopes ran on CUDA, while unscoped nodes, including the consumed
expand allocation check, ran on CPU. It does not establish that allocation
check on CUDA. During integration onto `4a4a480e6`, globally enabling CUDA gave
57 passes and one failure in that existing allocation check: the numerical
sum was correct, but allocated bytes grew by 68,157,440, exceeding 16,777,216.
The same node failed in isolation and with strict kernel math. These failures
are preserved under `_state/leaf-view-integration/`; no allocation threshold
was relaxed. The integrated public view/inplace/foreach contracts passed all
26 nodes on CPU and all 26 in the CUDA-selected run; native CPU gave 31 passes
and 27 CUDA-only skips, and CPU core gave 322 passes, 42 skips and one existing
xfail. An independent clean `4a4a480e628dae634a9d660f320c3aa957341e99`
worktree with a separate CUDA compilation cache reproduced the same numerical
result and 68,157,440-byte allocation failure. This is a pre-existing CUDA
full-reduction fusion boundary, tracked as KI-CODEGEN-005, rather than a new
leaf/view regression. Profile shows the default two-stage CUDA full sum
materializes the multiply result; explicitly naming both reduction axes keeps
the native fused consumer and allocates 1,048,576 bytes. The existing failing
test and budget are unchanged. Root integration structure/tools verification
also passed 1661 nodes with six environment skips. The source-bound ZeRO2
oracle measurements below remain pinned to `489417b36`.
The two reported structure skips about installed-package resolution are
environment checks, rather than executed tensor contracts.

After serial per-rank warmup, the real ZeRO2 BF16 case was rerun against the
clean `489417b36` source, with unique `stage2-leaf-owner-v1` outputs and a new
checkpoint directory. Both ranks again passed all 208 numeric fields against
the independent binary Torch oracle, all 112 gradient fields were bitwise
equal, and the largest normalized forward/state difference remained
`1.5522042910257974e-7`. Each rank's 56 strict save/load/resume fields were
bitwise equal within Jittor; all 166 master-partition elements changed on the
third update, with L2 deltas `0.009065359830856323` and
`0.02911890670657158`. Both ranks had fallback delta 0.

Unversioned evidence remains under the same task state root:
`multirank/leaf-owner-evidence-v1.json` binds 1531 source hashes and 28 evidence
hashes, including the clean-source pin, raw rank NPZ/logs, independent oracle,
comparison, strict resume and all unit/core/structure logs. It also retains
the pre-fix leaf-view and frozen-RHS counterexamples. The earlier v2 matrix
manifest remains unchanged. This is a source-bound dedicated CUDA result;
there is no equivalent world-size-two BF16 CPU run, so it cannot satisfy the
common scorecard's matching CPU ladder or count as accelerator breadth.

## Boundaries

This tiny two-rank case does not establish ZeRO3, CPU/NVMe offload, compiled
autograd, overlap modes, other optimizers in BF16, large models, throughput,
ROCm or NPU behavior. The existing differentiable/non-contiguous foreach
restrictions remain explicit. The leaf/view correction and its separate
verification scope are recorded above. A maintained two-GPU dedicated gate
and the second explicit input perturbation remain follow-up work; the original
single-rank FP32 gate is not a substitute for that protocol.
