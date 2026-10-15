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

## Boundaries

This tiny two-rank case does not establish ZeRO3, CPU/NVMe offload, compiled
autograd, overlap modes, other optimizers in BF16, large models, throughput,
ROCm or NPU behavior. The existing differentiable/non-contiguous foreach
restrictions remain explicit. The pre-existing grad-enabled leaf-view inplace
rejection gap requires its own shared-owner correction and is not represented
as a passing contract by this result.
