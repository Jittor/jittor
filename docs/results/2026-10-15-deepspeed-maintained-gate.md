# DeepSpeed: maintained two-rank BF16 Engine gate

- Status: bounded real Engine acceptance; whole-library coverage remains pending.
- Date: 2026-10-15
- Baseline: `f666c3f7190aeed39d7bb747dcf7f0753afaced8` (clean final matrix);
  CPU arithmetic fix `a441034299beb141785bbbffe7afaf03c99638c2`.
- Owner: Jittor core/compat maintainers.
- Review when: native Adam state arithmetic, view/storage publication, autograd hooks,
  optimizer loading, native collectives/Store, the runner/launcher, or the pinned
  DeepSpeed adapter/source changes.

## Scope and protocol

The maintained entry is `nox -s ecosystem_deepspeed_multirank`, calling
`test_two_rank_bfloat16_engine_training_and_resume`. It runs ZeRO 1/2 × AdamW ×
input perturbation 0/.125 on two actual CUDA ranks. Each rank has different inputs;
actual FP32 master partitions contain 166 of the model's 332 parameters. This is
DeepSpeedEngine.backward/step/save_checkpoint/load_checkpoint in a fresh Engine,
with six microsteps, GAS 2, three updates, save after update two and two resumed
microsteps. The tiny residual Linear/GELU/LayerNorm/Linear model is not a speed gate.

Both runtimes use official DeepSpeed v0.19.7 at
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`; the independently built source wheel has
SHA256 `467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`.
The oracle is binary Torch 2.11.0+cu128 with its actual `_C` extension and no shim
marker. The adapter remains separately installed and pins upstream source hashes.
No ordinary model runner or external SGD substitutes for the Engine.

Numeric comparison reuses the original shared harness: forward/state tolerance
2e-3, all-gradient tolerance 1e-2, the existing category floor 1e-3 times the maximum
reference magnitude, and the minimum scale 1e-6. These thresholds were unchanged.
Every incoming parameter gradient, both input gradients and the publicly available
full gradients are checked. Saved/restored and final/resumed optimizer states,
master partitions and all resumed output/loss/gradient/parameter fields are checked
bitwise within each runtime, independently of cross-runtime numeric tolerances.

Observed dtype matters: ZeRO1 BF16_Optimizer can return an existing BF16 param.grad
from safe_get_full_grad; ZeRO2's partition getter returns FP32 accumulation storage.
Raw metadata records each observed gradient dtype; both runtimes and resumed
fields must agree. Initial flags do not substitute for tensor residency checks.
All requested tensors are materialized on CUDA and fallback is forbidden/count 0.

## Reproduction and artifact safety

Install the verified source package and standalone adapter in the native environment;
use a separate binary Torch interpreter. Follow the
[runbook](../../agent/skills/deepspeed-torch-compat/SKILL.md) for dependency isolation.

```bash
export JITTOR_CI_PYTHON=<native-python>
export REAL_TORCH_PYTHON=<independent-binary-torch-python>
export JITTOR_TEST_DEVICES=cuda
export CUDA_VISIBLE_DEVICES=<allocated-device-a>,<allocated-device-b>
export JITTOR_DEEPSPEED_MULTIRANK_ROOT="$JITTOR_LAB_ROOT/_state/deepspeed/<fresh-run>"
python -m nox -s ecosystem_deepspeed_multirank
```

Cases include stage/optimizer/precision/world size/perturbation in their paths.
Pre-existing case directories, including empty ones, are rejected. Failed evidence
is retained. Each runtime/rank has private HOME/JITTOR_HOME/TMPDIR/cache directories.
The two native ranks share their case's communicator heartbeat/control directory;
that directory must not be split with rank caches. Cold core/NCCL warmups run in
sequence. A failed or timed-out rank terminates and reaps its live peer.
Source hashes and the current commit are checked before and after each case;
comparison.json binds NPZ/log/protocol/source hashes. No raw artifact is versioned.

## Validation

| Check | Result |
| --- | --- |
| Final maintained two-rank BF16 CUDA matrix, clean `f666c3f719` | 4 passed, 0 skipped |
| Maintained world-one FP32 CPU Engine matrix after FileStore fix | 12 passed, 0 skipped |
| Maintained world-one FP32 CUDA Engine matrix after FileStore fix | 12 passed, 0 skipped |
| Native Adam state/fetch-order regressions | 8 passed |
| Core, native and Torch modes | 322 passed, 42 skipped, 1 existing xfail |
| Full structure plus DeepSpeed protocol/skip-policy tests | 1,507 passed, 6 skipped |
| Focused launcher/nox host contracts | 23 passed |
| Repository layout | passed |

The full CPU/CUDA world-one gates were rerun after the arithmetic and FileStore
changes. Their source is the unchanged ancestor of clean `f666`; the subsequent
remote merge adds metric/TensorDict gates, inventory and documentation, without
changing the DeepSpeed runner or native Adam. The final four CUDA cells were run
on clean `f666`. The hot-cache four-cell pytest run took 83.84 s; this is execution
evidence only, with no comparative speed or throughput claim.

Read-only revalidation checks all 1,566 tracked runtime-source hashes and 1,792
numeric arrays across four cases/two ranks. All 1,024 gradient arrays are bitwise
identical to independent Torch. Maximum normalized forward/state error is
1.7584651928227423e-7; gradient error is zero under the unchanged comparator.
Every rank updates all 166 FP32 master-partition elements. Stage1 checks 64 strict
resume fields per rank/runtime and Stage2 checks 56, including restored optimizer
state and the continued training path. Both native ranks report fallback count 0.

Unversioned raw evidence is under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/multirank/`:

- `maintained-formal-v3/`: exclusive original case directories, rank NPZ/logs,
  checkpoints, FileStores, protocol/source manifests and comparisons.
- `maintained-deepspeed-v3-evidence.json`: 1,566 source hashes and 128 evidence
  hashes, including all 108 final raw case files; SHA256
  `425e8b405647c7aef75b88c28742598b43a60b87e508a690e829d8101b0da9df`.
- `maintained-nox-formal-v3.log`, `maintained-matrix-summary-v3.json` and
  `maintained-matrix-audit-v3.log`: final execution and recomputed comparisons.
- `maintained-filestore-nox-{cpu,cuda}-v1.log`,
  `maintained-postsync-core-cpu-v1.log`, `maintained-filestore-structure-v1.log`,
  `maintained-filestore-host-v1.log` and `maintained-filestore-layout-v1.log`:
  adjacent gates. The six structure/environment skips do not establish tensor
  execution; the actual Engine matrices have no skips.
- `upstream-protocol-inventory-v2.json`: bounded upstream protocol declarations,
  official source identity and four source-file hashes.

The code dependency order is `f4c1a00b7794` (maintained gate), `5da8ee2410a5`
(shared control-plane directory and observed dtypes), `a441034299be` (native CPU
Adam boundaries), then `0f1bc0de239a` (exclusive FileStore bootstrap). The arithmetic
fix is independently reviewable in its two-file commit. No source hash or baseline
is relabeled to a later integration commit.

Actual platform: Linux x86_64, RTX4090/SM89, NVIDIA driver 580.178.04,
Jittor pip-cu122 with nvcc 12.2.140 and GCC 11.5; independent Torch uses its CUDA12.8
binary distribution. Matmul/cuDNN TF32 are enabled, matmul precision high,
cuDNN benchmark false, four CPU threads per process. This is not a TF32-off claim.

## CPU state-rounding correction

Integration onto `26be435076cfd24ff9d0611fbe55f3a04085c298`, including the native
RNG repair, independently rehashed all 128 evidence files and 1,566 source hashes
and recomputed the four original CUDA comparisons. On the combined worktree,
the CPU and actual CUDA world-one Engine gates each passed all 12 cases with no
skips; native Adam's eight directed regressions also passed. Core passed 322
checks with 42 skips and one existing expected failure. Structure plus tools
passed 1,707 checks with six declared/environment skips. Four formatting-only
follow-ups preserve their source ASTs. Integration logs and reproduction
environments are unversioned under
`$JITTOR_LAB_ROOT/_state/deepspeed-integration-26be/`.

The first integration CPU attempt failed in the independent oracle before
training because its installed Ninja executable was absent from PATH. Adding the
existing private dependency bin directory made a single strict-resume case and
the full 12-case rerun pass. Original failure logs remain; no arithmetic or
threshold change was made for that environment repair.

The unchanged pre-maintenance runner on independent clean `4a4a480e6` reproduced
CPU ZeRO1 AdamW parameter fresh-resume divergence: 31/128 up.weight elements,
max absolute 1.4901161e-8. Saved/restored master and moments, resumed forward/loss,
and all incoming/full/input gradients were bitwise equal. Step-three Adam received
identical parameter, gradient, first/second moments, scalar hyperparameters and
step counts; final master and moments diverged by an ulp.

A no-DeepSpeed deterministic regression makes this observable by fetching updated
moments before the returned parameter versus fetching the parameter first.
The clean baseline's final regression differs at 86/332 FP32 parameters, max
2.9802322e-8. Intermediate outputs retained by state views/AD allow CPU contraction
across different boundaries. Preserving only final moment outputs fixes the old
parameter assertion but exposes moment differences under the new full-state strict
check; that failed attempt is retained rather than presented as acceptance.

Only CPU torch_math=True uses explicit lazy rounding boundaries for the optimizer
gradient, each state summand and updated moments. The shared native Adam formula
owns the change; default native math and CUDA's existing path keep their policy.
There is no graph-wide sync, global compiler-policy change or adapter math.
These boundaries may add CPU materializations; throughput was not measured.
Eight regressions cover Adam/AdamW × true FP32/FP64 and state/gradient fetch order.
Independent binary Torch checks actual same-dtype arithmetic on CPU and CUDA;
cross-runtime arithmetic remains numerically close rather than bitwise identical.
The true-FP64 supplementary oracle probe still observes up to about 2.19e-8
normalized residual from the existing scalar-coefficient precision policy; these
regressions assert same-runtime rounding stability, not full FP64 oracle fidelity.

## Boundaries and retained failures

The first maintained attempt failed honestly: two Stage1 oracle cells revealed a
wrong all-full-grad-FP32 assertion; two Stage2 native cells used separate heartbeat
rendezvous directories and the watchdog aborted. Dtype observations and shared
control-plane setup fixed these issues; the watchdog was not disabled.
A later single-rank CUDA attempt had 11 numeric passes and one independent Torch
TCPStore EADDRINUSE before training; exclusive per-runtime FileStore bootstrap
then passed both full 12-cell
world-one matrices and the final four-cell world-two matrix. FileStore is control
plane bootstrap; both CUDA ranks still perform real NCCL training collectives.
All failed logs and earlier raw matrices remain separate from the final run.

The earlier native 58-node view run did not globally enable CUDA. Its unscoped
allocation node ran on CPU. A fully CUDA-enabled run has the independently
baseline-reproduced KI-CODEGEN-005 full-sum allocation failure; this gate does not
change that threshold or claim the allocation check passed on CUDA.

The world-one FP32 CPU 12-cell Engine gate is separate evidence. It is not the
world-two BF16 CPU ladder. A matching world-two BF16 CPU result has not yet been
run; the current maintained
launcher is CUDA-specific, which does not establish that the actual CPU API is
impossible. A separate same-model/weights/spec CPU API probe is required. Common
scorecard admission stays pending and this CUDA result grants no accelerator
coverage numerator. The read-only upstream inventory records four unique ZeRO
stages (max_stage is an alias), three offload device values and nine declared
DeepSpeed optimizer names as separate bounded axes. Dynamic torch.optim lookup
and user optimizer instances/callables prevent treating that nine-name declaration
as the complete optimizer API denominator. The inventory binds the full upstream
SHA and four source-file hashes; central scorecard wiring is separate work.

Not validated: ZeRO3, CPU/NVMe offload, other optimizer classes in world-two BF16,
FP16 multi-rank, multi-node, pipeline/tensor/expert parallelism, DeepCompile,
elastic launch, performance, Windows CUDA, ROCm or NPU. The existing pinned
adapter still explicitly rejects unsupported compile/elastic paths. Four tiny
CUDA cells do not establish whole-library support.
