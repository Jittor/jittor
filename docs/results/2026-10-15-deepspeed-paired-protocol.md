# DeepSpeed paired BF16 Engine protocol

- Status: Verified bounded paired CPU/CUDA L3 protocol; no registry coverage claim
- Date: 2026-10-15
- Measurement commit: `f787cc0b34b8b24e04d79cc55aacf746c01036b8`, clean before/after every case
- Upstream: official DeepSpeed `f5af15c2b4da10411e340850c00ba0cb5b14fd87`
- Installed version: `0.19.7+f5af15c`; the development version is retained
- Owner: Jittor core / Torch compatibility maintainers
- Review when: Engine/ZeRO source, adapter pins, native autograd/storage/optimizer,
  BF16 Linear/LayerNorm, process groups, public launcher or protocol validator changes

## Verified scope

This is a bounded training/checkpoint protocol, with no registry denominator and no
performance claim. Both devices use the same residual Linear/GELU/LayerNorm model,
332 trainable BF16 parameter elements, two differentiable inputs, world size two,
ZeRO1/2, AdamW, GAS two, six microsteps and three updates. Both original input offsets
`0` and `.125` are exercised. Each rank has distinct inputs and a real 166-element
FP32 master optimizer partition. After microstep four, a new Engine loads the saved
checkpoint and repeats microsteps four/five. All observed optimizer state, master
partitions, outputs, loss, gradients and updated model parameters must match its
uninterrupted run bitwise.

The dedicated CPU counterpart uses actual OpenMPI 5.0.11; independent PyTorch uses
Gloo. This proves the CPU MPI backend for this protocol. It does **not** establish
native Gloo or CPU `jtrun` support, and does not recommend another training launcher.
The public `jtrun` owner currently supports NCCL/HCCL. CUDA candidate training uses
`python -m jittor.distributed.launch`, equivalent to the console entry, and actual
NCCL; independent binary PyTorch uses NCCL in separate processes.

Each candidate and oracle process actually sets seed zero, then fully overwrites
weights with the original deterministic fixture. The `.125` input offset is not
renamed to a different RNG seed. The snapshot observes actual initial BF16 training
weights after DeepSpeed conversion, before training; all 332 values match exactly.
The temporary FP32 fixture has an existing one-ULP fused/separate rounding difference
(up to `1.49e-8`) before BF16 conversion. The failed pre-conversion hash diagnostic
is retained; changing snapshot placement changes no model mathematics or weights.

Each microstep records the actual inputs/targets and all incoming parameter and input
gradients. ZeRO1's public full-gradient shortcut returns BF16; ZeRO2's full getter at
GAS boundaries returns FP32 accumulation storage. Gradient dtype and complete expected
schemas are checked independently of cross-runtime key equality. Initial BF16 output
can be invariant to the additive perturbation because of LayerNorm; later outputs,
every microstep's loss and gradients respond to the actual perturbation. The
validator checks the full training trajectory, without changing the original inputs.

## Results and platform

| Device / stage | Compared numeric tensors | Parameter / input gradient tensors | Bitwise gradients | Maximum scaled forward/state error | Maximum scaled gradient error |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / ZeRO1 | 1056 | 512 / 64 | 569 / 576 | 1.37151e-5 | 5.67769e-4 |
| CPU / ZeRO2 | 928 | 384 / 64 | 428 / 448 | 5.49490e-5 | 3.33526e-4 |
| CUDA / ZeRO1 | 1056 | 512 / 64 | 576 / 576 | 1.75847e-7 | 0 |
| CUDA / ZeRO2 | 928 | 384 / 64 | 448 / 448 | 1.75847e-7 | 0 |

Counts aggregate two perturbations and two ranks against the independent oracle.
Original shared budgets remain `.002` for forward/state and `.01` for gradients,
using each category's original comparison floor. CPU agreement is approximate;
CUDA gradients in this fixed matrix are bitwise equal. Every process passes 68
(ZeRO1) or 60 (ZeRO2) strict continuation fields and updates all 166 master elements.
All candidate runs execute under forbidden fallback, with fallback delta zero and
materialized requested-device inputs, parameters, outputs and gradients.

The host is Linux x86-64, AMD EPYC 9554, GCC 11.5.0 and Python 3.11. Native CPU uses
oneDNN 3.9.1. CUDA uses two allocated RTX 4090 cards, SM89, driver 580.178.04 and the
actual native pip-cu122/nvcc 12.2.140 provider. Oracle is independently loaded
PyTorch `2.11.0+cu128` with its real `_C` extension and SHA256. CUDA precision switches
report TF32 matmul/cuDNN enabled, cuDNN benchmark disabled and float32 matmul `high`;
CPU reports TF32 disabled and `highest`. These are recorded actual settings, not
claims about unrelated FP32 workloads. Per-process manifests retain GPU driver UUIDs,
allocation, compiler output and installed NVIDIA wheel metadata; installed metadata
alone is not treated as a loaded CUDA provider.

Both ranks share a case/runtime communicator control directory while HOME,
JITTOR_HOME and TMPDIR remain private. First core/extension builds are serial.
The CPU backend check explicitly preserves the parent's CPU affinity instead of
OpenMPI's default single-core binding. CUDA rank/device/control assignments come
from public `jtrun`; a `-S` wrapper selects caches before site hooks, then execs the
normal interpreter. Rank launch commands and placement controls are hashed raw data.

## Source and evidence

The official wheel is independently rebuilt from a clean detached upstream checkout
with binary Torch, `DS_BUILD_OPS=0` and the real CPU accelerator selection. The
original and rebuilt wheels contain exactly the same 1002 package files: 1001
tracked-source payloads plus upstream `setup.py`'s generated metadata. Every package
byte, including generated SCM version/installed-op/build-Torch metadata, matches.
The original wheel SHA256 is
`467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`.

The source descriptor expands official Git directory symlinks for accelerator,
op_builder and csrc. A Python-only source list would miss real JIT extension
sources. It records 34 official files excluded by the normal package-data rules
(docs, non-package AIO/GDS developer tests and an MPS `.metal` file); this is not MPS
support. Missing or modified C++/Python payloads, extra files, executable generated
metadata, dirty sources and nonofficial remotes are rejected. Independent rebuild
wheel bytes and build logs are retained outside Git; no new package version is
manufactured from installed dependency metadata.

Raw data is unversioned under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`:
`protocol-formal-v1/{cpu,cuda}/zero{1,2}/AdamW/bfloat16/world2/perturb-*` contains the
actual NPZ/log/launcher/protocol/comparison artifacts and complete source manifests.
Each manifest binds all 1574 tracked native/compat/backend/adapter/nox runtime files.
Historical failure diagnostics and old CUDA evidence retain their original baselines;
they are not relabeled as this clean public-launcher measurement.

Reproduce the dedicated gate with the installed independent interpreters, explicit
oneDNN/toolchain paths and fresh external artifact directories as described by the
[DeepSpeed runbook](../../agent/skills/deepspeed-torch-compat/SKILL.md). CUDA requires
explicitly allocated two-card `CUDA_VISIBLE_DEVICES`. CPU MPI backend verification
additionally requires a real OpenMPI toolchain; it is not a CPU `jtrun` command.
The maintained entry in either case is `nox -s ecosystem_deepspeed_multirank`.

`collect_deepspeed_scorecard.py` rechecks every actual rank/scenario artifact and
calls the common `audit_coverage.scorecard` protocol dispatcher. Protocol records use
`deepspeed-zero-training`, empty registry axes/references and identical full semantic
descriptors on CPU/CUDA; MPI/Gloo/NCCL and launchers stay in platform metadata.
Missing exact CPU counterparts must block CUDA admission. L4/L5 and a whole-library
support percentage are not inferred from these tiny L3 protocol measurements.
The actual common audit at `5cd468ac4f4f678cd0f73e62309a0ef348558fd2` accepted four
records, rejected zero and blocked zero: CPU and CUDA ZeRO1/2 are L3, with exact
semantic CPU counterparts for both CUDA records. The registry coverage array is
empty. Artifacts are in `paired-scorecard-v1/{evidence,scorecard}.json` under the
external state root; the complete source/raw/checkpoint/log seal is
`paired-protocol-v1-evidence.json`.

The final common audit on clean `c93eb2c0690b9541063330685fb7ac6560eb8812`
independently regenerates the same four accepted records, zero rejected/blocked,
and empty registry coverage in `paired-scorecard-v2/{evidence,scorecard}.json`.
The measured source remains `f787cc0b34b8b24e04d79cc55aacf746c01036b8`.
The immutable final evidence seal has SHA256
`f008faff4bfaa4cfd540b62067d69ec031b98749ac9ab04984bf52e3b6ae52cc`;
it binds 829 raw files, 1574 measured runtime source hashes, all 32 formal NPZ files
and 48 actual checkpoint files. It also retains the original diagnostics,
independent wheel rebuild, both common audits and all eleven negative controls.

Nine destructive tests on disposable copies of the actual NPZ/gate artifacts were
rejected by the common auditor after hashes were rebound: a gradient dropped from
both runtimes, one-ULP restored moment divergence, incorrect full-gradient dtype,
fallback, oracle shim identity, CPU backend mislabel, swapped CUDA UUID, private
CUDA launcher and stage mislabel. Thus ordinary hash mismatch was not the reason
for rejection. Separately, CUDA without CPU or with only a different-stage CPU
record stayed blocked, with admitted level null and independent L3 retained.

## Regression boundaries

Before source sealing: native/Torch core `322 passed / 42 skipped / 1 existing xfail`,
25 launch/control contracts and 30 focused structure checks passed. The full structure
run had 1502 passes and six skips: two xdist failures from disabled plugin autoload
were rerun successfully with the declared plugin; the two MPI/math report-registration
failures were resolved by subsequent upstream integration. Both original logs remain.
On clean measurement source, dedicated nox CPU and CUDA each passed all four cases
without skips. Launch/scorecard host tests passed 46 cases; with the shared
Accelerate protocol regression, 78 host cases passed. Ruff/layout passed.
After upstream sync and adding this report, full structure had 1504 passes and six
skips; its two failures are this new report's central index/MANIFEST registration,
which remains the integration owner's responsibility. No numerical test is skipped
or given a weaker threshold for those documentation failures.

The first enlarged single-rank FP32 regression exposed an overbroad new transport
assertion: the native singleton WORLD API retains its legacy default `mpi` label
even without an active MPI/NCCL communicator. CPU and CUDA each failed 12 cases at
that assertion after executing training/strict continuation. The comparator now
requires an actual matching native transport only for world size greater than one;
the full two-rank rules and numerical budgets are unchanged. Singleton labels do
not establish distributed capability. The original failure logs remain unmodified.

On clean `c93eb2c0690b9541063330685fb7ac6560eb8812`, the repaired old Engine
matrix passes all 12 CPU cases (118.99 seconds) and all 12 actual CUDA cases
(140.47 seconds), with zero skips and unchanged source/HEAD. The corresponding
`protocol-legacy-{cpu,cuda}-v2.log`, `legacy-regression-v2-source.json` and
`legacy-regression-v2-result.json` are sealed separately from the original failures.
The same clean source passes 78 host contracts, Ruff and layout. Retained direct
singleton Engine artifacts in `singleton-engine-v1/` verify real train/save/new-Engine
load on both devices, with 124 strict continuation fields per runtime and candidate
fallback delta zero. Their native `mpi` label coexists with `JT_BUILD_USE_MPI=0`;
that singleton metadata must not be read as evidence of an active MPI communicator.

Stage3, optimizer/parameter offload, other optimizers, pipeline/MoE/elastic/DeepCompile,
large models, arbitrary BF16 shapes, non-Linux platforms and CPU unified launching
remain outside this protocol. Existing approximate post-accumulate scheduling is not
promoted to exact PyTorch worker scheduling. Other accelerators remain unverified.

## Main-tree integration verification

Integrated onto synchronized `4ccd72fafe51496190671dc93548c852be2974c1`.
The original eight clean measurements retain `f787cc0b34b8b24e04d79cc55aacf746c01036b8`;
the singleton comparator repair is `6268a15d6`, and the shared collector/validator
delivery is `c93eb2c06`. No original numerical observation is relabelled.

An independent integration audit reopens all 32 rank/runtime/scenario NPZ files,
recomputes 3,968 numerical field pairs against the unchanged `.002`/`.01` budgets,
and checks 2,048 same-runtime continuation field pairs byte for byte. CPU has
997/1,024 bitwise gradient fields; CUDA has 1,024/1,024. The separate shared
collector also regenerates the full source/wheel closure and all four records:
four accepted, zero blocked or rejected, with empty registry coverage. The
official rebuild script/log and its wheel SHA256 were inspected; the rebuild
checkout remains clean at the pinned upstream commit.

On the integrated tree, old single-process FP32 Adam/AdamW × ZeRO0/1/2 × two
offset regressions pass all 12 CPU and 12 actual CUDA nodes, zero skips. Launch,
scorecard, Accelerate and registry host regressions pass 131 nodes. Full structure
passes 1,506 with six skips; Ruff for changed Python files and layout pass.
Report and generated-manifest registration resolve the two isolated-tree errors.

Integration environment failures are retained: the first CUDA run passed six
nodes and failed six when genuine Torch could not allocate a ZeRO bucket while
another process occupied about 18 GiB. Repeating the unchanged 12-node protocol
on an allocated card with sufficient free memory passes. The first CPU attempt
failed before numerical comparison because genuine Torch's extension builder
could not find Ninja; its initially missing private temporary directory also
allowed a temporary-path fallback. The final run creates that directory and
uses the existing genuine Ninja executable, compiling extensions serially in
the independent unit cache. These environment corrections change no model,
optimizer, bucket specification or numerical budget.

Unversioned independent and integration evidence is under
`$JITTOR_LAB_ROOT/_state/deepspeed-paired-integration-4ccd/`, including
`direct-array-audit.json`, `independent-scorecard-v1/`, source checks and gate logs.
This strengthens the bounded paired Engine result; it supplies no CPU `jtrun`,
fresh-process restoration, multi-host or whole-library conclusion.
