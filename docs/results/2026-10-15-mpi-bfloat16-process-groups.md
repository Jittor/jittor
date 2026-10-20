# CPU MPI BF16 communication and process groups

- Status: bounded communication acceptance; DeepSpeed CPU BF16 numerical ladder pending.
- Date: 2026-10-15
- Baseline: source commit `2de23a18e8788f901eaf08f9a0c4ef8e0d432876`;
  synchronized local merge `49a95fdde79869341af22eae98a7dd0972e01536` includes
  remote `824cb354d5cced0743b3207d50d518111ba59e76`.
- Owner: Jittor distributed and compat maintainers.
- Review when: MPI datatype/operation tables, communicator lifetime, collective
  gradients, native process groups, compat collective routing or storage publication change.

## Verified behavior

The native MPI owner now transports BF16 as opaque two-byte elements and uses a
separate BF16 SUM callback built from the existing core conversion/rounding type.
BF16 arithmetic never enters the FP16 callback. Actual MPI communicators come from
`MPI_Comm_split`, preserving requested rank order and independent contexts. All
WORLD ranks must create groups in the same order. Collective backward carries the
same communicator; mean uses its size and reduce/broadcast use its local root.
Communicators remain worker-owned until shutdown, matching existing group lifetime.

MPI gather is a composition of real per-group broadcasts. MAX/MIN/PRODUCT are
real gather plus existing native elementwise operations; they are explicitly
composite and reject differentiable inputs. This is not an MPI all-gather or
non-SUM fused kernel claim. Compatibility flat gather and grouped broadcast
forward to the native owner. Existing compat subgroup restrictions remain;
full-world clones required by DeepSpeed are covered here.

The reusable [collective oracle](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/collective_oracle.py)
executes the same public API body in isolated runtimes. Each runtime/device has
a fresh artifact directory; exclusive rank claim files reject repeated runs.
It checks BF16 raw payloads (signed zero, subnormals, infinities and NaN payloads),
finite SUM ties, uint8 overflow, FP32/BF16 SUM/MAX/MIN/PRODUCT, Parameter/data/detach/
view storage publication, immutable earlier clones, and differentiable SUM.

| Candidate | Independent oracle | Actual ranks | Compared fields | Result |
| --- | --- | --- | --- | --- |
| CPU / OpenMPI | CPU / Gloo | 2 | 36 per rank | Bitwise equal |
| CUDA / NCCL | CUDA / NCCL | 2 | 36 per rank | Bitwise equal |

All 144 compared numeric fields match. Metadata is checked separately; MPI is
not presented as native Gloo. Candidate inputs/outputs are materialized on their
requested device under the fallback-error policy with count delta zero. CUDA
regression exercises the shared non-SUM fold and grouped gather/broadcast routes.

## Platform and regression

Linux x86-64, Python 3.11.16, GCC 11.5; private OpenMPI 5.0.11. The official
[OpenMPI wheel](https://pypi.org/project/openmpi/5.0.11/) SHA256 is
`838de0ec7fd719af19cde664cd6bf76eb170ed4769c279040c2db434d6db11b9`.
Its complete `.data/data` layout was extracted to a private prefix, including
libraries omitted by the initial target-only installation. MPICH was not accepted:
the existing compiler probe uses OpenMPI-specific `--showme` options.
No shared environment was modified. oneDNN uses an explicit existing installation.

The independent oracle is binary Torch 2.11.0+cu128 with its real `_C` extension
and no shim marker. CUDA ran on two actual RTX 4090 / SM89 devices, driver
580.178.04, with the native pip-cu122 toolkit and nvcc 12.2.140. No performance
claim follows from these tiny communication fixtures.

- Actual two-rank MPI regression: **20 passed, 0 skipped per rank**, including
  BF16 reduce/mean with a nonzero group-local root and backward, reversed rank
  order, communicator independence, non-SUM AD rejection, and existing dtype,
  FP16 scalar/reduction and MPI operator tests.
- Compat routing owners: **7 passed**; native process-group host contracts:
  **13 passed**.
- CPU core: **322 passed, 42 skipped, 1 existing xfail**, both process modes.
- Full structure: **1497 passed, 6 skipped, 2 launcher failures** caused by
  disabling plugin autoload without explicitly loading xdist. With
  `PYTEST_PLUGINS=pytest_timeout,xdist.plugin`, that entire file passed **10/10**.
  The earlier core launcher failure likewise preceded test execution and was
  rerun with the timeout plugin. Original failure logs remain intact.
- Repository layout check passed. No smoke/full runtime or non-x86/other MPI
  implementation, HCCL, accelerator MPI subgroup or larger-world claim is made.

The final documentation/private-path check passes 18 tests with one pending
central integration check: this new report must be registered in the results
toctree by the parent integration owner. Shared indices and release manifests
were intentionally left to that owner.

## DeepSpeed CPU boundary

Official DeepSpeed 0.19.7 at `f5af15c2b4da10411e340850c00ba0cb5b14fd87`
can execute the existing real residual model on CPU BF16 with two Gloo ranks in
binary Torch. This was tested through actual Engine backward/step/save/load,
not inferred from the maintained runner's previous CUDA-only guard.

The same candidate model, deterministic weights, precision and six-microstep
protocol now execute on CPU MPI: ZeRO1 checks 64 strict-resume fields, ZeRO2 checks
56; all 166 local FP32 master elements update on each rank. These are API probes
with diagnostic metadata, not maintained-gate admissions. They still fail the
unchanged external numerical budgets (forward/state 0.002, gradients 0.01):

| Stage | Worst forward/state | Worst gradient, ranks 0 / 1 |
| --- | --- | --- |
| ZeRO1 | 0.0717949 | 0.662354 / 0.719285 |
| ZeRO2 | 0.0717949 | 0.294964 / 0.456353 |

Separate operator probes use the exact reference BF16 input and cotangent.
The up/down/head Linear outputs differ in 6/32, 2/16 and 4/8 elements respectively,
while their isolated input/weight/bias gradients are bitwise equal. LayerNorm's
isolated output is bitwise equal, but all 16 input-gradient elements and 4/8
weight-gradient elements differ (maximum absolute differences `2.32756e-5` and
`6.10352e-5`). Bias gradients match. These probes retain intermediates and are
explicitly diagnostic; the original Engine failures remain unchanged. Both
CPU math owners require separate investigation. No CPU paired scorecard, CUDA
coverage numerator or wider DeepSpeed support is granted by this MPI batch.

## Reproduction and evidence

Follow the [distributed verification skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)
for independent rank environments, serial first compilation and actual Torch
identity checks. Run the oracle in fresh roots with `--runtime jittor|torch` and
`--device cpu|cuda`; compare every numeric NPZ field exactly and audit metadata
separately. The native MPI regression entry is:

```bash
mpirun -n 2 <private-rank-environment> <jittor-python> <native-pytest-wrapper> -q \
  tests/backends/comm/mpi/test_mpi_bfloat16_groups.py \
  tests/backends/comm/mpi/test_mpi_dtypes.py::TestMpiCollectiveDtypes \
  tests/backends/comm/mpi/test_mpi_half_reduce.py::TestMpiHalfReduce \
  tests/backends/comm/mpi/test_mpi_half_scalar.py \
  tests/backends/comm/mpi/test_mpi_op.py::TestMpiOps
```

Native pytest runs with `JITTOR_TORCH_SHIM=0`; compat owner tests and structure
run in their separate Torch-mode process. Core uses
`python tools/run_test_suite.py --tier core --serial-compile`.

Unversioned evidence lives under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/cpu-bf16/`:
`mpi-bfloat16-groups-v3-evidence.json` has SHA256
`7e72563a214d02929b8856a030a974a428efdcd7f0c8011c92db7b0c4f6d5ed7`.
It binds 1,570 runtime source hashes to the source commit, the oracle script hash,
94 raw evidence hashes, and the four rank comparisons. `collectives-*-*-v2/`
contains final raw packets; earlier v1 artifacts remain untouched.
`audit-mpi-bf16-v3.py` recomputes source bytes and numeric equality.
`launch-collective-oracle.py`, `run-native-pair.sh`, `native-env.sh`,
`run-cuda-collectives.sh` and `run-cpu-gates.sh` retain exact local environments.

The manifest also preserves native Stage1/2 API packets, original unsupported
Gloo/BF16/group probes, their matched pre-commit MPI patch, and isolated
Linear/LayerNorm NPZ files. The independent Torch CPU Engine evidence remains
at its original `f945b223cd4316feaa733556ef1413344f98e11f` baseline in
`cpu-bf16-api-oracle-v1-evidence.json`; it is not relabelled as a final paired gate.

## Main-tree integration and launcher boundary

Integration starts from `0751e425b0f197b8e7b388f218613378c71f701a`, plus the
MPI source and evidence commits above. All 1,570 original runtime source hashes,
94 raw hashes and 144 numeric fields were independently rechecked. The main-tree
CPU run then executed two actual MPI ranks: all 72 numeric fields were byte-equal
to the preserved independent Torch/Gloo reference. The existing MPI regression
set passed 20 tests per rank without skips; native/compat owner contracts passed
13 and 7 tests. Core passed 322 tests with 42 skips and one existing xfail.

CUDA integration used the public `jtrun` owner through its documented equivalent
`python -m jittor.distributed.launch --standalone --backend nccl
--nproc-per-node 2` entry. The task-local rank wrapper only selected independent,
serially prewarmed caches; `jtrun` supplied rank, device and rendezvous settings.
Both actual CUDA ranks completed all 72 numeric fields byte-equal to the preserved
independent Torch/NCCL reference, with device residency and zero fallback checked.

The primary accelerator training entry remains `jtrun`. Its current supported
backends are NCCL and HCCL. The CPU `mpirun` evidence here validates the retained
MPI backend; it does not establish CPU support in `jtrun` or a new MPI training
entry. The CUDA launcher measurement above is separate from the child's original
private two-process launch and keeps its own command and logs.

One initially allocated CUDA device failed during serial warmup with
`cudaErrorECCUncorrectable` in `cudaMemGetInfo`, before any collective ran. The
failed warmup is preserved; no device reset was performed. A different allocated
device passed the actual UUID/residency check and the subsequent `jtrun` run.
The oracle's two formatting-only edits retain an identical AST to the originally
sealed script. Raw integration evidence and commands are unversioned under
`$JITTOR_LAB_ROOT/_state/mpi-integration-282/`; that directory names the integration
task, while the actual baseline is the full SHA above.
