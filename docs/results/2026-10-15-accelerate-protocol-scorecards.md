# Accelerate checkpoint protocol scorecards

- Status: independent CPU world-one and CUDA world-one/two correctness verified;
  shared admission accepts world-one CPU/CUDA, while CUDA world-two remains
  blocked by the missing matching CPU FSDP2 protocol.
- Measured source: `b7e5a07b32f300e6f348a2147e7fafd5f4704c96`.
- Synchronized upstream Jittor baseline: `0751e425b0f197b8e7b388f218613378c71f701a`.
- Maintainer: Jittor core/compatibility maintainers. Reviewed: 2026-10-15.
- Recheck when: Accelerate, binary Torch, FSDP2, optimizer/checkpoint/RNG owners,
  launcher, precision policy, or protocol validation changes.

## Scope and actual execution

The maintained gate calls existing Accelerate tests in two independent
interpreters. It adds observations and provenance, without changing model or
optimizer mathematics. Accelerate has no model registry denominator here:
records explicitly use `record_kind=protocol`, `axes=[]`, `registry_refs=[]`.
These results do not establish whole-library coverage.

Both interpreters execute Accelerate **1.10.1**, from official upstream commit
`623cc0ba588d192c77758939a6e340adf07edd4f` (tag `v1.10.1`). Every measured
process binds all **83 tracked package files** and all **83 installed package
files** to SHA256. The independent oracle is actual binary Torch
**2.11.0+cu128**, with its extension path/hash and interpreter PID recorded.
The candidate identifies the worktree's Jittor frontend, rather than borrowing
the oracle's tensor implementation.

CPU explicitly hides CUDA and sets canonical/legacy CUDA flags to zero.
The machine reports AMD EPYC 9554, Linux x86_64 and GCC 11.5. CUDA executes on
two RTX 4090 devices, with driver 580.178.04; candidate CUDA is the pinned
12.2 wheel/toolchain path, NVCC 12.2.140. The oracle's CUDA build is 12.8.
Installed wheel metadata is recorded separately and is not a claim that every
installed CUDA library was loaded. Actual CUDA UUID, logical ordinal and tensor
pointer device are checked before copying observations to host arrays.
Other users had concurrent GPU workloads, so these tiny correctness runs make
no throughput, latency, memory-peak or L4 claim.

Each precision/input seed uses a fresh launcher, with serial warm-up and
separate rank compilation caches. Candidate world-two runs use the public
`python -m jittor.distributed.launch --standalone --backend=nccl` entrypoint.
It is the same public owner registered as `jtrun =
"jittor.distributed.launch:main"` in `pyproject.toml`; these measurements retain
the actual module command and `[jtrun]` logs, rather than claiming an unexecuted
console argv. No independent `mpirun` training launcher was used. Oracle ranks
use `torch.distributed.run`; both CUDA runtimes report actual NCCL.

## Fixed protocols and independent results

All weights are explicitly overwritten with the maintained deterministic
initial values. Seed zero supplies the baseline; seed seven perturbs the
actual input with NumPy-generated Gaussian noise while preserving initial
weights. AdamW has two groups, learning rates 0.01/0.015, betas (0.7, 0.93),
epsilon 0.0001 and weight decay 0.02. World-one additionally checks StepLR.
Each process executes three updates before save, two expected updates and two
updates after load. All input/output/loss, full input/parameter gradients,
model values, Adam moments/steps and declared hyperparameters are captured.
The two resumed steps must equal the expected steps exactly within each runtime.
Python, NumPy, CPU Torch and device Torch RNG streams must also restore exactly.

The independent comparison retains the original elementwise budget
`abs(candidate-reference) <= 2e-6 + 2e-5 * abs(reference)`. It compares every
captured array; no BF16-specific relaxation was added. The normalized error
below is `max(abs(error)/(abs(reference)+0.1))`, consistent with that budget.

| Actual protocol | Array pairs | Parameter / input gradient tensors | Max absolute error | Max normalized error | Independent depth | Paired admission |
| --- | ---: | --- | ---: | ---: | --- | --- |
| CPU world1 FP32, Linear 3→2 | 438 | 28/28; 14/14 | 5.960465e-8 | 1.774503e-7 | L3 | accepted |
| CUDA world1 FP32, Linear 3→2 | 438 | 28/28; 14/14 | 2.980232e-8 | 1.986478e-7 | L3 | accepted |
| CUDA world2 FSDP2 FULL FP32, Linear 4→3 | 820 | 56/56; 28/28 | 1.490116e-8 | 8.923406e-8 | L3 | blocked, level null |
| CUDA world2 FSDP2 FULL BF16, Linear 4→3 | 820 | 56/56; 28/28 | 1.490116e-8 | 6.489249e-8 | L3 | blocked, level null |

These counts include both seeds and every rank, including repeated resumed
observations. The two runtimes each pass ten CUDA rank tests and two CPU tests:
**24 actual passed tests**, with no skipped required node. All candidate records
read back fallback policy `error` and fallback count zero.

The BF16 observer executes inside Linear before Accelerate's FP32 output
wrapper: all seven forwards per process observe actual BF16 input, weight,
bias and output on CUDA. FP32 master parameters, full gradients and Adam
moments remain FP32 on CUDA. Native and Torch local shard layouts may differ;
the validator checks each rank's owned element count and complete global
weight/bias coverage, rather than equating different legitimate layouts.
This is real FSDP2 parameter sharding, not an I/O-only checkpoint partition.

## CPU pairing and preserved failures

The same-source CPU world-two diagnostic initializes real Torch/Gloo on both
ranks, but Accelerate does not select FSDP2: both ranks report
`CPU world-two Accelerator did not use FSDP2`. MULTI_CPU/DDP is not a substitute
for this protocol. A separate candidate direct-init diagnostic stops at its
missing dynamic bootstrap; it does not establish failure of the native MPI
provider or test a complete candidate CPU two-rank training run.

Consequently CPU world1 only pairs with CUDA world1. The shared auditor
actually validates all four measured rows, accepts the world-one pair, and
places the independently verified world-two rows in `blocked`, retaining
`independent_verified_level=3` but setting admitted `level` and
`verified_level` to null. No registry numerator or denominator is invented.

Earlier immutable evidence remains under `formal-ee51` and `formal-297`.
The first warm-up used size-based `torch.ones(requires_grad=True)`, whose
candidate input gradient was absent, while the maintained checkpoint test's
explicit `torch.tensor` input did receive gradients. This is the existing
KI-COMPAT-004 boundary; the tool now uses the maintained tensor factory and
keeps explicit leaf/gradient assertions. It does not fix or hide the ones case.

The 297 warm worker also reset the assigned rank-one cache to rank zero.
Two FP32 launchers therefore timed out waiting for a still-compiling rank.
The fix passes an explicit warm cache rank without pretending to initialize a
distributed communicator. New clean-source CPU/CUDA measurements are entirely
fresh; older raw observations are not relabeled. A validator mismatch between
the distributed saved training observation and its model/optimizer-only
immediate restore snapshot was fixed by comparing exactly those restore fields;
all complete continuation and RNG checks remain strict and separate.

## Reproduction and raw bindings

Use private interpreters with matching Accelerate and the pinned official
checkout; install nothing into a shared environment. From the clean measured
source, prepare the CUDA toolchain/library environment, then run:

```bash
python tools/run_accelerate_gate.py --device cpu \
  --jittor-python <private-candidate-cpu-python> --oracle-python <private-binary-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cpu" \
  --cache-root <external-cpu-benchmark-cache>
python tools/run_accelerate_gate.py --device cuda --devices <gpu0>,<gpu1> \
  --jittor-python <private-candidate-cuda-python> --oracle-python <private-binary-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cuda" \
  --cache-root <external-cuda-benchmark-cache>
python agent/skills/torch-compat-repo-runbook/scripts/collect_accelerate_scorecards.py \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cpu" \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cuda" \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --baseline b7e5a07b32f300e6f348a2147e7fafd5f4704c96 \
  --report docs/results/2026-10-15-accelerate-protocol-scorecards.md \
  --command '<exact CPU and CUDA measurement commands>' --output <external-scorecard-directory>
```

Maintained nodes are the Accelerate method in `test_optimizer_checkpoint_state.py`
and `test_fsdp2_full_model_optimizer_checkpoint_resume[no|bf16]` in
`test_accelerate_distributed_integration.py`. `nox -s ecosystem_accelerate`
forwards gate arguments; `benchmark_accelerate` remains a separate performance gate.

Unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/accelerate-protocol/20261015/`:
`formal-b7e5/{cpu,cuda}/execution.json`, per-rank JSON/NPZ/JUnit and logs,
`formal-b7e5/cpu-provider/`, `scorecard-b7e5/`, full raw SHA256 manifest and
`registry-backward-compatibility-b7e5.json`. The latter compares **52 historical
registry records** against the pre-protocol auditor, with identical full output.
`source-b7e5.bundle` preserves the measured source commits; its prerequisites
are the published `0751e425b0f197b8e7b388f218613378c71f701a` and
`282783a3dfcd3521400869beadf56c6f0fe9a0f9` commits. `raw-bindings-b7e5.json`
binds measurements, actual commands, collector output, logs and this bundle.
Every scorecard recomputes clean upstream provenance and actual numerical
comparisons. Historical eight-rank checkpoint JSON is not relabeled or repaired.

This validates restoration into the same prepared model/optimizer objects;
fresh launchers do not prove fresh-object or fresh-process checkpoint restore.
The existing maintained node saves, trains, loads and resumes inside one worker.
It offers no separate producer/consumer launch phases: a later fresh-process
check must persist the actual checkpoint, terminate producer ranks, construct
new model/optimizer objects in new consumer ranks and compare the complete
resumed trajectory. This batch does not close that training-resume requirement.
Multi-host,
SHARDED/DCP planners, checkpoint interchange between runtimes, arbitrary
models, full Accelerate coverage, ROCm and NPU remain outside this result.

## Repository verification

Targeted protocol/registry negative tests and child-process contracts pass
**93 tests**. CPU `tools/run_test_suite.py --tier core --backend cpu` passes
**322 tests**, with 42 skips and one expected failure, using separate unit
caches after the benchmark processes terminate. Layout, Ruff and whitespace
checks pass.

The final full structure check records **1503 passed, 6 skipped, 2 failed**.
The two failures are the central results toctree entry for this new report and
generated `MANIFEST.in` entries for the report/tool; those files are owned by
the integrating maintainer and deliberately left for integration. No complete
structure-green claim is made for this isolated worktree.

## Main-tree integration verification

Integrated onto synchronized `d404c7027317c0200d648cb7b517bf578f9ec0f4`.
The original clean measurement remains `b7e5a07b32f300e6f348a2147e7fafd5f4704c96`.
An independent audit rehashes all 138 sealed raw artifacts, checks the twelve
delivered source/report hashes, and reruns the collector both from the archived
delivery and the integrated tree. Both recompute all 2,516 array pairs and give
two accepted, two blocked and zero rejected records, with no registry coverage.
Blocked world-two records retain null admitted levels despite independent L3
numerical agreement.

The integrated tree passes 111 scorecard/registry/Diffusers tool contracts,
actual CPU checkpoint contracts (seven nodes, zero skips), actual CUDA
checkpoint contracts (seven nodes, zero skips), and full structure (1,505 passed,
six skipped). The CPU checkpoint and final protocol-tool run together passes 39
nodes. Ruff for changed Python files and repository layout pass. Central report
and manifest registration resolve the isolated worktree's two structure errors.

Integration also corrects the reusable runner to give each runtime/device/rank
its own external `HOME`, alongside its existing independent `JITTOR_HOME` and
`TMPDIR`; the warm-rank contract verifies those directories. The original b7
runner inherited `HOME`, and its sealed observations did not capture that value.
Its actual historical `HOME` isolation is therefore unverified and is not filled
in retrospectively. Original source and numerical identities remain unchanged;
the runner correction is an integration change for subsequent measurements.

Unversioned independent and integration logs are under
`$JITTOR_LAB_ROOT/_state/accelerate-protocol/20261015/`, using the
`root-independent-*` and `root-main-*` prefixes. These checks do not extend the
documented same-object restoration scope or supply a missing CPU FSDP2 partner.
