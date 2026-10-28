# Native FSDP CUDA training and fresh-process continuation

- Status: verified bounded FP32 correctness; BF16 passes the fixed protocol and
  exact continuation, with the small-gradient numerical limitation below.
- Measured source: `ee24fac2be1e81ab6a2a14f32c869d6d17b600b1`, clean for the final
  independent oracle and all three launcher phases.
- Sync baseline: remote `cf8d2b6a961090c2b6ffd9f2139a6d21294108fc`, integrated as
  `7277b26f0a30ec43423c2914cd648a6d39a025d6` before the source commit.
- Date: 2026-10-15. Owner: Jittor core maintainers.
- Review when: FSDP storage, Module references, native backward, checkpoint,
  collective, RNG, Linear/BF16 or precision-policy ownership changes.
- Scope: CUDA/NCCL WORLD, two real ranks, fixed world size, native SGD/AdamW,
  FP32 master shards and either FP32 or BF16 forward. This does not close every
  native-FSDP requirement or establish performance or another backend.

The [implementation note](../notes/native-fsdp.en.md) describes ownership and API
limits. Native execution imports neither Torch nor `jittor.compat`; compatibility
FSDP delegates its existing sharding mathematics to the same native owner.

## Fixture, platform and device proof

The fixture is a real native `Linear(5, 7) -> tanh -> Dropout -> Linear(7, 3)`
module with a persistent gain buffer, four parameters and 66 complete parameter
elements. NumPy supplies fixed weights and rank-dependent `(3, 5)` inputs and
`(3, 3)` targets. Each rank owns 35 FP32 parameter elements, including collective
padding; checkpoint intervals contain only the 66 valid elements across ranks.
SGD uses momentum 0.9; AdamW uses betas `(0.85, 0.95)` and epsilon `1e-8`.
Two parameter groups use learning rates 0.013/0.007 and decay 0.02/0.

The native environment used Python 3.11.16, NumPy 1.26.4, CUDA 12.2.140,
NCCL 2.18.3 and two NVIDIA RTX 4090 devices. The independent interpreter used
real binary Torch 2.11.0+cu128 on CUDA, rather than the shim. Its metadata records
the actual `_C` binary, version and raw-data hash. Torch TF32 is disabled and
matmul precision is `highest`; native effective matmul and cuDNN tiers are both
asserted and recorded as `highest`. CUDA kernel compilation used strict math.

Every observed arithmetic/collective output is synchronized and must already
reside on CUDA before requesting its device pointer. Driver pointer attributes
then verify the actual logical CUDA device. The before/after observer probe
established that explicit `.numpy()` trace I/O can move the same Var's storage
to host without changing CUDA dispatch. Only a collective input carrying that
Var's own trace-readback tag may be uploaded again; unexplained host residency
fails. This does not grant CPU operator fallback. All final trajectories record
zero backend fallback and no frontend import. Existing unrelated GPU occupancy
means this run supports correctness conclusions only.

Actual Linear input and gathered weight dtypes are observed on every step.
BF16 runs use BF16 all-gathers and forward operands, FP32 reduce-scatter outputs,
FP32 local gradients, masters and moments. Forward policy does not change global
autocast or copy the optimizer update algorithms.

## Validation results

| Check | Result | Boundary |
| --- | --- | --- |
| Native FSDP + adjacent TP + ZeRO CPU tests | 51 passed, zero skipped | Clean measured source; 18 FSDP contracts |
| CPU core, native and Torch modes | 322 passed, 42 skipped, 1 expected failure | Precommit runtime bytes; explicit CUDA disabled |
| Complete structure gate | 1472 passed, 4 skipped | Synced runtime before final trace assertions; zero failures |
| Native CUDA parity | 7 passed per rank, zero skipped | Four optimizer/precision trajectories and three distributed negatives |
| Native CUDA save + live continuation | 4 passed per rank, zero skipped | Three steps, checkpoint, two further steps with real dropout |
| Fresh launcher restore | 4 passed per rank, zero skipped | Separate process group; last two steps exactly equal |
| Compatibility FSDP CPU ownership/flat-policy/math | 24 passed | Existing adjacent CPU coverage |
| Existing compatibility FSDP CUDA suite | 6 passed, 1 skipped, 2 failed per rank | Both failures also reproduced before these changes |
| Ruff, whitespace and repository layout | Passed | Owned files; no central registry edits |

The complete structure run includes a declared accelerator-dependent file with
zero collected cases and four declared/backend skips; it is not accelerator
acceptance. The compatibility CUDA skip requires at least four ranks and is not
covered by this two-rank run.

The independent oracle computes the dense two-rank mean loss, every parameter
gradient, rank-specific input gradient, output, loss, updated parameter and
optimizer moment for five steps. The final artifact audit also checks exact
expected key sets against that oracle. Four references contain 456 arrays;
the 24 candidate NPZ files contain 1,968 arrays across the three phases.

The save/restore fixture enables actual dropout with probability 0.25 and records
its input and output. Every phase verifies nontrivial dropped values. A new
launcher reconstructs new model and optimizer objects, loads the checkpoint and
executes steps four and five. All **368** resulting arrays equal uninterrupted
training exactly, including dropout output, input and parameter gradients,
parameters, moments and per-rank Python, NumPy, native CPU and CUDA RNG streams.
The RNG streams differ across ranks, so reproducing rank zero on every rank
would fail. This is stronger than loading into another model in the same process.
The instrumented phases execute 384 all-gather and 384 reduce-scatter local calls
across the two ranks.

Checkpoint manifests describe genuine model/moment shard intervals; the gain
buffer remains replicated. Model/optimizer/layout are not enclosed in a
whole-tree `RankLocal`. Distributed negatives reject inconsistent initial
parameters, optimizer controls and a replicated gain buffer changed on one rank.
CPU negatives cover stale/partial optimizers, graph/lifecycle misuse, invalid
layout and required-CUDA failure instead of skip.

## Fixed numerical protocol and BF16 limit

All comparisons require exact shapes and finite values. Elementwise tolerances
are `atol=2e-6, rtol=2e-5` for FP32 and `atol=4e-4, rtol=3e-2` for BF16.
Per-parameter step deltas additionally require L2 error at most
`2e-6 + 2e-5 * oracle_norm` for FP32 or `1e-6 + 0.03 * oracle_norm` for BF16.
An oracle update no larger than ten absolute floors is rejected as uninformative.
These thresholds are fixed before the shared Linear numerical repair and must
remain unchanged for that comparison.

Every actual parameter/step injects a zero gradient and a zero update and requires
rejection: **160 gradient and 160 update negatives**, each split evenly between
FP32 and BF16. Two host tests additionally zero only one parameter while the
other remains correct, and reject an update too small to distinguish from the
absolute floor. All per-step norms, errors, floors and budgets are retained.

| Optimizer / computation | Smallest oracle update L2 norm | Largest absolute-floor share | Largest update relative L2 error | Largest gradient relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| SGD / FP32 | 1.41889e-4 | 1.410% | 0 | 1.542e-7 |
| SGD / BF16 | 1.41218e-4 | 0.7082% | 0.5229% | 0.8657% |
| AdamW / FP32 | 4.45890e-3 | 0.04486% | 1.121e-6 | 1.104e-6 |
| AdamW / BF16 | 4.46419e-3 | 0.02241% | 0.5804% | **11.2242%** |

The BF16 AdamW maximum occurs on rank zero at `step/4/gradient/fc1.bias`:
oracle L2 norm `0.001057407702319324`, difference L2 norm
`0.00011868552246596664`. Its small coordinates pass the stated absolute
per-element tolerance. This is **not exact gradient agreement**, despite the
passing gate and exact same-runtime continuation. The largest BF16 AdamW
parameter absolute difference is approximately `2.7663e-4`.
The shared Linear/Lt precision investigation must be replayed with these same
inputs and thresholds before treating BF16 numerical agreement as final.

## Existing compatibility failures and preserved controls

The original compatibility suite fails its immediate old-shard weak-lifetime
assertion in `test_mesh_reordered_shards_custom_adam_and_lifetime`, after the
numerical checks pass. `test_nested_sharding_and_full_state_reload` reaches
`set_model_state_dict` with a native Module and raises because native
`load_state_dict` does not accept the Torch `strict` keyword.

Both failures were reproduced on two CUDA ranks after temporarily reversing
**all nine tracked runtime/test/skill changes** from this batch, leaving the
synced prechange runtime. The same two nodes fail on both ranks, with logs under
`logs/compat-prechange`. Reapplying the saved patch restored the FSDP work.
They remain identified existing failures in this source; follow-up repairs must
retain the prechange evidence and validate the shared Module/loading owners.

An earlier structure run had five failures: one duplicated helper introduced by
the lowering, two missing-xdist-plugin environment failures, and two TP central
document/manifest registrations. The helper now delegates, the environment loads
xdist explicitly, and syncing the remote registrations removed the latter two.
The subsequent complete structure run above is green. Earlier assertion-API and
regex mistakes in the new checkpoint negative are retained in failed logs and
were corrected without altering native tensor APIs or checkpoint validation.

## Reproduction and artifact binding

Use the [distributed verification skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)
with separate native and binary-Torch interpreters, private per-rank caches and
serial first compilation. CPU validation explicitly hides CUDA. The reusable
oracle tool is:

```bash
"$TORCH_PYTHON" agent/skills/jittor-distributed-verification/native_fsdp_oracle.py \
  --source-root "$PWD" --device cuda --kind sgd --dtype float32 \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/oracle"
```

Generate both optimizers and both dtypes. The native maintained test node is
`tests/backends/comm/nccl/test_native_fsdp_nccl.py`. Set
`JITTOR_REQUIRE_NATIVE_FSDP=1`, `JITTOR_TEST_REQUIRE_EXECUTION=1`,
`JITTOR_NATIVE_FSDP_TEST_DIR` and `JITTOR_NATIVE_FSDP_ORACLE_DIR`. Run the
native launcher on two assigned GPUs in three distinct invocations with
`JITTOR_NATIVE_FSDP_PHASE=parity`, then `baseline`, then `resume`; select
`-k test_oracle_and_fresh_launcher_resume` for the latter two phases. Every child
must set its private `HOME`, `TMPDIR` and `JITTOR_HOME` before importing Jittor.

Raw data and exact rank selectors/environments are external, under
`$JITTOR_LAB_ROOT/_state/native-fsdp/20261015/`. The final artifact directory is
`artifacts/clean-ee24fac2b`; `SOURCE.json` binds all 22 changed source files.
`AUDIT.json` independently rechecks complete key sets, numerical metrics, zero
injections, shard roles, dtype/device metadata and exact continuation.
`SHA256SUMS.json` binds 236 artifact files, 13 logs and five runner/environment
files. Its SHA-256 is:

```text
9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27
```

The earlier `formal-dirty-v3`, observer probe, failed logs and prechange control
remain separate. Final raw tensor data was not rewritten to correct metadata.
A future run must create a new artifact directory and source binding.

## Main integration checks

Integration over `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e` rehashed all
236 artifacts, 13 logs, five runner/environment files and the 22-file measured
source binding. Twenty files match that source exactly. Two small integration
changes affect host logic: optimizer managed-parameter identifiers are combined
once rather than rebuilt for every parameter, and FSDP checkpoint tensor-list
lengths are checked before encoding/decoding can truncate them with `zip`.
No sharding, gradient reduction or optimizer update mathematics changed.

The list-length defect was reproduced with six real malformed checkpoints,
covering gradients, first moments and second moments for ordinary and sharded
models. All six were wrongly accepted before the fix. Afterward, all six loads
raise before mutating target parameters, flags, optimizer learning rate/step or
native CPU RNG. Three additional malformed FSDP saves also refuse without a
complete manifest. The original CUDA evidence above remains tied to `ee24fac2b`;
it is not relabeled as a new device measurement of this host validation change.

Main integration CPU owner tests passed 78 cases without skips. The nine
corruption checks passed separately (six overlap with those 78). CPU core passed
322 tests with 42 skips and one existing xfail; adjacent FSDP math/flat-policy/
structure checks passed 24 tests. Complete Torch-mode structure passed 1470
tests with six skips. The additional two skips relative to the earlier report
are package-resolution checks in isolated interpreters without an installed
Jittor distribution. Documentation indices, generated manifest, layout and diff
checks were also completed. Raw integration evidence is unversioned under
`$JITTOR_LAB_ROOT/_state/native-fsdp-integration/`, including `artifact-audit.json`,
`source-equivalence.json`, `extra-list-before.log`, `extra-list-after.log`,
`cpu-owners.log`, `core.log`, `compat-owners.log` and `structure.log`.
