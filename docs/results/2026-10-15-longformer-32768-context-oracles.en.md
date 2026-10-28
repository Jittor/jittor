# Real 32768-token Longformer on CPU and CUDA

- Status: Reproduced full-context encoder correctness and three-step full-parameter training; tiny width, no performance grade.
- Measurement baseline: `cbe154c3b81b3f2cca65363e0bdc6bdcec60c191`, clean for every numerical and pytest child execution; includes the Longformer builder, explicit complete-gradient contract and opt-in test node.
- Upstream: clean Transformers `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`, release `4.56.2`.
- Maintainer: Jittor core / Torch compatibility maintainers.
- Review when: Longformer attention/position implementation, model parameter registration, downstream dependencies, compatibility tensor semantics or precision policy changes.

## Exact model and attention scope

The same builder executes the genuine upstream `LongformerModel` in independent binary PyTorch and
Jittor interpreters. It has one layer, hidden size 32, four heads, intermediate size 64, vocabulary 64,
batch 1, local attention window 256, maximum position embeddings 32770 and zero dropout. Inputs have
**32768 positions**, all valid, and the first token has actual global attention. No sequence shortening,
model download, adapter, copied attention algorithm, mock, no-op or downstream source patch is used.

The public `get_input_embeddings()` computes word embeddings, then a float noise tensor scaled by
0.01 is added and passed through the real `inputs_embeds` path. This preserves both word-embedding
parameter gradients and a differentiable input. Independent oracle observation confirms actual
position IDs 2 through 32769 and a `[1,1,32]` global-query input; read-only hooks preserve the upstream
computation. Both the complete `[1,32768,32]` hidden state and the real `[1,32]` pooler participate in the loss;
the returned flattened output contains 1048608 values. Local and global query/key/value parameters
and pooler parameters are explicitly included in the complete gradient contract.

Each comparison requires the output, **all 29 parameter gradients** and the `embedding_noise` input
gradient: 31 exact-shaped finite arrays on both sides. There are 29 trainable parameters and none
frozen. Model-width/layer sampling remains `shape_kind=tiny` despite the genuine 32768-token length.
The established scope is this full-parameter encoder scenario, not dense 32K attention, generation,
a production-sized pretrained checkpoint or every Longformer task head/API.

## Fixed platform and protocol

| Field | CPU | CUDA |
| --- | --- | --- |
| Oracle | Genuine binary PyTorch 2.11.0+cu128 | same, CUDA 12.8 |
| Candidate | Jittor 2.0.0 Torch compatibility layer | same, NVCC 12.2.140 |
| Dependencies | NumPy 1.26.4; Transformers 4.56.2 from the pinned source | same |
| Hardware | AMD EPYC 9554, accelerator visibility empty | one allocated NVIDIA RTX 4090, driver 580.178.04 |
| Precision | TF32 matmul/cuDNN OFF, matmul highest | TF32 matmul/cuDNN ON, matmul high |
| Other controls | float32, cuDNN benchmark OFF; two runtime threads | same; real CUDA dispatch, zero fallback |

Both interpreters load the identical pinned upstream checkout. Base seed 0 exports the complete
reference weights. Independent input seeds 7 and 19 reuse that same weights hash on both runtimes;
outputs must change and all gradient arrays must remain present. A separate public `training_probe.py`
run uses fresh seed-0 reference weights, SGD learning rate 0.001 and three deterministic input seeds
1, 2 and 3. Every step saves output, scalar loss, all gradients and every updated parameter: **183
arrays** total. This compares an update trajectory, not convergence.

The existing ecosystem harness defines normalized divergence as `maximum_absolute_error /
max(field_maximum_absolute_reference, 1e-3 * block_maximum_absolute_reference, 1e-6)`;
the gradient block contains all requested gradients, and the output has its own reference field. The report keeps
absolute errors as well as that exact normalized protocol. CPU limits are 2e-3 for outputs and 1e-2
for gradients; CUDA limits are 5e-3 and 2e-2. Updated parameters/loss use the output limit with their
own reference-array floor. These are the existing protocol thresholds, not post-hoc tuned tolerances.

## Results and evidence

| Device | Seed 0/7/19 arrays per seed | Max absolute / normalized error | Three-step arrays | Max absolute / normalized error |
| --- | --- | --- | --- | --- |
| CPU | 31 | 0.0166015625 / 8.67267123e-06 | 183 | 9.53674316e-07 / 9.84836796e-06 |
| CUDA | 31 | 0.078125 / 2.50184426e-05 | 183 | 1.43647194e-05 / 2.73647853e-05 |

All comparisons are finite and exact-shaped, with matching key sets and complete requested gradients.
Both reference and candidate update 28 parameters across three steps.
`key_global.weight` retains the same float32 values in this protocol; its full gradient and every
parameter snapshot remain checked. Mathematically near-zero attention-key bias gradients are also
retained rather than removed from the contract. Reference scalar loss trajectories are CPU: -0.000152084525, -0.00200377125, -0.000821642869; CUDA: -0.000152087901, -0.00200377079, -0.00082164153. The complete candidate losses and per-array errors are stored in the raw comparison.

- Actual opt-in gate: CPU **1 passed**, CUDA **1 passed**, zero skipped. CPU/CUDA JUnit node times are 297.888/454.044 seconds, including first compilation in distinct unit caches.
- Public scorecard: **two accepted L3 records, zero rejected**; 84 SHA256 artifact bindings independently checked.
- Host case / audit contracts: **37 passed**. CPU core: **322 passed, 42 skipped, 1 xfailed**.
- Full structure before adding this report: **1451 passed, 6 skipped, zero failed**. Lint, diff and layout passed.
- A false CUDA OFF/highest label is rejected with `Platform/measurement precision mismatch`; CPU remains accepted.

Resource diagnostics came from the earlier identical full-context prototype and retain those original
source baselines. Child-rusage maximum RSS values were about 1.93/2.22 GiB for Torch/Jittor CPU, and the
CUDA root-process sampled GPU values were 1950/2334 MiB for Torch/Jittor. These are diagnostic samples
and child rusage, not continuous whole-process GPU peaks. Formal CUDA allocation samples also show
other processes. The single-repeat benchmark times are retained only in raw verification JSON; neither
the sampling nor tiny-width timing establishes L4/L5 performance or resource acceptance.

The opt-in node is
`compat/tests/torch/test_ecosystem_longformer.py::EcosystemLongformer::test_transformers_longformer_32768`.
It requires actual execution, an independent Torch oracle, Transformers and a single explicit device.
Selecting the node with missing dependencies/oracle or mixed CPU/CUDA configuration fails. A speed
ratio request is rejected because this tiny-width case does not establish performance. Normal
unselected runs skip this expensive case visibly. The formal runs use distinct unit-test caches;
benchmark caches are reused only by benchmark/probe tools. Each real node is run directly under
pytest; no nox wrapper or scheduled CI invocation is claimed.

## Automatic denominator and limits

The public inventory regenerates Transformers model-family and task registries from the clean pinned
SHA. There are **374 model families** and **46 task registrations**. The scorecard accepts one named
Longformer full-parameter scenario on CPU and its matching CUDA scenario at **L3**: separately
**1/374** on the model-family axis; **0/46** on the task axis. Every verified row retains the concrete
case, full registry refs, tiny shape and all trainable parameter names. No other library/family, NPU,
ROCm, Corex, L4 speed or L5 resource result is established by this report.

Raw evidence is external at
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/longformer-32768-20261015/formal-cbe154/`.
Its schema-2 evidence document hashes every raw array, real platform declaration, comparison,
training metadata, command, JUnit result, inventory and source identities. A negative audit copies the
CUDA platform label, falsely changes actual TF32 ON/high to OFF/highest and recomputes that copied
artifact hash. The CUDA record must be rejected while the untouched CPU record is accepted; original
measurement metadata and tensors remain unchanged.

Key sealed artifact digests:

| External artifact | SHA256 |
| --- | --- |
| `scorecard-evidence.json` | `bd4f0a68f578be23bb6e2ce4016aa14b26b307ea7a07e035e1d06ea4c8406274` |
| `scorecard.json` | `a8a77219533559a151efe6cbe62a3634dff6d32196f4402394dbd6f8622f131c` |
| `transformers-inventory.json` | `f4efa7d5e6437ef219e69fa90287b88c4d1d7bee50228eff7df191251a5dc8da` |
| `negative-precision-scorecard.json` | `eb3534f437fa85c1adcbe183216f5d8b19215e120535a3664c4bcf73b317777f` |

## Reproduction

With independent provisioned NumPy-1/Transformers-4 interpreters and separate state/cache roots:

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 \
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases transformers_longformer_32768 \
  --oracle-python "$REAL_TORCH_PYTHON" --shim-python "$SHIM_PYTHON" \
  --device cpu --seed 0 --repeats 1 --no-memory --out "$RESULT_ROOT/cpu"
```

Repeat on allocated CUDA with `--device cuda`. CPU must explicitly mask accelerators and disable CUDA.
The actual precision/version/import-origin blocks must match between runtimes. For perturbations,
invoke public `_ecosystem_runner.py` at seeds 7 and 19 with the base `.ref.weights.npz` passed to both
runtimes. Invoke public `training_probe.py` with `--steps 3 --learning-rate 0.001`, passing the fresh
training reference weights to the candidate. Saved external command artifacts contain the complete
actual invocations and flags.

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_HOME="$UNIT_JITTOR_HOME" \
JITTOR_TORCH_CACHE_ROOT="$UNIT_SHIM_CACHE_ROOT" \
JITTOR_ECOSYSTEM_LONGFORMER_32K=1 JITTOR_REQUIRE_TRANSFORMERS=1 \
JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TEST_REQUIRE_EXECUTION=1 \
JITTOR_TEST_DEVICES=cpu JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
"$SHIM_PYTHON" -m pytest -q compat/tests/torch/test_ecosystem_longformer.py

python agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --upstream "transformers=$TRANSFORMERS_CHECKOUT" \
  --evidence "$RESULT_ROOT/scorecard-evidence.json" \
  --baseline cbe154c3b81b3f2cca65363e0bdc6bdcec60c191 \
  --output "$RESULT_ROOT/scorecard.json"
```

For the CUDA node set the single allocated device and `JITTOR_TEST_DEVICES=cuda`; use the same real
oracle, explicit required flags and an independent unit-test cache. The saved source hashes and
full commit bind this report to its measured implementation.

## Main-branch integration

Integrated on parent `23670917490c36f790e1bc24ec136cb1a5cb2230`, including the result
index and generated manifest registrations. The numerical records retain their clean
`cbe154c3` measurement baseline. Independent integration checks rehashed all 84 artifact
bindings, passed 37 host coverage/admission tests, core 322 passed / 42 skipped / one
existing xfail, and complete structure 1453 passed / 6 skipped / zero failed. The earlier
two central documentation registration failures are resolved. Integration logs are
unversioned under `$JITTOR_LAB_ROOT/_state/frozen-view-fix/longformer-*.log`.
