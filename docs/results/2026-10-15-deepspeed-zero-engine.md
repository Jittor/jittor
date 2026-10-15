# DeepSpeed Engine: bounded FP32 training and checkpoint acceptance

- Status: verified single-rank FP32 Engine paths; **R14 remains incomplete** pending
  actual multi-rank bf16 partition/reduction and checkpoint/resume acceptance.
- Date: 2026-10-15.
- Baselines: owner fixes `4bf21c9fe` / `a0c667538`, adapter/gate `67712827a`,
  synchronized tree `e082cf0a38aa195dc07c860231a454a097d02a68`
  (including remote `2b16004e3`), plus this report's commit.
- Owner: native autograd/Module, Torch compatibility and downstream adapter maintainers.
- Review when: native gradient formation/holder ownership, Module registration,
  optimizer parameter identity, DeepSpeed pinned source, or compat public
  `gradient_task.participating_tensors`, `owned_runtime_hook`, `transaction` or
  `module_patcher` changes.

This is a dedicated real `DeepSpeedEngine` gate, rather than an ordinary model
runner using an external optimizer. Official DeepSpeed `v0.19.7` is pinned to
`f5af15c2b4da10411e340850c00ba0cb5b14fd87` from
[the upstream repository](https://github.com/deepspeedai/DeepSpeed).
Both interpreters use the same source-built
`deepspeed-0.19.7+f5af15c-py3-none-any.whl`, SHA256
`467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`.
The oracle is genuine binary PyTorch 2.11.0+cu128 with its `_C` extension verified,
not a shim. Adapter admission checks the source version and six dependent file
hashes; unknown versions, changed source and activation after import fail closed.

## Verified protocol

The matrix is **12 / 12 explicit Engine configurations on CPU and 12 / 12 on
CUDA @ upstream `f5af15c2b4da10411e340850c00ba0cb5b14fd87`**:
ZeRO 0/1/2 × Adam/AdamW × input perturbation 0/0.125. This denominator is the
declared Engine test matrix, not a percentage of all DeepSpeed APIs or models.
Each configuration runs both independent interpreters on the requested device.

The deterministic residual model uses Linear(8,16), GELU, Linear(16,8), residual
addition, LayerNorm(8), and Linear(8,4), with identical explicit weights. Batch 2,
FP32, GAS 2 and clipping 1.0 produce three optimizer updates from six microsteps.
The actual Engine calls `backward`, `step`, `save_checkpoint`, and
`load_checkpoint` into a fresh Engine after update two. Resumed microsteps must
reproduce the uninterrupted output/loss/parameter trajectory within each runtime
at `rtol=2e-6`, `atol=2e-7`, including global step and client-state restoration.

All eight parameters' incoming gradients and both input gradients are mandatory
on every microstep. Accumulated full parameter gradients are also compared;
ZeRO 2 exposes its public full-gradient getter only at GAS boundaries. Missing
gradients, shape/schema differences, nonfinite observations, residual Engine
graph registrations, optimizer groups, restored moments/step, version/device or
execution-condition mismatches fail. The shared comparison uses separate
forward/state and gradient reference floors, with normalized tolerances `2e-3`
and `1e-2`; these are correctness thresholds, not performance budgets.

CUDA materializes and checks the actual input/parameter/output/gradient residency
under the error fallback policy, with zero backend-fallback attempts. CPU hides
all GPUs. Each runtime initializes its declared Gloo/NCCL world-size-one process
group; singleton operations do not exercise communication with another rank.
World-size-one ZeRO does **not** prove cross-rank partitioning or reduction.

## Fixes and adapter boundary

The earlier leaf-hook counterexample was resolved in the native gradient owner.
Hooks now run after a Var's complete gradient is formed, retain leaf identity,
validate replacement shape/dtype/device, survive parameter data rebinding and
honor removable-handle lifetimes. Independent Torch tests cover leaf/nonleaf
targets, loss hooks, deleted intermediate wrappers, unused inputs, repeated
backward, failures and simple closed-leaf closure collection. A hook closure
capturing a forward graph may retain that graph in both runtimes; remove its
handle explicitly. This is not a claim of general graph-cycle collection.

The public post-accumulate entry reuses the existing gradient/optimizer storage.
Its fidelity remains approximate: all gradient storage is published before
observers run, so cross-leaf observer scheduling differs from PyTorch worker
interleaving. Final callbacks drain FIFO after accumulation. The public
participation query reports actual current native backward participants only
inside post-accumulate execution; it does not expose fake worker graph IDs or
infer participants from stale `.grad`. Optimizer parameter replacement/reordering
rebinds the existing Adam state owner without adding another update algorithm.
The live native Module registration view remains the sole child storage owner.

The relocatable adapter handles only pinned library-private imports and selection:
DeepCompile eager imports are isolated and compile/elastic requests explicitly
rejected; hook selection uses real public post-accumulate behavior, and the
library's private worker-node count uses the public participation query. A CPU
Torch-ABI shared-memory extension unavailable to native Jittor returns upstream's
capability `None`, selecting genuine public Gloo communication. Binary Torch's
CPU reference builds and runs its actual extension. No fake Inductor/SymPy/private
Torch namespace or no-op training/communication implementation is supplied.
`GradBucket` and `Adagrad` remain importable but construction explicitly rejects;
they do not count as implemented functionality.

## Reproduction and regression

The reusable [DeepSpeed runbook](../../agent/skills/deepspeed-torch-compat/SKILL.md)
describes isolated package installation and the dedicated `nox -s
ecosystem_deepspeed` entry. It requires an independent oracle and actual dependency
execution; absent dependencies cannot turn into acceptance skips. This dedicated
Engine gate is not a fabricated entry in the shared ordinary-model `CASES` count.
Scheduled nightly registration is a separate integration step; this report does
not claim it has run automatically.

Platform: Linux x86_64, Python 3.11.16, NumPy 1.26.4, pytest 7.4.4, g++ 11.5,
RTX 4090/SM89, NVIDIA driver 580.178.04. Native CUDA uses actual nvcc 12.2.140
and pip-cu122; binary Torch uses cu128. TF32 and thread settings are matched by
the gate. Other devices and Torch/DeepSpeed versions have not been accepted here.

Raw logs and arrays are unversioned under the lab's
`_state/refactor-goal/deepspeed/`:

| Validation | Result | Raw log |
| --- | --- | --- |
| Formal CPU Engine matrix, both interpreters | 12 passed, 0 skipped | `cpu-formal-engine-gate-v1.log` |
| Formal CUDA Engine matrix, both interpreters | 12 passed, 0 skipped | `cuda-formal-engine-gate-v1.log` |
| Actual dedicated nox CPU CLI, independent editable installs | 12 passed, 0 skipped | `nox-engine-cpu-v3.log` |
| Actual dedicated nox CUDA CLI, genuine pip-cu122 build | 12 passed, 0 skipped | `nox-engine-cuda-v2.log` |
| Native/public owner tests, CPU and CUDA | 42 passed, 0 skipped | `cuda-hooks-owners-v2.log` |
| Independent binary Torch owner contracts | 19 methods passed on CPU; 19 on CUDA | `oracle-owners-final-{cpu,cuda}.log` |
| Expanded CUDA optimizer/autograd after remote initializer fix | 54 passed, 0 skipped | `cuda-optimizer-autograd-regression-after-sync.log` |
| CPU official core after hook changes | 322 passed, 42 skipped, 1 existing xfail | `cpu-core-after-hooks.log` |
| Adapter source/relocation/lifecycle contracts and existing vLLM contracts | 20 passed, 0 skipped | `adapter-contract-v3.log` |
| Dedicated nox entry isolation/configuration negatives | 6 passed, 0 skipped | `nox-engine-negative-v3.log` |
| Existing nox/environment and pytest contracts with the shared host recorder | 57 passed, 0 skipped | `nox-contract-v2.log` |
| Full structure at the synchronized owner baseline | 1457 passed, 6 skipped; 2 shared documentation registration failures | `structure-final-v3.log` |
| Repository layout | passed | `layout-final-v3.log` |

The two remaining structure failures are the documentation reachability and
generated packaging-manifest checks for the two new DeepSpeed reports. Their
shared index/manifest registration belongs to the integrating maintainer; this
report does not label that incomplete scan fully green. The first scan also
found a new gate test importing another test module; its host-only recorder was
moved to `_helpers`, and the repeated scan confirmed that contract is repaired.

The first isolated nox CPU run triggered a slow oneDNN source download; the session
now accepts the two explicit existing oneDNN install paths. Its next run correctly
rejected mismatched execution conditions: Jittor had four threads, while binary
Torch honored the inherited 128-thread MKL budget. The session now applies the
declared budget to every shared thread-pool environment name. The metadata check
was preserved, and the actual CPU CLI then passed all twelve cases. Failed and
interrupted raw logs remain available rather than replacing the record.

The private CUDA gate environment initially resolved newer CUDA metadata before
the pinned component target and selected a system SDK. That run was stopped
before accepting any matrix result. Prioritizing the private pinned component
target resolved `pipcu122_a89fa4fed08c`; the repeated actual CUDA CLI compiled
that provider and passed all twelve cases. The independent binary oracle package
and environment were unchanged. The oracle's formerly incomplete headers were
restored from the exact official PyTorch 2.11.0+cu128 wheel and checked against
its per-file RECORD hashes; no whole-wheel digest verification is claimed.

The expanded CUDA regression originally had 53 passes and one failure on a frozen
in-place initializer. An independent clean pre-hook baseline reproduced it; the
remote owner fix `236709174` prevents random initialization from gaining grad.
After preserving the original failure logs and synchronizing that fix, all 54
pass. The regression was not skipped or weakened.

Not accepted by this report: bf16, two-rank ZeRO/NCCL, ZeRO 3, model/tensor/pipeline
parallelism, offload/fused Adam/CPUAdam extensions, compiled/reentrant autograd,
DeepCompile, elastic launch, realistic-size performance, or the entire optimizer
and model namespace. The next R14 acceptance must run actual multi-rank bf16
training, partition/reduction and save/fresh-Engine resume against binary Torch.

## Combined main-tree integration verification

The integrating tree starts at `0cddc76749e4f881d587a1f9ee865b9129464883`
and includes the owner/Engine commits listed above, native ZeRO1 `0700dde342`,
and OpenMMLab `99ee01f7c8` / `2aa27ffe1c` / `cc7e09ff94`.
The uncommitted test source is bound by
`$JITTOR_LAB_ROOT/_state/deepspeed-integration/combined-source.json`; the commit
containing this section records the integrated source. Original numerical
measurements above retain their original baselines.

| Integrated validation | Result | Raw log under `_state/deepspeed-integration/` |
| --- | --- | --- |
| CPU core, native and Torch modes | 322 passed, 42 skipped, 1 existing xfail | `core-combined.log` |
| Gradient hooks, callback/participation, Module registration, optimizer rebinding and BatchNorm state, CPU | 32 passed, zero skipped | `owners-cpu.log` |
| Same owner regressions on real CUDA/pip-cu122 | 32 passed, zero skipped | `cuda/owners.log` |
| Native ZeRO and sharded checkpoint CPU regression | 45 passed, zero skipped | `native-zero-checkpoint.log` |
| Coverage audit, required-library skip policy and OpenMMLab host contracts | 61 passed, zero skipped | `host-combined.log` |
| Complete Torch-mode structure gate | 1460 passed, 6 skipped | `structure-combined.log` |

The central report indexes and generated manifests are now registered, closing
this integration's earlier documentation failures. The six structure skips
remain disclosed: two accelerator checks, two declared checks and two checks
whose isolated interpreter has no installed Jittor distribution. The CUDA run
uses a separate compilation cache on the same SM89 platform described above;
concurrent jobs exclude any performance conclusion. A final explicit import
re-export and this integration note do not alter the measured tensor algorithms.
Multi-rank bf16 DeepSpeed remains a separate incomplete acceptance task.
