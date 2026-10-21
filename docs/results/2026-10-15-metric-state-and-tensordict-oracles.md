# Metric state conversion and real TensorDict oracles

Status: verified for the selected tiny components on CPU and CUDA; no whole-library,
registry-coverage, memory, or performance claim. Maintainer: Jittor Torch compatibility
maintainers. Recheck when the listed upstream version, Module/tensor conversion owner,
runner precision policy, or selected operation changes.

The implementation is `cf601b9a88d959eb06e1e6e9e298359d27540497`. Formal component
measurements used clean merge baseline
`5ad8a1929f2e4a88db39b4f0e570e4901615487d`, which includes remote
`f02e7aec0b43daf1b275a97eab340de7c4e917d3` and that implementation. This preserves the
parent's full constraint-publication fix and tests. Original pre-fix/pre-registration
artifacts remain external under their original baseline; they are not relabeled.

## What is now verified

The new builders live in `compat/tests/torch/_ecosystem_cases.py` and use actual
upstream packages with no case-specific adapter. The public `verify_repo.py` driver,
shared `_ecosystem_runner.py`, and existing `training_probe.py` produced the results.
No `jittor-torch-adapters` distribution or `jittor.module_patches` entry point was
installed in this validation environment.

| Component | Exact selected scope | Checked tensors per base/perturb pass | Protocol evidence |
| --- | --- | --- | --- |
| `torchmetrics_regression` | Real MSE and MAE children; FP32 prediction/target `(4, 8)` | Output vector and both input gradients: 3 arrays, 0 parameter gradients | CPU/CUDA L2 component protocol; no trainable parameters |
| `tensordict_index_and_stack` | Real TensorDict batch `[2, 3]`, input `(2, 3, 4)`, integer/bool indexing, lazy stack, expand, Linear(4,3) | Output, 2 parameter gradients, 1 input gradient: 4 arrays | CPU/CUDA L3 component protocol with 3 explicit SGD steps |

Each case ran from independent reference initial weights at seed 0, then both
runtimes reran seed 7 inputs using the original weights. Reference outputs changed
under perturbation. Expected gradient names were enumerated from the actual reference
builder and checked in **both** output archives; two missing gradients cannot count
as agreement. All compared values were finite and the candidate reported the requested
actual device, `fallback_policy=error`, and zero fallback.

| Component/device | Worst absolute error, base or perturb | Worst scaled-relative error | Requested/checked parameter gradients | Requested/checked input gradients |
| --- | ---: | ---: | --- | --- |
| TorchMetrics CPU | 0 | 0 | 0/0 | 2/2 |
| TorchMetrics CUDA | 0 | 0 | 0/0 | 2/2 |
| TensorDict CPU | 9.536743e-7 | 6.690514e-8 | 2/2 | 1/1 |
| TensorDict CUDA | 1.595855e-3 | 1.021639e-4 | 2/2 | 1/1 |

The existing comparison accepts each tensor when absolute error is at most `2e-3`
**or** field-scaled relative error is at most `1e-2`. These are FP32 tiny-component
results under the actual precision settings below, not exact or strict-TF32-off
agreement for TensorDict CUDA.

TensorDict's three-step trajectory uses learning rate `1e-3`, seed 0, the same complete
initial weights on both sides, and train mode. All **21** arrays agree: each step's
output, scalar loss, two parameter gradients, input gradient, and two updated parameters.
The trainable list is exactly `projection.bias`, `projection.weight`; frozen list is
empty. Parameters actually change. Updates are explicit SGD in the shared probe and
do not validate Torch optimizer classes, Adam, an upstream Trainer, or distributed training.

| Device | Trajectory arrays | Worst absolute error | Worst scaled-relative error | Reference loss history | Candidate loss history |
| --- | ---: | ---: | ---: | --- | --- |
| CPU | 21 | 5.960464e-8 | 1.111221e-8 | 0.041279454, -0.167627022, -0.210601747 | 0.041279420, -0.167627022, -0.210601762 |
| CUDA | 21 | 1.291275e-3 | 2.406770e-4 | 0.041469362, -0.167696729, -0.210582942 | 0.041279450, -0.167627007, -0.210601747 |

A real TorchMetrics request to the same three-step probe fails with
`Training case has no trainable parameters` and creates no trajectory archive. Its
input backward success is not L3 training. At that measured baseline neither library
had an automatic registry denominator in the scorecard tool; those four records retain
`registry_verified_level=null` and `registry_coverage=null`. Registration of a manual
CASES entry alone does not add a pytest ecosystem CI method or award registry coverage.

## Actual platform and upstream identity

- Linux 6.1.188 x86_64, AMD EPYC 9554, Python 3.11, g++ 11.5.0, Jittor 2.0.0.
- Independent genuine Torch `2.11.0+cu128`, including its compiled `_C` extension;
  it is not the Jittor frontend. Both sides use runtime/thread environment count 2.
- CPU explicitly masks CUDA and sets `use_cuda=0`; actual TF32 matmul/cuDNN are
  disabled and matmul precision is `highest`.
- CUDA executes on one assigned RTX 4090, driver 580.178.04. Jittor uses the genuine
  NVCC 12.2.140 toolchain and cuDNN 9.19; the oracle Torch uses CUDA 12.8.
- CUDA JSON records actual TF32 matmul/cuDNN **enabled**, matmul precision **high**,
  cuDNN benchmark disabled. Each pair's complete `runtime_conditions` is equal;
  the platform block agrees with actual measurement JSON. Kernel math is `strict`,
  MKL/NCCL/MPI/CUTT and parallel op compilation are disabled.
- Cache roots for unit tests and component measurements are separate; CPU and CUDA
  each have their own cache and temporary output. No first JIT build was shared
  with another validation worker.

| Upstream | Pinned release | Clean checkout SHA | Installed tracked Python source parity |
| --- | --- | --- | --- |
| TorchMetrics | 1.7.4 | `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 341/341 files byte-identical |
| TensorDict | 0.10.0 | `c34264a58b12471dc3e2913ea586b32431872d4c` | 36/36 files byte-identical |

The parity check binds installed wheel Python files to the clean release checkout;
this does not claim parity for generated files or optional native extensions. Both
runtime result documents record the installed package version and origin.

## Minimal failures and owner fixes

The TorchMetrics failure split into two independent, framework-free API contracts:

1. `Module.to/float` walked registered tensors directly and bypassed custom `_apply`
   overrides. Metric's unregistered tensor/list accumulators stayed on the old device.
   The frontend now recursively invokes child overrides, deduplicates direct child
   aliases, and converts its own parameters/buffers while preserving parameter objects
   and gradients. Native Jittor owners are unchanged.
2. Augmented tensor operators inherited native stopped-gradient policy. In addition,
   reset state made through `detach().clone()` could lose the RHS graph permanently
   when `assign` copied its stopped flag into the replacement. The frontend's four
   augmented arithmetic operators now share the existing in-place owner; that owner
   rebinds a stopped non-view holder through native `_update` when it gains a
   differentiable RHS. No-grad writes remain frozen. This does not establish new
   view-gradient behavior or Torch future swap/set-data policies.

TensorDict previously failed during import because frontend cloning of a ModuleType
subclass missed constraints defined on the class. Parent commit
`9d9310e3c9a4af737702e12d2d6d47062fa015cd` publishes class and instance members while
preserving native owners. This batch consumes that remote implementation, rather than
keeping an older local copy. No additional TensorDict adapter was created.

## Regression checks and reproducibility

Before committing the implementation, the patched `db23449f8` worktree passed:

- CPU core: **322 passed, 42 skipped, 1 xfailed** (native 98; Torch mode 224).
- CUDA autograd/module/gradient-management neighbors: **193 passed**.
- Final new API cases, including child alias handling: CPU covered by **43** focused
  checks; CUDA **7 passed**. The same seven generic API cases independently passed
  against genuine Torch on CPU and CUDA: **14 passed**.
- Full structure: **1430 passed, 6 skipped, 2 failed**. Both failures were the existing
  downstream result page's missing shared toctree/MANIFEST registration, owned by the
  parent integration. This is not a claim that full structure passed in this worktree.

After syncing remote `f02e7aec0`, clean measurement baseline `5ad8a1929` passed **48**
CPU API/constraint/audit/training-registry/probe checks and **8** CUDA API/constraint
checks. Final full structure after the runbook/report update was again **1430 passed,
6 skipped, 2 failed**: the same toctree/MANIFEST contracts now identify both the
October 14 scorecard page and this new report, pending shared parent integration.
Repository layout and whitespace checks passed; both skills passed skill validation. The old GradScaler regression
now initializes `scale` before `unscale_`, as genuine Torch requires, retaining the
original item-count and Inf-skip assertions; the independent CPU/CUDA contract was
also checked.

Use the updated [TorchMetrics runbook](../../agent/skills/torchmetrics-torch-compat/SKILL.md)
and [TensorDict runbook](../../agent/skills/tensordict-torch-compat/SKILL.md) from the
repository's `agent/skills/` paths. Public entrypoints are:

```bash
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo torchmetrics --list-only
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo tensordict --list-only
```

The full commands, raw NPZ files, metadata, runtime precision, expected-gradient
checks and external driver source hashes are bound in
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/component-evidence-20261015.json`.
Its SHA-256 is
`6d26ff702720c75526da445053e326cc592d940a980cb1d70339903b8371efd0`.
The installed-source parity JSON SHA-256 is
`40840b7ac74a71f5046299d614cd9d4dc0b6cbc2fe7cfbd97d3976418396498a`.
Raw arrays, logs, wheels, caches and the local platform's physical device selection
stay outside the repository. Diagnostic runner timings were collected without a
controlled performance protocol; this report quotes no speed or memory ratio.

## Maintained oracle admission and bounded API inventories

The later [shared scorecard report](2026-10-15-metrics-tensordict-scorecards.md)
binds fresh clean-`90ee18a` measurements to these bounded axes. This page's older
component arrays retain their original baseline and are not relabeled.

The follow-up based on `b2aeec10874136c1b4b16c1598241252a5f92c87` adds actual
shared-oracle pytest methods for the two registered builders. They require both
TorchMetrics input gradients and all TensorDict parameter/input gradients.
The existing CUDA subclass inherits both tests. The source-only admission audit
now retains the enclosing class in selectors, so its reported paths identify
executable pytest nodes rather than unqualified unittest method names.

Both CPU nodes passed, followed by both actual CUDA nodes, with zero skips.
The genuine oracle remained Torch 2.11.0+cu128; both sides used TorchMetrics
1.7.4 and TensorDict 0.10.0. CPU explicitly hid CUDA. These new gate runs used
TF32 disabled, cuDNN benchmark disabled and strict native kernel math; they do
not relabel the earlier TF32-enabled CUDA artifacts. Each shared comparison
checks actual device, equal precision conditions and zero fallback. An initial
CPU attempt failed because the private interpreter exposed a SciPy stub;
prefixing an existing real SciPy installation fixed the environment. Original
failures remain in the external logs, and no package/math implementation was
changed for this admission follow-up.

`registry_inventory.py` also derives a `public_api` axis from every name in the
top-level package's source `__all__`, without importing either library:

| Repository | Full upstream SHA | Bounded public API denominator |
| --- | --- | ---: |
| TorchMetrics 1.7.4 | `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 105 |
| TensorDict 0.10.0 | `c34264a58b12471dc3e2913ea586b32431872d4c` | 60 |

The inventory retains the complete exported-name list, source hashes, real
version provenance and full tracked Python package hashes. Unknown mutations,
aliases or dynamic export construction fail closed. This axis covers the
top-level declared namespace; nested APIs, class methods and task variants are
outside it. It is not a model-family or whole-library denominator. The new
inventories and gate logs alone award no registry numerator; the historical
component records above keep their original source and scope until separately
converted with complete shared-schema evidence.

For the two CPU nodes, run the existing ecosystem pytest file selecting
`EcosystemParity::test_torchmetrics_regression` and
`EcosystemParity::test_tensordict_index_and_stack`, with an independent
`REAL_TORCH_PYTHON`, the actual package sites, `JITTOR_REQUIRE_REAL_TORCH=1`,
`JITTOR_REQUIRE_TORCHMETRICS=1` and `JITTOR_REQUIRE_TENSORDICT=1`. Explicitly hide
CUDA and set `JT_USE_CUDA=0`. For CUDA, expose the allocated GPU, set
`JT_USE_CUDA=1` and select the same methods on `EcosystemParityCUDA`. Use the
existing inventory CLI with `--repo torchmetrics` or `--repo tensordict` and
the corresponding clean upstream checkout. Fresh unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/metric-tensordict-gates/`; `oracle-cpu-real-scipy.log`
and `oracle-cuda.log` contain the actual four executions, and the two
`*-inventory.json` files bind the derived denominators.

Follow-up regression verification: core 322 passed, 42 skipped and one existing
expected failure; structure plus tools 1690 passed and six skipped. The latter
includes the corrected class-qualified Longformer selector contract. Repository
layout, changed-file Ruff and whitespace checks passed. The initial stale
selector assertion failure is retained in `structure-tools.log`; the passing
rerun is `structure-tools-corrected.log` in the same external evidence directory.
