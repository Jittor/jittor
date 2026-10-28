# Real MMEngine and MMCV training owners on CPU and CUDA

- Status: Genuine downstream forward/backward and three-step training reproduced on both devices; bounded tiny component scenarios, L3 only.
- Measurement baseline: `2aa27ffe1ce1699c888575f3b5fb3092b1fc6eff`, clean before and after every numerical child and dedicated nox execution.
- Compatibility prerequisite: `99ee01f7c8d2d29056347846b2c99e4775e46c2a`; the source baseline includes the training probes, explicit full-gradient contract, registry scanner and opt-in gate.
- Upstreams: MMEngine 0.10.7 at `390ba2fbb272816adfd2883642326d0fd0ca6049`; MMCV 2.2.0 at `3a5fb4b96ab1b241e7379adf7c326ef6edef6515`, both clean release checkouts shared by oracle and candidate.
- Maintainer: Jittor core / Torch compatibility maintainers.
- Review when: upstream registries, BaseModel/OptimWrapper/ConvModule implementations, BatchNorm state, compatibility optimizer/gradients, dependency versions or precision policy change.

## Actual downstream ownership and parameter scope

Both user models inherit the real upstream `mmengine.model.BaseModel`. Their unmodified
`BaseModel.train_step` owns preprocessing, `forward(mode='loss')`, scalar/list loss parsing,
backward, norm clipping through the registry-built `OptimWrapper`, real momentum SGD step and
zeroing. A small SGD subclass only observes gradients immediately before `super().step`; it
implements no optimizer mathematics. Each of three steps must call the loss forward and the
optimizer exactly once and return the real `loss`, `loss_regression`, `loss_regularization` logs.

`mmengine_train_step` uses LayerNorm(6), Linear(6,8), SiLU and Linear(8,2), input `[4,6]`,
output `[4,2]`, all **six** named parameters trainable. `mmcv_convmodule_train_step` builds two
actual `ConvModule` instances through `mmengine.registry.MODELS.build`: 3→4 channels with 3×3
convolution/BN/ReLU, then 4→4 with convolution/GN(two groups)/LeakyReLU, spatial mean and a
Linear(4,2) head. Input is `[2,3,8,8]`, output `[2,2]`, all **eight** named parameters trainable.
Targets are deterministic float32 tensors without gradients. The input sample is differentiable.
There are no frozen parameters, adapters, downstream source patches or substituted extensions.

Every trajectory compares the full input and targets, prediction, parsed logs, all post-clip
parameter gradients, input gradient, every updated parameter, every momentum buffer and every
named model buffer after every actual `train_step`. This yields **75 / 102 arrays** for the two
models. MMCV includes BN running mean, variance and scalar int64 batch counter at all three steps.
Complete symmetric keys, exact shape and dtype, finite values and identical fixed inputs are
mandatory. Ordinary CASES additionally establish direct forward/backward; ordinary generic SGD
training does not establish this downstream training owner.

## Fixed platform and protocol

| Field | CPU | CUDA |
| --- | --- | --- |
| Hardware | AMD EPYC 9554 64-Core Processor; accelerator visibility empty | one allocated NVIDIA GeForce RTX 4090, driver 580.178.04 |
| Oracle | independent binary PyTorch 2.11.0+cu128 with real `_C` | same, CUDA 12.8 |
| Candidate | Jittor 2.0.0 Torch compatibility layer | same, actual NVCC 12.2.140 |
| Dependencies | Python 3.11.16, NumPy 1.26.4, MMEngine 0.10.7, MMCV 2.2.0 | same pinned sources |
| Precision | TF32 matmul/cuDNN OFF, matmul highest | TF32 matmul/cuDNN ON, matmul high |
| Controls | float32; cuDNN benchmark OFF; two runtime threads, fixed two-CPU affinity | same; actual CUDA residency and zero backend fallback |

Seed 0 exports full reference weights and model buffers. Seeds 7 and 19 reuse the identical
weights hash on both runtimes; output and a requested gradient must change for each perturbation.
The separate actual training owner uses seed-0 reference state, three input seeds 1/2/3 and target
seeds 101/102/103, SGD learning rate 0.01, momentum 0.9 and clip norm 1.0. All six/eight parameters
must change after three steps. Each benchmark and unit/gate cache is separate; first compilations
are serial. Device allocation diagnostics and all visible NVIDIA wheel metadata remain in raw
platform JSON. Concurrent card occupancy is not treated as a performance measurement.

The shared harness measures `max_abs_error / max(reference_field_max, gradient_block_max*1e-3,
1e-6)`. Output, loss, updated parameters and buffers use their own reference field; all requested
gradients in a step share the gradient floor. Existing correctness limits remain 0.002 for
nongradient fields and 0.01 for gradients on both devices. No threshold or model reshape is added
to conceal a discrepancy. Integer BN state must also retain exact scalar shape and int64 dtype.

## Measured results

| Repo | Device | Arrays per seed 0/7/19 | Max absolute / normalized error across seeds | Three-step arrays | Max absolute / normalized trajectory error | Parameters actually updated |
| --- | --- | --- | --- | --- | --- | --- |
| mmengine | CPU | 8 | 2.38418579e-07 / 3.09323537e-07 | 75 | 1.1920929e-07 / 5.21069171e-07 | 6 / 6 |
| mmcv | CPU | 10 | 2.38418579e-07 / 1.54802998e-06 | 102 | 2.38418579e-07 / 6.47097259e-07 | 8 / 8 |
| mmengine | CUDA | 8 | 0.000469446182 / 0.000338873152 | 75 | 0.000156402588 / 0.000432249954 | 6 / 6 |
| mmcv | CUDA | 10 | 9.93609428e-05 / 0.00031569602 | 102 | 8.55773687e-05 / 0.000624192525 | 8 / 8 |

The public scorecard audit accepted **4 L3 records**, rejected 0, and rehashed **180 artifact bindings**. An honestly rehashed negative platform claiming CUDA TF32 OFF/highest was rejected for both CUDA rows with `Platform/measurement precision mismatch`; the two unchanged CPU rows remained accepted. Every original artifact binding was then rehashed unchanged.

The actual `ecosystem_openmmlab` nox session at the same clean baseline passed **5 tests, zero
skipped** independently on CPU and CUDA: both real model gates plus three BatchNorm state tests.
The real PyTorch interpreter independently passed the two relevant counter/state tests on each
device. The third test checks that frontend adoption leaves native Jittor checkpoint semantics
unchanged.

## Source-derived bounded denominator and honest breadth

The stdlib-only inventory scans all 153 / 128 tracked Python source files in the clean MMEngine /
MMCV packages, recording source hashes, registration expressions and branch/function/loop guards.
Literal registration loops, aliases, class/function decorators and optional compiled-module
branches remain in the declarations. Any unresolved registration in a claimed axis fails closed.
MMEngine has 25 resolved declaration sites (one conditional), MMCV 90 (nine guarded); no unresolved
bounded registration remains. MMEngine's 101 other registration sites, including dynamic optimizer
discovery, are disclosed without assigning them an invented denominator.

| Repo | Declared source axis | CPU L3 members / denominator | CUDA L3 members / denominator | Verified method context |
| --- | --- | --- | --- | --- |
| MMEngine | model_component | 1 / 7 | 1 / 7 | BaseDataPreprocessor in this actual train_step user model |
| MMEngine | optim_wrapper | 1 / 5 | 1 / 5 | registry-built OptimWrapper, clipping + momentum SGD |
| MMEngine | loop | 0 / 4 | 0 / 4 | no Runner loop acceptance |
| MMEngine | weight_initializer | 0 / 8 | 0 / 8 | no initializer-axis acceptance |
| MMCV | model_component | 1 / 70 | 1 / 70 | registry-built ConvModule with the stated BN/GN blocks |
| MMCV | hook | 0 / 1 | 0 / 1 | no hook acceptance |
| MMCV | transform | 0 / 19 | 0 / 19 | no transform-axis acceptance |

These denominators describe bounded source declarations at the stated upstream SHA, retaining
conditional and extension-dependent entries. They are not an installed runtime inventory or a
whole-library API denominator. The custom Regressor is not counted as an upstream registry member;
other norm/activation aliases used inside ConvModule are conservatively not counted separately.
Every accepted coverage row retains the case, complete registry references, tiny shape kind and
six/eight fully trainable parameter names. Other accelerator backends have zero accepted evidence.
Full Runner/dataset/hooks, distributed training, compiled `mmcv.ops`, production sizes, memory and
speed levels remain outside this result. Upstream inplace activation requests produce the existing
native allocation warning; output/gradient correctness is compared, allocation reuse is not graded.

## Compatibility defect and regression evidence

Before the fix, an independent public-API repro on CPU and CUDA found that frontend BatchNorm's
`num_batches_tracked` inherited native `(1,)` int32 nonpersistent state, while real Torch exposes a
scalar int64 persistent buffer. The fix is confined to the compatibility NN construction owner:
a true scalar backend `array(0, dtype='int64').stop_grad()` is registered on the adopted frontend
instance, or `None` when running statistics are disabled. Native Jittor state is unchanged.

The selected-device regression covers BatchNorm1d/2d/3d, counts 1/2/3 after real training forwards,
no update in eval, persistent state_dict and torch.save/load_state_dict roundtrip, resumed count 4,
float64 module casting preserving the int64 counter, parameter/buffer ownership and native/frontend
isolation. It does not claim untested running-statistics API behavior beyond this counter contract.

Development regressions ran before the source commits, on merge `3156ee1f7ac11cc82acb6920f9ec611da9299fee`
with this batch applied; the formal numerical and dedicated nox evidence separately pins the clean
measurement SHA above. CPU core: **322 passed, 42 skipped, one expected failure**; adjacent normalization/serialization/
counter tests: **56 passed, six accelerator skips**. Host/namespace/registry-negative contracts:
**68 passed**. Full structure with corrected external test environment: **1453 passed, four
skipped, two failed**, both existing local sealed-Longformer documentation index/manifest entries
for the previously sealed Longformer report on this source branch. These failures do not establish
a fully green structure suite. The 14 other documentation contracts passed after this report was
written; the reachability check remains part of central report registration. Layout, Ruff and diff
whitespace checks passed. Diagnostic logs retain the original environment failures separately
from the model acceptance evidence.

## Reproduction and artifact ownership

Reusable entrypoints are
[`openmmlab_training_probe.py`](../../agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py),
[`test_ecosystem_openmmlab.py`](../../compat/tests/torch/test_ecosystem_openmmlab.py) and the
`ecosystem_openmmlab` session. Provision the two pinned downstream checkouts in separate genuine
Torch and Jittor interpreters, then run the oracle probe first and supply its `.weights.npz` to the
candidate. Use the dedicated session with `REAL_TORCH_PYTHON`, `JITTOR_CI_PYTHON` and one selected
`JITTOR_TEST_DEVICES=cpu` or `cuda`; allocate an actual visible device and NVCC only for CUDA.
The session enables required dependency, real-oracle and nonzero-execution flags and forbids
backend fallback. CPU explicitly clears all CUDA visibility/enable variables before framework
import. Missing required libraries, incomplete gradients, fake owner/updates, wrong-device
metadata, missing optimizer state, asymmetric keys, broadcasting shapes and nonfinite data have
host negative tests.

Raw data, exact commands and private interpreter/cache locations remain under
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/openmmlab-20261015/formal-2aa27ffe1c/`:
`measurement-source.json`, `platform-{cpu,cuda}.json`, pinned inventories, raw seed/weight/train
NPZ and JSON, all comparison reports, nox logs/source checks, `scorecard-evidence.json`,
`scorecard.json`, `negative-precision-scorecard.json` and the negative result. This tracked report
contains no raw tensors. The snapshot skill/case discovery audit lists the two ordinary CASES;
the dedicated gate nodes, source SHA and actual nox execution are recorded and bound separately.
Static discovery alone never establishes an executed training gate.
