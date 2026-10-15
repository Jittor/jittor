# Installed-source OpenMMLab train_step on CPU and CUDA

- Status: Both original tiny train_step scenarios pass complete installed-source CPU/CUDA gradients, perturbations and three-step training; bounded component L3 only.
- Date: 2026-10-15.
- Measurement baseline: clean `9345f99aad2a67ec7c043427a1dfd661c53c4369`, synchronized with published `593249c5f0e82febb931234aa35003d0d41c8b33` before execution. Later maintained-gate/postprocessor source: `99120d4a5c1f1814833e15e16e90bc993cf5f1b8`.
- Schema prerequisites: `fc8bbc41dcce2986de4276e2c7b10b154ea0fcf3` and `8018b73c85f3f46966da2a7751aca9890626fca9`.
- Maintainer: Jittor Torch compatibility and downstream validation maintainers.
- Review when: upstream pins/registries/setup declarations, installed payloads, providers, precision, component builders, training owners, proof schemas or maintained gates change.

This fresh measurement preserves the complete [original OpenMMLab protocol](2026-10-15-openmmlab-training-oracles.md).
The original four rows and their raw files retain the original `2aa27ffe...` identity; no provenance
was backfilled. The new records independently bind actual installed packages. No adapter, copied
training algorithm, upstream patch, model download or fake compiled extension was used.

## Upstream and actual installations

| Component | Clean upstream / module version | Actual wheel METADATA Name / Version | Python files / complete installed payload |
| --- | --- | --- | --- |
| MMEngine | `390ba2fbb272816adfd2883642326d0fd0ca6049` / 0.10.7 | `mmengine` / 0.10.7 | 153 / 157 |
| MMCV | `3a5fb4b96ab1b241e7379adf7c326ef6edef6515` / 2.2.0 | `mmcv-lite` / 2.2.0 | 128 / 531 |

Both are self-built wheels from unmodified clean source archives, **not official PyPI wheels**.
MMEngine wheel SHA256 is `6e4c14ecdb9030e2ce1a0ee2b4169200ed961c93b8193b6364372f0a105113c4`;
MMCV-lite is `0da8de72a749ef224f5ab7772e516ec262f868a3c82dbc2de760dcdb6d98e764`.
MMCV has 534 tracked package files; three unshipped non-Python files are retained in the inventory,
not represented as installed. All tracked Python files match their clean source bytes.

The finite distribution-name proof is derived from the actual tracked `setup.py`, binding its hash,
complete supported conditional expression and every declared branch. MMEngine declares `mmengine`
or `mmengine-lite` through `MMENGINE_LITE`; MMCV declares `mmcv` or `mmcv-lite` through `MMCV_WITH_OPS`.
Unresolved declarations or dropped branches fail closed. The actual wheel's unique METADATA Name,
Version and all package bytes are independently checked, and every runtime phase explicitly records
the actual `distribution_name` and `distribution_version`. No version string is modified. The legacy
artifact role suffix `-official-wheel` is a schema label, not a claim about the origin of these wheels.

The MMCV record additionally binds its actual MMEngine dependency, complete source/payload and registry
references. Merely binding MMCV's own files would omit the owner of its `train_step` and optimizer wrapper.
No `_ext` stub exists and no compiled `mmcv.ops` acceptance is inferred from the lite package.

## Complete original training scope and budgets

| Case | Actual model and input | Requested full gradients | Three-step arrays |
| --- | --- | --- | --- |
| `mmengine_train_step` | LayerNorm(6), Linear(6,8), SiLU, Linear(8,2); sample 4×6 | all 6 named parameters and sample | 75 |
| `mmcv_convmodule_train_step` | registry-built ConvModule BN/ReLU + GN/LeakyReLU, Linear head; sample 2×3×8×8 | all 8 named parameters and sample | 102 |

The original real `BaseModel.train_step` owns preprocessing, loss forward, scalar/list log parsing,
backward, norm clipping through the registry-built `OptimWrapper`, actual momentum SGD step and
zeroing. `RecordingSGD` only observes the post-clip gradients before invoking the real SGD owner.
Each of three steps must call loss forward and optimizer update exactly once. Every named parameter
is trainable and changes; targets are fixed float32 inputs without requested gradients.

Base seed 0 and perturbations 7/19 use identical full initial parameters and buffers. CUDA reuses the
CPU reference initial state. Both runtimes must change output and at least one requested gradient
for each perturbation; all 16 such runtime/case/device/perturbation observations were independently
verified. Training retains input seeds 1/2/3, target seeds 101/102/103, learning rate 0.01, momentum 0.9
and clip norm 1. Each step compares prediction, all parsed losses, input, targets, input gradient,
every post-clip parameter gradient, updated parameter, momentum buffer and named buffer. MMCV's BN
counter is a true scalar int64 and must be exactly 1/2/3, including on the actual CUDA device.

Every phase retains the original normalized metric:
`max_abs_error / max(reference_field_max, gradient_block_max * 0.001, 1e-6)` strictly below 0.002
for nongradient fields or 0.01 for gradients. The gradient block is restricted to the current step;
nongradients use their own field scale. Inputs, targets and counters must also match exactly.
Symmetric complete keys, exact shapes/dtypes and finite values are mandatory. There is no tolerance
change, broadcast allowance, aggregate parameter-scale dilution or substitution of ordinary SGD
training for the downstream-owned `train_step`.

The post-formal `99120d4...` change makes the already-satisfied perturbation-gradient condition an
explicit maintained-gate and collector guard. It changes four postprocessing/test files only; all
measurement producers, builders and computation remain byte-identical to the measured source.
The raw 9345 identity is retained. Later nox execution has its own actual 99120 source identity.

## Actual platform and numeric results

Both runtimes use Python 3.11, NumPy 1.26.4 and the same actual installed MM packages. Oracle is
independent official binary PyTorch 2.11.0+cu128; the candidate is Jittor's Torch compatibility layer.
CPU explicitly masks CUDA and disables its compiler/backend. CUDA runs on an allocated NVIDIA RTX
4090 using real NVCC 12.2.140. Precision is FP32: CPU TF32 OFF / matmul `highest`, CUDA matmul/cuDNN
TF32 ON / matmul `high`, cuDNN benchmark OFF. Both use two runtime threads, matching recorded CPU
affinity, strict native kernel math and offline controls. Platform JSON preserves provider metadata
and actual card occupancy; this is correctness evidence with no performance or L4 conclusion.

| Component / device | Array pairs across all phases | Maximum absolute error | Maximum normalized error |
| --- | --- | --- | --- |
| MMEngine / CPU | 99 | 2.38418579e-7 | 5.21069171e-7 |
| MMCV / CPU | 132 | 2.38418579e-7 | 1.54802998e-6 |
| MMEngine / CUDA | 99 | 4.69446182e-4 | 4.32249954e-4 |
| MMCV / CUDA | 132 | 9.93609428e-5 | 6.24255403e-4 |

All 462 array pairs were independently recomputed with NumPy under the unchanged original budgets.
There are 32 runtime-phase metadata files and 956 actual tensor observations. Candidate observations
check physical CPU/device residency and dispatch after synchronization, without moving the tensor to
manufacture residency. Candidate backend fallback policy is `error`, with zero actual fallbacks.
CPU/CUDA benchmark caches and maintained unit caches are separate.

The read-only oracle provider remains the complete official wheel with SHA256
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`; actual `_C` SHA256 is
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`.
A post-formal rehash verifies all 11,841 original installed wheel files (1,624,262,090 bytes), excluding
the pip-rewritten RECORD as in the original provider proof, and the exact 2,227 Python source set.
The existing private CUDA 12.8 provider layout is reused read-only. No shared packages were repaired.

## Maintained gates, proof negatives and bounded coverage

Actual clean `99120d4...` nox execution finished **2 passed / 0 skipped** on CPU (179.02 s pytest)
and **2 passed / 0 skipped** on CUDA (423.47 s pytest); cold sessions took approximately eight and
thirteen minutes. The independently isolated canonical CPU core completed **322 passed / 42 skipped /
1 existing xfail**, with native and Torch modes both exit 0.

The new opt-in `ecosystem_openmmlab_installed` session executes both complete original protocols,
three input seeds and all real training steps. It requires both actual wheel/checkouts, independent
Torch, installed dependency versions, explicit CPU/CUDA allocation and zero fallback. It rejects
positional filtering and clears `PYTEST_ADDOPTS`; numerical failures propagate beyond worker exit
status. The actual class methods are `EcosystemOpenMMLabInstalled::test_mmengine_train_step` and
`::test_mmcv_convmodule_train_step` in `compat/tests/torch/test_ecosystem_openmmlab_installed.py`.
These opt-in executions alone do not establish nightly scheduling.

All 32 actual metadata packets passed complete proof. On copies only, 384 missing/wrong Name,
METADATA version, source, non-Python payload or wheel-hash mutations were rejected, separately for
both referenced libraries where applicable. Original raw arrays and metadata remain unchanged.
Host contracts cover incomplete tensors, shapes/dtypes/nonfinite values, original phase budgets,
BN scalar/counter trajectory, unchanged perturbation gradients, observer physical residency and
numerical failure propagation. All 71 focused host contracts pass; full `tests/structure` is **1510 passed / 4 skipped**,
and Ruff/layout checks pass. Host/structure ran on 9345 with the exact postprocessing patch later
committed as 99120; core and nox ran on clean 99120.
The final documentation-only check has **14 passes and one pending central toctree registration**
for this new report. The integrator owns that index and MANIFEST; this is not claimed as a fully
green documentation suite before registration.

The common auditor accepts **4 L3 records**, rejects 0 and checks **102 artifact bindings**.
It independently rejects both honestly rehashed CUDA platform copies claiming TF32 OFF/highest
against the actual ON/high measurements; unchanged CPU rows remain accepted. Three additional
common-auditor attacks each reject the affected MMCV row: the base oracle omits its MMEngine
distribution name, the seed-19 candidate omits its own name, or the training candidate omits its
MMEngine dependency name. All original scorecard/raw bytes remain unchanged.

| Automatic source declaration axis | CPU L3 members / denominator | CUDA L3 members / denominator |
| --- | --- | --- |
| MMEngine model_component | 1 / 7, BaseDataPreprocessor | 1 / 7 |
| MMEngine optim_wrapper | 1 / 5, OptimWrapper | 1 / 5 |
| MMEngine loop / weight_initializer | 0 / 4, 0 / 8 | 0 / 4, 0 / 8 |
| MMCV model_component | 1 / 70, ConvModule | 1 / 70 |
| MMCV hook / transform | 0 / 1, 0 / 19 | 0 / 1, 0 / 19 |

These bounded source axes retain conditional declarations and extension-dependent entries; they
are not a whole-library runtime registry or API denominator. Both MMCV references to MMEngine use
the same already-counted members, not additional breadth. Each verified scenario retains the exact
case, all registry refs, tiny shape kind and full trainable/input-gradient scope. No full Runner,
dataset/hook/loop, distributed training, compiled mmcv.ops, larger model, other accelerator, storage
reuse, speed or whole-library coverage is granted.

## Reproduction and immutable evidence

Provision actual packages in independent interpreters first. The wheel and source paths are required;
the maintained session does not download or install downstream packages.

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_ECOSYSTEM_PACKAGE_SITE="$MM_SITE" JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$MM_SITE" \
  JITTOR_MMENGINE_CHECKOUT="$MMENGINE_CHECKOUT" JITTOR_MMENGINE_WHEEL="$MMENGINE_WHEEL" \
  JITTOR_MMCV_CHECKOUT="$MMCV_CHECKOUT" JITTOR_MMCV_WHEEL="$MMCV_WHEEL" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_openmmlab_installed
# CUDA: allocate CUDA_VISIBLE_DEVICES explicitly, use a separate unit cache,
# set JITTOR_TEST_DEVICES=cuda and provide real nvcc_path/JT_BUILD_NVCC_PATH.
```

Source-backed raw data and exact command/environment artifacts are unversioned under
`$JITTOR_LAB_ROOT/_state/openmmlab-source-bound/20261015/`, with measurements in
`formal-9345f99aad2a-v1/{cpu,cuda}`. `frozen-source-9345f99aad2a-v1.json` and the immutable Git
ref bind the original source; `post-formal-guard-source-v1.json` distinguishes later guard validation.
`formal-independent-array-audit-v1.json`, `formal-installation-independent-negatives-v1.json` and
`formal-gradient-perturbation-independent-v1.json` preserve independent checks.

`scorecard-v1.json` is the final four-record scorecard. The successfully reread raw seal is
`mm-sealed-raw-manifest-v2.json`: **1,021 files / 927,817,307 bytes**, SHA256
`bdc6ff7a996db0d2e93df636bb77f9f2a14fecf52696fa3c323d7dafadc4790a`.
It binds the original measurements, comparisons, providers, complete installed MM payloads,
commands, diagnostics and both source snapshots. The final evidence manifest additionally binds
the scorecard and documentation commit. Sealed files are not rewritten. The first seal attempt
included its own still-open log and is retained as a failed diagnostic; v2 excludes writer logs
and independently rehashes every entry.

Initial environment/setup diagnostics are retained separately: missing private dependencies were
installed into this task's own prefix, and the candidate core editable installation was corrected
with the existing `editable_mode=compat` contract. The first structure attempt had one namespace
resolution failure; both targeted checks and subsequent complete structure runs passed. An extra
core-v1 launch omitted the task-local lab root and was stopped after partial native results; it is
not counted as complete core. Core-v2 explicitly isolates the lab root and passes both modes.
These setup/validation diagnostics do not relabel the formal measurements or repair shared packages.

## Main integration verification

The integrating maintainer synchronized published
`cd7d45ab99e086f2b027be35bffa66d3b33efdd7` and retained the original measured
source identities above. Clean integrated source
`c4935e9061964f17fa7d8ac4fb9930e1b84f5bfd` adds the maintained gates, source
contracts, discovery and central results/MANIFEST registration. All 73 existing
nox function ASTs are unchanged; only `ecosystem_openmmlab_installed` is added.

On this integrated source, host contracts for provisioning, Swift, distribution
names, installed source/version proofs, OpenMMLab, shared coverage, VERL, PEFT and
Transformers pass **228 / 0 skipped**. Canonical core passes **322 / 42 skipped /
1 existing xfail**, with both native and Torch mode exits zero. Complete
`tests/structure` plus `compat/tests/structure` passes **1622 / 6 skipped** in
235.97 seconds. The six skips retain their accelerator/declared/uninstalled
native-distribution boundaries; no installation or hardware claim is inferred.
Layout, generated manifest, Ruff and whitespace checks pass.

The maintainer independently rehashed all 313 nightly evidence files and all
1033 OpenMMLab evidence files: no mismatches. Original failed attempts and
measurement/source identities are preserved. New isolated maintained numerical
runs at c4935 are in progress; these integration checks do not relabel the old
numerical results as c4935 measurements. Unversioned integration records are in
`$JITTOR_LAB_ROOT/_state/maintained-gates-main-integration-cd7/`.
