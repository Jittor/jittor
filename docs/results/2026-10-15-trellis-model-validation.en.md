# TRELLIS.2 dense model training validation

- Status: actual independent CPU and CUDA L3 validation complete, with zero candidate fallback in all six scenarios. This is a bounded model gate, not a pretrained 3D pipeline or performance acceptance.
- Date: 2026-10-15.
- Baseline: `6d39728fffc37d4618c1cb91a9caa1d67beec311` (clean committed gate/runner; source and HEAD hashes unchanged throughout formal validation). Initial CPU `ae1a9bf1f045b11cf49f1665ad4d4e5d9892f787` and CUDA warm-up `862e6395d614d70763fc4a6f5ef050200fbb2510` are preserved as separate earlier evidence.
- Upstream: Microsoft TRELLIS.2 `75fbf0183001ed9876c8dbb35de6b68552ee08bd`.
- Owner: downstream compatibility maintainers.
- Review when: upstream declarations/model source, Torch shim attention/Conv3d/normalization/autograd/optimizer semantics, backend precision/device checks, or this gate changes.

## Scope and protocol

The gate constructs the real exported `SparseStructureFlowModel`, `SparseStructureEncoder`, and `SparseStructureDecoder` from the clean pinned upstream checkout. Their tiny architectures keep the actual attention, convolution, normalization and residual paths. The dense attention backend is the upstream-supported `sdpa` option, with FP32 parameters/computation and TF32/cuDNN benchmarking disabled in both runtimes.

The Flow model's upstream zero-initialized output layer produces an identically zero initial output. Merely importing it or checking that output would not establish numerical support. This gate instead transfers deterministic nonzero FQN-bound weights and all buffers from an independent binary Torch process. It checks every named parameter gradient, every requested input gradient, output/loss, updated parameter and SGD momentum buffer for all three steps. Each requested input is then perturbed independently and the complete training comparison repeated; the output must change while other inputs remain fixed. There is no adapter, monkeypatched upstream implementation, synthetic extension or copied model.

The oracle imports a real Torch C extension and asserts that Jittor was not imported. The candidate activates the shim before importing the same upstream model. Both processes record full source identities/hashes, the exact weights/probe hashes, precision and thread/affinity settings. CPU explicitly disables CUDA before import. Accelerator checks synchronize each observed tensor, verify its actual device location and logical CUDA ordinal, and require the backend fallback policy `error` and count zero.

Numerical budgets are inherited unchanged from the existing ecosystem comparison: forward/loss/parameter/momentum scaled relative error at most `2e-3`; parameter/input gradient at most `1e-2`. Each field uses a reference-only scale floor equal to `1e-3` of the largest reference magnitude, with the shared `1e-6` absolute denominator floor. They were not selected from TRELLIS candidate errors. All compared arrays must be finite and have identical complete schemas/shapes.

## Automatically derived coverage

The source-only inventory enumerates all literal lazy export maps and sampler exports, retaining missing definitions and optional dependency paths. Separate axes are not summed into a whole-library percentage.

| Axis | Declared members | Actual verified members in this report |
| --- | ---: | ---: |
| models | 9 | 3 |
| pipelines | 2 | 0 |
| modules.sparse | 33 | 0 |
| samplers | 4 | 0 |
| datasets | 11 | 0 |
| trainers | 15 | 0 |
| renderers | 4 | 0 |
| representations | 4 | 0 |
| exported_modules | 4 | 0 |
| package_api | 2 | 0 |

Eight declared exports have no matching definition at this source pin: `datasets.TextConditionedSparseStructureLatent`; `modules.sparse.SerializeMode`, `sparse_serialized_scaled_dot_product_self_attention`, `SparseSubdivide`, `sparse_nearest_interpolate`, `sparse_trilinear_interpolate`, `encode_seq`, and `decode_seq`. They remain in their denominators. The abstract `Sampler` export also remains in the sampler axis. Trace calls do not award separate component coverage.

This checkout has no published package version field. Its `source@75fbf0183001ed9876c8dbb35de6b68552ee08bd` identity denotes only the exact clean source commit, not a release version or validation of every future TRELLIS version.

## Results and environment

The independent Torch runtime is Python 3.11.16 / Torch `2.11.0+cu128` with its real `_C` shared library. The private candidate interpreter uses read-only dependencies and Jittor from the stated worktree. NumPy is `1.26.4`, einops `0.8.2`, easydict `1.13`, and the actual upstream source pin is identical in both processes. Upstream setup recommends Torch 2.6.0; this report validates the installed 2.11.0 runtime, not a separate 2.6.0 combination.

Formal CPU: three cases passed, no skip, in 44.57 seconds. Formal CUDA: three cases passed, no skip, in 61.65 seconds. The earlier CPU run took 92.79 seconds and is retained separately. The single CUDA card is an RTX 4090 with driver 580.178.04, NVCC 12.2.140 and g++ 11.5.0; the oracle build is CUDA 12.8. These are process-isolated runtimes, not a claim that the toolchains are identical. Each baseline compares 294 (Flow), 300 (Encoder), or 300 (Decoder) arrays. Independent input variants bring the total to 2,376 array pairs per device. Wall time is gate duration and is not a speed comparison.

The default Flow path initially failed in the independent Torch environment because `flash_attn` was absent. Selecting upstream-supported `sdpa` permits the dense model path. Actual upstream `SLatFlowModel.forward` and `SparseConv3d` construction separately fail on absent `flash_attn` and `flex_gemm` in the independent reference environment. Separate real import probes find missing `cumesh` for `FlexiDualGridVaeDecoder` and missing `trimesh` at the full pipeline import boundary. Successful imports of `SLatFlowModel` and `SparseUnetVaeEncoder` establish no forward/backward support. These are first dependency breakpoints, not proof that every CPU path is unsupported or that CUDA kernels work.

The Cu122 wheel directory reports cuDNN distribution metadata `9.26.0.51`, while the actual selected `libcudnn.so.9` API returns `92400` (9.24.0). The CUDA runtime API returns `12020`. Both the declared and actual versions are recorded in platform evidence, together with the selected library file hashes; no shared dependency was modified. The actual CUDA Conv3d forward/backward scenarios above passed against this selected binary combination. Replacing those binaries or resolving the metadata collision requires revalidation. An initial private-environment import failed because Cu128 metadata shadowed the Cu122 stack and `cublas.h` was not found; the isolated candidate path was corrected to select existing read-only Cu122 components with strict wheel checking. The failed log remains part of the raw evidence.

| Device | Model | Compared raw array pairs, including independent perturbations | Maximum absolute difference | Maximum scaled relative difference |
| --- | --- | ---: | ---: | ---: |
| CPU | SparseStructureFlowModel | 1,176 | 3.42727e-7 | 2.68249e-6 |
| CPU | SparseStructureEncoder | 600 | 8.82149e-6 | 2.48022e-5 |
| CPU | SparseStructureDecoder | 600 | 1.88276e-5 | 2.56591e-4 |
| CUDA | SparseStructureFlowModel | 1,176 | 2.30968e-7 | 1.99612e-6 |
| CUDA | SparseStructureEncoder | 600 | 6.43730e-6 | 1.35543e-5 |
| CUDA | SparseStructureDecoder | 600 | 3.94583e-5 | 3.93669e-4 |

The maximum columns may refer to different tensors. Both use the unchanged field-specific budgets above. Every case checks 30 or 32 parameter FQNs; the Flow model requests gradients for `x`, `t`, and `cond`, and each VAE requests `x`. Each baseline/perturbed specification checks all three training steps.

An earlier clean `7493641ba8bc9ecab3d93f45141efaedde0aef3d` run also passed three CPU and three CUDA numerical scenarios, but its Git pathspec omitted `trellis2/__init__.py` from source metadata (92 files). The collector rejected that incomplete source binding. Those raw JSON/NPZ files and the rejection log are preserved unchanged; the accepted matrix above was rerun at the source-enumeration fix, rather than patching old metadata.

## Reproduction

Use a clean official upstream source checkout at the full SHA above, a private Python environment with the recorded dependencies, and an independent binary Torch interpreter. Set all state/cache/model/log paths outside the repository. Use a separate `JITTOR_HOME` per device and serialize first compilation.

```bash
export JITTOR_TRELLIS_CHECKOUT="$JITTOR_LAB_ROOT/trellis-validation/TRELLIS.2"
export REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_TRELLIS=1
export JITTOR_TORCH_SHIM=1 PYTHONPATH=python
export JITTOR_TEST_DEVICES=cpu
export JT_BACKEND=cpu JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' nvcc_path=''
export JITTOR_TRELLIS_EVIDENCE_DIR="$JITTOR_LAB_ROOT/_state/trellis-validation/fresh-cpu"
<jittor-python> -m pytest -q -s compat/tests/torch/test_ecosystem_trellis.py
```

For CUDA select exactly one allocated visible card, an isolated CUDA cache and a complete supported CUDA component stack; set `JITTOR_TEST_DEVICES=cuda JT_BACKEND=cuda JT_USE_CUDA=1 use_cuda=1`, and a new evidence directory. Both runtime processes execute the same versioned runner. Raw directories are created exclusively; reusing one is an error rather than silently accepting stale files.

## Evidence and limits

Raw logs, binary arrays, metadata, source/API snapshots and hashes are unversioned under `$JITTOR_LAB_ROOT/_state/trellis-validation/`. `frozen-formal-source-complete.json` records the clean test/runner/shared-harness hashes; `accepted-formal-source-complete.json` verifies they and HEAD remained unchanged through execution. The measurement baseline remains the historical `6d39728fffc37d4618c1cb91a9caa1d67beec311`; the collector and report were subsequently integrated against origin `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` (local merge `f2d4bc30f53c950e8939eeff6daf6999c78ea04e`). This does not re-label the earlier numerical execution as a newer-source run. Each case directory contains baseline and per-input-perturbation reference/candidate NPZ and JSON, comparison summaries and a SHA256 artifact manifest. `scorecard-source-complete-final.json` contains six accepted L3 records and zero rejected records, derived from all 93 tracked upstream Python files including the package root. The common downstream scorecard validates the platform/precision/runtime identities, raw hashes and source denominator; the TRELLIS collector only turns those concrete model executions into L3 records.

No trained checkpoint, pretrained weights, image/text conditioning stack, CuMesh/FlexGEMM/nvdiffrast/o-voxel/FlashAttention extension execution, 4B pipeline, mesh geometry, BF16/FP16, NPU/ROCm, multiple cards, whole-process peak memory or speed acceptance is established. The historical report-derived full-pipeline ratio `1.0926x` is not reproduced here and remains outside this acceptance. L3 plus observed zero fallback does not imply L4 performance acceptance or a whole-library L5 result.

## Maintenance checks

After synchronizing origin `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6`, the core runner passed 322 tests, with 42 declared skips and one existing expected failure. Its native and Torch sessions used separate CPU compilation caches; first compilation is included in the raw logs and is not a performance result. The source inventory, comparison and collector contract tests cover stale/missing source identity, missing gradients, nonfinite values, nonbinary oracle/fallback, duplicate scenarios, changed raw files and incomplete artifact manifests. The final host contract suite passed 38 tests; Ruff and repository layout passed. The required full structure run finished with 1,476 passed, six skipped and two failures: this new report is not yet in the central results toctree, and the generated root manifest needs the same new report. Both central registration files belong to the integration owner and were intentionally left for that integration; no other structure failure occurred. Raw commands/results are in `logs/final-host-complete.log`, `logs/final-core.log`, `logs/final-structure.log`, `logs/final-ruff.log`, and `logs/final-layout.log` under the external evidence root.


## Fresh installed-source proof and maintained replay (2026-10-16)

Status: the same three tiny dense models pass complete independent CPU and CUDA
training again, with installed-source proof for every runtime and input variant.
Measurement source is clean `b6947541b5bddfa87098d40c6f4e09ad0e704c54`, based on
published `ffbb9087610dd737edb7ab758449c1f8f645437e`; the generic persistent-directory
helper has exactly the bytes in `cc86b8674f713b1924affb954aecfc6fad6bd57a`.
The earlier measurements retain their original identities and bytes. In particular,
the current common auditor rejects all six historical 6d397 rows because they lack
actual installed-source bindings; a complete old source-hash list alone does not
satisfy that newer proof requirement. This replay does not backfill those records.

The official Microsoft checkout remains clean
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`. It has no root packaging declaration,
published package version or wheel. The actual installation is its real checkout
on the import path, explicitly recorded as `installation_kind=source-checkout` and
`source@75fbf0183001ed9876c8dbb35de6b68552ee08bd`. There is no invented PyPI release,
METADATA, wheel or modified `__version__`. Both processes capture the actual resolved
`trellis2.__file__`, independently hash every installed package file, compare the
complete **93 Python / 93 payload** set against the regenerated clean inventory,
and verify it again after training. Unexpected, missing, changed or escaping files
fail closed; generated Python bytecode is excluded. The collector independently
checks every base and per-input perturbation metadata packet, and separate
`perturbN-comparison` roles make every variant visible to the common auditor.

All original computation and budgets remain unchanged: 30 Flow parameters and
three differentiable inputs, 32 Encoder parameters and one input, 32 Decoder
parameters and one input; deterministic nonzero named initial weights, seed 0,
three SGD steps (`lr=.003, momentum=.2, weight_decay=.01`), output/loss, all parameter
and input gradients, updated parameters and momentum. Each input independently
receives `+.125`, repeating the full three-step trajectory. These are input variants,
not additional random seeds. All arrays have exact schemas/shapes, finite float32
values, the original forward `2e-3` and gradient `1e-2` field criteria and reference
scale floors. No new adapter, forced alternate attention oracle or mathematical
implementation is introduced. The upstream-supported dense `sdpa` choice remains.

Actual opt-in `ecosystem_trellis` nox execution passes **3 / 0 skipped** on CPU
(197.76 s pytest, eight minutes including cold setup) and **3 / 0 skipped** on CUDA
(469.06 s pytest, thirteen minutes including cold setup). Positives require actual
numerical comparisons, not just successful workers. The new host negative retains
successful-worker output while rejecting a failed comparison, without writing a
passed completion packet. The explicit persistent-directory helper rejects existing
or repository-local output; all 136 final raw files were written normally without
polling or copying JIT products. The CUDA session leaves all 68 CPU file hashes
unchanged. All 191 cross-device initial-state and saved-input fields match exactly.

| Device | Model | Independently recomputed array pairs | Maximum absolute error | Maximum scaled error |
| --- | --- | ---: | ---: | ---: |
| CPU | SparseStructureFlowModel | 1176 | 3.42726707e-7 | 2.68249228e-6 |
| CPU | SparseStructureEncoder | 600 | 8.82148743e-6 | 2.48021997e-5 |
| CPU | SparseStructureDecoder | 600 | 1.88276172e-5 | 2.56591094e-4 |
| CUDA | SparseStructureFlowModel | 1176 | 2.30967999e-7 | 1.99611972e-6 |
| CUDA | SparseStructureEncoder | 600 | 6.67572021e-6 | 1.80022135e-5 |
| CUDA | SparseStructureDecoder | 600 | 3.79979610e-5 | 3.85377480e-4 |

Independent host replay recomputes all **4752 array pairs / 32 runtime metadata
packets**, including every requested input variant; all saved comparison metrics
match the recomputation exactly. Complete current common admission accepts **six
L3 records / zero rejected**: models **3 / 9** separately on CPU and CUDA, all other
axes zero. The trainer axis remains **0 / 15**; ordinary real-model SGD trajectories
do not establish an upstream Trainer or pretrained pipeline. On copies only, all
192 actual per-phase missing/changed-source, incomplete/extra-payload, version and
origin mutations are rejected. Three honestly rehashed common-auditor attacks target
the CUDA Flow base, third perturbation and training-reference metadata; each rejects
only that affected row while the other five remain accepted.

Both sides use fresh private Python 3.11 environments, NumPy 1.26.4, einops 0.8.2
and easydict 1.13. The independent Torch 2.11.0+cu128 prefix is newly extracted from
the complete official wheel SHA256
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`;
all **11842 / 11842** original files, including RECORD, are verified before and after
execution. Actual `_C` SHA256 is
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`.
No shared package was repaired or replaced. CPU masks CUDA before imports. CUDA
uses one explicitly allocated RTX 4090, driver 580.178.04 and real NVCC 12.2.140;
the oracle build is CUDA 12.8. Both use strict native kernel math, FP32, TF32 OFF,
matmul `highest`, cuDNN benchmark OFF, two threads and matching actual affinity.
Actual process environments prove distinct CPU/CUDA unit caches, structure cache
and unused benchmark root. Explicit candidate library probes again return cuDNN
9.24.0 / CUDA runtime 12.2; the cuDNN distribution metadata difference remains visible.
No timing above grants a speed or L4 conclusion.

Focused host contracts pass **83 / 0 skipped**, including framework-free `python -S`
proof execution. Full source structure checks pass **1624 / 4 skipped**; two native
checks are enabled by this private editable core installation. Ruff and layout pass.
All 2882 tracked files and link bytes remain unchanged through the completed
measurement and independent audits. The initial host-only manifest enumerations
rejected Git-quoted non-ASCII names and a tracked directory symlink before writing;
NUL-separated names and Git link bytes resolved that diagnostic without modifying
source or runtime evidence. The 64 test warnings are the unchanged upstream invalid
escape sequences; they do not hide skips or failed nodes.

Reproduce using the existing skill and `ecosystem_trellis` session, fresh private
interpreters, the same clean upstream pin and a new absolute external
`JITTOR_TRELLIS_EVIDENCE_DIR`. Exact commands/environments, actual platform files,
original numerical arrays and all proof/negative results are unversioned under
`$JITTOR_LAB_ROOT/_state/trellis-installed-source-fresh/20261016/`.
`scorecard-installed-source-v1.json` binds the six accepted rows;
`independent-{cpu,cuda}-raw-audit-v1.json` and
`accepted-frozen-source-and-cross-device-v1.json` preserve independent results.
The twice-reread `sealed-source-bound-raw-manifest-v1.json` binds
**179 files / 842528632 bytes**, including 136 original raw files, complete provider
proof and the official Torch wheel. Its SHA256 is
`afe80a51c9c9df8846d9bc7a0c7413bbdfb3a2222286174aa62ec2e086f50436`.
The subsequent documentation change does not relabel these b694 measurements.
All full-pipeline, sparse-extension, pretrained, additional precision/backend,
whole-process memory and performance boundaries above remain open.

## Central integration review (2026-10-16)

Owner: Jittor core maintainers. Integration source is clean
`e05eaa4c0`; measured source remains the original full `b6947541b5bddfa87098d40c6f4e09ad0e704c54`
above. Recheck when the producer, collector, source proof, upstream pin or shared
admission rules change. The integration adds no new numerical measurement.

The integration owner independently recomputed all **4752 array pairs and 32
metadata packets**, without importing Jittor, using the original complete CPU/CUDA
artifacts and original criteria. The integrated collector and common auditor again
accept **six L3 records, zero rejected**, with CPU/CUDA models each **3 / 9** and
trainer each **0 / 15**. All **2882 original Git blobs**, including tracked symlink
target text, match the frozen measurement snapshot. All **11842 actual Torch
provider files** match their original proof. Both the original 179-file raw seal
and final **184 files / 842586487 bytes** seal were independently rehashed without
mismatch; the latter SHA256 is
`1451445897974e5e4339dcb159f271aae1216c5231902b30c1aa3dee9edffe47`.

The integrated main tree passes 68 TRELLIS host contracts plus 15 persistent-output
contracts, core **322 passed / 42 skipped / one existing expected failure**,
`tests/structure` **1508 passed / 6 skipped**, and `compat/tests/structure`
**114 passed / zero skipped**. Layout, generated manifest check, changed-Python
Ruff and diff checks pass. The six structure skips retain their actual reasons,
including two interpreter installation-resolution checks; import-path execution
is not relabeled as a private editable installation. Root verification outputs,
terminal logs and the fresh common scorecard are unversioned under
`$JITTOR_LAB_ROOT/_state/trellis-main-source-integration-dfe/20261016/`.
No old metadata or raw measurement was changed, and no broader pipeline,
pretrained, trainer, backend or L4 conclusion is granted.
