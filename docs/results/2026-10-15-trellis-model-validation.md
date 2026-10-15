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
