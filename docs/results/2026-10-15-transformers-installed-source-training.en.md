# Transformers: full installation provenance and original training-matrix re-measurement

- Status: CPU ① and real CUDA ③ pass over the original full scope; the independent unified scoring gives 20 accepted, 0 blocked, 0 rejected.
- Date: 2026-10-15.
- Measured source: `b4e46151dff102b0fe92697d7c1a7e53acaf7a3b`, all formal processes on the same clean commit.
- Remote synced at kickoff: `9ba68263b267c78799ec254e2b71abd865f41094`, plus explicit integration of the completed PEFT tooling as a prerequisite.
- Maintainer: Jittor Torch compatibility maintainers.
- Recheck conditions: changes to the pinned upstream or wheel, model constructors, the shared runner/training probe, computation implementations, provenance capture, or the scorer.

This run re-measures the original TF18 and Longformer2 **10 cases × CPU/CUDA × four phases × two independent runtimes**, and completes the binding of the actual installed packages and upstream source. The original models, all parameters and floating-point input gradients, the two input perturbations, three-step training, and numerical budgets are retained; no model math, shared runner, compatibility computation, or new adapter was modified. The old twenty rows lacked a complete installation provenance at the time, and the fact that the current scorer rejects their new-provenance admission is separately sealed; the old numerical evidence is neither rewritten nor backfilled. This report describes only the new formal processes and raw artifacts.

## Pinned source and actual platform

| Item | Actual identity |
| --- | --- |
| Transformers | Official 4.56.2, clean source `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| Standalone oracle | Python 3.11, NumPy 1.26.4, complete official binary Torch `2.11.0+cu128`, private prefix |
| CPU | AMD EPYC 9554, Linux/glibc 2.34; both sides explicitly CPU, the candidate disables CUDA/oneDNN/MPI/NCCL |
| CUDA | One NVIDIA GeForce RTX 4090, driver 580.178.04; all 80 formal processes use the same physical card |
| Compilation and precision | GCC 11.5.0, candidate NVCC 12.2.140; FP32 requested, TF32 matmul/cuDNN enabled and benchmark disabled on the CUDA side of both, TF32 disabled on CPU |
| Isolation | 2 threads each; runtime × device each with its own HOME/JITTOR_HOME/TMPDIR/XDG/shim cache; no shared unit-test cache |

Each formal process checks the actual import root, the preferred distribution, the official METADATA/RECORD, and all **2173 Transformers payload files** before and after computation, including 2147 Python files and C++/CUDA/header files. The upstream package has 2175 files: GPT-2 `CONVERSION.md` and T5 `download_from_gcp.sh` do not enter the official wheel, and the inventory explicitly records these two non-runtime files without falsely recording them as installed. Each worker binds **1562 framework/compatibility-layer/related-tool source hashes**; in addition, the mode/blob/byte hashes of all **2842 Git files** in the measured commit are recorded, symbolic links are handled as their target text, and the actual compat owner is bound.

Official Transformers wheel SHA256:
`79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`.
The oracle uses the complete official Torch wheel, SHA256:
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`,
verifying all **11842 payload files**, the RECORD hashes and sizes one by one, and checking the actual private prefix again. The actual `_C` SHA256 is `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`.
The oracle has no shim marker; no Torch wheel bytes were changed and no version was faked.

The actually mapped `libcudart` API query is **12020** for the candidate and **12080** for the oracle, each retaining its own real path, hash, and preferred NVIDIA metadata. The candidate's cuBLAS/runtime/cuDNN metadata are 12.2.5.6 / 12.2.140 / 8.9.7.29, and the oracle's are 12.8.4.1 / 12.8.90 / 9.19.0.56. The private oracle uses the standard adjacent NVIDIA dependency layout and references the real cu128 namespace and real distribution metadata read-only; cu128 is not relabeled as cu122. The candidate's actual NVCC/G++, key headers, runtime, and metadata/RECORD are also bound. A mapped library does not mean the model uses the library's full compute capability; this run makes no communication, NCCL, or jtrun capability claims.

## Original complete numerical scope

Four phases per device: base seed 0, input perturbation seed 7, input perturbation seed 19, three-step training; two real processes per phase. Both sides load the complete oracle initial parameters and buffers. An additional independent check confirms that the **20 sets of initial weights, 870 parameter/buffer arrays, are byte-identical value by value** on CPU/CUDA, with identical model configurations, input specs, and phase seeds.

| Case | Parameters / trainable / frozen | base and each perturbation array | three-step arrays | total arrays per device |
| --- | ---: | ---: | ---: | ---: |
| BERT | 39 / 39 / 0 | 40 | 240 | 360 |
| CLIP multimodal | 78 / 78 / 0 | 80 | 477 | 717 |
| GPT-2 | 28 / 28 / 0 | 29 | 174 | 261 |
| Llama | 21 / 21 / 0 | 22 | 132 | 198 |
| Llama RoPE 2048 | 12 / 12 / 0 | 13 | 78 | 117 |
| Mixtral MoE | 41 / 41 / 0 | 42 | 252 | 378 |
| T5 | 47 / 47 / 0 | 48 | 288 | 432 |
| ViT | 40 / 40 / 0 | 42 | 249 | 375 |
| Whisper | 89 / 88 / 1 | 91 | 540 | 813 |
| Longformer 32768 | 29 / 29 / 0 | 31 | 183 | 276 |
| Total | | | | **3927** |

RoPE keeps the real length **2048**, max position 512, and linear factor 4; CLIP includes both towers and pixel input gradients; Mixtral keeps four experts/top-2 and gradients of all experts and the router; Whisper requests gradients for all 89 parameters in base/perturbation, keeps 88 trainable and one frozen positional embedding in training, and checks feature input gradients. Longformer keeps `[1,32768]` tokens, `[1,32768,32]` embedding noise and its gradients, window 256, global attention, and gradients for all 29 parameters, and the actual primary output has **1048608** elements. No shorter sequence was substituted for it.

The public `training_probe.py` is reused: three steps of manual SGD, learning rate 0.001, the original fixed complete weights, step input seeds 1/2/3. Each step compares the output, loss, gradients of all trainable parameters, all floating-point input gradients, and all parameters; frozen parameters must be byte-identical value by value, trainable parameters must actually update, and the two perturbations must actually change both sides' outputs. In addition, the 32 actual special-scenario artifacts per device are checked for nonzero gradients of the four experts, Longformer global q/k/v and noise gradients, floating-point input gradient shapes, and Whisper's frozen invariants; no worker or coverage numerator was added.

## Original budgets and results

TF9 keeps the original **whole-phase reference-array maximum-magnitude** normalization: `max_abs <= atol` or `max_abs / phase_reference_max <= rtol`. base is 0.002/0.01 on both devices; CPU perturbation/training remain 0.002/0.01, and CUDA perturbation/training remain 0.005/0.02. This is not an elementwise relative-error claim.

Longformer keeps the original per-array rule: gradients/input gradients are normalized by
`max(reference_array_max, 0.001 * step_gradient_block_max, 1e-6)`; outputs, loss, and parameters use their own array magnitude. CPU forward/backward is limited to 0.002/0.01 and CUDA to 0.005/0.02, all retaining the original strict-less-than criterion. Huge outputs cannot mask small gradients. The new comparator additionally recomputes read-only all **7854 historical array pairs** and confirms the budgets are unchanged; these old arrays do not count as the new-provenance measurements of this run.

| Formal device | Cases / independent processes / array pairs | TF9 maximum original whole-phase normalized error | Longformer maximum original array/gradient-block normalized error | Candidate fallback |
| --- | ---: | ---: | ---: | ---: |
| CPU ① | 10 / 80 / 3927 | 1.7956447e-6 | 9.8483680e-6 | error / 0 |
| CUDA ③ | 10 / 80 / 3927 | 1.1524663e-2 | 2.7364785e-5 | error / 0 |

The maximum CUDA TF9 difference is in the **gradient of `decoder.block.0.layer.2.DenseReluDense.wi.weight` at T5 seed 19**: maximum absolute error **0.3834403**, original-phase-magnitude normalized **1.1524663e-2**, within that phase's original **2e-2** budget. This actual precision boundary is retained and is not reduced by changing inputs, disabling TF32, or loosening the budget. The maximum CPU/CUDA Longformer absolute errors are **0.0166015625 / 0.060546875** respectively; their normalization rule differs from TF9. CUDA checks the residency, driver pointer attributes, and UUID of the actual input/output/parameter/all requested gradients; all formal phases use the same physical card. The oracle's fallback field is a placeholder from the old runner and is not treated as a Jittor fallback measurement.

After the independent collector/unified scorer re-read all artifacts: **20 accepted, 0 blocked, 0 rejected, 7854 array pairs**. The 11842 files of the complete official Torch are fully re-verified after the formal run. Both formal maintenance nox runs exited successfully, with full CPU/CUDA runs taking about 33/40 minutes respectively, including provenance checks and first JIT; the raw per-worker timings are retained as diagnostics only and do not grant L4 or performance ratios.

The coverage axes of the pinned source remain **9/374 model families, 2/46 task mappings @ cd74917f**; the two Llama scenarios count as one model family. Longformer keeps the original broad, and CLIP/MoE/RoPE keep the tail. More complete provenance does not mean added models or whole-library capability.

## Maintained entry point and independent recomputation

Added `tools/run_transformers_gate.py`, a dedicated collector/helper, and
`nox -s ecosystem_transformers_source`, reusing the existing case/runner/train math. The maintenance entry point enforces all ten cases, four phases, and 80 processes; each case automatically checks full provenance, weights, observations, perturbations, frozen parameters, and all numeric values, and a worker rc0 cannot pass on its own. A non-empty output is refused for reuse; `--preflight` is only two BERT base processes and does not grant L3. nox only allows passing through output/cache paths and refuses shrinking cases/phases.

```bash
export JITTOR_CI_PYTHON=<candidate-python> REAL_TORCH_PYTHON=<genuine-binary-torch-python>
export JITTOR_TRANSFORMERS_CHECKOUT=<clean-official-transformers-checkout>
export JITTOR_TRANSFORMERS_WHEEL=<official-transformers-wheel>
export JITTOR_TEST_DEVICES=cpu
<audit-python> -m nox -s ecosystem_transformers_source -- \
  --output <new-external-cpu-output> --cache-root <private-benchmark-cache>
# CPU 完成后，显式分配一张真实 CUDA 卡和实际 nvcc/provider，使用新的 output。
export JITTOR_TEST_DEVICES=cuda
<audit-python> -m nox -s ecosystem_transformers_source -- \
  --output <new-external-cuda-output> --cache-root <private-benchmark-cache>
```

The original `test_ecosystem_parity.py` and `test_ecosystem_longformer.py` nodes keep their own existing parity scope; they cannot be treated as the complete L3 automatic gate above. This run does not claim that a scheduled nightly has selected the new nox session. Independent recomputation command:

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_transformers_scorecards.py \
  --run "$TF_STATE/formal-v5/cpu" --run "$TF_STATE/formal-v5/cuda" \
  --transformers-checkout "$JITTOR_TRANSFORMERS_CHECKOUT" \
  --transformers-wheel "$JITTOR_TRANSFORMERS_WHEEL" \
  --baseline b4e46151dff102b0fe92697d7c1a7e53acaf7a3b \
  --report docs/results/2026-10-15-transformers-installed-source-training.md \
  --output <new-external-audit-output>
```

`TF_STATE=$JITTOR_LAB_ROOT/_state/transformers-source-bound/20261015` (unversioned).
The formal `formal-v5/{cpu,cuda}`, the independent `independent-scorecard-v5`, the source/package/library bindings, the failure history, and the final `evidence-manifest-v5.json` are all there; the canonical environment entry points are `run-formal-cpu-v5.sh` and `run-formal-cuda-v5.sh`, and the independent recomputation is wrapped as `recheck-v5.sh`.
Raw logs, binaries, and caches do not enter the repository.

## Counterexamples, failure history, and boundaries

12 host contracts pass, covering both numerical normalizations, the Longformer gradient floor, real provenance paths, private cache/device routing, and controller failure due to a numerical error; they are not treated as real L3 evidence.
In addition, actual metadata copies are used to verify nine rejection counterexamples: missing pixel gradient, missing output, missing complete installation provenance, wrong RECORD, nonzero fallback, HOME deviation, missing installation hash at seed19, changed precision at seed19, and changed training source hash.
The dedicated independent collector verifies every real process of base/seed7/seed19/train and does not rely on the old central scorer's hole where the roles of additional perturbations had not yet been verified.

preflight-v1 failed at the provenance path of the extra platform helper and did not compute; after the fix a direct file-path contract was added.
v3's complete CPU matrix and CUDA base preflight passed, but the first formal CUDA oracle then failed to import because the shared Torch symlink target lacked `compile_time_profiler.py`. The original failure, directories, and link identity are retained, and no inference is made about who changed anything.
The new complete official Torch private prefix's first import then exposed ELF lookup falling into the old CUDA runtime; that v4 undefined-symbol failure and the loader trace are retained. After adopting the real adjacent cu128 dependency layout, CPU→CUDA was fully rerun with the new v5 output, without changing Torch wheel files and without relabeling the old CPU numbers as v5.
After the initial card's preflight only about 3.56 GiB of memory remained, so another idle card was coordinated and the preflight rerun before the formal CUDA run started; all 80 formal CUDA processes stayed on the same card throughout. Optional gdb/path diagnostics are kept in raw, and success is judged by the actual rc/numeric values.

This run closes only the provenance and complete maintenance-gate gap of the original ten-scenario FP32 L3; it does not grant Trainer, generation, optimizer-class, checkpoint/resume, full pretraining-family, BF16, multi-rank, NPU/ROCm, or L4 performance capability.


## Mainline integration and historical-provenance re-check

Mainline first synchronized remote `593249c5f0e82febb931234aa35003d0d41c8b33`, then integrated the tooling and the report; the clean integrated source is `09e9e8eb8f67f4b32e3b46e05174013c1216983f`. The ASTs of the existing 72 nox functions (Swift, VERL, PEFT, etc.) stay consistent, and a complete Transformers session is added.
On this source the host contracts are **91 passed, 0 skipped**, CPU core is **322 passed, 42 skipped, 1 existing xfail**, and the complete structure check is **1622 passed, 6 skipped**; Ruff, layout, MANIFEST, and diff checks pass. These checks do not replace full model measurements on that source.

An independent read-only review of the originally sealed **1627 raw/auxiliary files, 14100 external files, and 2842 original Git files** found all hashes consistent; recomputation in the original measurement worktree is still **20 accepted, 0 blocked, 0 rejected, 7854 array pairs**. The sealed manifest SHA256 is
`7545c8ed8d91462cc14fb061b5eededb5a3c256e72852cafff137e47d389d946`.
In addition, the original B4 Git blobs were read one by one and the 1562 framework/tool source hashes captured by each worker were checked, with zero differences.

The historical collector command above must run in the measurement worktree whose corresponding provenance is unchanged. The current mainline has 11 changed captured paths and 9 newly added captured paths relative to B4; recomputing the original measurement directly on the current mainline is expected to be rejected with `Measured framework/tool bytes changed`. That failure is retained; the check was not loosened and the old metadata was not rewritten.
The complete CPU/CUDA measurements of the current source run in a new isolated worktree, and the results must record their own actual provenance and cannot borrow the historical numbers above. The mainline re-check logs and path differences are in
`$JITTOR_LAB_ROOT/_state/transformers-main-integration-593/` (unversioned).
