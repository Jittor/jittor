# Transformers model-family CPU/CUDA verification matrix

- Status: measured; nine tiny scenarios reach L3, complete Transformers support is still unfinished.
- Date: 2026-10-15, Asia/Shanghai.
- Measurement baseline: `8e388d108eb83c1c438936709f29855f47ba2a69`, working tree clean before the run.
- Maintainer: Jittor Torch compatibility maintainers.
- Recheck conditions: changes to model builders, the compatibility layer, the training probe, Transformers, Torch, drivers, or precision settings.

## Scenarios and denominators

Both sides use Transformers 4.56.2, clean upstream source commit
`cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`. The denominator re-extracted from that version's registry is
374 model families and 46 task classes. CPU and CUDA each verify **8/374 model families, 2/46 task classes**;
366 model families and 44 task classes respectively remain unverified in this round. NPU, ROCm, and Corex have zero verification this round.
The two Llama scenarios contribute only one model family. Sampled by scenario: the original six core items plus three new tail items,
each run on CPU/CUDA, hence 12 core cells and 6 tail cells, with broad at zero.

All scenarios are small configurations with fixed random weights, running forward and backward with the real upstream model; the original
weights are then fixed, the inputs are changed with seeds 7 and 19 respectively, and it is confirmed that both sides' outputs really change, the requested gradients are complete, and the values are finite.
In addition, three steps of SGD with learning rate 0.001 are run, comparing each step's loss, parameter gradients, floating-point input gradients, and parameter trajectory.
The training input seeds are 1/2/3; training keeps the original model's frozen partition. The tensor counts in the table below are per-side archived array counts,
and the training error is the training comparator's maximum field-scaled relative error.

| Scenario | Trainable / frozen parameters | Base arrays | Three-step arrays | Training error CPU / CUDA |
| --- | --- | --- | --- | --- |
| bert | 39 / 0 | 40 | 240 | 2.18e-07 / 8.92e-07 |
| clip_multimodal | 78 / 0 | 80 | 477 | 6.62e-07 / 0.00114 |
| gpt2 | 28 / 0 | 29 | 174 | 2.38e-07 / 0.000168 |
| llama | 21 / 0 | 22 | 132 | 2.09e-07 / 0.000161 |
| llama_rope_2048 | 12 / 0 | 13 | 78 | 1.79e-07 / 0.000115 |
| mixtral_moe | 41 / 0 | 42 | 252 | 1.19e-07 / 8.45e-05 |
| t5 | 47 / 0 | 48 | 288 | 3.36e-07 / 0.000355 |
| vit | 40 / 0 | 42 | 249 | 5.17e-07 / 6.12e-05 |
| whisper | 88 / 1 | 91 | 540 | 3.83e-07 / 0.000336 |

- GPT-2, Llama, Mixtral, and RoPE Llama correspond to `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`.
  BERT, ViT, T5, Whisper, and CLIP correspond to `MODEL_MAPPING_NAMES`.
- BERT and ViT check both hidden states and pooler output, so the pooler parameters participate in backward.
- CLIP uses a real text and image dual tower, text shape `(2,8)`, image `(2,3,16,16)`, checking text-image logits,
  all parameters, and pixel gradients. hidden 32, two layers, four heads, projection 16, patch 8.
- Mixtral uses two layers, hidden 32, intermediate 64, four experts with top-2, input `(2,32)`;
  the gradients of each layer's per-expert w1/w2/w3 and the router are all required to exist. This round does not certify the router auxiliary loss.
- RoPE Llama actually takes input length 2048; original max position 512, linear factor 4, single layer hidden 32.
  This does not prove availability of 32K context, generation, or full-size models.
- T5/Whisper are base encoder-decoder models, with no LM head/generation verification.
  The Whisper base comparison re-enables gradients for all parameters; training keeps one frozen positional embedding,
  so the base and training gradient parameter denominators differ. Floating-point input gradients include Whisper features and ViT/CLIP pixels.

## Platform and acceptance

Python 3.11.16, NumPy 1.26.4, Jittor 2.0.0; the standalone binary oracle is
PyTorch 2.11.0+cu128, with the native `_C` extension checked and the shim marker false.
Linux 6.1.188/glibc 2.34, AMD EPYC 9554, CUDA uses a single RTX 4090, driver 580.178.04.
The candidate has CUDA toolkit 12.2.140 and GCC 11.5; the oracle has CUDA 12.8 and cuDNN 91900.
The candidate's actual cuDNN API/header is 92400 while the installation metadata declares 9.26.0.51;
the report retains this discrepancy, the external artifacts bind the actual library hashes, and the metadata is not taken as the loaded library version.

CPU disables CUDA/ACL, with TF32 off and matmul precision highest; CUDA enables TF32 matmul/cuDNN,
precision high, and cuDNN benchmark off. The actual Jittor auto graph replay is 1, retain budget
268435456 bytes, and CUDA kernel math is strict. Both sides use 2 threads, with identical run conditions per phase.
The device mapping, complete wheel provenance, binary paths, complete model configurations, and all parameter names are kept in the external platform artifacts.

The base `verify_repo.py` uses the existing default atol 0.002 / rtol 0.01; training and perturbation comparisons follow
CPU 0.002/0.01 and CUDA 0.005/0.02. Each phase is accepted array by array against its own threshold first; the scorecard summary's upper bound
takes the maximum threshold across phases and stores `stage_tolerances`; the summary upper bound cannot be used to explain the stricter base phase.
The CUDA T5 seed 19 perturbation has a maximum scaled relative of 0.011524663, matching the preset 0.02,
not the base phase's 0.01, so both conventions are explicitly retained. No threshold was loosened for this round's results.
These field-scaled metrics differ from the shared pytest per-tensor divergence metric.

The actual pytest ecosystem gate was additionally run as CPU 9 passed / CUDA 9 passed, both with zero skips,
with CUDA's minimum actual execution count required to be 9. The independent training and perturbation processes enable fallback=error, with an actual count of zero.
This proves zero fallback for the measured scenarios; since no L4 real-size performance acceptance was done, the highest level in this round remains L3.
The shared machine, single diagnostic timing run, and lack of complete process peak sampling do not satisfy the conditions for a speed conclusion.

## Reproduction and evidence

Use the [unified runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md)'s standalone interpreter,
same-weights procedure, and the `training_probe.py` three-step training command. Warm up serially first, use independent caches for unit tests and benchmarks,
and independent caches for CPU/CUDA and parallel tasks as well. The base command is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_TRANSFORMERS=1 \
JITTOR_ECOSYSTEM_CACHE_ROOT="$RUN_ROOT/benchmark-cache" \
<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases <case> --device <cpu-or-cuda> \
  --oracle-python <real-torch-python> --no-memory --repeats 1 --out "$RUN_ROOT/results"
```

CPU explicitly sets `CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path=''`; CUDA uses the
allocated device mapping and explicitly sets the compilation toolchain, and must not infer it from the default device alone. The full names of the nine cases are all
`transformers_` plus the scenario in the table. Perturbations reuse the original `.ref.weights.npz` and do not regenerate weights.
The complete dual-interpreter commands for each training and perturbation run are written into the corresponding comparison JSON.

Unversioned raw evidence is in `$JITTOR_LAB_ROOT/_state/transformers-matrix/final-8e388d108/`.
`source-manifest.json` pins the SHA256 of six executed source files; `independent-artifact-audit.json`
checks item by item the 18 scenarios, 468 artifact bindings, complete parameter gradient sets, non-empty input perturbation operations, training parameter partitions,
actual device/precision, dependency versions, and zero fallback. `scorecard-evidence-v2.json` binds the original measurements,
the auditor accepts 18 rows and rejects zero. The negative check correspondingly rewrote the CUDA platform JSON, labels, and their SHA while retaining the actual
measurements; all nine CUDA rows were rejected for a precision mismatch; the nine CPU rows continued to pass.

Key artifact SHA256:

- `independent-artifact-audit.json`: `bfd36e44cc7a34fddb2d07eebb73a9f7faf068c01490ea27acbdf77936e21304`.
- `scorecard-evidence-v2.json`: `3827caa84e8b37e75ca2226482748a5be5a3ab47ac5fc07bbe355b039024f13c`.
- `scorecard-v2.json`: `5239ab929958cfff3d22079295ce39b1409b2845334d7263c817796e2b45b87b`.
- `negative-precision-scorecard.json`: `5638335206d146066659389339cbe6a45e47f1bc860081aa9a527ad6f5a33003`.

## Conclusion boundaries

L3 refers only to the fixed configurations in the table and the three-step SGD numerical trajectory; it cannot be generalized to Trainer, AdamW, task convergence, real
checkpoints, bf16, multi-GPU training, or whole-family support. The 32K Longformer verification is a separate follow-up task and is not counted in this table.
The historical [real-size model report](2026-09-24-torch-compat-real-models.en.md) and
[H3 report](2026-09-12-minimax-h3-torch-compat.en.md) retain their own versions, baselines, and known issues,
and cannot be merged into new measurements of this commit or current family-coverage numerators.
