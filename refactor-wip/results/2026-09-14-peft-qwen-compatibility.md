# PEFT 0.17.1 Qwen compatibility validation

## Scope

This validation targets PEFT 0.17.1 and Transformers 4.56.2 through the independent Jittor Torch frontend. The only base architecture used for acceptance is Qwen2: a fixed tiny `Qwen2ForCausalLM` configuration for the method matrix and `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987` for the public-checkpoint CUDA workflow. The oracle is PyTorch 2.6.0+cu124. Results do not cover quantization, distributed training, `torch.compile`, NPU/ROCm, or newer PEFT/Transformers releases.

PEFT registers 26 `PeftType` values in this version. Native PEFT rejects `ADAPTION_PROMPT` for `model_type=qwen2`; the other 25 types are usable upstream. rsLoRA and DoRA are LoRA configuration variants and are reported outside that denominator.

## CPU method matrix

All 25 Qwen2-usable types complete the standard CPU protocol: construction, deterministic forward/backward, all expected finite trainable gradients, one public SGD step, frozen-base equality, adapter save, same-base fresh load, and post-load forward. CPT has one exact upstream exception, `prompt_encoder.default.embedding.weight: grad=None`. rsLoRA and DoRA complete the same protocol. The reusable gate is `compat/tests/torch/test_peft_qwen_methods.py`; the full experiment artifact is `/home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/jittor/strict_complete_matrix_final.json`, with XLORA in `jittor/xlora_reload_diagnostic_v3.json`.

Method-specific checks cover AdaLoRA rank allocation, Multitask Prompt Tuning and Poly task IDs, CPT token masks, VBLoRA vector-bank perturbation, FourierFT and RandLoRA seeded reconstruction, SHIRA sparse updates, and XLORA routing through two distinct nonzero experts. The LoRA+LoHa `PeftMixedModel` test initializes every multiplicative LoHa factor to a nonzero value and observes nonzero LoRA-only, LoHa-only, and joint deltas with 12 finite gradients. PEFT 0.17.1 does not expose MixedModel save support, so no checkpoint claim is made for that type.

VeRA is accepted in the main matrix with `save_projection=False`, which reconstructs projections from `projection_prng_key`. PyTorch and Jittor both reload this configuration. PEFT 0.17.1 defaults `save_projection=True`; when projections are unchanged, same-seed construction reloads, but deliberately changed projection keys are saved without the target `.default` component and are ignored by upstream `strict=False` loading in both runtimes. That upstream behavior is not reported as complete persistence support.

## Task and lifecycle coverage

The same tiny Qwen2 fixture passes LoRA training and fresh reload for sequence classification, token classification, question answering, and feature extraction. Transformers 4.56.2 has no Qwen2 mapping in `AutoModelForSeq2SeqLM`, so fixed-Qwen2 Seq2Seq is an upstream interface boundary.

Adapter lifecycle coverage includes add/set/disable/delete/load, `modules_to_save`, config and state-dict APIs, AutoPeftModel loading, merge/unmerge/unload, atomic safe-merge NaN rejection, weighted linear/cat/SVD merges, and CPU/CUDA placement. PyTorch-to-Jittor-to-PyTorch safetensors keys, shapes, dtypes, and tensor hashes agree; output error is at most `6.33e-8`. Cross-runtime resumed-step gradient error is at most `2.68e-7` and update error at most `3.73e-9`. Raw evidence is under `/home/cgq/jittor-lab/_state/PEFT/adapter_lifecycle_20260913/`.

## Public checkpoint CUDA result

The FP32 Qwen2.5-0.5B LoRA workflow uses q/v projections in all 24 layers, rank 2, alpha 4, and dropout 0. Both runtimes complete three AdamW steps, produce finite gradients for all 96 trainable tensors, leave every frozen tensor unchanged, reload adapter logits exactly, continue training, and execute greedy and beam generation, cache prefill/decode, and `inputs_embeds`. Jittor cache length changes from 5 to 6, cached/full next-token logits differ by `2.34e-5`, and input-id versus embedding logits are exact. Jittor's observed process allocation was about 4.83 GiB; its CUDA memory API did not return a peak counter.

After aligning the public AdamW default weight decay to PyTorch's `0.01`, Jittor updates 96/96 trainable tensors at every step. The final PT v5 and JT v2 artifacts each contain the same 1,728 trajectory keys: all 96 step-0 parameters are exact, and the 288 optimizer step values are 96 copies of 1, 2, and 3 at the corresponding steps. Per-step gradient maximum absolute differences are `6.08e-5`, `4.03e-5`, and `1.14e-4`; per-step post-update parameter differences are `3.01e-4`, `3.06e-4`, and `3.08e-4`. Ordinary end-to-end gradient differences near AdamW's `eps=1e-8` are amplified by normalization. A separate same-gradient optimizer replay agrees to `3.73e-9`, which validates optimizer semantics but does not replace the end-to-end result. The workflow is functionally complete; it is not reported as exact parameter-trajectory parity. Final artifacts are `pytorch/real_qwen_cuda_v5_fixed_steps_and_initial/` and `jittor/real_qwen_cuda_v2/` under `/home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/`; PT trajectory SHA256 is `9d535fed...`.

The public 0.5B run explicitly loads FP32, does not use autocast or GradScaler, and uses eager Qwen2 attention. PyTorch has matmul TF32 disabled; Jittor uses `JT_CUDA_KERNEL_MATH=strict`. Separate tiny-Qwen2 LoRA lifecycle probes cover FP16 and BF16: PEFT autocast keeps LoRA A/B in FP32, and both probes pass finite gradient, frozen-state, update, and same-runtime reload checks. Their cross-runtime logits errors are `2.62e-4` and `2.05e-3`, respectively. No low-precision 0.5B run was performed.

The final tiny CUDA method batch uses one fixed order: construct the base on CPU, restore the fixture, inject PEFT, and then explicitly move the model and all inputs to CUDA. It covers 24 non-XLORA usable PeftTypes plus rsLoRA and DoRA. PyTorch passes all 26 execution lines. Jittor initially passes 24/26; OFT then passes its strict gate after advanced-index placement alignment, and BOFT passes after installing the fixed official CuPy CUDA provider dependency. Every final Jittor line preserves parameter identity, reports backend-1 GPU placement, is synchronized on the target device, and reloads with zero output error. A separate fixed two-expert XLORA CUDA fixture also passes with a cross-runtime routing-delta difference of `1.02e-8`, 2/2 finite classifier gradients, unchanged frozen parameters, and zero reload error. These results establish CUDA evidence for all 25 Qwen2-usable PeftTypes; rsLoRA and DoRA remain variants rather than additional PeftTypes.

## Compatibility changes and boundaries

The work repaired frontend protocol and native fidelity used by PEFT: module representation, CPU/CUDA placement during load, frontend Linear MRO and execution, live ModuleDict views, sparse in-place graph preservation, deepcopy data ownership, `Tensor.uniform_(generator)`, `remainder`, `normal`, `triu_indices`, native CPU Generator/randperm state, `torch.linalg.solve(left=False)`, and the public AdamW default. Directed tests cover each shared behavior rather than patching PEFT.

The protocol bindings, module/load behavior, overloads, generator consumption, public optimizer default, and public seed barrier are C2 compatibility repairs. Right-side `linalg.solve` and sparse-to-dense addition reuse existing differentiable device operations and are C3 compositions. Exact CPU Generator/randperm state and value generation required native C4 support.

Two serialization/device boundaries remain explicit. Constructing these seeded methods on CPU and then moving them is supported, while an explicitly CUDA-bound `torch.Generator` is rejected rather than silently emulated. Adapter safetensors round-trip in both directions and same-checkpoint optimizer continuation are proven, but a Jittor optimizer `.pt` pickle is not readable by native PyTorch (`Invalid magic number`); this does not affect adapter safetensors portability.

Top-k/top-p sampling executes and returns three score tensors of shape `[2, 41]`. The initial eager-style failure was caused by pending random weight graphs consuming a later sampling seed. After the public seed barrier repair, the original unmaterialized protocol produces exact repeated sequences and scores without a user warmup; CPU evidence is `/home/cgq/jittor-lab/_state/PEFT/core_training_20260913/sampling_probe_seed_barrier.json`, while CUDA rand/multinomial and aliases are included in the final 23-case CUDA gate. BOFT's optional upstream `fbd_cuda` extension does not compile in either environment and both use PEFT's public fallback. That fallback is now proven on CUDA: fixed `cupy-cuda12x==13.6.0` and `fastrlock==0.8.3` wheels enable the official `numpy_code` GPU provider; batched left/right solve, analytic gradients, the original `block_diag`, and the full BOFT train/save/reload path pass. The environment manifest is `/home/cgq/jittor-lab/_state/PEFT/environment_cupy_20260914/manifest.json`.

## Validation

The complete PEFT source gate passes `2 passed` with no skips. Its first unittest node executes 26 subtests (24 non-XLORA PeftTypes plus rsLoRA and DoRA); its second node executes XLORA, for 27 method execution lines. The final-source rerun took 646.64 seconds because `peft_core_training_rng_cpu/cfgcfcd30e6` compiled previously unseen method-shape operators; Jittor was explicitly set to CPU after import. The harness reported bounded FFT cache and process flag state left by this single test file, but did not classify them as failures:

```bash
python -m pytest -q compat/tests/torch/test_peft_qwen_methods.py
```

The method, sparse, and solve directed selection passes `10 passed` with no skips in 17.50 seconds:

```bash
python -m pytest -q \
  compat/tests/torch/test_peft_qwen_methods.py \
  compat/tests/torch/test_sparse_dense_add.py \
  compat/tests/torch/test_torch_compat_linalg.py \
  -k 'peft or sparse or solve'
```

The final combined CUDA compatibility selection passes `23/23` in 7.37 seconds from `/home/cgq/jittor-lab/_state/PEFT/core_training_20260914_final/cuda_combined_23.log`; the affected structure selection passes `33/33`, and repository layout passes. The repository's root structure command, `python -m pytest -q tests/structure`, passes `1331` tests with two accelerator skips and no failures in 220.89 seconds; its log is `/home/cgq/jittor-lab/_state/PEFT/core_training_20260914_final/tests_structure_root_final.log`. A separate, broader compatibility-specific run under `compat/tests/structure` is `105 passed, 9 failed`; all nine failures reproduce on the untouched `b67236dc` baseline (`9/9` failed for the same reason in 265.63 seconds, log `/home/cgq/jittor-lab/_state/PEFT/core_training_20260914_final/structure_base_b672_9nodes.log`). These are distinct test trees: the required root structure gate is green, while the nine extra compatibility failures are recorded as pre-existing baseline behavior rather than hidden or claimed green. This report records measured support only and keeps low-bit quantization, distributed training, compilation, and unrelated model families outside its claims.
