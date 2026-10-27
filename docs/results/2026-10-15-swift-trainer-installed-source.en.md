# Registered Swift FULL SFT with complete installed-source evidence

- Status: Validated for the bounded CPU/CUDA scenario below; L3, with zero observed backend fallback.
- Date: 2026-10-15
- Baseline commit (actual measurements): `606b602e7e79b5bfa5573b9693144f8d224c6a07`.
- Collector hardening: `f8731d454f95070c018af7ece9068a182f49f359`; this independently checks the stored wheel and complete payload after measurement, without changing tensors or their source identity.
- Shared auditor: `ac4c39d9da541c71e4ff9d083e55c0b69f52a660` (including the shared release-proof change `1ecea7d5879761ec487d2497a25ef07bad47ae17`).
- Owner: Jittor maintainers.
- Review when: Swift/Transformers/PEFT/Accelerate versions, model loading/template/Trainer selection, Torch hooks/autograd/optimizer/device behavior, or the source-proof and gate contracts change.

## Result and breadth

The official Swift model loader, Llama template and `SwiftSft.main()` / `Seq2SeqTrainer.train()` complete three real FULL SFT updates in independent genuine Torch and Jittor processes. The observer only attaches public forward hooks and a Trainer callback; Swift/Transformers/Accelerate own loss, backward, optimizer creation and stepping. There is no adapter or replacement training loop.

The clean Swift 4.5.2 checkout is `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` from `modelscope/ms-swift`. Denominators are regenerated from that exact checkout. Each row below is one scenario on each device, not a statement about every API or production checkpoint within a family.

| Declared registry axis | CPU | CUDA | Actual scenario |
| --- | --- | --- | --- |
| model_family | 1/222, L3 | 1/222, L3 | `llama`, real `LlamaLoader`, full tiny Llama |
| trainer | 1/13, L3 | 1/13, L3 | `causal_lm`, registered `swift.trainers.Seq2SeqTrainer` |
| task | 1/5, L3 | 1/5, L3 | `causal_lm` |

`FULL` is the official `tuner_type='full'` argument. Coverage of the separate 11-entry `SWIFT_MAPPING` tuner axis is 0/11 in this scorecard. CLI, RLHF, Megatron, distributed training, generation and larger models are unverified here. The historical Swift LORA scenario remains a separate case and baseline in [the earlier oracle report](2026-10-14-downstream-oracle-scorecards.en.md).

## Fixed protocol and provenance

The local checkpoint is created by genuine Transformers on CPU with seed 7: one layer, width 32, intermediate width 64, four query heads/two KV heads, vocabulary 64, 128 positions, untied embeddings and zero attention dropout. A real WordLevel tokenizer and the upstream Llama template produce 120 tokens per batch. There are 12 fully trainable parameter tensors, 13,408 scalar parameters, and one rotary buffer; no frozen parameters.

All runs start from the same six checkpoint/tokenizer artifacts. Training uses FP32, batch one, three updates, SGD at `.001`, a constant scheduler, zero weight decay, no gradient clipping, no AMP/checkpointing and no sample shuffling. The SFT pipeline encodes three actual message records. Training/data-loader seeds stay 0; input-record seeds are 0, 7 and 19. Perturbations preserve all initial parameters/buffers and change actual tokens and outputs.

`input_ids`, `attention_mask` and `labels` are integer model inputs. They have zero requested input gradients. Their dtype, shape, values and SHA256 are observed at the real model call and compared exactly. An additional retained embedding-output gradient is compared on every step; it is labelled as an intermediate gradient. Integer token magnitudes do not enter the floating error scale.

The selected class, loader, template and trainer identities come from the actual model/Trainer objects. Python call observations additionally confirm the real registered loader and template entry points and three calls each to `Seq2SeqTrainer.compute_loss` and `Template.compute_sft_loss`. The factory observation contains one official selection and one identity-verification call.

Swift is installed from an actual wheel built from the clean pinned checkout. Its SHA256 is `ae40b628d3b45e4916d07debdee091b6862ad6fedabf50eb3be526b6d0b7a9ae`. This is a source-built wheel, not a claim about a published PyPI wheel. All 517 Python source files and all 532 package payload files (including 15 data files) match the checkout, wheel and actual installation. Both runtimes hash the installed package before and after training. The collector independently reads wheel METADATA and every payload member and rejects missing, extra, stale or untracked Python contents.

Python is 3.11.16; Transformers is 4.56.2, PEFT 0.17.1, Accelerate 1.10.1 and NumPy 1.26.4. Both sides also record complete installed file hashes for Swift, Transformers, PEFT and Accelerate. The upstream denominator and release-wheel proof in this report concern Swift.

| Platform fact | CPU | CUDA |
| --- | --- | --- |
| Host | AMD EPYC 9554 | Same host |
| Accelerator | Disabled explicitly | One allocated RTX 4090 |
| Genuine oracle | Torch 2.11.0+cu128, CPU tensors | Torch 2.11.0+cu128, CUDA tensors |
| Candidate toolkit | CPU build | CUDA 12.2.140 / cu122 provider, SM89 |
| Driver | Not applicable | 580.178.04 |
| Compiler | GCC 11.5.0 | GCC 11.5.0 / NVCC 12.2.140 |
| Precision | TF32 off; `highest` | TF32/cuDNN TF32/autotuning off; `highest`; strict native CUDA kernel math |
| Native graph flags | Default replay, retained budget 268435456 bytes | Same |

The oracle has no shim installation context and loads an actual Torch `_C` extension with SHA256 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`. Platform artifacts include both interpreters' NVIDIA distribution inventories and actual device allocation. Every observed candidate parameter, output, loss, gradient and updated parameter is synchronized and checked for physical residency and native dispatch before host serialization; fallback is forbidden during the real training scope.

## Numerical and maintained-gate evidence

Each runtime trace contains 94 floating arrays: 13 initial state arrays plus every step's output, loss, extra intermediate gradient, 12 parameter gradients and 12 updated parameters. There are six paired comparisons / 12 raw NPZ files, 564 field comparisons and 1128 actual floating arrays. All arrays are finite and have matching exact shapes and dtypes. The original acceptance budget remains `max_abs <= .002 OR scaled_relative <= .01` for every field; no tolerance was widened.

| Device | Compared fields over three input seeds | Maximum absolute error | Maximum scaled relative error | Actual parameter updates audited |
| --- | --- | --- | --- | --- |
| CPU | 282 | 4.768371582e-7 | 1.146249123e-7 | 108 pairs; all nonzero |
| CUDA | 282 | 4.768371582e-7 | 1.147065891e-7 | 108 pairs; all nonzero |

Every requested parameter/intermediate gradient is present and nonzero in these measurements. The smallest reference parameter-update L2 norm is `4.945786551e-7` on both devices. Per-parameter/per-step norms and update errors are stored separately. The CUDA training scope observes zero fallback.

The actual `nox -s ecosystem_swift` session runs the concrete node `compat/tests/torch/test_ecosystem_swift.py::EcosystemSwift::test_ms_swift_trainer_full_llama`. CPU: **1 passed, 0 skipped**, 139.36 seconds; CUDA: **1 passed, 0 skipped**, 327.53 seconds. Each node independently recreates the genuine CPU checkpoint and executes both runtimes for all three input seeds. These times include first model-kernel compilation in separate unit caches. They are not speed ratios or L4 evidence. An external GPU process was present; this report grants correctness/trajectory evidence only.

The post-measurement source-proof change adds an independent wheel/payload audit and negative tests without changing the producer, model, arrays or numerical budget. Tool regressions after collector hardening: **60 passed**; the final combined source-proof/tool selection on the shared-auditor baseline passes **67 tests**; complete structure after the collector change: **1510 passed, 4 hardware/declared skips**. Ruff and repository layout checks pass. An earlier structure run exposed that this source-only venv had received the compat editable package without an installed native core; provisioning the native core in that same private venv fixes the two packaging-resolution checks. Its failed log and successful reruns are preserved. No shared environment was changed. The standard CPU seed path uses the public [no-device CUDA seed contract](2026-10-15-cuda-seed-without-devices.en.md); the Swift seed routine is unchanged.

The final worktree structure run after adding this report records **1506 passed, 4 skipped, 4 failed**. Two failures await the integrator-owned results index and generated MANIFEST entries. The other two occur because this invocation disabled pytest plugin autoload, leaving its xdist fixture children without `-n`/`--dist`; explicitly loading `xdist.plugin` passes the complete affected file (**10 passed, 0 skipped**). No source workaround was applied. Both post-seal logs and their hashes are retained in `swift-post-seal-maintenance-v1.json`; the 170-file numerical evidence seal is unchanged. The integrator must rerun structure after central registration.

The strict shared scorecard admits **two L3 records, zero rejected and zero blocked**, one per device. A separate read-only audit recalculates all 564 field pairs from the 1128 raw arrays and verifies every referenced artifact hash; a copied record with false TF32 metadata is rejected on each device. Full coverage rows retain the case, registry references, tiny shape and complete trainable-parameter names. The [shared release-source audit](2026-10-15-release-installed-source-proof.en.md) independently validates the same complete binding. The synthetic incomplete-payload fixture is kept separate: it preserves the original arrays but removes installed-file declarations. The old collector accepted it; the hardened collector and shared release-source checker reject it.

## Reproduction and sealed artifacts

Use two separate interpreters with the pinned dependencies; the hardware interpreter needs the native core installed as well as the compat source. Keep artifacts, model files, HOME/TMPDIR, JIT state and dependency targets under `$JITTOR_LAB_ROOT`. The opt-in gate requires a clean pinned Swift checkout and the real installed wheel:

```bash
export JITTOR_CI_PYTHON=<jittor-python>
export REAL_TORCH_PYTHON=<real-torch-python>
export JITTOR_SWIFT_UPSTREAM=<clean-ms-swift-checkout>
export JITTOR_SWIFT_WHEEL=<actual-ms_swift-4.5.2-wheel>
JITTOR_TEST_DEVICES=cpu nox -s ecosystem_swift
CUDA_VISIBLE_DEVICES=<allocated-gpu> JT_BUILD_NVCC_PATH=<nvcc> \
  JITTOR_TEST_DEVICES=cuda nox -s ecosystem_swift
```

The reusable `swift_training_probe.py prepare` creates the local checkpoint in genuine Torch CPU; `run` executes the real Trainer. `collect_swift_scorecard.py` rechecks all seed pairs, historical producer bytes, full wheel/source/payload provenance and the shared scorecard schema. Its `--run cpu=<directory> --run cuda=<directory>` inputs contain `seed{0,7,19}-{torch,jittor}.{npz,json}`. Full argv/environment records and exact commands are stored with the raw artifacts rather than requiring machine paths in this report.

Unversioned artifacts are at `$JITTOR_LAB_ROOT/_state/downstream-training-evidence/20261015/`: `formal-606b602e7e79/`, `swift-scorecard-final-v1.json`, wheel/source proofs, actual gate logs and the synthetic negative fixtures. The sealed manifest is `swift-sealed-evidence-manifest-v1.json`, 170 files / 38361434 bytes, SHA256 `e22d32542a3a05ce29ef4834faa5d277e2a6c91f20c8e0c7476b2ec518309a58`. It excludes caches and environment trees; numerical artifacts keep their original `606b602e` identity.

The final scorecard SHA256 is `70837b67cb082ffdd41284f1de44d1d002075c5b0c5e6456486523a766b615ae`. Synthetic incomplete-payload and platform-precision failures are retained as negative evidence and do not contribute coverage.

## Main-branch integration and independent recheck

The integration first synchronized `origin/2.0-refactor` at `8929633bef1e11294c3f0dfbef01bab9c0e015fe`. Both maintained device sessions then ran on the clean, frozen integrated source `fee77cad724f33a19ea858c69639a09defffe5e4`: **CPU 1 passed, 0 skipped, 149.55 seconds; CUDA 1 passed, 0 skipped, 332.18 seconds**. Each session recreates the checkpoint and checks both runtimes at seeds 0/7/19 for all three Trainer updates. Fifteen integrated file hashes were unchanged across both terminal runs. Separate CPU/CUDA unit caches and a new private candidate environment preserve the original producer's environments and evidence. Cold compilation is included in the session duration; these remain correctness results without a speed claim.

The new main-source gate logs are distinct from the persistent `606b602e` numerical traces. The maintained node removes its temporary arrays on success, so its pass is not presented as a new persistent-array scorecard. The original measurements retain their historical producer SHA. An independent recheck under shared auditor `8a7bb3873aa3066509f2518dda0669b1ab84dc5a` regenerates the Swift denominator, validates the complete actual wheel/payload and all three seed pairs, and accepts **two L3 records, zero rejected and zero blocked**. A separate NumPy-only absolute-error calculation checks all **564 pairs**, exact shapes/dtypes and finiteness: maximum absolute error remains `4.76837158203125e-7` on each device. All **170 sealed files** were rehashed before and after collection without changes. Writable comparison regeneration uses byte copies outside the sealed tree.

A negative copy exposed that the shared auditor previously ignored additional perturbation metadata: changing only the second perturbation's reference precision still passed that auditor. The dedicated Swift collector had already checked that phase. The shared fix now validates every additional `perturb<digits>-comparison` pair, including the existing `perturb2-comparison` and `perturb19-comparison` spellings. Declaring `perturb_seed2` without an extra comparison is rejected; historical single-perturbation protocols remain valid. Both runtime source/precision/device bindings, passed execution and candidate zero-fallback checks apply to the extra pairs. The same synthetic invalid-precision copy is now rejected; it contributes no numerical coverage. Combined tool regressions pass **135 tests**, with no skips and unchanged tolerances.

The integration preserves the remote registry inventory's expanded package-file declarations and the existing PEFT session. The source-only coverage audit now recognizes Swift's real class-method node and its dedicated nox session, while keeping `verified_level` unset until actual evidence is supplied. Central results-index and MANIFEST registration resolve the two integration-owned documentation failures above. On the integrated source before the extra auditor hardening, core passes **322 tests / 42 skips / one existing xfail**, and combined structure passes **1622 tests / six classified skips**. The skips are two hardware, two declared and two native-distribution-not-installed checks in that validation environment; they are unrelated to the required Swift device nodes, which execute without skips.

New unversioned integration evidence is under `$JITTOR_LAB_ROOT/_state/swift-main-integration-892/`: `clean-source-freeze.json`, `cpu-nox.log`, `cuda-nox.log`, `main-validation.json`, `reaudited-scorecard-fixed.json`, `reaudit-fixed-summary.json`, `absolute-audit.json`, and the separately labelled `perturb2-gap-reproduction.json` / `perturb2-gap-fixed.json`. Initial private-environment provisioning failed before any numerical work; that attempt and the successful provisioning are retained in `provision-history.json`.

The scheduled ecosystem workflow currently selects the older general `ecosystem` session, not this dedicated session. The real maintained manual gate and bounded numerical evidence above are available; dedicated nightly provisioning/selection remains outstanding. This report does not upgrade the historical LORA case, distributed Swift, untested trainers/tuners, or whole-library integration.

Final combined structure after the additional-perturbation fix and report/index registration: **1622 passed, six classified skips**, 226.80 seconds. Repository layout, generated MANIFEST consistency, Ruff and whitespace checks pass. These checks validate the integrated code and documentation; they do not broaden the bounded numerical or hardware conclusions.
