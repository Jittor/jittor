# Four-library real CPU/CUDA oracle and three-step training scorecard

- Status: CPU and CUDA numerics for the four specified samples and three-step training actually passed; six registry L3 records passed machine audit, plus two Diffusers **component** training evidence entries, not counted as pipeline coverage. All results are manual acceptance. The earlier schema v1 scorecard that mislabeled CUDA TF32 as disabled via a manual platform snapshot has been withdrawn; this report uses corrected schema v2 bound to per-stage raw metadata.
- Date: 2026-10-14 (Asia/Shanghai).
- Baseline commit: `230ba7a0305091fb64ec81a679f32dcaefcea518`; this commit fixes the training probe's parameter snapshot variable on `45d08301f3357bf462d88f9be543bba85fb84063` after synchronizing the remote.
- Auditor commit: `822632d097ab848d3263ff69272aaf993a95dc64`, schema v2; compute commit remains the baseline above.
- Scope: Transformers GPT-2, Diffusers UNet2DModel, PEFT LoRA/Llama, ms-swift LoRA/Llama; tiny FP32, independent PyTorch binary oracle and Jittor shim in separate processes.
- Owner: Torch compatibility layer and downstream ecosystem maintainers.
- Review when: the compat/core gradient semantics involved in the above commits, shared ecosystem builder/runner, training probe, downstream versions, or platform/precision configuration change.

## Fixed platform and real upstreams

CPU is AMD EPYC 9554 64-Core Processor, Linux 6.1.188, Python 3.11.16, NumPy 1.26.4, GCC 11.5.0. Both sides' compute threads and OMP/MKL/OpenBLAS threads are fixed at 2, with the same process CPU affinity. The CPU process explicitly sets `CUDA_VISIBLE_DEVICES=''`, `use_cuda=0`, `JT_USE_CUDA=0` and disables nvcc; measured Jittor `use_cuda=false`, `use_acl=false`.

CUDA uses one NVIDIA GeForce RTX 4090, driver 580.178.04, with the allocated physical card mapped to in-process `cuda:0`. Jittor toolchain NVCC 12.2.140, actual cuBLAS API returns 12.2.5, cuDNN API returns 91900; both sides actually **allow matmul/cuDNN TF32, `matmul_precision=high`**; Jittor native HF32 off, native `cuda_kernel_math=strict` flag recorded separately, which cannot be interpreted as the Torch frontend matmul being strict. Default graph regime, MKL/NCCL/MPI/CUTT and parallel op compiler all off. Oracle is real `torch._C` binary PyTorch **2.11.0+cu128**, no shim marker; its build CUDA version and Jittor compiler toolchain recorded separately, cannot be written as both built by CUDA 12.2.

Each side uses an independent interpreter and runtime environment; CPU, CUDA, structure tests use independent JIT caches. Both sides load the same clean release source in the table below; the fixed-version source registry was also automatically enumerated from that checkout, not carrying over the larger denominator of the development branch.

| Library | Actual version | Complete upstream SHA |
| --- | --- | --- |
| Transformers | 4.56.2 | `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| Diffusers | 0.35.1 | `0f252be0ed42006c125ef4429156cb13ae6c1d60` |
| PEFT | 0.17.1 | `53c25fe4fdd7c6aa4b40db0560815ea570d32303` |
| ms-swift | 4.5.2 | `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` |

ms-swift's real dependencies also include ModelScope 1.33.0, Datasets 3.6.0, PyArrow 19.0.1 and pandas 2.2.3. Dependency restoration writes only to the external private environment: Python source files missing from the existing environment are restored from a private mirror of their original CPython 3.11 bytecode; missing cuDNN CNN/ADV libraries and headers are extracted from the official **9.19.0.56** wheel, verifying each member's SHA256 against the wheel RECORD. Shared installations, downstream source or framework semantics were not modified to bypass verification. Actual library resolution paths, wheel sources, member hashes and raw logs of failed attempts are all kept in the external artifacts.

## Compute protocol and results

First reuse the [shared runner](../../compat/tests/torch/_ecosystem_runner.py) to compare outputs, all requested parameter gradients and floating-point input gradients. Both sides initialize from the oracle-saved complete same-name parameters and buffers. Then fix these weights, change only the input random seed from 0 to 7, verify the output really changes with the input, and re-compare all gradients. Integer tokens do not request input gradients.

Training uses the [committed training probe](../../agent/skills/torch-compat-repo-runbook/scripts/training_probe.py), `model.train()`, 3 steps, fixed SGD update `p -= 0.001 * grad`, with input seeds 1, 2, 3 per step, fixed random loss weights averaged; outputs, loss, all requested gradients and all parameters saved per step. Materialize all old gradients before updating lazy Vars; keep upstream original frozen state, check frozen parameters unchanged each step, and at least one trainable parameter changes after training. LoRA's numerical gradient test deliberately requests all 29 parameters' gradients; actual training updates only upstream-declared 8 adapter parameters, with 21 base parameters frozen. This probe does not verify Adam, Trainer, scheduler or distributed training.

| Sample | Tensors per numerical parity | Parameter gradients / input gradients | Trainable / frozen parameters | Three-step trajectory arrays | CPU / CUDA |
| --- | ---: | ---: | ---: | ---: | --- |
| `transformers_gpt2` | 29 | 28 / 0 | 28 / 0 | 174 | all passed |
| `diffusers_unet2d` | 146 | 144 / 1 | 144 / 0 | 873 | all passed |
| `peft_lora_llama` | 30 | 29 / 0 | 8 / 21 | 117 | all passed |
| `ms_swift_lora_llama` | 30 | 29 / 0 | 8 / 21 | 117 | all passed |

Comparison follows this run's `verify_repo.py` FP32 criteria: each tensor's maximum absolute error not exceeding `2e-3`, **or** error relative to the same-class reference field scale not exceeding `1e-2`; not elementwise `np.allclose` relative error. All tensors must be finite, no missing gradients. The first two items below take the worst value from the baseline and fixed-weight input perturbation rounds; trajectory items cover three complete update steps:

| Device / sample | Numerical max absolute error | Numerical max field-scaled relative error | Three-step trajectory max absolute error |
| --- | ---: | ---: | ---: |
| CPU / GPT-2 | 2.67029e-5 | 2.83476e-7 | 2.38419e-7 |
| CPU / UNet2D | 4.67300e-5 | 8.86536e-7 | 1.37091e-6 |
| CPU / PEFT LoRA | 1.43051e-5 | 3.08378e-7 | 2.38419e-7 |
| CPU / Swift LoRA | 1.43051e-5 | 3.08378e-7 | 2.38419e-7 |
| CUDA / GPT-2 | 9.65500e-3 | 1.34945e-4 | 1.68085e-4 |
| CUDA / UNet2D | 1.25256e-2 | 2.37624e-4 | 3.41415e-4 |
| CUDA / PEFT LoRA | 1.03002e-2 | 2.21989e-4 | 5.96279e-7 |
| CUDA / Swift LoRA | 1.03002e-2 | 2.21989e-4 | 5.96279e-7 |

The four CUDA cases in baseline, perturbation and training stages have both sides actually on `cuda`; Jittor fallback policy `error`, count increment **0**. The four CPU cases likewise measured CUDA disabled. Some raw gradient absolute differences on CUDA exceed `2e-3`, and what passes is the explicitly recorded field-scaled relative criterion; this report does not claim all absolute errors are below that threshold.

## Registry denominators, grading and gate boundaries

The [scorecard auditor](../../agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py) re-enumerates the upstream complete registry, checking this Git SHA, upstream SHA/version, platform, real binary oracle, requested/actual device, complete gradients, input perturbation, multi-step trajectory, raw artifact hashes and existing specific gate nodes per entry. CPU and CUDA each accept three **broad/L3** records for GPT-2, PEFT LoRA, Swift LoRA, six in total; each CUDA record has a corresponding independent CPU parity.

| Library / registry axis | CPU verified members / complete denominator | CUDA verified members / complete denominator | Actual evidence this run |
| --- | ---: | ---: | --- |
| Transformers / model family | 2 / 374 | 2 / 374 | GPT-2 full-parameter SGD; Llama **only PEFT/Swift LoRA**, dense Llama full-parameter training not verified |
| Transformers / task | 1 / 46 | 1 / 46 | causal LM mapping |
| PEFT / method | 1 / 26 | 1 / 26 | LORA |
| PEFT / task | 1 / 6 | 1 / 6 | CAUSAL_LM |
| ms-swift / model family | 0 / 222 | 0 / 222 | did not call Swift's registered model loader |
| ms-swift / method | 1 / 11 | 1 / 11 | Swift native LoRA tuner |
| Diffusers / pipeline module family | 0 / 79 | 0 / 79 | this run executed only the UNet2DModel component |
| Diffusers / pipeline class | 0 / 249 | 0 / 249 | did not execute a full pipeline |
| Diffusers / auto pipeline family | 0 / 37 | 0 / 37 | did not execute an auto pipeline |
| Diffusers / auto pipeline task | 0 / 3 | 0 / 3 | did not execute an auto pipeline |

**UNet2DModel's CPU/CUDA evidence satisfies L3's compute and training conditions, but does not belong to the existing inventory's pipeline registration axes.** The two component evidence entries are stored separately, not forging DDPM pipeline references, not reducing denominators, nor sent into the registry coverage acceptance list. The Swift builder directly constructs Transformers Llama and hands it to the Swift tuner, so it counts toward the Swift LORA method and Transformers Llama, not toward Swift's own 222 registered model families.

The 2/374 above is a method-context family touch count, **not full-parameter L3 acceptance of two dense models**. The GPT-2 sample updates all 28 parameters; the Llama sample updates only 8 adapter parameters via the PEFT/Swift LORA method. Raw accepted records keep both the Transformers family and corresponding LoRA method registry_refs; when reading the flattened X/N one must look at these scenarios together, and cannot extrapolate to dense Llama, other tuners or all APIs of the family.

The existing specific gate is at `test_ecosystem_parity.py::EcosystemParity::test_<case>` and its CUDA subclass; the gate covers shared one-step numerical computation. The fixed-weight perturbation and three-step training this run are supplemented by a manual probe, **not an L3 gate wired into every CI run**. No independent CPU/CUDA oracle for the other nine libraries this round; TorchMetrics, TensorDict, MMCV, MMEngine, FlashAttention, vLLM, vLLM-Omni, VERL, TRELLIS not executed. For the scope of other historical results see the per-library runbooks; they cannot be merged into this new grading.

No real-size performance acceptance this run, no whole-process memory protocol, nor train/eval or other graph regimes compared, so **no L4 or L5 granted**. Zero fallback is a direct fact of the executed CUDA samples, not bypassing L4's cumulative conditions. The short samples' `seconds` are kept only as diagnostic data, and the training probe's NumPy materialization and transfer are not used for performance conclusions.

## Reproduction and artifacts

The actual release checkouts are in `$JITTOR_LAB_ROOT/downstream-coverage/releases/<repo>`; raw data in `$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/`, including independent interpreters, run conditions, actual dependency origins, per-round NPZ/JSON/log, complete source registry inventory, platform and CUDA provider map. Raw data and absolute environment paths do not enter Git.

First prepare two independent environments for the real PyTorch oracle and shim per the versions above, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and thread environment, CPU explicitly disables CUDA. CUDA explicitly sets `JITTOR_ECOSYSTEM_TF32=1`, `JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=0` to reproduce this run's actual high regime; switching to strict is another set of measurements needing a rerun. Baseline numerical computation example:

```bash
<shim-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases transformers_gpt2 --device cpu \
  --oracle-python <oracle-python> --shim-python <shim-python> \
  --no-memory --repeats 3 --out <outside-checkout>/cpu
```

Three-step training single-side command as follows; CUDA changes both `--device cpu` to `cuda`, using the allocated card and full toolchain, independent of the CPU compile cache. The same command is used in turn for the four cases in the table; the two NPZs' all arrays must be compared one by one per the thresholds above, not just the `TRAINING_RESULT` marker:

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case transformers_gpt2 --runtime torch --device cpu --steps 3 \
  --output <outside-checkout>/cpu/transformers_gpt2.train.ref.npz
<shim-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case transformers_gpt2 --runtime jittor --device cpu --steps 3 \
  --weights <outside-checkout>/cpu/transformers_gpt2.train.ref.weights.npz \
  --output <outside-checkout>/cpu/transformers_gpt2.train.shim.npz
```

Fixed-weight perturbation uses the shared `_ecosystem_runner.py`, both sides `--seed 7 --weights <base-reference-weights.npz>`; full commands, per-stage comparison details and raw hashes are saved with the external evidence JSON. Re-audit this run's acceptance list:

```bash
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --baseline 230ba7a0305091fb64ec81a679f32dcaefcea518 \
  --upstream transformers=<releases>/transformers \
  --upstream peft=<releases>/peft \
  --upstream diffusers=<releases>/diffusers \
  --upstream ms-swift=<releases>/ms-swift \
  --evidence <real-runs>/scorecard-evidence-v2.json \
  --output <outside-checkout>/scorecard-v2.json
```

The command explicitly re-audits the measurement commit `230ba7a0305091fb64ec81a679f32dcaefcea518`'s historical evidence via the full `--baseline`; the output also records the new auditor commit. When the parameter is not passed by default, it still rejects old evidence from a different HEAD, so the original measurement baseline cannot be rewritten to obtain a pass. Explicit historical audit does not prove the current tree has been rerun. Audit output: six accepted, zero rejected; two Diffusers component records listed separately in `component-evidence-v2.json`. The `scorecard_eligible=false` in raw stage markers is retained: a single `verify_repo.py` inherently lacks perturbation, platform and training evidence, and this report composes independent stage artifacts to establish L3.

This real execution also reproduced and fixed a variable-name shadowing bug in the training probe: the first formal oracle reported `TypeError` at the first step and stopped, failure log retained; after the fix, all stages of the four libraries and two devices were fully rerun on this report's baseline. All 29 host-side tests of the audit and training probe passed, related pytest/child-process contracts and tool targeted re-verification **71 passed**; Ruff 0.15.22 passed; layout and internal link checks of this report and the initial audit report passed. This baseline's full structure test was actually **1425 passed, 6 skipped, 3 failed**: old `DropoutACL` gradient inventory mismatch, and two integration gaps where the new report is not yet in the results toctree and the packaging MANIFEST. This task did not modify the shared index or MANIFEST; cannot claim the complete structure gate is all green.

This round's metadata correction keeps the original base/perturb/training NPZ and comparison JSON, does not change the compute baseline or raw numbers. The error source is that the external platform collector hardcoded the TF32 label as false, while the CUDA driver did not explicitly set `JITTOR_ECOSYSTEM_TF32=0`, and the shared runner's real default is ON/high. v2 directly reads both sides' `tf32`, `runtime_conditions`, actual device from the three stages, and compares one by one with the platform snapshot; training additionally binds actual trainable/frozen parameter names. Old schema v1 and wrong platform snapshots kept in external `history/schema-v1/`, explicitly withdrawn. Correction source and each raw measurement hash in `precision-correction-provenance.json`.

The counter-example of the same real raw set was also executed: re-supplying three CUDA rows with hash-correct platform snapshots labeled OFF/highest, the v2 audit actually returns nonzero, all three rejected with `Platform/measurement precision mismatch`; the three CPU rows unaffected. Re-auditing with the correct ON/high platform accepts six and rejects zero. Coverage also keeps each row's `verified_scenarios` case, repo, L, complete registry_refs, tiny/real, actual trainable/frozen parameter names and counts, preserving the Llama+LoRA method boundary.

The SHA256 of this run's external total artifacts as follows; each raw NPZ, platform, comparison and initial weight file hashes continue to expand in the evidence's artifacts.

| External artifact | SHA256 |
| --- | --- |
| `scorecard-evidence-v2.json` | `d351f0669721b7d033c6c1a006181219e6ab68096c6f424d6dd7b621654ed2b4` |
| `component-evidence-v2.json` | `0dcba24e0102dfd920dc07980da3afb679d3ad1c7dc459c617636c4e75d78ad9` |
| `scorecard-v2.json` | `5c1fc98b1445cf0f55480ccf9c9b29384a4f77d1df3dc14d210622c456109fde` |
| `precision-correction-provenance.json` | `8e3ffa63924cfeb0b2a346cfa71a0daff2e244448bb3b3d316ebf5ca7f4a2be1` |
| `audit-validation-v2.json` | `126660351d4a17cd1ce40b0061f3bf3538948a76d5a8cd7fc53c29dc643d7a39` |
