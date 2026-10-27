# PEFT LoRA: complete installed source and maintained training gate

- Status: the original complete tiny LoRA scope passes on CPU ① and real CUDA ③; the maintained nox performs numerical acceptance automatically.
- Date: 2026-10-15.
- Measurement source: `1738f4a09d178becfad989f7bef7915729735890`, the same clean commit on both devices.
- Maintainer: Jittor core maintainers.
- Review when: the PEFT/Transformers pinned source or wheel, the shared case/runner, the training probe, compatibility computation, provenance collection or the scorer changes.

This re-measures the complete `peft_lora_llama` scope already claimed by the old report, and completes the
full installation package and upstream source bindings for every actual compute process. No identity was backfilled into the old raw data, no framework mathematics or tolerance was adjusted, and no adapter was added.
The old [2026-10-14 numerical report](2026-10-14-downstream-oracle-scorecards.en.md) keeps its own source and evidence;
the new provenance fields it lacks cannot be retro-proven from this round's installation directory.

## Fixed provenance and actual runtime environment

| Item | Actual identity |
| --- | --- |
| PEFT | 0.17.1, official source `53c25fe4fdd7c6aa4b40db0560815ea570d32303` |
| Transformers | 4.56.2, official source `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| Independent oracle | Python 3.11, binary PyTorch `2.11.0+cu128`; the real `_C` extension path and SHA256 are recorded per process |
| CPU | AMD EPYC 9554, Linux/glibc 2.34; both sides explicitly CPU, the candidate disables CUDA/oneDNN/MPI/NCCL |
| CUDA | NVIDIA GeForce RTX 4090, driver 580.178.04; both sides' actual allocation pointers belong to the same physical card |
| Compilation and precision | GCC 11.5, candidate real NVCC 12.2; FP32, TF32 matmul/cuDNN ON for both sides on CUDA with benchmark OFF, TF32 OFF for both sides on CPU |
| Threads and isolation | 2 threads per side; each runtime × device has its own HOME/JITTOR_HOME/TMPDIR/XDG/shim cache, with identical paths before and after computation |

The API query of the actually mapped `libcudart` is **12080** for the oracle and **12020** for the candidate.
The candidate uses a separate real cu122 package prefix; the cu128 libraries are not relabelled as cu122.
Each process records the actual preferred NVIDIA metadata and the absolute paths/hashes of loaded libraries; the candidate has cuBLAS
12.2.5.6, runtime 12.2.140 and cuDNN 8.9.7.29, and the oracle 12.8.4.1, 12.8.90 and 9.19.0.56 respectively.
This case is single-process, so no NCCL communication support is claimed on that basis; mapping a library at import time also does not mean its compute path is used.

SHA256 of the two original official PyPI wheels:

- PEFT: `3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e`.
- Transformers: `79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`.

Every formal worker re-checks, before and after computation, the actual import root, the `importlib.metadata` preferred distribution,
the official wheel's METADATA/RECORD bytes and all installed package payloads: PEFT **161** files
(158 Python, plus typed/C++/CUDA), Transformers **2173** files (2147 Python, including C++/CUDA/headers and so on).
The two extra upstream Transformers files, GPT2 `CONVERSION.md` and T5 `download_from_gcp.sh`, did not enter the official wheel and
keep an explicit exclusion reason in the inventory; they are not falsely recorded as installed runtime files.
Each worker binds **1557** framework/compatibility-layer/related-tool source hashes, including the target text of directory symlinks and
the actual compat owner. It also seals the mode/blob/byte hashes of all **2835** Git files of the measured clean commit.

## Original complete numerical scope

Reuses the existing public case, ecosystem runner and `training_probe.py`; no model mathematics was copied.
The model is a two-layer tiny Llama: hidden 64, intermediate 128, 2 attention/KV heads, vocab 128;
`q_proj`/`v_proj` use LoRA rank 4, alpha 8, dropout 0. Input is `(2, 8)` int64 tokens, the
output is `(2, 8, 128)` FP32 logits, and it runs offline.

Six real processes per device: the two runtimes of base, the two runtimes of the seed 7 input
perturbation, and the two runtimes of three-step training. All processes load the same oracle initial weights, with **29 parameters and 1 rotary buffer**.
base/perturb follow the runner's original contract and temporarily request **all 29 parameter gradients**; training keeps the original PEFT partition,
with **8 LoRA parameters trainable and 21 parameters frozen**, three manual SGD steps at learning rate 0.001.
Input tokens are integers, so input grad is not requested; the 0 requests are not written up as a floating-point input-gradient capability.
Each step checks the output, loss, all 8 parameter gradients and all 29 updated parameters, and confirms byte-for-byte that frozen parameters are unchanged and
trainable parameters actually updated. The input perturbation must actually change the output.

| Device | base: output + 29 gradients | seed 7 perturbation | three-step trajectory | maximum absolute error | maximum normalized-by-original-magnitude error |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU ① | 30 | 30 | 117 | 1.4543533e-5 | 3.1351752e-7 |
| CUDA ③ | 30 | 30 | 117 | 1.0300159e-2 | 2.2198918e-4 |

After the independent collector and the current unified auditor regenerated all provenance inventories and recomputed: **2 accepted, 0 blocked, 0 rejected**,
with **354 array pairs** across both devices. The original budgets are unchanged: `max_abs <= 0.002` or
`max_abs / maximum magnitude of the original reference array group <= 0.01`; this is not a per-element relative-error claim.
The CUDA maximum difference comes from the perturbation group; the three-step trajectory maximum absolute error is only **5.9627928e-7**.
The CPU three-step trajectory maximum absolute error is **2.3841858e-7**.
Every candidate phase uses `fallback_policy=error` with count **0**, while observing the actual input/output/parameter/gradient
devices; CUDA additionally checks the actual allocation through driver pointer attributes and UUID. The oracle's fallback field is only an old runner
placeholder; the real oracle identity is proven by the independent binary and the actual devices, and that placeholder is not treated as a Jittor fallback measurement.

Coverage is counted separately against the fixed real registries: PEFT methods **1/26 @ 53c25fe4** (LORA), tasks **1/6**
(CAUSAL_LM); the attached Transformers model family **1/374 @ cd74917f** (llama), tasks **1/46**
(causal LM mapping). These axes cannot be added up and do not represent whole-library support.

## Reusable entry points and independent recomputation

Added `tools/run_peft_gate.py` and the maintained entry `nox -s ecosystem_peft`.
The controller must complete all six workers and automatically invoke the same strong collector to check all **177** numerical fields,
the input perturbation, the frozen parameters and the provenance bindings before it exits successfully. The preflight `--preflight` runs only the two base processes and explicitly grants no L3;
nox refuses to pass through scope-narrowing arguments such as phase/preflight and allows only the output and cache paths to be specified.

The actual formal-v5 used the following public entry; the environment and complete commands are kept in the external `run-gate-v5.sh`:

```bash
export JITTOR_CI_PYTHON=<candidate-python> REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_PEFT_CHECKOUT=<clean-official-peft-checkout>
export JITTOR_TRANSFORMERS_CHECKOUT=<clean-official-transformers-checkout>
export JITTOR_PEFT_WHEEL=<official-peft-wheel>
export JITTOR_TRANSFORMERS_WHEEL=<official-transformers-wheel>
export JITTOR_TEST_DEVICES=cpu  # 完成 CPU 后以 cuda 独立运行
# CUDA 另显式分配 CUDA_VISIBLE_DEVICES 和真实 nvcc/provider；首次 JIT 串行。
<audit-python> -m nox -s ecosystem_peft -- \
  --output <new-external-output> --cache-root <private-benchmark-cache>
```

The PEFT CPU/CUDA nodes of the ordinary `test_ecosystem_parity.py` remain the existing base/one-step parity gate;
they cannot be taken as the complete automatic L3 gate. The actual entry for this complete L3 is the separate nox session above,
and no claim is made yet that a scheduled nightly workflow selects that new session.

After all formal measurements reach their final state, re-read the NPZ/JSON, complete source, wheels and provenance records with an independent process:

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_peft_scorecards.py \
  --run "$PEFT_STATE/formal-v5/cpu" --run "$PEFT_STATE/formal-v5/cuda" \
  --peft-checkout "$JITTOR_PEFT_CHECKOUT" --transformers-checkout "$JITTOR_TRANSFORMERS_CHECKOUT" \
  --peft-wheel "$JITTOR_PEFT_WHEEL" --transformers-wheel "$JITTOR_TRANSFORMERS_WHEEL" \
  --baseline 1738f4a09d178becfad989f7bef7915729735890 \
  --report docs/results/2026-10-15-peft-installed-source-training.md \
  --output <new-external-audit-output>
```

`PEFT_STATE=$JITTOR_LAB_ROOT/_state/peft-source-bound/20261015` (unversioned).
formal-v5's execution/validation, the 12 actual worker JSON/NPZ/log files, the independent scorecard,
the source/environment/library bindings and the failure history are bound by that directory's final manifest; neither raw data nor environments are put in the repository.

## Negatives and retained boundaries

13 host contracts pass, covering non-Python payload/RECORD/metadata misplacement, missing actual observations,
complete scope and nox argument constraints, and numerical errors causing the maintained controller to fail.
Separately, **7 rejection scenarios** were verified on external copies of the real formal CPU metadata/NPZ: missing Transformers source,
missing RECORD, missing output residency, deviation from the actual private HOME, wrong gradient values, missing requested gradients, and a frozen parameter
changed by only 1 ULP (the transparent copy updates its own digest so the frozen invariant is tested in isolation). The original raw data is always read-only.

The first formal-v1 reported `IsADirectoryError` at the source snapshot of a directory symlink, before any computation; the original rc/source/log are retained.
It was fixed with the dual binding of target text plus the real compat file. preflight-v2 exposed that the old collector validated RECORD/complete observations
insufficiently, which was fixed with negatives; the base output's actual key is `__output__`, and a check that wrongly used `output` was also corrected.
formal-v3's CPU six workers and the external numerical acceptance passed, but the controller did not yet validate numbers automatically at that time, so that old nox exit code
does not grant automatic L3. The ea897 change then added the automatic checks; after the remote source-audit safety-sync fix, the same-source formal-v5 was re-run.
All old data keeps its original baseline and is not unified with this round.

This round closes only the provenance and reusable complete-training-gate gap for this tiny LoRA L3 scope. It grants no L4 performance, adapter
save/restore, Trainer, BF16, distributed training, quantization, generation, other tuners or real large-model capability;
NPU/ROCm were not verified. A missing dependency environment, missing real devices and mismatched provenance must all fail explicitly; an import,
a host contract or a CPU fallback cannot stand in for numerical acceptance.

## Main-tree integration re-verification

After synchronizing `9ba68263b267c78799ec254e2b71abd865f41094`, the code and this page were integrated as
`508cdd07e67f0c5f1fbc6aa1411185a024a38068`. On a separate frozen worktree of that clean commit,
with a fresh private benchmark cache, the complete maintained CPU/CUDA nox was re-run: all 12 real workers succeeded,
and each device automatically checked 177 arrays; independent-process recomputation again gave 2 accepted / 0 blocked / 0 rejected and
354 array pairs. The errors match the table above, the actually mapped CUDA runtime remains 12080/12020 on the two sides, and
every candidate phase has fallback=0. Each worker binds 1558 runtime source files, and the complete Git inventory is 2838 files.
This new set of measurements keeps the 508c identity, does not relabel the original 1738 record, and does not add coverage numerators again.

On the main tree, the targeted 77 items pass and core is 322 passed / 42 skipped / 1 existing xfail.
The first complete structure check found two violations in the new host tests — loading modules during collection and referencing the interpreter directly;
these were fixed with a shared deferred-loading base class and the standard child-process helper. An intermediate change that made only one test class deferred caused
three fixture failures; the original logs are retained. The final tests and structure contracts pass 55 items, and the complete structure is
1622 passed / 6 skipped. The six skips are two accelerators, two declarations, and two where the current interpreter has no native package installed.
These fixes touch tests only; the run tooling and compute source of the frozen measurement worktree never changed.

The main tree's new measurements, independent scorecard, complete source and raw-evidence inventory are in the unversioned
`$JITTOR_LAB_ROOT/_state/peft-main-integration-9ba/`; the original 1738 evidence stays read-only,
and 235 raw/supporting files, 2375 actual installed files/libraries and all 2835 Git files' mode/blob/hashes were independently checked as well.
