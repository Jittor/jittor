# Transformers: complete source-bound training recheck after mainline integration

- Status: independent mainline integration measurement, complete scope passes CPU ① / CUDA ③.
- Date: 2026-10-16 (started 15 October, completed across the day boundary).
- Measured source: `09e9e8eb8f67f4b32e3b46e05174013c1216983f`; every formal process uses the same clean commit.
- Start-of-work sync: the fetch yielded `593249c5f0e82febb931234aa35003d0d41c8b33`, which the integration already contains; the frozen commit remains `09e9e8eb8`.
- Maintainer: Jittor Torch compatibility maintainers.
- Review when: the framework, compat, model/runner/training probes, source capture/comparator, upstream/wheels or the actual provider change.

This report is a **new independent measurement** of the [original B4 complete source matrix](2026-10-15-transformers-installed-source-training.en.md)
after its integration into the mainline; it does not relabel the old raw artifacts with a new
commit number. Compared with the old B4's 1562 runtime source files, this run actually binds
1571 files: 11 content changes and 9 additions. The strict scorer correctly refused to let the
old source artifacts impersonate the current tree; the original working tree, the original B4/v5
numerics and every source identity are left untouched.

## Environment and actual execution

An independent worktree, a new candidate Python prefix and empty CPU/CUDA benchmark caches were
created from the specified integration commit; no old JIT binaries were copied. First compilation
ran serially; every controller, worker, HOME/JITTOR_HOME/TMPDIR/XDG and raw artifact lives in the
external lab state. The sealed official Torch oracle/provider and the Transformers wheel
installation are reused read-only. This run changed no computation implementation, case, shared
runner/training probe, collector/scorer or numerical budget.

The pinned upstream is official Transformers 4.56.2 source
`cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`, with official wheel SHA256
`79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`.
Every formal worker verifies the complete 2173 actually installed files, the official
METADATA/RECORD, and 1571 framework, compat and selected tool source files. The independent oracle
is complete official Torch `2.11.0+cu128` / NumPy1.26.4; the complete 11842 wheel payload files are
fully rehashed before and after the run, and the real binary `_C` SHA256 is still
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`.

Both CPU sides explicitly disable CUDA, and the candidate disables oneDNN/MPI/NCCL; all 80 formal
CUDA processes use the same RTX4090 and verify actual input/output/parameter/all requested gradient
residency, driver pointer attributes and physical UUIDs. The candidate uses NVCC12.2.140/GCC11.5.0
mapping to runtime API12020; the oracle is 12080, and the real cu122/cu128 library paths, hashes and
NVIDIA distribution metadata are retained separately without relabeling the provider. Both CUDA
sides keep the original TF32-on policy. This run is single-process/single-device and claims no MPI,
jtrun, NCCL or multi-rank capability.

## Complete scope and original budgets

All ten cases are retained: BERT, the CLIP two-tower model, GPT-2, Llama, RoPE2048, Mixtral
four-expert/top-2, T5, ViT, Whisper and a real 32768-token Longformer. Each case/device has
independent genuine Torch and candidate processes for base0, input perturbations 7/19 and three
training steps, for **160 formal workers and 7854 array comparisons** in total. The three manual
SGD steps, lr0.001, step input seeds 1/2/3, the complete initial weights, all parameters and
floating-point input gradients, loss and parameter trajectories are unchanged. Whisper keeps its
base/perturbation 89 parameter gradients and training 88 trainable + 1 frozen; CLIP/ViT pixel
gradients, Whisper feature gradients, all MoE expert/router fields and Longformer's
`[1,32768,32]` noise input gradients are all checked.

TF9 keeps the original per-phase whole-field reference magnitude: base on both devices 0.002/0.01,
CPU perturbation/training 0.002/0.01, CUDA perturbation/training 0.005/0.02, passing on either
absolute or whole-phase normalized error. Longformer keeps its original per-array/gradient-block
floor normalization and strictly-less-than rule, CPU0.002/0.01, CUDA0.005/0.02; TF9's
large-output magnitudes must not mask its small gradients.

| Device | cases / formal workers / array pairs | TF9 largest original whole-phase normalized error | Longformer largest original array/gradient-block normalized error | Candidate fallback |
| --- | ---: | ---: | ---: | --- |
| cpu | 10 / 80 / 3927 | 1.79564466e-06 | 9.84836796e-06 | error / 0 |
| cuda | 10 / 80 / 3927 | 0.011524663 | 2.73647853e-05 | error / 0 |

The worst CUDA T5 entry is `grad::decoder.block.0.layer.2.DenseReluDense.wi.weight` in
`perturb19`, with a largest absolute error of 0.383440316 and a phase-normalized error of
0.011524663; the original 2e-2 budget for that phase is retained, without claiming per-element
relative precision or strict IEEE. The largest CPU/CUDA Longformer absolute errors are
0.0166015625 / 0.0439453125 respectively.

Both public maintained nox runs finished with rc0, taking about 33.58 / 39.27 minutes on
CPU/CUDA (including the first JIT for each shape; the independent base preflight is counted
separately); they are diagnostic only and grant no L4.

The independent collector and unified scorer: **20 accepted, 0 blocked, 0 rejected, 7854 array
pairs**. A separate host-only check verified 64 actual phase/runtime cases for the two sides'
special scenarios, and that 20 complete sets of CPU/CUDA initial weights, 870 arrays in total,
are byte-identical. The coverage axis remains 9/374 model families and 2/46 task mappings at the
fixed upstream, with the two Llama scenarios counting as one model family; the source
supplementation added no model capability.

## Reproduction and evidence

`TF_MAIN_STATE=$JITTOR_LAB_ROOT/_state/transformers-main-validation/09e9e8eb8` (unversioned).
The actual commands, the complete environment entry `run-gate-v1.sh`, the serial execution record
`complete-execution-v1.json`, `formal-v1/{cpu,cuda}`, the independent scoring, the source diff and
whole-Git snapshot, the complete oracle/provider recheck and the final `evidence-manifest-v1.json`
all live in that state. Raw artifacts, installed packages and compilation caches do not enter the
repository.

The public entry remains `nox -s ecosystem_transformers_source`; configure the candidate/genuine
Torch interpreters and the fixed TF checkout/wheel, run CPU first and then an explicit single CUDA
device with real NVCC, passing a new external `--output` and a private `--cache-root`. Cases and
phases must not be reduced, the maintained entry performs the numerical decision itself, and a
worker rc0 is not sufficient to pass. The independent read-only recomputation (no GPU/JIT needed)
uses the state's `recheck-v1.sh <new-empty-audit-output>`; the baseline must remain the complete
`09e9e8eb8f67f4b32e3b46e05174013c1216983f` and must not be changed to a later report commit.

This run adds no model family or adapter and grants no Trainer, generation, checkpoint/resume,
other optimizer, BF16, full pretraining scale, multi-rank, NPU/ROCm or performance capability; it
does not conflate ordinary ecosystem pytest nodes with the complete nox gate, and does not claim
that a scheduled nightly has selected this new session.

## Pre-commit checks

The TF tools' 12 host contracts pass (1.185 s); repository layout and diff checks pass. The
prescribed Torch-mode structure check uses independent unit-test HOME/JITTOR_HOME/TMPDIR and a
fresh CPU cache: 1506 passed, 6 skipped, 2 failed (270.55 s). The two failures are that the new
report is not yet in the results toctree and that MANIFEST.in does not register this file; central
registration belongs to the integration step and neither changes computational budgets nor hides
failures. The 6 skips comprise 2 accelerator-hardware items, 2 declared-category items, and 2
package-resolution checks for which the source-only candidate environment has no installed Jittor
distribution metadata; real source import and CPU/CUDA computation are proven separately by the
complete maintained gate. The original logs are the state's `unit-structure-v1.log`,
`unit-execution-v1.json` and `host-tools-v1.log`.

## Central integration review

On the published `0f2f9481cd59072cc02a1edff4bc16bc12d1b071` the integrator added only this report,
the results index and the generated MANIFEST, without changing measurement code. An independent
reread of 1106 raw/auxiliary files, 17197 external files, 922 actually compiled binaries and 2865
original measurement Git files matched every hash; the original seal SHA256 is
`125c2ffecefacb94680981424908d433a3ab5397f698d4ddb4deaf264d26ff06`.
Source symlinks were verified against the target text stored by Git; compiled binaries were not
copied or relabeled.

After registration, the main tree's complete `tests/structure + compat/tests/structure` reports
1622 passed and 6 skipped in 211.12 s, and layout and whitespace checks pass. The original
independent tree's two registration failures keep their original logs and are not retroactively
rewritten as passes; all numbers above still correspond to 09e9 and cannot be extrapolated to this
report's commit. The central review records are in the unversioned
`$JITTOR_LAB_ROOT/_state/transformers-main-source-doc-integration-0f2/`.
