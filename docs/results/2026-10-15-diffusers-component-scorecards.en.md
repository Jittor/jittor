# Diffusers UNet component evidence in the shared scorecard

- Status: Two historical CPU/CUDA UNet L3 component records migrated and accepted; no pipeline or new numerical execution claimed.
- Date: 2026-10-15.
- Measurement baseline: `230ba7a0305091fb64ec81a679f32dcaefcea518`, unchanged from the original independent oracle runs.
- Migration tools and formal auditor: clean `f61555a818acbe9f61b4061b896ff5d59b341b6f`, based on synchronized origin `824cb354d5cced0743b3207d50d518111ba59e76`.
- Upstream: Diffusers 0.35.1, clean `0f252be0ed42006c125ef4429156cb13ae6c1d60`, official Hugging Face repository.
- Owner: Torch compatibility and downstream validation maintainers.
- Review when: Upstream model exports, the UNet builder, training probe, dependency identity, original artifacts, numerical budgets or scorecard rules change.

The [original four-library report](2026-10-14-downstream-oracle-scorecards.en.md)
established real `UNet2DModel` CPU/CUDA numerical and three-step training evidence.
Its two component records had `level=null` and no registry references, because the
then-existing Diffusers inventory described pipelines only. They were not accepted
registry rows. This migration adds an honest independent component export axis;
it does not turn UNet execution into pipeline execution.

## Complete, independent public-export denominator

The central inventory now also parses every `_import_structure` declaration in
`src/diffusers/models/__init__.py`, without importing the package or evaluating
dependency availability. At the pinned source there are 80 module declarations
and **87 distinct public names**. Flax exports, helpers, mixins, enums and functions
remain in this denominator. It is a bounded **model-package public API axis**,
not 87 model families, all component APIs or a whole-library support fraction.

Every conditional declaration retains its module, names, source line and guards.
The same public name from the same owner in alternative branches counts once,
with both declarations preserved. Duplicate exports or overwritten assignments
in the same branch, conflicting module owners, aliases that permit untracked
mutation, loops and unknown dynamic registration fail explicitly. Optional
dependencies never shrink the denominator.

| Axis | CPU L3 | CUDA L3 | Actual scope |
| --- | --- | --- | --- |
| `model_api` | 1 / 87 | 1 / 87 | `UNet2DModel`, the original tiny FP32 case |
| pipeline module family (`model_family`) | 0 / 79 | 0 / 79 | No pipeline execution |
| `pipeline_class` | 0 / 249 | 0 / 249 | No pipeline execution |
| `auto_family` | 0 / 37 | 0 / 37 | No AutoPipeline execution |
| `task` | 0 / 3 | 0 / 3 | No AutoPipeline execution |

The four pipeline denominators and their members are unchanged. All untested
model exports remain in the new denominator; no smaller UNet-only list is used.

## Preserved measurements and independent artifact recheck

The reusable collector checks every original artifact's SHA256 before reading
the measurements. It reads the original framework Git objects to verify that
the measured case imported and constructed the public `UNet2DModel`, and that
both training metadata records match the archived training program hash. It
retains the original framework SHA, upstream identity, package versions/origins,
platform, inputs, numerical budgets, commands, report and original artifacts.
The derived fields are `level=3` and the single `model_api` reference; new
migration comparisons, inventory and original evidence-file hashes are appended.
No historical JSON or NPZ is rewritten.

The original configuration has sample size 16, three input/output channels,
one layer per block, channels 16/32, DownBlock/AttnDownBlock and matching up
blocks, eight normalization groups and zero dropout. Inputs are FP32
`sample[2,3,16,16]` and integer `timestep[2]`. Each numerical comparison checks
the output, all **144 parameter gradients** and the `sample` gradient: 146
fields. Seed 7 changes only inputs while retaining the original seed-0 weights.
Both independent runtimes' output perturbations are verified from their NPZs.

Each of three real SGD updates uses learning rate 0.001 and records output,
loss, all parameter/input gradients and all updated parameters, **873 fields**.
The complete initial snapshot has all 144 parameters; the archived baseline
snapshot, perturbation weight source and both training weight bindings have
the same original hash. Baseline transfer follows the archived shared harness;
no missing per-process baseline weight field is backfilled. Both training trajectories contain actual
parameter updates. Original trainable/frozen scope, exact key sets, shape,
dtype and finite values are checked again.

| Device | Phase | Fields rechecked | Max absolute error | Max field-scaled relative error |
| --- | --- | ---: | ---: | ---: |
| CPU | Base | 146 | 4.673004e-5 | 8.865361e-7 |
| CPU | Seed-7 perturbation | 146 | 4.196167e-5 | 8.466126e-7 |
| CPU | Three-step training | 873 | 1.370907e-6 | 9.395992e-7 |
| CUDA | Base | 146 | 1.252556e-2 | 2.376241e-4 |
| CUDA | Seed-7 perturbation | 146 | 9.673119e-3 | 1.951558e-4 |
| CUDA | Three-step training | 873 | 3.414154e-4 | 2.339958e-4 |

All 2,330 array pairs pass the original `absolute <= 0.002 OR relative <= 0.01`
budget. Relative error uses the original verifier's maximum magnitude across
the entire reference field, with floor `1e-12`; this is not elementwise
`allclose`. The CUDA base absolute error exceeds 0.002 and passes the declared
relative criterion. No threshold or model mathematics changed.

The platform remains the original measurement: Linux, AMD EPYC 9554, Python
3.11.16, NumPy 1.26.4, GCC 11.5; independent binary Torch 2.11.0+cu128.
CPU explicitly hid CUDA and read back `use_cuda=false,use_acl=false`.
CUDA used one RTX 4090, driver 580.178.04, Jittor NVCC 12.2.140; actual
matmul/cuDNN TF32 was **ON/high**, CPU OFF/highest. Actual candidate computation
had fallback policy `error`, count zero, in every measured phase. Migration
only runs NumPy/stdlib checks and does not allocate a CUDA device or repeat
those numerical executions on the newer framework.

Original measurements record the pinned dependency version and source origin,
but **do not record complete per-process installed-package hashes**. This
limitation is retained. The freshly regenerated registry identifies its current
clean upstream; it is not a retroactive measurement of the old installation.
The existing release-version schema accepts this bounded evidence. A stronger
installed-source claim needs a new instrumented independent CPU/CUDA run;
today's installation hashes cannot be appended to the old raw metadata.

## Reproduction and verification

Raw artifacts are unversioned. Original component evidence remains at
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/component-evidence-v2.json`.
Migration outputs and logs are under
`$JITTOR_LAB_ROOT/_state/diffusers-scorecard/20261015/`; the formal clean-tool
output is `formal-f61555a81/scorecard.json`. It accepts **2 L3 records, 0 rejected**.
All **33 unique artifact bindings** rehash unchanged. A separate preservation
check confirms that only the declared derived fields and append-only artifacts
change; original numerical, platform and source metadata remain identical.
The unchanged original records still fail registry admission without migration;
their manual evidence is preserved separately.

```bash
<numpy-python> agent/skills/torch-compat-repo-runbook/scripts/collect_diffusers_scorecard.py \
  --root <checkout-at-f61555a81> --upstream <clean-diffusers-0.35.1-checkout> \
  --evidence "$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/component-evidence-v2.json" \
  --baseline 230ba7a0305091fb64ec81a679f32dcaefcea518 \
  --output-dir "$JITTOR_LAB_ROOT/_state/diffusers-scorecard/<fresh-run>"
```

The original measurement commit must be available as Git objects; a shallow
checkout must obtain that exact history rather than substituting current source.
Reproducing the original **model execution** additionally requires its original
checkout and two independent environments described in the original report.

Targeted registry/collector/common scorecard contracts pass **91 tests**. Negative
cases cover duplicate/ambiguous/dynamic exports, aliases and hidden mutations,
stale framework/upstream, fake pipeline references, incomplete gradient scope,
weakened budgets, one-step training, changed original files and mismatched
archived training code. Shared array-contract tests reject missing/extra fields,
wrong shapes, nonfinite values and numerical failures. Three actual-data negative
checks also reject invented pipeline references, weakened budgets and a candidate
NPZ with one gradient deleted **after its altered contents are honestly rehashed**.
Ruff, repository layout and diff whitespace checks pass.

Full structure validation finished with **1,497 passed, 6 skipped, 2 failed** in
494.26 seconds, including serial cold CPU feature configurations. Both failures
are this new report's central toctree entry and generated MANIFEST, intentionally
left to the integration owner. There are no other failures. The six skips are
two accelerator-only, two declared, and two package-resolution checks because
the private interpreter lacks installed Jittor distribution metadata; source
paths are pinned to this worktree. Original logs remain in `structure.log`.

This result does not establish a full generation pipeline, production-scale
training, other exported models, a current-framework numerical rerun, complete
installation provenance, NPU/ROCm/Corex, speed, memory, L4 or L5.

## Main-tree integration

Integration on `282783a3dfcd3521400869beadf56c6f0fe9a0f9` plus the two migration
commits independently rehashed all 19 delivery files and all 33 unique evidence
bindings. The integrated collector recomputed all 2,330 original array pairs and
the full upstream denominator again: two accepted records, zero rejected;
CPU/CUDA each retain `1/87` model API coverage and zero pipeline coverage.
The central results toctree and source manifest now register this report.
Integration outputs are unversioned under
`$JITTOR_LAB_ROOT/_state/diffusers-integration-d87/`; the directory name identifies
the start of the integration task, not a new numerical measurement baseline.
The historical installed-package provenance limitation above remains unchanged.
