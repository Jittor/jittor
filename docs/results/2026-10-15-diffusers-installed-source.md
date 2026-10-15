# Fresh Diffusers UNet evidence with installed-source binding

- Status: Verified CPU/CUDA L3 for one bounded tiny UNet component; no pipeline claim.
- Date: 2026-10-15.
- Measurement source: clean `1a92aae04f4c70fa44e251f8fafdc7e2cd77beda`, based on synchronized origin `4ce19215c`.
- Upstream: Diffusers 0.35.1, clean `0f252be0ed42006c125ef4429156cb13ae6c1d60`, official Hugging Face source.
- Owner: Torch compatibility and downstream validation maintainers.
- Review when: UNet builder, source/installation identity, precision, gradient/training protocol or scorecard rules change.

This is a fresh numerical execution with instrumented package identity, separate
from the [historical component migration](2026-10-15-diffusers-component-scorecards.md).
That migration's framework `230ba7a0`, TF32 ON/high CUDA measurements and missing
historical full installed hashes remain unchanged. No new installation hashes
are attached to old raw records.

## Installation and source identity

A wheel was built from an archive of the exact clean upstream commit and
installed with `--no-deps` into a task-owned target. It neither replaced the
independent Torch nor modified shared environments. The wheel is
`diffusers-0.35.1-py3-none-any.whl`, SHA256
`fe763b8bb4381794f64c7a33ffdd29b3e63266c0013194317521331b3a15bdeb`.
All 695 package Python files match clean source, wheel and installation bytes.
The complete non-bytecode package payload is 697 files: those Python sources,
`py.typed`, and `utils/model_card_template.md`.

The existing `component_measure.py` now supports Diffusers and records the
actual loaded package path/version, all 695 clean-source hashes and all 697
installed payload hashes inside every base, perturbation and training process.
The shared collector checks installed-file hashes against the upstream source
mapping and requires identical complete package payloads between interpreters.
Source attestation does not establish execution of every package API. It also
does not claim full provenance for every transitive dependency or caches and
installation metadata outside the package directory.

The oracle is an independent binary Torch `2.11.0+cu128`; candidate imports the
requested worktree. Both use actual Diffusers `0.35.1`. CPU explicitly hides CUDA
and reads back `use_cuda=false`; CUDA uses one explicitly allocated RTX 4090,
NVCC 12.2.140 on candidate and Torch's CUDA 12.8 stack on oracle. TF32 and cuDNN
benchmark are OFF, matmul precision is `highest`, threads are fixed to one, and
candidate CUDA kernel math is strict. Actual candidate `auto_graph_replay=1`
and retain budget 268435456 bytes are recorded; effective capture or replay
performance is not separately established. Backend fallback is forbidden and zero.
These actual conditions differ from historical ON/high and are not relabelled.

## Numerical and training scope

The unchanged maintained `diffusers_unet2d` builder constructs a real
`UNet2DModel`: sample size 16, channels 3, one layer per block, block channels
16/32, Down/AttnDown and matching up blocks, eight norm groups, dropout zero.
The FP32 sample is `[2,3,16,16]`; timestep is integer `[2]`. All 144 parameters
are trainable and the sample input requires a gradient. No adapter is added.

Base seed 0 and perturbation seed 7 retain the same complete original weights.
CPU and CUDA oracle initial states also match all 144 arrays bitwise. Both
runtimes' actual outputs respond to the input perturbation. Each comparison
checks output, 144 parameter gradients and one sample gradient: 146 fields.
Three SGD steps at learning rate 0.001 retain complete output, loss, gradients
and updated parameters: 873 fields. The collector derives parameter names from
the complete measured snapshot and rejects a changed 144-parameter protocol.

The original criterion is unchanged: maximum absolute error <= 0.002 **or**
maximum error divided by the largest reference magnitude across that phase's
whole field <= 0.01, with floor 1e-12. Key sets, shapes, dtypes and finite values
must match exactly; missing gradients cannot become zeros or skips.

| Device | Base / perturb / training arrays | Largest absolute error | Largest scaled relative error |
| --- | --- | --- | --- |
| CPU | 146 / 146 / 873 | 4.9591064453125e-5 | 9.408136315500259e-7 |
| CUDA | 146 / 146 / 873 | 3.4332275390625e-5 | 6.740602497111345e-7 |

All 1,165 field pairs per platform pass the original budget. Each of the twelve
actual processes attests to the same 695 source and 697 installed payload hashes.

Actual update observations distinguish missing gradients from legitimate small
updates. All requested gradients are present. Some attention key-bias gradients
are near numerical zero and their SGD update rounds away in FP32 on both sides;
this is not described as every parameter changing every step. Exactly 140/144
parameter arrays change at each step on each runtime/device; four attention
`to_k.bias` arrays remain unchanged. All 48 such rows independently equal the
FP32-rounded SGD formula; their largest theoretical update/ULP ratio is
0.000375000003259629. Per-parameter, per-step gradient/update norms are retained.

## Coverage and maintained gates

The shared schema-v2 auditor accepts the two fresh CPU/CUDA L3 records with
zero rejections against the explicit clean measurement baseline. Each device
counts `model_api 1/87`, solely `UNet2DModel`, with tiny shape and all 144
trainable parameters retained in `verified_scenarios`. The four pipeline axes
remain `0/79`, `0/249`, `0/37`, and `0/3`.

The denominator is the unchanged complete `models._import_structure` public
export axis, with optional branches, Flax APIs, helpers and mixins retained.
It is not a model-family count. The four pipeline axes remain independent and
zero; constructing and training UNet is not an Omni/generation pipeline,
pretrained checkpoint, other dtype, NPU, performance, memory, L4 or L5 result.
The same UNet member is not counted twice because historical and fresh records
have different baselines.

Both maintained required parity nodes pass with zero skips:
`compat/tests/torch/test_ecosystem_parity.py::EcosystemParity::test_diffusers_unet2d`
(CPU, 80.72 s) and `EcosystemParityCUDA::test_diffusers_unet2d`
(CUDA, 350.58 s including first compilation). `JITTOR_REQUIRE_REAL_TORCH=1`,
`JITTOR_REQUIRE_DIFFUSERS=1`, `JITTOR_TEST_REQUIRE_EXECUTION=1`, and both explicit
device selectors are set. Unit caches are separate from measurement caches.

New and existing component/historical host tests pass 44 cases. The cleanup and
pytest structure contracts pass 47 cases after removing a duplicate test loader;
Ruff and repository layout pass. The first full structure run had 1504 passes,
four skips and three failures: the duplicate loader was corrected; two remaining
central report-index/MANIFEST registrations belong to integration, not numerical
validation. The documentation contracts outside those two central registrations
then pass 13 cases, with those two deliberately deselected. No core math or
adapter changes are part of this tool-only batch.

Host contracts reject changed nested installed files, inconsistent complete
installed/source hashes, incomplete parameter snapshots, invalid arrays and
unrelated pipeline claims. Six additional mutations of copies/in-memory metadata
from these actual CPU/CUDA runs reject missing source entries, inconsistent full
installed-source hashes, and a removed real parameter-gradient array. Sealed
originals are unchanged. Existing historical migration tests remain valid.

## Reproduction and evidence

Use the existing source-bound `component_measure.py` launchers around
`verify_repo.py` and `training_probe.py`, followed by the shared
`collect_component_scorecard.py`. Pass `--repo diffusers` to the measurement
wrapper, clean `--checkout` at the full upstream SHA, and `--root` at the clean
measurement commit. Select `diffusers_unet2d` only; use separate CPU/CUDA
processes and unit/measurement caches, `JITTOR_ECOSYSTEM_TF32=0`, explicit
`JITTOR_TEST_DEVICE` and `JITTOR_TEST_DEVICES`, offline inputs and an independent
`REAL_TORCH_PYTHON`. No dependency upgrade is part of the measurement.

Unversioned raw files live under
`$JITTOR_LAB_ROOT/_state/diffusers-installed-source/20261015/`. Exact argv and
environment, interpreter/package/source metadata, original complete weights,
all arrays, independent update/source audits and gate logs are retained there.
The wheel and full-source proof are separate artifacts; they are not retroactive
proof for the historical 230ba7a0 installation.

The immutable `sealed-numerical-manifest-v1.json` binds 95 raw files,
45,307,571 bytes, SHA256
`6d9446897a2aea5b75317edecc0a8945b1d0944db2496d3015742020acc7f394`.
Its `component-records-final-v1.json` contains the shared evidence rows. The final
scorecard is generated read-only from those existing rows, clean source inventory
and original arrays, with `--baseline` set to the full measurement SHA; its
separate auditor/documentation commit does not relabel the measured program.

The measured collector writes its derived `complete-comparison.json` on each
collection. Each platform was collected once before sealing. Do not rerun that
version on a sealed directory; use the auditor on existing records or a fresh
copy. The integrator's subsequent overwrite protection has a separate baseline.
