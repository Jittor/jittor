# vLLM 0.29.0 primitive CPU and CUDA scorecards

- Status: current installed-source L2 replay verified below; historical 72e rows fail current strict source admission. No engine acceptance.
- Date: 2026-10-16; original 72e measurement sections retain their 2026-10-15 identity.
- Original measured source (sections before the fresh replay): `72e12a9c662b46d5a5389f7e031c3997520d4562`, clean throughout capture.
- Owner: Jittor compatibility maintainers.
- Review when: the pinned upstream, builder, runtime, adapter, precision policy or gate changes.

## Identity and coverage

The independent interpreters execute the same vLLM 0.29.0 package and
Transformers 5.13.0. The oracle imports the real PyTorch 2.13.0+cu130 binary;
the candidate imports the Jittor Torch frontend. The clean upstream checkout is
`98dff2a81d747d1dba01a47f939f48c3526d4206`, the exact official `v0.29.0`
tag. The installed generated version reports `0.29.0` and commit marker
`g98dff2a81`. The original vLLM wheel SHA256 is
`09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`.
All 2312 Python package source files enumerated from that checkout match the
installed package byte for byte; the inventory also binds 589 registry/target
source files. Package installation alone is not numerical evidence.

The automatic source registry has **376 model families and 40 CustomOp
registrations**, separate bounded axes. Each device's one scenario verifies the
native paths of `silu_and_mul`, `rms_norm` and `rotary_embedding`: **3/40
CustomOps, 0/376 model families**. Runtime plugins remain an open set. The older
`51a99565c398c8320de8131e07731c75c52eb87c` inventory has different denominators
and is not used here.

| Device | Scenario | Depth | CustomOp numerator | Model numerator | Gradient scope |
| --- | --- | --- | --- | --- | --- |
| CPU | `vllm_decoder_primitives`, tiny FP32 | L2 | 3/40 | 0/376 | all four input gradients; zero trainable parameters |
| CUDA | same fixed weights and shapes | L2 | 3/40 | 0/376 | all four input gradients; zero trainable parameters |

## Scenario and numerical protocol

The maintained builder constructs the real public `SiluAndMul`, `RMSNorm` and
`RotaryEmbedding` objects under an actual vLLM configuration with compilation
and custom kernels disabled. `RMSNorm(16, has_weight=False)` has an intentionally
unused weight. Inputs are gates `[5,32]`, hidden states/query/key `[5,16]` and
five integer positions with maximum position 32. Rotary head dimensions are
8. The four outputs concatenate to a 320-element vector. This verifies upstream
native component execution; it does not verify a vLLM `_C` kernel.

Both interpreters receive the same serialized initial state and NumPy-generated
inputs. The base seed is 0. Seeds 7 and 19 change inputs while preserving that
initial state, and each oracle output is checked to differ from the base.
Every run compares exactly five arrays: output and gradients of `gates`,
`hidden_states`, `query`, and `key`. Symmetric key sets, exact shapes and finite
values are required. Thus each device has 15 compared arrays with no missing
gradients. There is no parameter update or training claim.

The frozen per-tensor normalized maximum error uses the oracle's maximum
absolute magnitude with a `1e-8` floor. Limits are `0.002` for output and `0.01`
for input gradients, unchanged across devices and perturbations.

| Device | Seed 0 maximum | Seed 7 maximum | Seed 19 maximum | Maximum absolute error across all arrays |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1.0609394e-7 | 9.7602168e-8 | 1.9150882e-7 | 4.7683716e-7 |
| CUDA | 6.7617129e-8 | 9.7602168e-8 | 1.1146150e-7 | 4.7683716e-7 |

Both candidate devices report fallback policy `error`, count zero, and the
requested device. CPU explicitly hides CUDA and reads back `use_cuda=0`.
CUDA reads back `use_cuda=1`, `use_acl=0`. Both runtime metadata records agree
on TF32 disabled, cuDNN benchmark disabled, highest matmul precision and one
BLAS/OpenMP thread. Jittor kernel math is strict.

The machine uses an AMD EPYC 9554 CPU; the CUDA process sees one NVIDIA GeForce
RTX 4090. Actual runtime capture records driver 580.178.04, oracle CUDA 13.0
and candidate nvcc 12.2.140, plus package origins and GPU identity. These are
correctness results. No performance budget, process memory peak, exclusive
machine use or L4 claim is inferred from this tiny fixture.

## Maintained gate and reproduction

The exact gate node is
`compat/tests/torch/test_ecosystem_vllm.py::EcosystemVLLM::test_vllm_decoder_primitives`.
The dedicated `ecosystem_vllm` nox session requires preprovisioned real packages
and an independent oracle. Its required flags reject missing dependencies,
wrong interpreter identity and wrong device. CPU and CUDA formal nox runs each
passed one test with zero skips before source sealing; those logs retain their
original baseline. Eight host contracts cover the session wiring.
The corresponding shared-owner fix and its wider regression results are in
[RMSNorm contracts and Triton pointer aliases](2026-10-15-rmsnorm-triton-alias.md).

```bash
# Provision real oracle/candidate dependencies and set verified source/wheel inputs.
export JITTOR_VLLM_CHECKOUT="$VLLM_CHECKOUT"
export JITTOR_VLLM_WHEEL="$OFFICIAL_VLLM_WHEEL"
export JITTOR_VLLM_EVIDENCE_DIR="$FRESH_EXTERNAL_EVIDENCE_DIR"
REAL_TORCH_PYTHON=<oracle-python> JITTOR_CI_PYTHON=<candidate-python> \
  JITTOR_TEST_DEVICES=cpu nox -s ecosystem_vllm
# Select the allocated GPU externally; keep an independent cache for each device.
REAL_TORCH_PYTHON=<oracle-python> JITTOR_CI_PYTHON=<candidate-python> \
  JITTOR_TEST_DEVICES=cuda nox -s ecosystem_vllm

# Historical generic diagnostic; strict admission uses the maintained source-bound gate.
<host-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo vllm --cases vllm_decoder_primitives --device <cpu-or-cuda> \
  --oracle-python <oracle-python> --shim-python <candidate-python> \
  --repeats 1 --seed 0 --no-memory --out <outside-checkout>
```

Raw artifacts live under
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`:

- `vllm-clean-72e12a9c6/manifest.json` binds capture commands, clean measured
  source, both devices' NPZ/metadata/logs and source hashes.
- `vllm-central-inventory-72e.json` and
  `vllm-installed-source-proof-72e.json` bind the registry and installed source.
- `platform-vllm-{cpu,cuda}.json` records actual platform and precision values.
- `vllm-scorecard-evidence-72e.json` binds the two numerical records to their
  exact gate, full registry references and raw artifact hashes;
  `vllm-scorecard-72e.json` is the regenerated central auditor output.
- `replay_vllm_primitives_v2.py` preserves fixed-weight perturbation commands;
  `assemble_vllm_scorecards.py` assembles rather than invents numerical evidence.

The central audit accepts both records. Negative checks reject an unknown
registry member, a missing input gradient, and a platform precision block that
claims TF32 enabled even when its artifact hash is valid but actual measurement
metadata says disabled. All 44 capture artifacts were independently rehashed.

The scorecard explicitly uses historical `--baseline
72e12a9c662b46d5a5389f7e031c3997520d4562`. Later integration and documentation
commits do not change that measurement identity. An initial collector used an
obsolete report filename after a successful CPU base run; its failed log is
retained, and the corrected collector reused that unchanged base artifact.

## Engine and Omni boundaries

A genuine vLLM 0.29.0 CUDA oracle executed the saved tiny Qwen3 engine for two
prompts, three tokens each, with log probabilities. The candidate engine stops
at the real `torch.fx._graph_pickler.GraphPickler`/`Options` import. No empty
GraphPickler implementation was added. The oracle log is
`logs/oracle029-engine-cuda-v5.log`; the candidate failure is
`logs/shim029-engine-cuda-initial.log`. Component success does not remove this
engine blocker or establish model coverage, serving performance or training.

vLLM-Omni at source `446c2b5474dabffccf9a0034836eeab77c62b4b8` is a separate
pending validation. The actual SCM version is
`0.29.0rc2.dev58+g446c2b547`. At the time of this sealed capture, its tiny MiniMaxH3DiTModel CUDA inference
had a reproduced RMSNorm precision-contract difference, while the genuine CPU
path rejects the CUDA-only Omni platform. Neither path is included in these
vLLM records. The original failures and numerical artifacts are retained;
there is no Omni model, pipeline, training or CPU coverage award here.

## Integration audit

Integration synchronized `824cb354d5cced0743b3207d50d518111ba59e76` and
preserves the clean `72e12a9c662b46d5a5389f7e031c3997520d4562` measurement
identity. Independent review rehashed all 44 raw capture artifacts and nine
final delivery bindings, matched the eight measured source files against that
Git commit, and re-derived the complete registry from the clean official
release checkout. It also rehashed all 2312 tracked package Python files against
the actual installed package used by both runtimes, recomputed the 30 numerical
array comparisons and verified that both perturbations changed the output.
The current shared auditor accepted two records with zero rejections.

The sealed scorecard SHA256 is
`24d170691d2ea4efbb772644d146b33f6c0dd5a45692413f2d23d9a5bcc4fea1`;
the final delivery manifest SHA256 is
`42dab72c4bd2ea9a0d07e884cf19e8b176b67385f6fefdd5d9316b30bda8ed03`.
Central results navigation and the generated package manifest are registered
by this integration. Independent checks and logs are unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-scorecard-integration-824/`.
Later Omni precision and storage-contract investigations remain separate work;
no subsequent model acceptance is implied by this component report.

## Fresh complete installed-source replay (2026-10-16)

Measured source is clean `c9978992512024a4951a139e4c3d221aaa2584b1`.
Owner: Jittor compatibility maintainers. Recheck when the source/wheel pin,
producer, adapter carrier, nox session, collector, precision or admission rules
change. The original 72e records remain unchanged: the current common auditor
independently rejects both for missing actual distribution/source binding.
The new result below comes from complete new CPU and CUDA execution.

The clean official upstream remains `98dff2a81d747d1dba01a47f939f48c3526d4206`,
with the original official vLLM wheel SHA256 above. Every worker binds the actual
`vllm.__file__`, real distribution METADATA/version and legitimate installed
RECORD to the full original wheel and **2312 Python / 4904 package payload files**.
Actual Transformers 5.13.0 payloads must also match between the two runtimes.
The existing candidate `jittor-torch-adapters` 2.0.0 carrier is installed normally
and bound to its actual bootstrap, entry point, METADATA, RECORD and editable
source URL. No new adapter or mathematical implementation is added. The probe
hashes all **1604** relevant framework, adapter and maintained gate source files
before and after execution, requiring a clean unchanged measured HEAD.

The original scenario and budgets remain unchanged: FP32, fixed `[32,8]` rotary
cache, output `[320]`, all four requested input gradients, seeds 0/7/19, output
normalized maximum `<= .002` and gradient `<= .01`, oracle scale floor `1e-8`.
Each perturbation changes output and at least one gradient on both sides, with
unchanged initial weights and distinct worker PIDs. CPU/CUDA oracle initial cache
arrays are also exactly equal. There are zero trainable parameters or updates.

Actual public nox execution passes **one test / zero skipped** on each device:
CPU **246.02 s**, CUDA **353.91 s** pytest time. Independent host replay without
importing Jittor rechecks all **30 array pairs**, **12 distinct workers**, **60
physical tensor observations** and every runtime source/installation packet.
CPU explicitly hides CUDA and reports CUDA unavailable; CUDA actual pointer
attributes resolve to its explicitly allocated device. Candidate fallback remains
`error`, count zero. Matching actual precision and runtime conditions require
TF32 and cuDNN benchmark OFF, highest matmul precision and one thread.

| Device | Seed 0 maximum normalized error | Seed 7 | Seed 19 | Largest absolute error |
| --- | ---: | ---: | ---: | ---: |
| CPU | `1.0609394e-7` | `9.7602168e-8` | `1.9150882e-7` | `4.7683716e-7` |
| CUDA | `6.7617129e-8` | `9.7602168e-8` | `1.1146150e-7` | `4.7683716e-7` |

The corrected collector and ordinary common auditor accept **two L2 records,
zero rejected, zero blocked**. Each device remains **CustomOp 3/40, model 0/376**.
The postprocessor's zero-parameter count/list fields were initially reversed;
a new positive-case schema assertion reproduced the failure before correcting
only those output fields at `ffff94ee2`. Its final 77 focused contracts pass.
This postprocessing change retains the original c997 source capture and uses the
frozen measured checkout for validation; it does not rewrite worker metadata,
retag measurements, change criteria or grant a training grade.

The actual platform is an EPYC 9554 and one RTX 4090, driver 580.178.04,
GCC 11.5.0, candidate NVCC 12.2.140 and genuine Torch 2.13.0+cu130. Private fresh
interpreters use the existing read-only genuine provider. The full official Torch
wheel SHA256 is `60fcdcb2f3876e21146cb4524ef06397d727ca9ad5f020818547e25075fe3cb7`;
all **12117 original installed files**, the legitimate pip-rewritten RECORD and
original wheel are rehashed after execution. No shared package was repaired.
Candidate NVIDIA metadata is separately recorded: cuDNN metadata says 9.26.0.51,
while an explicit real provider API probe reports **9.24.0**; CUDA runtime reports
**12.2**. The ten inherited NVIDIA provider metadata directories lack RECORD;
that fact is preserved, with no invented RECORD or full NVIDIA wheel-integrity
claim. These explicit provider probes do not assert every library was loaded by
every component worker. Complete platform files retain the original oracle
capture and add these candidate facts in new files.

The first fresh CPU attempt stopped before numerics because the existing adapter
entry point was absent. Two subsequent CUDA launches stopped before numerics
because the old nox session forced CUDA wheels OFF and the system toolkit lacked
`cublas.h`. The maintained session now honors the explicit option with default
wheel discovery ON. All failed logs remain. Private unit and per-device nox
caches are separate; task-owned caches are reused normally, without copying JIT
binaries or calling a later run another cold build. The CUDA wheel fingerprint's
first compilation ran serially. Full-engine dependency `pip check` remains
unsatisfied for unused engine extras/pins; this component result does not certify
an installable full-engine dependency closure.

All **2890 original Git blobs and post-run working-tree files**, including symlink
target text, match the frozen source snapshot. The later CUDA run leaves the
20 original CPU raw files unchanged. Forty original c997 raw files and eight
fresh derived comparison files are retained. The complete seal covers **199 files /
1035348925 bytes**, SHA256
`4b67d5ebc7a1fdd1fd871fa5271b46a31535429babb94495ac41eddb8a95cc7a`.
Unversioned commands, actual platform files, raw arrays, independent rechecks,
failed attempts, official wheel inputs and `scorecard-installed-source-v4.json`
are under `$JITTOR_LAB_ROOT/_state/vllm-installed-source/20261015/`.
Reproduce through the updated [vLLM skill](../../agent/skills/vllm-torch-compat/SKILL.md)
with real source/wheel inputs, fresh independent interpreters and new external
output paths. This result grants no whole model, engine, Omni, training,
pretrained, additional backend or performance support.

Main integration at `2c9b2176d` preserves the other **73 nox functions** exactly
and matches all 11 other delivered task files to c997. After the postprocessing
fix at `ffff94ee2`, focused host checks pass **77 / zero skipped**, core passes
**322 / 42 skipped / one existing expected failure**, and complete structure plus
compat structure passes **1627 / six skipped** (206.34 s). Two structure skips
are interpreter installation-resolution checks; no editable installation is
inferred from source-path imports. Central terminal logs and integration source
checks are unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-main-integration-5def/20261016/`.
