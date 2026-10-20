# SiLU precision and the MiniMaxH3 inference boundary

- Status: SiLU/SwiGLU contracts verified; Omni model admission blocked
- Date: 2026-10-15
- Baseline: public SiLU `3fe6a787a2a58ca8f7da4edd9a0fae0e91ea5be9`; final clean combined source `ac4d4cf28f7d2c68c0d4d7256fb1e6f5a3f92b1f`
- Owner: Jittor core, CUDA and vLLM adapter maintainers
- Review when: SiLU/SwiGLU precision, adapter dispatch, SDPA backend selection, H3 upstream source, or its autograd contract changes

The public low-precision SiLU path now evaluates its operands in FP32 and rounds
its result to the requested dtype. CUDA inference SwiGLU preserves that SiLU
rounding boundary before multiplying its second operand. The vLLM adapter keeps
the upstream `forward_native` method and the private CUDA method's actual
no-autograd contract. These repairs pass their public and real-library CPU/CUDA
checks. They do **not** establish whole-model MiniMaxH3 correctness or training:
the original default-attention comparison still fails, and both genuine Torch
and candidate lack most model gradients.

## Fixed source and environment

The independent oracle uses genuine Torch `2.13.0+cu130` and its actual compiled
vLLM extension, not the shim. Both sides execute vLLM `0.29.0`, exact source
`98dff2a81d747d1dba01a47f939f48c3526d4206`, and Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8`, installed SCM version
`0.29.0rc2.dev58+g446c2b547`. The older runbook's cu129 environment and older
source SHAs supply no numerical evidence for this pair. Candidate compilation
uses CUDA 12.2.140; device processes use separate caches, one host thread, eager
execution and strict CUDA kernel math. TF32 is disabled and Torch matmul
precision is `highest`. Candidate snapshots assert actual physical device before
host observation; backend fallback is forbidden and its counter remains zero.
No performance, L4, multi-rank or engine result is reported.

The H3 fixture is the real `MiniMaxH3DiTModel` with one main block, one token
refiner, hidden=32, four heads, head dimension=8, FFN=64, and 33268 parameters.
All 36 named parameters and one buffer are initialized identically from NumPy
seed 3. Inputs use seeds 7 and 19: two packed text, two video and two audio
tokens. This is a tiny model configuration, not a pretrained checkpoint or an
Omni pipeline. The original forward protocol freezes parameters while its four
floating inputs require gradients; it is not a `torch.no_grad()` measurement.

## Verified repairs

The saved H3 SiLU input first distinguished premature narrow sigmoid rounding
from legitimate SDPA algorithm differences. Before the repair, BF16 public SiLU
already differed in 74/256 output values. After the repair, FP16/BF16 public SiLU
and SiLU-times-up outputs are bitwise equal to genuine Torch on the saved inputs
on CPU and CUDA. First and second derivatives pass the frozen 1% scaled budget;
they are not described as exact (maximum scaled error: CPU 0.719424%, CUDA
0.675676%). FP32/FP64 behavior and analytic second derivatives remain covered.

The actual vLLM `SiluAndMul` matrix uses three dtypes and both input gradient
states. `custom_ops=['none']` exercises the original upstream native method;
default CUDA exercises the real compiled `_C` oracle. Low-precision native
outputs and CUDA native input gradients match bitwise; CPU native gradients
retain a maximum scaled error of 0.359712%. Default CUDA outputs match all six
arrays bitwise and both sides expose no autograd connection, even when the input
requires gradients. The adapter does not silently turn the native path into
that inference-only method. CPU default `_C` on the installed CUDA wheel fails
as an unavailable kernel, while the real native CPU path executes.

| Check on the source-fix batch | Result |
| --- | --- |
| Public SiLU tests on original owner batch | CPU 6 passed; the claimed CUDA six-node run selected CPU inside the tests, so it supplies no CUDA evidence |
| Original CUDA-environment neighbor run | 15 passed; per-node device execution was not sufficiently recorded, so this count supplies no additional CUDA proof |
| Combined adapter/kernel/public directed tests | CPU 12 passed; the original CUDA process reported 27 passes but its six public SiLU nodes executed on CPU and cannot be counted as CUDA |
| Maintained `nox -s ecosystem_vllm` | CPU 1 / CUDA 1 passed, zero skips |
| Native and Torch CPU core | 322 passed, 42 declared skips, 1 existing xfail |
| Full structure after the kernel/adapter fix | 1507 passed, 4 declared skips, zero failures |

The existing component gate still covers the same three CustomOp members;
adding precision configurations does not add model-family coverage. Related
shared repairs retain their own source and evidence identities:
[RMS/alias contracts](2026-10-15-rmsnorm-triton-alias.md),
[like factories](2026-10-15-like-factory-leaves.md), and
[SDPA opmath](2026-10-15-sdpa-opmath.md).

## Original H3 default comparison remains blocked

All 37 initial state arrays match exactly. The original 0.2% forward budget is
unchanged. Maximum scaled output error against genuine default attention is:

| Input seed | Video | Audio | Loss |
| --- | ---: | ---: | ---: |
| 7 | 0.120073% | **0.583711%** | **1.701884%** |
| 19 | 0.181590% | **0.449398%** | **0.265659%** |

Four of six fields fail. A separate genuine-Torch MATH calibration, with actual
math operators observed and fused alternatives absent, is bitwise equal to the
candidate in all six final fields. Of 194 intermediate snapshots, 184 are
bitwise equal; ten FP32 time-embedding/projection aliases differ by at most
1.4901161e-8. Every observed BF16 intermediate is exact to that MATH calibration.
This identifies the remaining difference on this fixture as also present in
**genuine default versus genuine MATH**. It does not replace the original
reference, alter its budget, or turn a failed default gate into a pass.

The all-gradient audit separately makes all 36 parameters and four floating
inputs trainable. In both real Torch and candidate, on both seeds, only the
four final video/audio head parameters have gradients. **32/36 parameter
gradients and 4/4 input gradients are absent**. The diagnostic writes actual
arrays and the missing-name ledger, then exits nonzero. No fabricated backward,
zero-filled gradients or head-only SGD trajectory is accepted as full-model L3.

## CPU and coverage boundary

CPU was attempted first with GPUs hidden and candidate CUDA explicitly disabled.
The installed genuine CUDA wheel's H3 constructor requests CUDA capability and
fails; candidate rejects unavailable Gloo. Those wheel failures alone do not
prove the absence of all CPU implementations. The pinned upstream separately
supports CPU packaging, but its built-in Omni diffusion platform plugins are
CUDA, ROCm, NPU, XPU and MUSA. `UnspecifiedOmniPlatform` declares device `cpu`
without implementing the required diffusion-attention selector. No built-in H3
CPU diffusion-attention runtime is declared by this source snapshot; external
plugins and future CPU builds remain outside this source-bounded conclusion.

The source-regenerated Omni scorecard contains **no admitted numerical record**:
model families **0/92**, diffusion models **0/68**, pipeline model-type keys
**0/64**, each bound to the full `446c2b...` SHA. These are separate axes, not
additive coverage. The forward diagnostic is not an admission collector and is
not registered as a passing shared ecosystem model case. R27/R28 remain open;
there is no new Omni L2/L3, CPU runtime, engine or performance claim.

## Reproduction and raw evidence

The separate maintained-tool source `5216ac20da92d67fbf374fdd5c0898ff2ec84fbb`
contains `agent/skills/vllm-omni-torch-compat/scripts/probe_h3.py`,
with separate runtime/device, fixed clean upstream pins, overwrite protection,
source identity, physical-device checks, and `--mode grad-audit` failure on any
missing gradient. Genuine-only `--torch-math` creates a separate calibration.
Use its commands and environment prerequisites in the
[Omni skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/vllm-omni-torch-compat/SKILL.md).

Run public regressions with explicit device selection and the selected runtime:

```bash
JITTOR_TEST_DEVICE=cuda PYTHONPATH=python \
  <jittor-python> -m pytest -q tests/nn/test_silu_opmath.py \
  tests/backends/cuda/test_swiglu_dtype_boundaries.py
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda PYTHONPATH=python:adapters \
  <jittor-python> -m pytest -q adapters/tests/vllm/test_layers.py
```

Use separate CPU processes with GPUs hidden and `JT_USE_CUDA=0`; CUDA-only tests
are checked on the target device, not counted as CPU execution. The maintained
real-library gate requires independent `REAL_TORCH_PYTHON`, installed matching
vLLM, `JITTOR_TEST_DEVICES=cpu` then `cuda`, and runs `nox -s ecosystem_vllm`.

Unversioned artifacts live at
`$JITTOR_LAB_ROOT/_state/vllm-evidence/20261015/`. The final production evidence
manifest `silu-h3-final-evidence-manifest-ac4-v1.json` binds 173 files, 4,403,832
bytes, SHA256 `afcaf5b5b7c2225b7849f2e6ea5cbb7d1e691507ab93ebc3961114d26a8c965e`.
It includes before/after arrays, actual library call routes, fixed-input H3 raw
and missing-gradient ledgers, source snapshots and gate logs. Original metadata
remains authoritative: earlier `86073` and `cea664` plus explicit owned-source
hashes are not relabelled as clean `ac4`. Original CPU failures, default failures
and the two earlier central-document registration failures are retained; the
latter were cleared by the subsequent published central registration.


The tool was then run from clean `5216ac20da92d67fbf374fdd5c0898ff2ec84fbb`,
which preserves the production fixes and integrates published CPU MPI source
`01351fe61`. CPU was retried first and retained the real platform/Gloo errors.
On CUDA, each runtime reproduced its original clean-ac4 six forward fields and
37 initial states bitwise. Each runtime's separate gradient audit also reproduced
all 14 available arrays and 37 states bitwise, recorded the same missing names,
and exited nonzero. The two source identities remain separate.

The tool's eight host contracts pass. Full structure initially passed 1506 tests
with four skips and rejected collection-time dynamic import in the new host
test; moving that import into `setUpClass` fixed the test. All 42 affected pytest
and child-process structure contracts then passed. No structure rule was
relaxed. Layout and Ruff passed. These tool-only results do not supersede the
production batch's core or real-library gates above.

The additional `maintained-h3-tool-final-manifest-5216-v1.json` binds 25 raw
files and the three tool/test/skill source hashes, SHA256
`49d0d6a0688b17517514abb5685305c1cc15938e8658ba954774ad7f013db8d9`.

## Main-tree integration and corrected device evidence

Integration starts from synchronized `4ce19215cba8fb15d7b347f53fda012ced9b4143`.
The original runner set the process CUDA flag but omitted
`JITTOR_TEST_DEVICE=cuda`; the public SiLU test's local scope therefore selected
CPU. Its original six-node CUDA claim, and the insufficiently recorded neighbor
claim, are withdrawn above. Original logs remain unchanged. The separate
saved-input and real-library probes have explicit device arguments and physical
residency assertions and retain their own bounded evidence.

The first integration run with actual CUDA selection exposed float64 SiLU error
of `1.55156931e-7`: the shared CUDA sigmoid expansion called `expf` even for
double inputs. It now selects double `exp` for float64, retaining the existing
float32 path. Public SiLU's original `2e-13` double tolerance is unchanged, and
its test now checks sigmoid directly as well as forward, gradient and second
derivative. This is an additional core repair beyond `ac4`, not a relabelling of
the earlier source measurements.

After that repair, actual CUDA public SiLU/SwiGLU/unary regressions pass all 29
nodes, zero skipped; adapter contracts pass six nodes separately in Torch mode.
CPU SiLU/unary regression passes 11 nodes with 15 declared CUDA skips, and the
host probe/adapter contracts pass 14 nodes. An initial CPU erfinv reference check
failed because the environment contained only a partial SciPy namespace;
repeating with the existing genuine SciPy 1.13.1 installation passes. The
initial mixed native/adapter command was rejected by the process-mode policy;
the final runs use separate native and Torch processes.

Core passes 322 nodes with 42 skips and one existing xfail; full structure
passes 1,505 with six skips. An independent integration audit rehashes all 173
production and 25 tool evidence files plus the three exact tool sources, and
recomputes 144 saved-input/component array pairs within the original budgets.
The six H3 outputs and 37 weights are independently rechecked: the original
default reference still fails four fields, while all six MATH-calibration fields
are byte-equal. No new full-model admission follows.

Unversioned integration evidence is under
`$JITTOR_LAB_ROOT/_state/silu-integration-4ce/`; the independent H3 forward and
production hash audit is under
`$JITTOR_LAB_ROOT/_state/cpu-bf16-math-integration-013/next-silu-*.json`.
