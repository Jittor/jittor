# MiniMaxH3 autograd boundary in the real upstream CUDA path

- Status: Root cause verified; complete-model gradient/training admission remains blocked.
- Date: 2026-10-15.
- Initial diagnostic source: clean `2857a614b30d5315da74621c662a759e3f187ea4`.
- Maintained diagnostic source: `521314943b480517b6fb422c09359fc521081a8f`.
- Owner: Downstream validation and vLLM adapter maintainers.
- Review when: pinned Omni operation dispatch, Triton/CUDA autograd registration, supported platform or H3 training API changes.

This follow-up preserves the [SiLU/H3 report](2026-10-15-h3-silu-contracts.md),
its original default-oracle failures and all sealed raw files. It investigates
why both runtimes lacked 32/36 parameter gradients and all four floating-input
gradients; it does not replace the original scaled-relative 0.002 (0.2%) forward
budget, select a MATH oracle, count a component as a model, or train only the
final heads.

## Actual identity and protocol

The clean upstreams remain vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`
(exact release 0.29.0) and Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8` (actual installed SCM
`0.29.0rc2.dev58+g446c2b547`). The independent interpreter is binary Torch
`2.13.0+cu130`, with actual extension path/SHA256 captured inside the new
maintained run. This is not the Diffusers Torch 2.11+cu128 interpreter. No new
trace changes an old measurement's actual version or source identity.

The tiny real `MiniMaxH3DiTModel` retains one main block, one refiner, hidden32,
33268 scalars across 36 named parameters, fixed initial seed3 and input seeds7/19.
All 36 parameters and the four declared differentiable inputs (`x`, audio,
text and time) are explicitly enabled for the full-gradient audit; masks and
integer indices remain fixed control data. Initial constructor trainability is
separately recorded; it never reduces the audit's expected list. Evaluation and the public
`model.train(True)` path are separate real executions, using default attention.

| Request | Real outcome |
| --- | --- |
| Genuine CPU | H3 constructor requests CUDA capability; `No CUDA GPUs are available` with GPUs hidden |
| Candidate CPU | Real upstream initialization rejects absent Gloo: `Fallback Gloo backend is not available` |
| Genuine CUDA eval / train | Both seeds in both runs miss 32 parameter and all four input gradients |
| Candidate CUDA eval / train | Same missing names and counts in both runs; physical CUDA snapshots and zero fallback |

All six processes preserve nonzero exit status. The four CUDA runs save their
actual missing-name ledgers before rejection; each has only 14 available arrays,
not the complete 86-field output/loss/gradient request. The constructor has 19
true and 17 false parameter requires-grad flags on both runtimes. The audit
records all 36 parameter and four input requests as enabled, with global
`grad_enabled=True`; partial constructor freezing does not explain away the
observed graph breaks. The genuine extension SHA256 is
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`.

All four runs retain identical 37-array serialized initial states. Each runtime's
eval/train pair saves the same 14 arrays bitwise. The genuine observational
wrappers and subsequent maintained probe also match those 14 arrays bitwise in
each mode. This verifies that the observation/flag/metadata changes preserve the
recorded calculation; **it does not compare a complete candidate gradient set
with an independent oracle or establish full-model numerical correctness**.
Candidate uses the CUDA 12.2.140 toolchain and strict CUDA kernel math, with TF32
OFF/highest. Graph capture/replay and performance are not separately validated.

## First graph breaks

Observational hooks on the genuine model preserve original operations and
record 92 module/function boundaries per two-input execution. The earliest
observed true-to-false gradient boundary is the token refiner's RMSNorm, whose
actual selected method is Omni `RMSNorm.forward_cuda`. It allocates an output,
passes `weight.data` to the compiled `_C` RMS kernel, and returns an output
without an autograd edge. Omni `SiluAndMul.forward_cuda` likewise writes an
allocated output using the real `_C` kernel.

`VllmConfig(custom_ops=['none'])` controls vLLM's CustomOp registry. Omni's
separate `diffusion.layers.custom_op.CustomOp.dispatch_forward` does not read
that setting; it selects `forward_cuda` from the actual platform. Interpreting
the configuration as disabling every Omni fused kernel was incorrect. The
maintained diagnostic now records the request and each actual selected method
separately; it does not reroute them.

The real CUDA modulation functions independently establish more breaks:
`rms_norm_indexed_scale_shift`, `indexed_gate`, and
`indexed_gate_rms_norm_scale_shift` allocate `empty_like` outputs and fill them
with bare Triton launches, without autograd registration. In genuine Torch,
all three return `requires_grad=False` despite trainable inputs. The final
`indexed_scale_shift_` preserves a pre-existing graph on its destination but
adds no edges for shift/scale; non-None output gradients alone cannot prove
that function's full derivative contract.

These are upstream inference operation contracts. They are not fixed by
setting additional requires-grad flags, calling `train()`, changing the
candidate's core mathematics, inventing a backward, or reporting only the four
head parameters as complete training.

## CPU and official alternatives

Eight real-source function cases per device cover FP32/BF16. The three
out-of-place CPU branches have actual gradients for all their inputs.
The final CPU in-place `indexed_scale_shift_` instead raises a genuine Torch
version-counter error during backward. The corresponding CUDA cases expose
no new shift/scale graph edges. These isolated function results do not establish
an executable complete-model CPU route.

The pinned Omni builtin platform registry declares CUDA, ROCm, NPU, XPU and
MUSA; its fallback is `UnspecifiedOmniPlatform`, which declares a CPU device but
does not implement the required diffusion-attention selector.
No CPU platform is fabricated here, and a CUDA-wheel failure does not prove
that every possible out-of-tree CPU plugin is unsupported.

The actual H3 pipeline `forward` is decorated with `torch.no_grad()`. The
source's official regional compilation helper compiles only H3's declared
`MiniMaxH3DiTBlock` regions, leaving final normalization outside those regions.
It is an inference compilation feature, not evidence of a complete training
contract. No alternate full-training API was found in the audited pinned H3
model/pipeline and compilation owners. Experimental whole-model compile and
out-of-tree plugins are not granted or ruled out by this bounded audit.

## Maintained diagnostic changes and validation

Only the existing probe and its host contracts change. `--model-mode eval|train`
defaults to the previous eval behavior and calls the real Module flag API.
Metadata now distinguishes the vLLM request from actual operation methods,
records constructor and requested trainability, global grad state, executable,
and genuine extension SHA256, and checks the additional executed upstream
activation/dispatch sources. Missing gradients still save their real ledger
and cause a nonzero exit. No optimizer trajectory is run after this failure.

The ten probe host contracts pass, including observation without invoking or
replacing selected methods and rejection of an invalid callable route. The
complete structure suite together with those ten tests passes 1,518 cases,
with four declared/accelerator skips and zero failures. Ruff, diff checks and
repository layout pass. The later documentation contracts pass 13 cases, with
two central index/MANIFEST registrations deselected for the integrator. Only a
diagnostic tool and its host tests change; there is no new native calculation,
compat behavior or adapter repair.

The initial host attempt used the automatic shim cache and was terminated
before completion; the successful host/structure runs use the explicit isolated
unit cache. All fresh model diagnostics use separate CPU/CUDA benchmark caches,
source remains clean throughout, and first compilation is serial.

The existing source-regenerated inventory remains `model_family 0/92`,
`diffusion_model 0/68`, and `pipeline_model_type 0/64` at Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8`, on both CPU and CUDA. No new
complete model or pipeline L2/L3 record is admitted, and no performance/L4
conclusion is drawn. Another process occupied the allocated accelerator; only
correctness diagnostics were attempted. The open R27/R28 goals are not closed.

## Evidence

Raw files are unversioned under
`$JITTOR_LAB_ROOT/_state/omni-h3-autograd/20261015/`: exact commands/environment,
source observations, CPU/CUDA isolated-function records, original full-model
arrays/weights and missing ledgers, graph observations, and validation logs.
The initial hooks use the clean synchronized source above; the subsequent
maintained probe runs have their own clean source SHA. Existing ac4/5216 and
Diffusers 1a92 sealed files were neither overwritten nor relabelled.

`sealed-evidence-manifest-v1.json` binds 51 original files, 1,377,904 bytes,
SHA256 `2c02826b8defd601421c94fc44cbc86f52b7755c08183b696ed058c045fcbdd8`.
It includes both CPU setup failures, all four CUDA rejected gradient audits,
original arrays/weights, independent graph/source observations, exact commands
and environment, the actual-reason audit and validation logs. Cache/venv trees
and this later documentation are excluded. Full raw manifest hashes and owned
diagnostic source hashes are checked again after sealing.
