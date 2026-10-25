# Pinned H3 upstream inference kernels and the training boundary

- Status: Original private CUDA paths lack complete AD; training opt-in is design only.
- Date: 2026-10-16.
- Measured baseline: clean `aa9e669f514432a3aae77840d1604458c1b9f3ff`.
- Documentation baseline: synchronized `00ceb26fd395707fb33649cc43709af045110a9a`.
- Owner: Downstream validation and vLLM adapter maintainers.
- Review when: pinned Omni dispatch/kernels, the proposed training contract, or compat public `owned_runtime_hook`, `transaction`, `module_patcher` APIs change.

The fixed upstream **inference implementation does not provide complete-model
training AD**. Calling `train()` and enabling all parameter/input gradients does
not supply derivatives for its output-buffer CUDA/Triton kernels. This follows
the [original graph-boundary diagnosis](2026-10-15-h3-autograd-boundary.md); it
neither changes those raw files nor treats matching missing gradients as L2/L3.
The separate aa9 attention work improves the original default forward and
primitive gradient precision; the model's missing 32 parameter and four input
gradients remain an independent, unresolved boundary.

## Fresh actual diagnosis

Upstreams are unchanged: Omni `446c2b5474dabffccf9a0034836eeab77c62b4b8`
(actual SCM distribution `0.29.0rc2.dev58+g446c2b547`) and vLLM
`98dff2a81d747d1dba01a47f939f48c3526d4206` / `0.29.0`.
The real independent oracle is binary Torch `2.13.0+cu130`, extension SHA256
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`.
Installed source, complete wheel payload, METADATA/RECORD and source identities
are checked before and after execution. Omni's fixed C6 wheel is a reviewed
local source-built artifact, **not a PyPI official wheel**. No package, version,
upstream implementation, candidate core or adapter is modified.

| Actual request | Outcome |
| --- | --- |
| CPU, 16 component cases | 14 complete gradient sets; two final in-place cases fail backward with a version-counter error; worker exits 1 |
| CUDA, same 16 cases | Zero complete gradient sets; original private kernels actually execute; worker exits 1 |
| CPU / CUDA tiny Q/K control, two dtypes each | Both actual eager calls produce all four requested q/k/norm-weight gradients; each worker exits 0 |

Cases cover FP32/BF16, RMSNorm and SiluAndMul with both real eval/train flags,
and all four H3 modulation functions. Global grad is enabled and inference mode
is disabled. Actual CUDA input/output pointers belong to the allocated card;
RMSNorm/Silu kernels appear in the actual operator trace and modulation Triton
launches complete. These are genuine upstream component diagnostics, not a
candidate comparison or complete-model CPU support claim. CPU and CUDA use
separate processes and private caches; their exact commands and environments
are saved. No fresh process library-map snapshot was collected in this minimum.

RMSNorm passes `weight.data` to real `_C::rms_norm`, fills an allocated output,
and returns `requires_grad=False, grad_fn=None`; both x and the registered
weight requested gradients. SiluAndMul and the three out-of-place modulation
functions likewise return outputs without AD edges. The final in-place
modulation retains its destination's old `MulBackward0`, but gives no shift or
scale gradient. A non-None output `grad_fn` therefore does not prove complete AD.
On CPU the original RMS kernel attempt fails and upstream's own native branch
runs; this is recorded separately from a completed CUDA kernel.

## Complete bounded edge inventory

The following owners exhaust the private kernel interfaces reached by the
**one-main-block / one-refiner / hidden32 / 33268-scalar / 36-parameter** recipe.
The external inventory records file hashes, exact function line ranges and the
full remaining 32+4 names. It does not enumerate every possible future bug in
public operations once these graph breaks are repaired.

| Owner relative to `vllm_omni/diffusion/` | Actual fixed-path edge / remaining obligation |
| --- | --- |
| `layers/custom_op.py`: `CustomOp.dispatch_forward` | Selects platform CUDA methods independently of vLLM `custom_ops=['none']`; train mode does not switch it |
| `layers/norm.py`: `RMSNorm.forward_cuda`, `_forward_fused` | Refiner pre-norms/final norm, refiner Q/K norms and final-layer norm lose activation and weight edges |
| `layers/activation.py`: `SiluAndMul.forward_cuda` | Both MLP gates need derivatives to both halves of the fused fc1 output |
| `attention/ops/minimax_h3_modulation.py`: `rms_norm_indexed_scale_shift` | Main pre-attention path needs x, norm weight, indexed shift and scale derivatives |
| Same file: `indexed_gate_rms_norm_scale_shift` | Residual plus gate branch and normalized second output need all six input derivatives, including indexed reductions |
| Same file: `indexed_gate` | Main residual/gated MLP merge needs residual, gate and branch derivatives |
| Same file: `indexed_scale_shift_` | Final AdaLN needs x/shift/scale derivatives **and** correct in-place alias/version semantics; CPU currently errors, CUDA silently lacks shift/scale edges |
| `layers/fused_qk_norm_rope.py` | Tiny head_dim8/rotary6 takes actual eager `F.rms_norm` plus RoPE; all four edges are observed. The 128/96 fused Triton branch is outside this recipe, not a demonstrated tiny-model break |

The remaining public-operation chain is also mandatory in eventual full AD:
video/audio patch and text projections; time sin/cos and both projections;
QKV split/view, default SDPA and output projections; MLP projections; AdaLN
SiLU/projection/chunk and repeated-index reduction; BF16/FP32 casts;
`index_select`/`index_add_` multimodal packing; final output selection/masks;
and world1 SP prepare/gather identities. Default unquantized vLLM Linear routes
to public `F.linear`; SDPA retains the original default backend. These source
facts do not substitute for a complete candidate/oracle gradient comparison.
Quantized GEMM, larger fused RoPE, TP/SP/ring >1, multi-request packing,
compile/capture/replay and performance are outside this bounded diagnosis.

The original complete-model aa9 ledger has the same missing names in all eight
runtime/mode/seed rows: 10 embedding/time parameters, nine refiner parameters,
10 main-block parameters and three final norm/AdaLN parameters. Missing floating
inputs are x, audio, text and time. Only the four final video/audio head
parameters have gradients. It remains a failed full-gradient audit. This fresh
component run does not re-execute or relabel the earlier complete-model runs.

## Explicit training opt-in: proposal, not an implementation

Under the [adaptation skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/downstream-library-adaptation/SKILL.md),
these library-private kernels belong to the third-row adapter/upstream-fix
category. They do not establish a missing public Torch/core capability.
Prefer an upstream explicit differentiable training API; otherwise propose a
separate versioned, **default-off** opt-in in
`adapters/jittor_adapters/vllm/omni_training.py` with colocated
`adapters/jittor_adapters/vllm/tests/test_omni_training.py`.
Registration/bootstrap changes would be separately scoped. This report creates
none of these files and grants no training support.

The proposed implementation owns only the six private operation contracts
above. It must either compose existing public differentiable operations with
verified original rounding, or use an explicitly specified public autograd
wrapper around the real forward with independently validated derivatives.
Calling `forward_native` alone is insufficient: preserve BF16/FP32 rounding and
the original forward budget, save stable values for backward, validate repeated
indices and both output cotangents, and define alias/version behavior for the
underscore operation. Replacing it with an out-of-place function without an
explicit new API is not an equivalent repair. No fabricated backward for bare
`empty`/Triton, detached weights, framework-private state, compiler-state spoof,
CPU masquerade, forced MATH oracle or global no-grad workaround is admissible.

Before admitting that future opt-in:

1. Pin both full upstream SHAs, source-built artifact and installed payloads;
   use `_common.require_version` / `SUPPORTED_VERSIONS` and reject unknown
   versions/source layouts. Keep original inference imports and defaults inert.
2. Inherit relocatable/public-framework-entry contracts and negative tests;
   installation, double activation, rollback/uninstall, import order and partial
   failure must preserve ownership. Removing `adapters/` must leave core gates
   intact. Do not edit `torch`/Jittor version or global framework attributes.
3. Define an independently executable genuine-Torch training reference under
   the same **new** explicit opt-in. Its original upstream has no full gradient
   oracle, so new training evidence cannot be labelled original default AD.
   Validate each derivative in FP64/FP32 against independent public mathematics
   or finite differences before BF16 real-CUDA comparisons; presence alone is
   insufficient. Preserve backward input/output dtypes and in-place alias rules.
4. Attempt real CPU first, retain any upstream platform refusal, then real CUDA
   with independent interpreters, physical residency and zero candidate fallback.
   CPU component successes do not invent a complete-model CPU platform or a
   paired scorecard. Missing required libraries must be visible skips ordinarily
   and hard failures on the acceptance machine.
5. Keep the complete tiny architecture, mixed dtypes, original weights seed3,
   inputs7/19 and default attention. Compare all outputs at original scaled
   relative 0.002, every one of 36 parameter and four input gradients in both
   modes, with gradient budgets fixed before execution. The existing primitive
   0.01 contract is not permission to omit model gradients or substitute 0.02.
6. Only after full L2 succeeds, validate multi-step loss, every updated parameter,
   optimizer/scheduler state and fresh-process save/resume including all RNG
   streams for L3. Head-only updates do not qualify. Any new measured training
   protocol needs separate source/raw bindings and normal CPU/CUDA pairing;
   original blocked records remain unchanged.

## Evidence and independent recheck

Unversioned fresh evidence is under
`$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary/20261016/`.
`independent-component-audit-v1.json` checks all 32 component rows, 166 actual
component arrays and four additional tiny-QK controls / 40 arrays. It checks
finite values, exact saved gradient presence, real expected failures and
installed/source identity consistency; it grants no numerical training level.
`bounded-ad-edge-inventory-v1.json` binds exact upstream owner line ranges and
the previous full-model missing-name ledger.

`evidence-manifest-v1.json` SHA256
`e0bef985feb0f83bf2ed64d339e1b160f552256d224c7ff72049bada277e1e24`
binds 12,010 artifact paths / 1,198,351,359 bytes and all 2,912 Git mode/blob/content
rows of the unchanged measured tree. It includes the immutable prior full-model
seal/ledger references; those are explicitly historical, not fresh model runs.

A read-only `recheck-evidence-v1.py` and its result are under the sibling
`$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary-review/20261016/`. Run:

```bash
<NumPy-python> "$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary-review/20261016/recheck-evidence-v1.py" \
  "$JITTOR_LAB_ROOT/_state/omni-h3-ad-boundary/20261016/evidence-manifest-v1.json"
```

The rechecker SHA256 is
`9361714feeba3598fd567f7aaba85e3b9ac0cd9be16a70169423ece4149c8006`.
It independently passed all bytes, Git blobs and actual arrays, writes only
stdout, and imports neither runtime. Fourteen pure-host documentation contracts
pass; only integrator-owned index reachability registration is excluded. Layout
and diff checks pass; no numerical production owner changed in this report. The producer commands are retained for fresh,
exclusive-output reproductions, not for overwriting these results. The first
seal attempt rejected the repository's directory symlink as a regular file;
the preserved failure is corrected by hashing the symlink target text, without
changing any measured source or arrays. No new production calculation, compat
behavior or adapter implementation is part of this report.
