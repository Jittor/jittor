# NPU Qwen2.5-3B batch-4 prefill layer probe — 2026-10-16

After hw00 SSH recovered, native vLLM-Ascend 0.24rc1 and Jittor vLLM core 0.24 were run sequentially on NPU6 with the same Qwen2.5-3B-Instruct BF16 TP1 fixed batch4, repeated 128-token prompt (SHA256 `642b80747d8ec7d15b291ef59e952d3c5d1aa5b2a3bbe49a3677c92bf3afd13b`), and 16 greedy output tokens. Both produced identical 16-token sequences `[374,13,362,15626,374,264,35972,1633,429,4586,4756,279,7286,315,24004,1561]` and shut down gracefully. These 16 tokens do not invalidate the earlier uninstrumented 128+64 failure at zero-based output index 47.

An opt-in `model.forward` call-0 layer probe recorded the final prompt position's 1×2048 BF16 embedding and hidden/residual after layers 0, 1, 2, 4, 8, 16, 35. The embedding was exactly equal in all 2048 elements. After layer 0, hidden differed (359/2048 exactly equal, MAE 0.00172182, max absolute difference 0.009765625), and residual differed (412/2048 equal, MAE 0.00137722). Hidden MAE at layers 1/2/4/8/16/35 was 0.00079159/0.00068778/0.00268138/0.00387848/0.00863066/0.03252311; at layer 35 hidden 281/2048 elements were equal. This localizes the first *observed* numerical difference to the first decoder layer or its input/forward boundary, not the captured embedding row or final argmax alone. Because probe readbacks can perturb Jittor lazy-graph lifetime, it does not prove that the uninstrumented path has identical per-layer values or that layer 0 alone causes the later token mismatch. The next test should split the layer-0 RMSNorm, QKV, attention and MLP outputs while checking that generated IDs stay the same as the uninstrumented run.

Logs: `_state/qwen25-b4-128x16-layer-{native6,jittor6}-20261016.log`. Comparison helper: `_state/compare_npu_layer_logs_20261016.py`. The existing 128+64 default-path correctness counterexample remains unresolved; no serving default changed.

## AddRMSNormBias diagnostic, later on 2026-10-16

The native vLLM-Ascend path uses its packaged `aclnnAddRmsNormBias` custom
operator. Jittor's existing decomposed and CANN `aclnnAddRmsNorm` paths differ
at BF16 precision. On deterministic 512×2048 BF16 inputs, Jittor's existing
fused path matched only 798,344/1,048,576 native-custom normalized elements.
An **opt-in diagnostic** Jittor ACL runner was added behind
`JITTOR_NPU_EXPERIMENTAL_CUSTOM_ADD_RMS_NORM=1`. It loads the native custom
`libcust_opapi.so` specified by `JITTOR_NPU_ADD_RMS_NORM_BIAS_LIBRARY`, and
requires that library's vendor directory in `ASCEND_CUSTOM_OPP_PATH` and its
compatible runtime `libstdc++` in `LD_LIBRARY_PATH`. The isolated output then
matched native exactly: 1,048,576/1,048,576 normalized elements and all
residual elements. This is a diagnostic dependency on vLLM-Ascend, **not a
shippable Jittor-only solution**.

An uninstrumented TP1 batch4 Qwen2.5-3B 128+64 run with only this custom
operator still differed on all four requests at zero-based output index 22:
native 5080 versus Jittor 7512. Log:
`_state/normcustom-b4-128x64-20261016.log`. Therefore matching this one
operator does not fix the complete-model correctness bug; it only isolates
one contributor. Serving defaults remain unchanged. Additional candidate
combinations and layer-by-layer localization are in progress.

Combining the custom AddRMSNormBias path with the opt-in fused linear and
batched FIA prefill paths made this same request *worse*: all four sequences
first differed at output index 12 (native 7286, Jittor 18940). Log:
`_state/normlinear-fia-b4-128x64-20261016.log`. Those flags must not be
enabled as a correctness workaround. The next probe targets the first bad
decode step, without enabling the fused-linear or batched-FIA candidates.

### Decode call 22 instrumentation warning

At 128+32, a broad probe reading embedding, layers 0/1/35 and seven
layer-0 submodules made all 32 native and Jittor tokens equal, although the
uninstrumented custom-AddRMSNormBias 128+64 run differed at output index 22.
A narrower Jittor probe that **skipped embedding** and read only layer 35 at
model-forward call 22 preserved the same mismatch at output index 22
(5080 versus 7512). Thus output length 32 by itself does not explain the
change. Early readbacks can perturb lazy execution and must not be treated
as passive measurements. Logs:
`_state/npu6-layercall22-{native,jittor-custom}-20261016.log` and
`_state/npu6-layer35call22-jittor-custom-20261016.log` (the latter used
`JT_NPU_PROBE_SKIP_EMBEDDING=1`).

A Jittor probe reading **only first-layer QKV at call 22**, also skipping
embedding, preserved the same four-way token mismatch at index 22. Its QKV
row matched native in just 1208/2560 BF16 elements (MAE 0.0019955), so a
numerical difference is already observable upstream of attention in the
failure-preserving path. This does not by itself prove QKV is the only cause:
the incoming hidden state and earlier-step cache can also differ. Log:
`_state/npu6-qkvcall22-jittor-custom-20261016.log`. The next isolated A/B
enables only the existing fused-Addmm linear candidate alongside the custom
AddRMSNormBias diagnostic, leaving prefill/attention candidates disabled.

## 2026-10-17: prefill exact, first decode attention differs

With the diagnostic custom AddRMSNormBias, experimental fused Addmm and
experimental batched FIA prefill enabled together, a call-0 probe of the same
TP1 batch4, 128-input/16-output Qwen2.5-3B case found *bitwise identical*
native/Jittor BF16 values in the last prompt row at embedding; layer-0 input
norm, QKV, RoPE, attention, o-projection, post norm and MLP; and layer
0/1/2/4/8/16/35 hidden and residual. Log:
`_state/npu6-prefill-custom-linear-fia-20261017.log`. The uninstrumented
combination still differs at output index 12, so the probe is diagnostic, not
proof of overall correctness.

The same configuration probed only at model-forward call 1 (first decode)
preserved the index-12 token mismatch: native 7286, Jittor 18940. At layer 0,
hidden matched only 417/2048 BF16 elements (MAE 0.0012394). Logs:
`_state/npu6-decode1-{native,jittor-combo}-20261017.log`.

A more detailed call-1 layer-0 probe found input RMSNorm 2048/2048 exact,
QKV 2560/2560 exact and both RoPE outputs exact (2048/2048 and 256/256).
The first observed difference was attention: 1780/2048 exact, MAE
0.000029081, maximum absolute error 0.0078125. O-projection and later
operations consequently differed. **This heavier probe changed the generated
16-token sequences to exact**, so the operator-level result must not be
treated as a non-perturbing localization of the uninstrumented failure.
Logs: `_state/npu6-decode1-layer0-{native,jittor}-20261017.log`.

Synthetic BF16 [4,2048] Addmm with a [2560,2048] weight matched native
10240/10240 exactly; ordinary Jittor `nn.linear` matched only 7437/10240.
This confirms the fused linear's isolated numerical behavior but not its
whole-model sufficiency. Forcing cache-write synchronization left the
custom-norm-plus-fused-linear index-22 failure unchanged; the experimental
sync change was reverted. A synthetic BF16 argmax tie chose the same first
index in native and Jittor, so no tie-breaking implementation bug was found.

Next correctness gate: without any layer probe, test the existing default-off
`JITTOR_NPU_PAGED_DECODE_FIA=1` path (CANN fused infer attention) together
with the above diagnostic combination for 128+64, then compare all token IDs
against the independently captured same-shape native oracle. Do not enable
these flags by default unless the uninstrumented long-output gate passes.

The attempted gate failed unchanged: all four 64-token outputs first differed
at index 12, native 7286 versus Jittor 18940, log
`_state/npu6-fia-decode-combo-20261017.log`. Inspection explains why the flag
did not help: `_paged_attention_decode_acl` refused batch 4 unless an older
separate experiment was enabled, and that experiment explicitly excluded
Qwen2.5's two KV heads. The batch-4 request therefore fell back to Jittor's
generic paged-cache gather/FP32 attention path. A debug run did not report
any FIA dispatch. A new *default-off* bounded diagnostic
`JITTOR_NPU_EXPERIMENTAL_LOOPED_FIA_DECODE=1` has now been added in
`backends/acl/kernels/kv_cache.py`: for batch 2–4 it invokes the existing
single-request ACL decode once per request, requiring
`JITTOR_NPU_PAGED_DECODE_FIA=1`. Its attempted uninstrumented 128+64 gate
also differed at output index 12 for all four sequences; log:
`_state/npu6-looped-fia-decode-combo-20261017.log`. Even with the FIA debug
flag, no FIA dispatch marker appeared. **Neither FIA-flag run actually enabled
FIA**: the reused `_state/hw_batched_fa_ab_20261016.sh` launcher silently
overwrites `JITTOR_NPU_PAGED_DECODE_FIA=0` after inheriting environment
variables. These two runs cannot judge FIA's numerical correctness. A direct
launcher preserving `JITTOR_NPU_PAGED_DECODE_FIA=1` is now under test. The
candidate remains default-off and is **not** a verified fix.

The corrected direct-launch 128+64 test then confirmed actual C++
`paged decode FIA dispatch=1` with per-request query `[1,16,1,128]` and
cache `[113,2,128,2,128]`. It moved the first mismatch from output index 12
to index 29 on each of four requests, but still failed: native 32676, Jittor
6961. Log: `_state/npu6-looped-fia-correct-flags-20261017.log`. This is
evidence that the ACL/FIA path changes numerics in the right direction, not
that it has fixed correctness. Native vLLM-Ascend calls CANN FIA with causal
`sparse_mode=3`; Jittor's opt-in single-request FIA used `0`. A separate
default-off `JITTOR_NPU_PAGED_DECODE_FIA_SPARSE_MODE3=1` experiment is now
being tested against the same 128+64 oracle. The initial version failed at
CANN workspace-size query with `When attnMask is not provided, sparseMode
must be 0` (log `_state/npu6-looped-fia-sparse3-20261017.log`). This is a
missing argument, not evidence about generated-token parity. Native
vLLM-Ascend creates a fixed 2048×2048 int8 upper-triangular mask and passes
it to causal FIA. Jittor's default-off sparse-mode-3 experiment was then
updated to pass the same mask shape and values.

The corrected mask-bearing sparse-mode-3 run completed and exited normally,
but all four outputs still first differed at index 29 (native 32676 versus
Jittor 6961), exactly the same point as sparse mode 0. Log:
`_state/npu6-looped-fia-sparse3-mask-fixed-20261017.log`. The preceding
mask-bearing attempt failed before generation on the C++ wrapper's old
three-input guard; that guard was temporarily relaxed for this A/B. Sparse
mode 3 plus native-style mask did **not** solve this case, so this failed
candidate should not become a serving default. The remaining discrepancy
requires inspecting cache content/layout or later decode-stage numerical
operations with non-perturbing probes.

The failed sparse-mode-3/mask candidate was removed from the serving code;
the default-off looped-FIA batch route remains available for diagnosis.
Adding only `JITTOR_VLLM_NPU_KEEP_PREFILL_ROOT_NORM=1` to that looped-FIA
configuration left the same four-way index-29 mismatch unchanged, despite
confirmed FIA dispatch and graceful shutdown. Log:
`_state/npu6-looped-fia-rootnorm-20261017.log`. The next isolated test toggles
attention-output rebind, not both lifetime flags together.

For context, the independent *default-path* Jittor 128+64 run first differed
at index 47 (the 48th generated token), so the looped-FIA diagnostic's
index-29 mismatch is **not an improvement over the serving default**. The
index-12 to index-29 comparison only holds within the separate custom-norm,
fused-linear, batched-FIA-prefill diagnostic combination.

The isolated `JITTOR_NPU_ATTENTION_OUTPUT_REBIND=1` run also completed with
confirmed FIA dispatch and graceful shutdown, but all four outputs still
first differed at index 29 (native 32676 versus Jittor 6961). Log:
`_state/npu6-looped-fia-rebind-20261017.log`. Neither lifecycle toggle is a
sufficient fix for this TP1 batch4 128+64 case. The next diagnostic should
compare cached K/V after prefill and at the first divergent decode step,
using probes that preserve the uninstrumented token sequence.
