# Training Qwen3-0.6B on Ascend 910B3: correctness and performance

- Status: FP32 forward/loss/backward accepted (about `1.07x-1.12x` native `torch_npu`);
  BF16 single-step forward accepted as element-wise aligned; **the BF16 exact-path
  performance gate and cross-framework training trajectories remain open**
- Date: 2026-08-30 (BF16 part through 2026-09-02)
- Baseline commit: FP32 and the slicing follow-ups are `d02a72ed` plus the consecutive
  slicing changes in this report; BF16 semantic alignment is `8ab4d2b5`
- Validation scope: one Ascend 910B3, driver 25.5.1 / CANN 9.0.0; Qwen3-0.6B (596,049,920
  parameters); Transformers 4.56.2 for FP32 and 5.5.3 for BF16; eager attention, sequence
  length 8, 2 warm-ups, 5 synchronised timed runs taking the per-stage median. Not covered:
  FP16, Qwen3-8B training, multiple cards, long-running trajectories, checkpoint restore,
  sampling and quantisation
- Maintainers: Torch compatibility layer and Ascend backend maintainers
- Review triggers: changes in CANN, the Transformers Qwen3 modules, the ACL lowering of
  embedding/RMSNorm/RoPE/slicing, dtype, the optimizer, or the timing protocol

## Conclusion

In FP32 the full forward, causal-LM loss and backward run on the NPU; both steady-state runs
give `cpu_compile_count=0` and `fallback_count=0`, loss `3.2688751221`, and the sampled
embedding, first-layer attention and last-layer MLP gradients are present and finite. When
it first ran it was `1.87x` native; the dominant cost was not GEMM but device fallbacks and
composite operators in the training graph: the label's constant pad went through reindex
onto a CPU path, RMSNorm/RoPE backward was not wired to CANN, and embedding backward used
the generic IndexPut on the `[151936, 1024]` weight. After wiring in a restricted ACL concat
and CANN RMSNorm/RoPE/embedding forward and backward:

| Implementation | Forward | Backward | Total | vs `torch_npu` |
| --- | ---: | ---: | ---: | ---: |
| Native `torch_npu` | 87.348 ms | 103.230 ms | 190.579 ms | 1.000x |
| Jittor, constant pad fixed only | 161.586 ms | 195.253 ms | 356.839 ms | 1.872x |
| Jittor, fully wired in (two runs) | 109.9–114.0 ms | 94.5–100.1 ms | 204.4–214.1 ms | 1.073x–1.123x |

The FP32 timings exclude `optimizer.step()`. Every new operator path was validated against
an independent NumPy reference for outputs and gradients: `aclnnRmsNormGrad`,
`aclnnRotaryPositionEmbeddingGrad` (q/k/cos/sin), `aclnnEmbeddingDenseBackward` (repeated
token accumulation, `padding_idx`).

## BF16: semantic alignment

The first three BF16 semantic differences have been fixed: operations between a Python
float scalar and a BF16 tensor keep BF16 per `torch.result_type`; RMSNorm keeps PyTorch's
order of "FP32 normalisation, BF16 rounding, BF16 affine"; BF16 SiLU now uses `aclnnSwish`
(the old `aclnnSilu` deviated on boundary values such as `5.9375`). After the fix, with the
same checkpoint and inputs, the 29 hidden states and the final logits are element-wise
identical to `torch_npu`, the initial loss is `3.377089977` versus `3.377089262`, and the
backward error accumulates layer by layer to a relative L2 of `0.022717` at
`hidden_grad_0`. **This accepts the single-step forward and finite gradients; it does not
mean the long-run training trajectory is aligned.**

With an explicit `fused=True`, AdamW goes through CANN `aclnnApplyAdamWV2`, and after two
steps the BF16 parameters, first and second moments are element-wise identical to
`torch.optim.AdamW(fused=True)`; the default `fused=None` keeps the foreach-style rounding
semantics, so a single-rounded result is not passed off as the default path.

## BF16: performance (still open)

Transformers 5.5.3, BF16 eager, explicit fused AdamW, run back to back on the same card:

| Path | Forward | Backward | AdamW | Full step | vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native `torch_npu` | 81.120 ms | 111.297 ms | 19.006 ms | 211.424 ms | 1.000x |
| Jittor `d02a72ed` | 120.805 ms | 106.521 ms | 25.275 ms | 252.601 ms | 1.195x |
| Jittor, with contiguous-slice backward (median of three processes) | 120.817 ms | 79.881 ms | 24.139 ms | 224.837 ms | 1.063x |

Two slicing changes: a full slice (`begin=0, end=shape, step=1`) returns a clone that
shares storage, and backward no longer does a memset plus `StridedSliceAssignV2`; for a
slice that is only a contiguous subrange of the last dimension, backward uses a single CANN
Cat to splice `dout` back together with the cached zero blocks into the original shape. All
112 `StridedSliceAssignV2` calls disappear from the profile, backward device time goes
`88.666 → 68.203 ms`, and the 61 hidden/logits/gradient arrays are bit-identical to the
already-accepted exact snapshot. The remaining gap is mainly in the forward pass and the
Python state shuffling around the optimizer (930 parameters).

Rejected or withdrawn attempts: using CANN RoPE directly reaches `0.988x`, but logits have a
relative L2 of `0.04270` and the worst hidden gradient is `0.23168`, changing the BF16
rounding trajectory, so it is not the default; doing half-slice backward with memset+2D
memcpy degrades the full step to `272.504 ms`; doing rotate-half with a non-contiguous ACL
tensor view triggers a vendor `libnnopbase` segmentation fault in the minimal case. All three
were fully withdrawn, with no fallback used to paper over them.

## Reproduction

The training probe is not in the repository (the in-repo
`tests/backends/acl/manual/run_qwen3_transformers.py` only does inference). Protocol: the
compatibility layer uses a fully deployed shim site rather than only setting environment
variables.

```bash
python -m jittor.compat.shim.deploy --target "$RUN_ROOT/shim-site"
export PYTHONPATH="$RUN_ROOT/shim-site:$JITTOR_SOURCE/python"
source "$CANN_SET_ENV"; export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
# Transformers 5.5.3 的异步加权重线程没有 ACL 线程上下文
export HF_DEACTIVATE_ASYNC_LOAD=1
JITTOR_TORCH_SHIM=1 python <training-probe> --backend jittor --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw --fused --warmups 2 --repeats 5
python <training-probe> --backend torch --model "$QWEN3_MODEL" \
  --sequence-length 8 --dtype bfloat16 --optimizer adamw --fused --warmups 2 --repeats 5
```

Both sides use the same checkpoint, token ids, dtype, attention implementation and
synchronisation boundaries; timing excludes loading. Transformers 4.56.2's Qwen3 inlines a
composite RoPE and does not call `jt.nn.rotary_emb`, so the FP32 fused RoPE path requires
the probe to explicitly replace `modeling_qwen3.apply_rotary_pos_emb`. The native and Torch
compatibility regression files must run in separate processes.

## Still open

- The BF16 exact-path full step is still about 6.3% slower than native (the performance
  gate has not passed);
- Cross-framework BF16 training trajectories, FP32 optimizer and checkpoint restore are not
  yet validated;
- The embedding fast path does not cover `scale_grad_by_freq=True`, `max_norm` or sparse;
- The numbers only represent the single-card, sequence-length-8 synchronised eager
  protocol.

Checked on the current tree (`e3c369acb`), these paths are still present (the AdamW,
embedding, norms and SiLU runners under `backends/acl/kernels/native/`, and the slicing
lowering in `getitem_op.py`), there has been no new NPU measurement since this report, the
known-issues ledger has no separate entry, and this report is the sole record of that
performance gate.
