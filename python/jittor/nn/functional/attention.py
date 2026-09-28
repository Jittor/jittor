"""Canonical scaled dot-product attention."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import math

import jittor as jt
from jittor._core.flags import _output_requires_grad
from jittor._runtime.dispatch import select_kernel, try_dispatch


def _attention_bias(scores, attn_mask, is_causal):
    """``scores`` with the masks applied as ``-inf`` / additive terms."""
    if is_causal:
        query_length, source_length = int(scores.shape[-2]), int(scores.shape[-1])
        causal = jt.triu(jt.ones((query_length, source_length), dtype="bool"),
                         diagonal=1)
        scores = jt.ternary(causal, jt.array(float("-inf")).cast(scores.dtype)
                            .broadcast(scores.shape), scores)
    if attn_mask is not None:
        if _jittor_dtype_name(attn_mask.dtype) == "bool":
            scores = jt.ternary(attn_mask, scores,
                                jt.array(float("-inf")).cast(scores.dtype)
                                .broadcast(scores.shape))
        else:
            scores = scores + attn_mask.cast(scores.dtype)
    return scores


def _product(a, b, trans_a=False, trans_b=False):
    """``a @ b`` reading either operand transposed in place.

    ``matmul`` folds a transpose into the kernel only when it is a view of a
    stored tensor; the probabilities and score gradients below are
    intermediates, and their ``.transpose(-2, -1)`` became a full copy of an
    ``Lq x Lk`` matrix on every call -- 2.5 ms of a 21 ms attention step at
    4096 tokens. The batched kernel takes the flags directly, as matmul's own
    backward does.
    """
    if len(a.shape) == len(b.shape) >= 3 and a.shape[:-2] == b.shape[:-2]:
        kernel = select_kernel("batched_matmul", a, b, trans_a, trans_b)
        if kernel is not None:
            return kernel(a, b, trans_a, trans_b)
    return jt.nn.matmul(a.transpose(-2, -1) if trans_a else a,
                        b.transpose(-2, -1) if trans_b else b)


class _MemoryEfficientAttention(jt.Function):
    """Attention whose backward recomputes the probabilities.

    The composite below keeps the ``[..., Lq, Lk]`` softmax for its backward,
    and autodiff keeps the scores that produced it: one attention call at the
    SD1.5 UNet's 64x64 resolution (4096 tokens, 8 heads) retained 522 MB
    after its forward and peaked at 1.5 GB in its backward, where PyTorch's
    memory-efficient kernel retains 5 MB. Five such layers per forward is how
    an SD1.5 UNet training step ran out of a 24 GB card PyTorch trains it on
    in 17 GB. Here the forward keeps q, k, v and the output; the backward
    recomputes the probabilities from q and k, and holds the ``Lq x Lk``
    tensors only while that one layer's gradient is computed.

    The softmax is built from max/exp/sum ops rather than the backend's fused
    softmax kernel the composite uses. The kernel is faster in isolation, but
    it is opaque to fusion, so its input and output are both materialized;
    with it the SD1.5 UNet step peaked at 21.1 GB and ran out of the card
    again, against 20.2 GB and a 5% faster step without it.

    Rows every key is masked out of produce 0, as the composite's
    ``zero_all_neg_inf`` softmax does, and get zero gradient.
    """

    def execute(self, query, key, value, attn_mask, is_causal, scale):
        self.attn_mask, self.is_causal, self.scale = attn_mask, is_causal, scale
        out = _product(self._probabilities(query, key).cast(value.dtype), value)
        out = out.cast(query.dtype)
        self.saved = (query, key, value, out)
        return out

    def _probabilities(self, query, key):
        """Softmax over keys, in float32."""
        scores = _product(query * self.scale, key, trans_b=True)
        scores = _attention_bias(scores.float32(), self.attn_mask, self.is_causal)
        top = scores.max([-1], keepdims=True)
        top = jt.ternary(jt.isinf(top), jt.zeros_like(top), top)
        weights = (scores - top).exp()
        total = weights.sum([-1], keepdims=True)
        # A fully masked row is all zeros here; dividing by 1 keeps it so.
        return weights / jt.ternary(total > 0, total, jt.ones_like(total))

    def grad(self, grad_out):
        query, key, value, out = self.saved
        probabilities = self._probabilities(query, key)
        grad_out = grad_out.cast(value.dtype)
        grad_value = _product(probabilities.cast(value.dtype), grad_out, trans_a=True)
        grad_prob = _product(grad_out, value, trans_b=True).float32()
        row = (grad_out.float32() * out.float32()).sum([-1], keepdims=True)
        grad_scores = (probabilities * (grad_prob - row)).cast(query.dtype)
        grad_query = _product(grad_scores, key) * self.scale
        grad_key = _product(grad_scores, query * self.scale, trans_a=True)
        return (grad_query.cast(query.dtype), grad_key.cast(key.dtype),
                grad_value.cast(value.dtype), None, None, None)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """Compute scaled dot-product attention with Torch-compatible masks."""
    query_dtype = _jittor_dtype_name(query.dtype)
    if _jittor_dtype_name(key.dtype) != query_dtype or _jittor_dtype_name(value.dtype) != query_dtype:
        raise RuntimeError("query, key and value must have the same dtype")
    probability = float(dropout_p or 0.0)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("dropout probability must be between 0 and 1")
    if attn_mask is not None:
        mask_dtype = _jittor_dtype_name(attn_mask.dtype)
        if _jittor_dtype_name(mask_dtype) != "bool" and "float" not in _jittor_dtype_name(mask_dtype):
            raise AssertionError("only bool and floating attention masks are supported")
        allowed_mask_dtypes = {query_dtype}
        if _jittor_dtype_name(query_dtype) in {"bfloat16", "float16", "float64"}:
            allowed_mask_dtypes.add("float32")
        if _jittor_dtype_name(mask_dtype) != "bool" and _jittor_dtype_name(mask_dtype) not in allowed_mask_dtypes:
            raise RuntimeError("attention mask dtype must match query dtype or be float32")
    fast = try_dispatch(
        "nn.scaled_dot_product_attention", query, key, value,
        attn_mask=attn_mask, dropout_p=probability, is_causal=is_causal, scale=scale)
    if fast is not None:
        return fast
    # A fused kernel never writes the [..., Lq, Lk] scores to memory; see
    # backends/cuda/kernels/nn/cudnn_attention_cuda.py. It declines what it
    # cannot run, and everything below is the path for that.
    fused = try_dispatch(
        "nn.fused_attention", query, key, value,
        attn_mask=attn_mask, dropout_p=probability, is_causal=is_causal, scale=scale)
    if fused is not None:
        return fused
    query_length = int(query.shape[-2])
    scale_factor = 1.0 / math.sqrt(int(query.shape[-1])) if scale is None else scale
    if probability == 0.0 and _output_requires_grad(query, key, value):
        return _MemoryEfficientAttention.apply(
            query, key, value, attn_mask, bool(is_causal), float(scale_factor))
    block = _query_block(query, key, is_causal)
    if block is None:
        return _composite(query, key, value, attn_mask, probability, is_causal,
                          scale_factor, query_dtype)
    # Softmax is per query row, so blocks of rows are exact; each block's
    # scores are dead once its output is, and the executor runs the blocks in
    # the order they were built.
    outputs = []
    for start in range(0, query_length, block):
        stop = min(start + block, query_length)
        mask = attn_mask
        if mask is not None and len(mask.shape) >= 2 and int(mask.shape[-2]) == query_length:
            mask = mask[..., start:stop, :]
        outputs.append(_composite(query[..., start:stop, :], key, value, mask,
                                  probability, False, scale_factor, query_dtype))
    return jt.concat(outputs, dim=-2)


#: Score bytes, counted at 4 per element, above which attention without a
#: backward runs in blocks of queries. The composite builds the whole
#: ``[..., Lq, Lk]`` score matrix and its softmax: at the SD1.5 UNet's 64x64
#: resolution that is 1 GiB per call where PyTorch's fused kernel holds none,
#: and a 20-step sampling run peaked 0.9 GB above PyTorch's.
#:
#: Blocks are no slower on the device, but they cost the pipelined step:
#: at 64 MiB a Qwen3-0.6B prefill (28 calls, 4 blocks each) took 64.1 ms
#: instead of 51.5, and in a 20-step SD1.5 sampling loop 2 or 3 blocks per
#: call measured 1528-1534 ms against 1535 unsplit while 4 and 8 blocks took
#: 1649-1685 ms. Why four is the step was not isolated (auto-flush places its
#: cuts differently), so blocks are kept large: that prefill is not split,
#: and the SD1.5 call is split in three, peaking at 394 MiB instead of 1064.
_SCORE_CHUNK_BYTES = 384 << 20


def _query_block(query, key, is_causal):
    """Queries per block when the scores are too large to build at once, else None.

    Causal attention is not split: its mask is laid out for the whole query
    range.
    """
    if is_causal:
        return None
    rows = 1
    for size in query.shape[:-2]:
        rows *= int(size)
    per_query = rows * int(key.shape[-2]) * 4
    if per_query * int(query.shape[-2]) <= _SCORE_CHUNK_BYTES:
        return None
    return max(1, _SCORE_CHUNK_BYTES // per_query)


def _composite(query, key, value, attn_mask, probability, is_causal, scale_factor,
               query_dtype):
    """Attention built from matmuls and a softmax, for calls without a backward."""
    query_length = int(query.shape[-2])
    source_length = int(key.shape[-2])
    if scale_factor != 1.0:
        # Fold the softmax scale into Q rather than into the N x N scores.
        # `(q*s) @ k^T` equals `s * (q @ k^T)` exactly, but the multiply then
        # touches batch*heads*Lq*head_dim elements instead of
        # batch*heads*Lq*Lk. On the MiniMax-H3 video VAE's 32-head, 1797-token
        # decoder blocks that is 3.7M elements instead of 103M, and the scores
        # multiply was the largest single non-GEMM kernel in its profile. The
        # usual 1/sqrt(head_dim) is a power of two for the 64-wide heads here,
        # so the fold is exact in float16 as well.
        query = query * scale_factor
    scores = _product(query, key, trans_b=True)
    softmax_options = dict(log=False, zero_all_neg_inf=attn_mask is not None, dim=-1)
    fast_softmax = select_kernel("nn.softmax", scores, **softmax_options)
    zero_fully_masked = fast_softmax is not None and attn_mask is not None
    skip_row_valid = fast_softmax is not None and (
        is_causal or attn_mask is not None
    )
    negative = jt.array(float("-inf") if zero_fully_masked else -1e30).cast(
        scores.dtype
    )
    valid_positions = None

    if is_causal:
        causal = jt.triu(
            jt.ones((query_length, source_length), dtype="bool"),
            diagonal=1,
        )
        scores = jt.ternary(
            causal,
            negative.broadcast(scores.shape),
            scores,
        )
        if not skip_row_valid:
            valid_positions = jt.logical_not(causal)

    if attn_mask is not None:
        if _jittor_dtype_name(attn_mask.dtype) == "bool":
            if not skip_row_valid:
                valid_positions = (
                    attn_mask
                    if valid_positions is None
                    else valid_positions & attn_mask
                )
            scores = jt.ternary(
                attn_mask,
                scores,
                negative.broadcast(scores.shape),
            )
        else:
            if not skip_row_valid:
                negative_infinity = jt.isinf(attn_mask) & (attn_mask < 0)
                finite_positions = jt.logical_not(negative_infinity)
                valid_positions = (
                    finite_positions
                    if valid_positions is None
                    else valid_positions & finite_positions
                )
            scores = scores + attn_mask

    row_valid = (
        valid_positions.sum(-1, keepdims=True) > 0
        if valid_positions is not None
        else None
    )
    if row_valid is not None:
        scores = jt.ternary(row_valid, scores, jt.zeros_like(scores))
    weights = (
        fast_softmax(scores, **softmax_options)
        if fast_softmax is not None
        else jt.nn.softmax(scores, dim=-1)
    )
    if row_valid is not None:
        weights = jt.ternary(row_valid, weights, jt.zeros_like(weights))
    if probability > 0.0:
        weights = jt.nn.dropout(weights, p=probability, is_train=True)
    if _jittor_dtype_name(weights.dtype) != _jittor_dtype_name(value.dtype):
        weights = weights.cast(value.dtype)
    output = jt.nn.matmul(weights, value)
    return output if _jittor_dtype_name(output.dtype) == query_dtype else output.cast(query_dtype)


__all__ = ["scaled_dot_product_attention"]
