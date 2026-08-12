"""Canonical scaled dot-product attention."""

import math

import jittor as jt


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
    query_dtype = str(query.dtype)
    if str(key.dtype) != query_dtype or str(value.dtype) != query_dtype:
        raise RuntimeError("query, key and value must have the same dtype")
    probability = float(dropout_p or 0.0)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("dropout probability must be between 0 and 1")
    if attn_mask is not None:
        mask_dtype = str(attn_mask.dtype)
        if mask_dtype != "bool" and "float" not in mask_dtype:
            raise AssertionError("only bool and floating attention masks are supported")
        allowed_mask_dtypes = {query_dtype}
        if query_dtype in {"bfloat16", "float16", "float64"}:
            allowed_mask_dtypes.add("float32")
        if mask_dtype != "bool" and mask_dtype not in allowed_mask_dtypes:
            raise RuntimeError("attention mask dtype must match query dtype or be float32")
    query_length = int(query.shape[-2])
    source_length = int(key.shape[-2])
    scale_factor = 1.0 / math.sqrt(int(query.shape[-1])) if scale is None else scale
    scores = jt.nn.matmul(query, key.transpose(-2, -1)) * scale_factor
    negative = jt.array(-1e30).cast(scores.dtype)
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
        valid_positions = jt.logical_not(causal)

    if attn_mask is not None:
        if str(attn_mask.dtype) == "bool":
            valid_positions = attn_mask if valid_positions is None else valid_positions & attn_mask
            scores = jt.ternary(
                attn_mask,
                scores,
                negative.broadcast(scores.shape),
            )
        else:
            negative_infinity = jt.isinf(attn_mask) & (attn_mask < 0)
            finite_positions = jt.logical_not(negative_infinity)
            valid_positions = (
                finite_positions if valid_positions is None else valid_positions & finite_positions
            )
            scores = scores + attn_mask

    row_valid = valid_positions.sum(-1, keepdims=True) > 0 if valid_positions is not None else None
    if row_valid is not None:
        scores = jt.ternary(row_valid, scores, jt.zeros_like(scores))
    weights = jt.nn.softmax(scores, dim=-1)
    if row_valid is not None:
        weights = jt.ternary(row_valid, weights, jt.zeros_like(weights))
    if probability > 0.0:
        weights = jt.nn.dropout(weights, p=probability, is_train=True)
    if str(weights.dtype) != str(value.dtype):
        weights = weights.cast(value.dtype)
    output = jt.nn.matmul(weights, value)
    return output if str(output.dtype) == query_dtype else output.cast(query_dtype)


__all__ = ["scaled_dot_product_attention"]
