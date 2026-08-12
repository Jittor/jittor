"""Canonical functional multi-head attention."""

import jittor as jt


def _append_mask_column(mask):
    if mask is None:
        return None
    shape = list(mask.shape)
    shape[-1] = 1
    return jt.concat([mask, jt.zeros(shape, dtype=mask.dtype)], dim=-1)


def _validate_mask_dtype(mask, name):
    if mask is None:
        return
    dtype = str(mask.dtype)
    if dtype != "bool" and "float" not in dtype:
        raise AssertionError("only bool and floating types of {} are supported".format(name))


def _validate_mask(mask, target_length, source_length, batch_heads):
    if mask is None:
        return
    shape = tuple(int(size) for size in mask.shape)
    if len(shape) == 2:
        expected = (target_length, source_length)
    elif len(shape) == 3:
        expected = (batch_heads, target_length, source_length)
    else:
        raise RuntimeError("attention mask must be 2-D or 3-D")
    if shape != expected:
        raise RuntimeError("attention mask shape {} does not match {}".format(shape, expected))


def _validate_shape(tensor, expected, name):
    actual = tuple(int(size) for size in tensor.shape)
    if actual != expected:
        raise AssertionError("{} shape {} does not match {}".format(name, actual, expected))


def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True,
    is_causal=False,
):
    """Torch-compatible functional multi-head attention."""
    dropout_probability = float(dropout_p or 0.0)
    if dropout_probability < 0.0 or dropout_probability > 1.0:
        raise ValueError("dropout probability must be between 0 and 1")
    if query.ndim not in (2, 3):
        raise AssertionError("query must be a 2-D or 3-D tensor")
    is_batched = query.ndim == 3
    if key.ndim != query.ndim or value.ndim != query.ndim:
        raise AssertionError("query, key and value must have matching ranks")
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    _validate_mask_dtype(attn_mask, "attn_mask")
    _validate_mask_dtype(key_padding_mask, "key_padding_mask")
    if is_causal and attn_mask is None:
        raise RuntimeError("Need attn_mask if specifying the is_causal hint")

    target_length, batch_size, embed_dim = (int(size) for size in query.shape)
    if embed_dim != int(embed_dim_to_check):
        raise AssertionError(
            "query embedding dimension {} does not match {}".format(embed_dim, embed_dim_to_check)
        )
    if embed_dim % int(num_heads) != 0:
        raise AssertionError(
            "embedding dimension {} is not divisible by {} heads".format(embed_dim, num_heads)
        )
    head_dim = embed_dim // int(num_heads)
    if static_k is None and int(key.shape[1]) != batch_size:
        raise AssertionError("query and key batch sizes must match")
    if static_v is None and int(value.shape[1]) != batch_size:
        raise AssertionError("query, key and value batch sizes must match")
    if in_proj_bias is not None:
        _validate_shape(in_proj_bias, (3 * embed_dim,), "input projection bias")

    if use_separate_proj_weight:
        if q_proj_weight is None or k_proj_weight is None or v_proj_weight is None:
            raise AssertionError("separate projection weights are required")
        if key.shape[:2] != value.shape[:2]:
            raise AssertionError("key and value sequence dimensions must match")
        _validate_shape(q_proj_weight, (embed_dim, embed_dim), "query projection weight")
        _validate_shape(
            k_proj_weight,
            (embed_dim, int(key.shape[-1])),
            "key projection weight",
        )
        _validate_shape(
            v_proj_weight,
            (embed_dim, int(value.shape[-1])),
            "value projection weight",
        )
        if in_proj_bias is None:
            bias_q = bias_key = bias_value = None
        else:
            bias_q = in_proj_bias[:embed_dim]
            bias_key = in_proj_bias[embed_dim : 2 * embed_dim]
            bias_value = in_proj_bias[2 * embed_dim :]
        q = jt.nn.linear(query, q_proj_weight, bias_q)
        k = jt.nn.linear(key, k_proj_weight, bias_key)
        v = jt.nn.linear(value, v_proj_weight, bias_value)
    else:
        if in_proj_weight is None:
            raise AssertionError("in_proj_weight is required")
        if key.shape != value.shape:
            raise AssertionError("key and value shapes must match")
        if int(key.shape[-1]) != embed_dim:
            raise AssertionError("key and value embedding dimensions must match query")
        _validate_shape(
            in_proj_weight,
            (3 * embed_dim, embed_dim),
            "packed projection weight",
        )
        bias_q = bias_key = bias_value = None
        if in_proj_bias is not None:
            bias_q = in_proj_bias[:embed_dim]
            bias_key = in_proj_bias[embed_dim : 2 * embed_dim]
            bias_value = in_proj_bias[2 * embed_dim :]
        q = jt.nn.linear(query, in_proj_weight[:embed_dim], bias_q)
        k = jt.nn.linear(key, in_proj_weight[embed_dim : 2 * embed_dim], bias_key)
        v = jt.nn.linear(value, in_proj_weight[2 * embed_dim :], bias_value)

    if (bias_k is None) != (bias_v is None):
        raise AssertionError("bias_k and bias_v must be provided together")
    if bias_k is not None:
        if static_k is not None or static_v is not None:
            raise AssertionError("static key/value cannot be combined with bias")
        k = jt.concat([k, bias_k.repeat(1, batch_size, 1)], dim=0)
        v = jt.concat([v, bias_v.repeat(1, batch_size, 1)], dim=0)
        attn_mask = _append_mask_column(attn_mask)
        key_padding_mask = _append_mask_column(key_padding_mask)

    q = q.reshape(target_length, batch_size * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.reshape(int(k.shape[0]), batch_size * num_heads, head_dim).transpose(0, 1)
    else:
        k = static_k
    if static_v is None:
        v = v.reshape(int(v.shape[0]), batch_size * num_heads, head_dim).transpose(0, 1)
    else:
        v = static_v
    if k.ndim != 3:
        raise AssertionError("static key must be a 3-D tensor")
    if v.ndim != 3:
        raise AssertionError("static value must be a 3-D tensor")
    if int(k.shape[0]) != batch_size * num_heads or int(k.shape[2]) != head_dim:
        raise AssertionError("static key has an incompatible shape")
    if int(v.shape[0]) != batch_size * num_heads or int(v.shape[2]) != head_dim:
        raise AssertionError("static value has an incompatible shape")
    if int(k.shape[1]) != int(v.shape[1]):
        raise AssertionError("static key and value source lengths must match")

    if add_zero_attn:
        zeros = jt.zeros((batch_size * num_heads, 1, head_dim), dtype=k.dtype)
        k = jt.concat([k, zeros], dim=1)
        v = jt.concat([v, zeros.cast(v.dtype)], dim=1)
        attn_mask = _append_mask_column(attn_mask)
        key_padding_mask = _append_mask_column(key_padding_mask)

    source_length = int(k.shape[1])
    _validate_mask(attn_mask, target_length, source_length, batch_size * num_heads)
    if key_padding_mask is not None:
        expected = (batch_size, source_length)
        actual = tuple(int(size) for size in key_padding_mask.shape)
        if actual != expected:
            raise AssertionError(
                "key padding mask shape {} does not match {}".format(actual, expected)
            )

    if not need_weights:
        q = q.reshape(batch_size, num_heads, target_length, head_dim)
        k = k.reshape(batch_size, num_heads, source_length, head_dim)
        v = v.reshape(batch_size, num_heads, source_length, head_dim)
        if is_causal and key_padding_mask is None:
            attention_bias = None
            causal_hint = True
        else:
            attention_bias = None
            causal_hint = False
            if attn_mask is not None:
                if str(attn_mask.dtype) == "bool":
                    zero = jt.zeros(attn_mask.shape, dtype=q.dtype)
                    attention_bias = jt.ternary(
                        attn_mask,
                        zero + float("-inf"),
                        zero,
                    )
                else:
                    attention_bias = attn_mask
                if attention_bias.ndim == 2:
                    attention_bias = attention_bias.reshape(1, 1, target_length, source_length)
                else:
                    attention_bias = attention_bias.reshape(
                        batch_size,
                        num_heads,
                        target_length,
                        source_length,
                    )
            if key_padding_mask is not None:
                if str(key_padding_mask.dtype) == "bool":
                    zero = jt.zeros(key_padding_mask.shape, dtype=q.dtype)
                    padding_bias = jt.ternary(
                        key_padding_mask,
                        zero + float("-inf"),
                        zero,
                    )
                else:
                    padding_bias = key_padding_mask
                padding_bias = padding_bias.reshape(batch_size, 1, 1, source_length)
                attention_bias = (
                    padding_bias if attention_bias is None else attention_bias + padding_bias
                )
        output = jt.nn.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=dropout_probability if training else 0.0,
            is_causal=causal_hint,
        )
        output = output.permute(2, 0, 1, 3).reshape(target_length, batch_size, embed_dim)
        output = jt.nn.linear(output, out_proj_weight, out_proj_bias)
        if not is_batched:
            output = output.squeeze(1)
        return output, None

    q = q * (float(head_dim) ** -0.5)
    scores = jt.nn.matmul(q, k.transpose(1, 2))
    negative = jt.array(float("-inf")).cast(scores.dtype)
    if attn_mask is not None:
        if str(attn_mask.dtype) == "bool":
            scores = jt.ternary(
                attn_mask,
                negative.broadcast(scores.shape),
                scores,
            )
        else:
            scores = scores + attn_mask
    if key_padding_mask is not None:
        scores = scores.reshape(batch_size, num_heads, target_length, source_length)
        padding = key_padding_mask.reshape(batch_size, 1, 1, source_length).broadcast(
            [batch_size, num_heads, target_length, source_length]
        )
        if str(key_padding_mask.dtype) == "bool":
            scores = jt.ternary(
                padding,
                negative.broadcast(scores.shape),
                scores,
            )
        else:
            scores = scores + padding
        scores = scores.reshape(batch_size * num_heads, target_length, source_length)

    weights = jt.nn.softmax(scores, dim=-1)
    if dropout_probability > 0.0 and training:
        weights = jt.nn.dropout(weights, p=dropout_probability, is_train=True)
    if str(weights.dtype) != str(v.dtype):
        weights = weights.cast(v.dtype)
    output = jt.nn.matmul(weights, v)
    output = output.transpose(0, 1).reshape(target_length, batch_size, embed_dim)
    output = jt.nn.linear(output, out_proj_weight, out_proj_bias)

    result_weights = None
    if need_weights:
        result_weights = weights.reshape(batch_size, num_heads, target_length, source_length)
        if average_attn_weights:
            result_weights = result_weights.mean(dim=1)
    if not is_batched:
        output = output.squeeze(1)
        if result_weights is not None:
            result_weights = result_weights.squeeze(0)
    return output, result_weights


__all__ = ["multi_head_attention_forward"]
