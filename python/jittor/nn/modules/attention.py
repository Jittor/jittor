"""Stateful multi-head attention module."""

import jittor as jt


def _dtype_name(dtype):
    if dtype is None:
        return "float32"
    if isinstance(dtype, str):
        return dtype
    if callable(dtype):
        return dtype.__name__
    return str(dtype).replace("torch.", "") or "float32"


def _legacy_positional_dtype(device, dtype):
    if dtype is not None or device is None:
        return dtype
    name = str(device).replace("torch.", "")
    if callable(device) or name in {
        "bfloat16",
        "float16",
        "float32",
        "float64",
    }:
        return device
    return dtype


class MultiheadAttention(jt.Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        dtype = _legacy_positional_dtype(device, dtype)
        del device
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                "embed_dim and num_heads must be positive, got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        self.embed_dim = embed_dim
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        if self.head_dim * num_heads != embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")

        dtype_name = _dtype_name(dtype)
        if self._qkv_same_embed_dim:
            self.in_proj_weight = jt.empty((3 * embed_dim, embed_dim), dtype=dtype_name)
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            self.in_proj_weight = None
            self.q_proj_weight = jt.empty((embed_dim, embed_dim), dtype=dtype_name)
            self.k_proj_weight = jt.empty((embed_dim, self.kdim), dtype=dtype_name)
            self.v_proj_weight = jt.empty((embed_dim, self.vdim), dtype=dtype_name)
        self.in_proj_bias = jt.empty((3 * embed_dim,), dtype=dtype_name) if bias else None
        self.out_proj = jt.nn.Linear(embed_dim, embed_dim, bias=bias)
        if str(self.out_proj.weight.dtype) != dtype_name:
            self.out_proj.weight = self.out_proj.weight.cast(dtype_name)
            if self.out_proj.bias is not None:
                self.out_proj.bias = self.out_proj.bias.cast(dtype_name)
        self.bias_k = jt.empty((1, 1, embed_dim), dtype=dtype_name) if add_bias_kv else None
        self.bias_v = jt.empty((1, 1, embed_dim), dtype=dtype_name) if add_bias_kv else None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters(dtype_name)

    def _reset_parameters(self, dtype_name="float32"):
        if self._qkv_same_embed_dim:
            jt.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_weight = self.in_proj_weight.cast(dtype_name)
        else:
            jt.init.xavier_uniform_(self.q_proj_weight)
            jt.init.xavier_uniform_(self.k_proj_weight)
            jt.init.xavier_uniform_(self.v_proj_weight)
            self.q_proj_weight = self.q_proj_weight.cast(dtype_name)
            self.k_proj_weight = self.k_proj_weight.cast(dtype_name)
            self.v_proj_weight = self.v_proj_weight.cast(dtype_name)
        if self.in_proj_bias is not None:
            jt.init.constant_(self.in_proj_bias, 0.0)
            jt.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            jt.init.xavier_gauss_(self.bias_k)
            self.bias_k = self.bias_k.cast(dtype_name)
        if self.bias_v is not None:
            jt.init.xavier_gauss_(self.bias_v)
            self.bias_v = self.bias_v.cast(dtype_name)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        self.__dict__.update(state)

    def execute(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        is_batched = query.ndim == 3
        if self.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        output, weights = jt.nn.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.is_training(),
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


__all__ = ["MultiheadAttention"]
