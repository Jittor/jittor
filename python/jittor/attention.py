"""Compatibility facade for the canonical :mod:`jittor.nn` attention API."""

from .nn import (
    MultiheadAttention,
    baddbmm,
    multi_head_attention_forward,
    pad,
    scaled_dot_product_attention,
)


__all__ = [
    "MultiheadAttention",
    "baddbmm",
    "multi_head_attention_forward",
    "pad",
    "scaled_dot_product_attention",
]
