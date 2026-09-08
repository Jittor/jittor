"""Dense rotary embedding compatible with ``flash_attn.ops.triton.rotary``."""

import torch


def apply_rotary(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    if cu_seqlens is not None or max_seqlen is not None:
        raise NotImplementedError("Jittor flash-attn rotary does not support packed sequences")
    if not isinstance(seqlen_offsets, int) or seqlen_offsets != 0:
        raise NotImplementedError("Jittor flash-attn rotary requires zero sequence offsets")
    rotary_dim = int(cos.shape[-1]) * 2
    if rotary_dim > int(x.shape[-1]):
        raise ValueError("rotary table is wider than the input head dimension")

    cos_value = cos
    sin_value = -sin if conjugate else sin
    while cos_value.ndim < x.ndim:
        axis = -2 if cos_value.ndim == x.ndim - 1 else 0
        cos_value = cos_value.unsqueeze(axis)
        sin_value = sin_value.unsqueeze(axis)
    cos_value = cos_value.to(x.dtype)
    sin_value = sin_value.to(x.dtype)

    rotated_input = x[..., :rotary_dim]
    if interleaved:
        first = rotated_input[..., ::2]
        second = rotated_input[..., 1::2]
        rotated = torch.stack(
            (first * cos_value - second * sin_value, second * cos_value + first * sin_value),
            dim=-1,
        ).flatten(-2)
    else:
        first, second = torch.chunk(rotated_input, 2, dim=-1)
        rotated = torch.cat(
            (first * cos_value - second * sin_value, second * cos_value + first * sin_value),
            dim=-1,
        )
    if rotary_dim < int(x.shape[-1]):
        rotated = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    if inplace:
        x.copy_(rotated)
        return x
    return rotated


__all__ = ["apply_rotary"]
