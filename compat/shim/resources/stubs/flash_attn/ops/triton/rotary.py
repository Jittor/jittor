"""Rotary application helper matching flash_attn.ops.triton.rotary.apply_rotary."""
from __future__ import annotations

import torch


def apply_rotary(x, cos, sin, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None,
                 interleaved=False, inplace=False, conjugate=False):
    """Apply rotary position embeddings using public torch tensor operations."""
    rotary_dim = int(cos.shape[-1]) * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    offset = int(seqlen_offsets) if isinstance(seqlen_offsets, int) else 0
    if cu_seqlens is not None:
        # Varlen unpadded layout: x has shape (total_nnz, nheads, headdim)
        cu = cu_seqlens.to(device=x.device)
        seqlens = cu[1:] - cu[:-1]
        total_nnz = int(x.shape[0])
        starts = torch.repeat_interleave(cu[:-1], seqlens)
        pos = torch.arange(total_nnz, device=x.device, dtype=cu.dtype) - starts
        if offset:
            pos = pos + offset
        cos_slice = cos.to(device=x.device, dtype=x.dtype)[pos].unsqueeze(1)
        sin_slice = sin.to(device=x.device, dtype=x.dtype)[pos].unsqueeze(1)
    else:
        seq_dim = -3 if x.ndim >= 4 else -2
        seqlen = int(x.shape[seq_dim])
        cos_slice = cos[offset:offset + seqlen].to(device=x.device, dtype=x.dtype)
        sin_slice = sin[offset:offset + seqlen].to(device=x.device, dtype=x.dtype)
        if x.ndim == 4:
            cos_slice = cos_slice.unsqueeze(0).unsqueeze(2)
            sin_slice = sin_slice.unsqueeze(0).unsqueeze(2)
        elif x.ndim == 3:
            cos_slice = cos_slice.unsqueeze(1)
            sin_slice = sin_slice.unsqueeze(1)
    if conjugate:
        sin_slice = -sin_slice
    if interleaved:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        o1 = x1 * cos_slice - x2 * sin_slice
        o2 = x1 * sin_slice + x2 * cos_slice
        out_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
    else:
        half = rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        out_rot = torch.cat((x1 * cos_slice - x2 * sin_slice, x1 * sin_slice + x2 * cos_slice), dim=-1)
    if inplace:
        x[..., :rotary_dim] = out_rot
        return x
    return torch.cat((out_rot, x_pass), dim=-1) if x_pass.shape[-1] > 0 else out_rot


__all__ = ["apply_rotary"]
