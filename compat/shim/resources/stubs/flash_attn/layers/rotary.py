"""Rotary embedding helpers matching flash_attn.layers.rotary on JTorch."""
from __future__ import annotations

import torch
from ..ops.triton.rotary import apply_rotary


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0,
                     cu_seqlens=None, max_seqlen=None):
    return apply_rotary(
        x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace,
    )


class RotaryEmbedding(torch.nn.Module):
    """Minimal flash_attn.layers.rotary.RotaryEmbedding compatible module."""

    def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False,
                 scale_base=None, pos_idx_in_fp32: bool = True, device=None):
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.interleaved = bool(interleaved)
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = bool(pos_idx_in_fp32)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None):
        if (seqlen > self._seq_len_cached or self._cos_cached is None
                or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            self._cos_cached = torch.cos(freqs).to(dtype=dtype)
            self._sin_cached = torch.sin(freqs).to(dtype=dtype)

    def forward(self, qkv, seqlen_offset=0, max_seqlen=None, cu_seqlens=None):
        seqlen = int(max_seqlen) if max_seqlen is not None else int(qkv.shape[1])
        self._update_cos_sin_cache(seqlen + (int(seqlen_offset) if isinstance(seqlen_offset, int) else 0),
                                   device=qkv.device, dtype=qkv.dtype)
        return apply_rotary(
            qkv, self._cos_cached, self._sin_cached,
            seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen, interleaved=self.interleaved,
        )


__all__ = ["RotaryEmbedding", "apply_rotary_emb"]
