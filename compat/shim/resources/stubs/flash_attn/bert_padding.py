"""Padding helpers used by Transformers' FlashAttention integration.

These functions operate on the public torch API, which is provided by JTorch
when this stub is deployed.  They keep the same tensor layouts as flash-attn:
``(batch, sequence, ...)`` for padded tensors and ``(tokens, ...)`` for packed
tensors.
"""
from __future__ import annotations

import torch


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Flatten valid positions and return their packed sequence metadata."""
    if hidden_states.ndim < 2 or attention_mask.ndim != 2:
        raise ValueError("hidden_states must be (batch, sequence, ...) and attention_mask must be rank 2")
    if tuple(hidden_states.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("attention_mask must match the batch and sequence dimensions")
    if unused_mask is not None and tuple(unused_mask.shape) != tuple(attention_mask.shape):
        raise ValueError("unused_mask must match attention_mask")

    selected_mask = attention_mask if unused_mask is None else (attention_mask + unused_mask)
    lengths = selected_mask.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(selected_mask.reshape(-1), as_tuple=False).reshape(-1)
    cumulative = torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    packed = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return packed, indices, cumulative, max_length, used_lengths


def pad_input(hidden_states, indices, batch, seqlen):
    """Scatter packed token states back into a zero-padded batch."""
    batch, seqlen = int(batch), int(seqlen)
    if batch < 0 or seqlen < 0 or hidden_states.ndim < 1:
        raise ValueError("batch and seqlen must be non-negative and hidden_states must have a token dimension")
    if int(indices.numel()) != int(hidden_states.shape[0]):
        raise ValueError("indices length must equal the packed token count")
    if int(indices.numel()) and (int(indices.min().item()) < 0 or int(indices.max().item()) >= batch * seqlen):
        raise ValueError("indices are outside the padded batch")
    shape = (batch * seqlen, *hidden_states.shape[1:])
    output = torch.zeros(shape, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


__all__ = ["pad_input", "unpad_input"]
