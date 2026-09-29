"""Padding helpers exposed by flash-attn's ``bert_padding`` module.

These helpers keep the public packing contract used by Transformers vision
encoders.  The fused CUDA kernels are optional; indexing and scatter are
delegated to the active Torch/Jittor tensor owner.
"""

import torch


def index_first_axis(input, indices):
    """Select flattened rows from ``input`` using a one-dimensional index."""
    return input[indices]


def unpad_input(hidden_states, attention_mask):
    """Remove masked tokens and return the flash-attn varlen metadata."""
    batch_size, seqlen = attention_mask.shape[:2]
    flat_mask = attention_mask.reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    unpadded = index_first_axis(hidden_states.reshape(batch_size * seqlen, *hidden_states.shape[2:]), indices)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return unpadded, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch_size, seqlen):
    """Scatter packed rows back into a dense ``[batch, seqlen, ...]`` tensor."""
    output = hidden_states.new_zeros((batch_size * seqlen,) + tuple(hidden_states.shape[1:]))
    output[indices] = hidden_states
    return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])


__all__ = ["index_first_axis", "unpad_input", "pad_input"]
