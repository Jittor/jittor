"""Device-preserving FlashAttention padding utilities using public Torch ops.

These helpers pack tensors; they do not implement or claim a fused attention
kernel. Autograd is provided by the underlying gather/scatter operations.
"""

import torch

__all__ = ["index_first_axis", "pad_input", "unpad_input"]


def index_first_axis(input, indices):
    """Select rows of a ``(rows, ...)`` tensor, preserving trailing axes."""
    if input.ndim < 2 or indices.ndim != 1:
        raise ValueError("index_first_axis requires input.ndim >= 2 and 1-D indices")
    return torch.index_select(input, 0, indices)


def pad_input(hidden_states, indices, batch, seqlen):
    """Restore packed tokens at unique flattened mask indices; pad with zeros."""
    if hidden_states.ndim < 2 or indices.ndim != 1:
        raise ValueError("pad_input requires hidden_states.ndim >= 2 and 1-D indices")
    if hidden_states.shape[0] != indices.shape[0]:
        raise ValueError("pad_input indices and packed rows must have equal length")
    if batch < 0 or seqlen < 0:
        raise ValueError("pad_input batch and seqlen must be nonnegative")
    tail = tuple(hidden_states.shape[1:])
    output = torch.zeros((batch * seqlen,) + tail,
                         dtype=hidden_states.dtype, device=hidden_states.device)
    expanded_indices = indices.reshape((indices.shape[0],) + (1,) * len(tail)).expand_as(hidden_states)
    output = output.scatter(0, expanded_indices, hidden_states)
    return output.reshape((batch, seqlen) + tail)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Return packed tokens, indices, int32 offsets, max length, used lengths.

    ``attention_mask`` marks used tokens; optional ``unused_mask`` additionally
    allocates slots in the packed output. The final return value counts only
    used tokens, matching the upstream five-result padding interface.
    """
    if hidden_states.ndim < 3 or attention_mask.ndim != 2:
        raise ValueError("unpad_input requires (batch, seqlen, ...) and a 2-D mask")
    if tuple(hidden_states.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("attention_mask must match the first two hidden-state axes")
    if unused_mask is not None and tuple(unused_mask.shape) != tuple(attention_mask.shape):
        raise ValueError("unused_mask must match attention_mask")
    all_masks = attention_mask if unused_mask is None else attention_mask + unused_mask
    lengths = all_masks.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_length = int(lengths.max().item())
    offsets = torch.cat((torch.zeros((1,), dtype=torch.int32, device=lengths.device),
                         torch.cumsum(lengths, dim=0, dtype=torch.int32)), dim=0)
    flat = hidden_states.reshape((hidden_states.shape[0] * hidden_states.shape[1],)
                                 + tuple(hidden_states.shape[2:]))
    return index_first_axis(flat, indices), indices, offsets, max_length, used_lengths
