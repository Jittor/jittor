"""Order-statistic reductions through backend sorting and indexing."""

from collections import namedtuple
from operator import index

import jittor as jt

from ...context import get_install_context


_Mode = namedtuple("mode", ("values", "indices"))


def mode(input, dim=-1, keepdim=False, *, out=None):
    torch = get_install_context(jt).target_namespace
    if input.is_complex():
        raise RuntimeError("mode does not support complex tensors")
    if out is not None:
        values, indices = out
        if torch.is_grad_enabled() and any(value.requires_grad for value in (input, values, indices)):
            raise RuntimeError("mode with an out argument does not support automatic differentiation")
        if values.dtype != input.dtype or indices.dtype != torch.int64:
            raise RuntimeError("mode out tensors must use the input dtype and int64 indices")
        if values.device != input.device or indices.device != input.device:
            raise RuntimeError("mode out tensors must be on the input's device")
    dim = index(dim)
    rank = max(input.ndim, 1)
    if not -rank <= dim < rank:
        raise IndexError(f"mode dimension {dim} is out of range for {input.ndim} dimensions")
    dim %= rank

    if input.ndim == 0:
        result = _Mode(input.clone(), torch.zeros((), dtype=torch.int64, device=input.device))
    else:
        size = input.shape[dim]
        if size == 0:
            raise IndexError("mode cannot reduce an empty dimension")
        if input.numel() == 0:
            shape = list(input.shape)
            if keepdim:
                shape[dim] = 1
            else:
                del shape[dim]
            result = _Mode(input.reshape(shape),
                           torch.empty(shape, dtype=torch.int64, device=input.device))
        else:
            result = _reduce_mode(torch, input, dim, keepdim, size)
    if out is None:
        return result
    for target, value in zip((values, indices), result):
        if tuple(target.shape) != tuple(value.shape):
            # Rebind the destination holder when its storage cannot fit
            # the result. A correctly shaped destination keeps its storage.
            target.data = value
        else:
            target.copy_(value)
    return _Mode(values, indices)


def _reduce_mode(torch, input, dim, keepdim, size):
    rows = torch.movedim(input, dim, -1).contiguous()
    ordered = torch.sort(rows.detach(), dim=-1).values.contiguous()
    # The difference of the two insertion positions is each value's
    # frequency. Both searches run on the source tensor's device.
    counts = torch.searchsorted(ordered, ordered, right=True) - torch.searchsorted(ordered, ordered)
    if input.is_floating_point():
        counts = torch.where(torch.isnan(ordered), 1, counts)
    positions = torch.arange(size, dtype=torch.int64, device=input.device)
    winners = counts == counts.amax(dim=-1, keepdim=True)
    prefer_last = input.device.type == "cuda" and size <= 2048
    if prefer_last:
        winner = torch.where(winners, positions, -1).amax(dim=-1, keepdim=True)
    else:
        winner = torch.where(winners, positions, size).amin(dim=-1, keepdim=True)
    selected = ordered.gather(-1, winner)
    matching = rows == selected
    if input.is_floating_point():
        matching = matching | (torch.isnan(rows) & torch.isnan(selected))
    if input.device.type == "cuda" and not prefer_last:
        indices = torch.where(matching, positions, size).amin(dim=-1, keepdim=True)
    else:
        indices = torch.where(matching, positions, -1).amax(dim=-1, keepdim=True)
    # Gather from the original input so the selected index also owns the
    # gradient. Sorting and counting operate on detached values.
    values = rows.gather(-1, indices)
    if keepdim:
        return _Mode(torch.movedim(values, -1, dim), torch.movedim(indices, -1, dim))
    return _Mode(values.squeeze(-1), indices.squeeze(-1))
