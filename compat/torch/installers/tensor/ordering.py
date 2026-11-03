"""Torch tensor ordering ownership."""

def sort(input, dim=-1, descending=False, **kwargs):
    """Return Torch's ``(values, indices)`` pair for a sort along ``dim``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    indices, values = _owner._NATIVE_ARGSORT(input, dim=dim, descending=descending)
    return _owner._Sort(values, indices.int64())


def argsort(input, dim=-1, descending=False, **kwargs):
    """Return only the sorting indices, as Torch's ``argsort`` does."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._NATIVE_ARGSORT(input, dim=dim, descending=descending)[0].int64()


def topk(input, k, dim=-1, largest=True, sorted=True):
    """Return the ``k`` largest (or smallest) values and their indices.

    Backend selection is shared with native topk. The portable fallback keeps
    the existing argsort/gather path because native ACL slicing is restricted.
    """
    from importlib import import_module as _import_module
    from jittor.ops.sorting import _select_topk_kernel
    _owner = _import_module(__package__)
    kernel = _select_topk_kernel(input, k, dim, largest, sorted)
    if kernel is not None:
        return _owner._TopK(*kernel(input, k, dim, largest, sorted))
    indices, _ = _owner._NATIVE_ARGSORT(input, dim=dim, descending=largest)
    rank = input.ndim
    axis = dim if dim >= 0 else dim + rank
    window = [slice(None)] * rank
    window[axis] = slice(0, k)
    indices = indices[tuple(window)]
    return _owner._TopK(_owner._NATIVE_GATHER(input, axis, indices), indices.int64())


def median(input, dim=None, keepdim=False):
    """Return Torch's lower median, with indices when ``dim`` is given."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if dim is None:
        return _owner._NATIVE_MEDIAN(input, keepdim=keepdim)
    axis = dim if dim >= 0 else dim + input.ndim
    if axis < 0 or axis >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{input.ndim}, {input.ndim - 1}], but got {dim})")
    indices, values = _owner._NATIVE_ARGSORT(input, dim=axis)
    lower = (input.shape[axis] - 1) // 2
    window = [slice(None)] * input.ndim
    window[axis] = slice(lower, lower + 1) if keepdim else lower
    window = tuple(window)
    return _owner._Median(values[window], indices[window].int64())
