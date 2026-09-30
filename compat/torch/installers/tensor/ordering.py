"""Torch tensor ordering ownership."""

def sort(input, dim=-1, descending=False, **kwargs):
    """Return Torch's ``(values, indices)`` pair for a sort along ``dim``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    from jittor._runtime.dispatch import dispatch_context
    if dispatch_context(input).backend == "acl":
        from jittor.backends.acl.kernels.ops.sort_op import sort_acl
        values, indices = sort_acl(
            input, dim=dim, descending=descending,
            stable=kwargs.get("stable", False))
        return _owner._Sort(values, indices)
    indices, values = _owner._NATIVE_ARGSORT(input, dim=dim, descending=descending)
    return _owner._Sort(values, indices.int64())


def argsort(input, dim=-1, descending=False, **kwargs):
    """Return only the sorting indices, as Torch's ``argsort`` does."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    from jittor._runtime.dispatch import dispatch_context
    if dispatch_context(input).backend == "acl":
        from jittor.backends.acl.kernels.ops.sort_op import sort_acl
        return sort_acl(
            input, dim=dim, descending=descending,
            stable=kwargs.get("stable", False))[1]
    return _owner._NATIVE_ARGSORT(input, dim=dim, descending=descending)[0].int64()


def _acl_small_topk(input, k, axis, largest):
    """Select a small number of extrema on ACL without materializing argsort."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    jt = _owner.jt
    chosen = jt.zeros(input.shape, dtype="float32")
    values, indices = [], []
    sentinel = float("-inf") if largest else float("inf")
    reducer = _owner._NATIVE_ARGMAX if largest else _owner._NATIVE_ARGMIN
    for _ in range(k):
        candidates = jt.where(chosen > 0, sentinel, input)
        index, extremum = reducer(candidates, dim=axis, keepdims=True)
        # When all remaining values equal the sentinel, selecting the old
        # index again would duplicate it. Pick the first unused index.
        unused_index, _ = _owner._NATIVE_ARGMAX(1 - chosen, dim=axis, keepdims=True)
        index = jt.where(extremum == sentinel, unused_index, index)
        values.append(_owner._NATIVE_GATHER(input, axis, index))
        indices.append(index.int64())
        chosen = jt.scatter(chosen, axis, index,
                            jt.ones(index.shape, dtype="float32"))
    if k == 1:
        return _owner._TopK(values[0], indices[0])
    return _owner._TopK(jt.concat(values, dim=axis),
                        jt.concat(indices, dim=axis))


def topk(input, k, dim=-1, largest=True, sorted=True):
    """Return the ``k`` largest (or smallest) values and their indices.

    On ACL, small FP32 selections compose device reductions and gather.
    Other backends retain the existing argsort implementation.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    rank = input.ndim
    axis = dim if dim >= 0 else dim + rank
    from jittor._runtime.dispatch import dispatch_context
    from jittor._core.dtypes import dtype_name
    if (dispatch_context(input).backend == "acl"
            and dtype_name(input.dtype) == "float32"
            and 0 < k <= min(16, input.shape[axis])):
        return _acl_small_topk(input, k, axis, largest)
    if dispatch_context(input).backend == "acl":
        from jittor.backends.acl.kernels.ops.sort_op import sort_acl
        values, indices = sort_acl(input, dim=axis, descending=largest)
        window = [slice(None)] * rank
        window[axis] = slice(0, k)
        return _owner._TopK(values[tuple(window)], indices[tuple(window)])
    indices, _ = _owner._NATIVE_ARGSORT(input, dim=dim, descending=largest)
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
