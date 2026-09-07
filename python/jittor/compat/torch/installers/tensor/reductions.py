"""Torch tensor reductions ownership."""

def corrcoef(x, *args, **kwargs):
    """Return a CPU NumPy correlation matrix as a Jittor tensor."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    result = _owner.np.corrcoef(x.float32().numpy())
    return _owner.jt.array(_owner.np.ascontiguousarray(result))


def broadcast_shapes(*shapes):
    """Return Torch-compatible broadcasted shape metadata."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    norm = [
        (int(shape),) if isinstance(shape, (int, _owner.np.integer))
        else tuple(int(dim) for dim in shape)
        for shape in shapes
    ]
    return _owner._TorchSize(_owner.np.broadcast_shapes(*norm)) if norm else _owner._TorchSize(())


def amax(input, dim=None, keepdim=False, keepdims=None, axis=None):
    """Return a values-only maximum reduction through the native owner."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._NATIVE_AMAX(
        input, dim if axis is None else axis,
        keepdim=keepdim, keepdims=keepdims)


def amin(input, dim=None, keepdim=False, keepdims=None, axis=None):
    """Return a values-only minimum reduction through the native owner."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._NATIVE_AMIN(
        input, dim if axis is None else axis,
        keepdim=keepdim, keepdims=keepdims)


def count_nonzero(input, dim=None):
    """Count non-zero entries through the native owner."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._NATIVE_COUNT_NONZERO(input, dim)


def _reduce_index(result):
    """Return only the index half of a Jittor arg-reduction, as int64."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if isinstance(result, (tuple, list)):
        result = result[0]
    return result.int64()


def argmax(input, dim=None, keepdim=False, keepdims=None, axis=None):
    """Return the indices of the maxima, as Torch's ``argmax`` does."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if axis is not None:
        dim = axis
    if keepdims is not None:
        keepdim = keepdims
    if dim is None:
        return _owner._reduce_index(_owner._NATIVE_ARGMAX(input.reshape(-1), 0))
    try:
        result = _owner._NATIVE_ARGMAX(input, dim, keepdims=keepdim)
    except TypeError:
        result = _owner._NATIVE_ARGMAX(input, dim, keepdim=keepdim)
    return _owner._reduce_index(result)


def argmin(input, dim=None, keepdim=False, keepdims=None, axis=None):
    """Return the indices of the minima, as Torch's ``argmin`` does."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if axis is not None:
        dim = axis
    if keepdims is not None:
        keepdim = keepdims
    if dim is None:
        return _owner._reduce_index(_owner._NATIVE_ARGMIN(input.reshape(-1), 0))
    try:
        result = _owner._NATIVE_ARGMIN(input, dim, keepdims=keepdim)
    except TypeError:
        result = _owner._NATIVE_ARGMIN(input, dim, keepdim=keepdim)
    return _owner._reduce_index(result)


def _maxmin(which, x, *args, **kwargs):
    """Shared body of Torch's ``max``/``min``, which have three shapes."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    # jittor-internal callers use the `keepdims` kwarg (with an 's') and
    # expect values-only semantics; delegate straight to the native op so
    # we don't break jittor's own softmax/layernorm/etc.
    if "keepdims" in kwargs:
        native = _owner._NATIVE_MAX if which == "max" else _owner._NATIVE_MIN
        return native(x, *args, **kwargs)
    if "axis" in kwargs and "dim" not in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    dim = kwargs.get("dim", None)
    keepdim = kwargs.get("keepdim", False)
    other = kwargs.get("other", None)
    pos = list(args)
    if pos:
        if isinstance(pos[0], _owner.jt.Var):
            other = pos[0]
        else:
            dim = pos[0]
            if len(pos) > 1:
                keepdim = pos[1]
    if other is not None:
        if which == "max":
            return _owner._NATIVE_MAXIMUM(x, other)
        return _owner._NATIVE_MINIMUM(x, other)
    if dim is None:
        # native scalar reduction via the captured METHOD (0-dim scalar);
        # NOT x.max(), which now routes back into this wrapper (recursion).
        if which == "max":
            return _owner._NATIVE_MAX_METHOD(x)
        return _owner._NATIVE_MIN_METHOD(x)
    index_of = _owner.argmax if which == "max" else _owner.argmin
    idx = index_of(x, dim=dim, keepdim=keepdim)
    if getattr(_owner.jt.compiler, "has_acl", 0):
        native = _owner._NATIVE_MAX if which == "max" else _owner._NATIVE_MIN
        val = native(x, dim, keepdims=keepdim)
    elif keepdim:
        val = _owner._NATIVE_GATHER(x, dim, idx)
    elif x.ndim == 1:
        val = x[idx]
    else:
        val = _owner._NATIVE_GATHER(x, dim, idx.unsqueeze(dim)).squeeze(dim)
    return _owner._MinMax(val, idx.int64())


def max(input, *args, **kwargs):
    """Return Torch's ``max``: a scalar, a ``(values, indices)`` pair, or a
    pairwise maximum, depending on how it was called."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._maxmin("max", input, *args, **kwargs)


def min(input, *args, **kwargs):
    """Return Torch's ``min``, in the same three shapes as ``max``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner._maxmin("min", input, *args, **kwargs)


def _correction_to_unbiased(unbiased, correction):
    """Collapse Torch's legacy ``unbiased=`` and modern ``correction=``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if correction is not None:
        return correction != 0
    if unbiased is not None:
        return bool(unbiased)
    return True                       # torch default


def _multidim_var(input, dims, unbiased, keepdim):
    """Variance over a LIST/TUPLE of axes, which the native op cannot do.

    jittor's native ``var()`` ``dim=`` slot is scalar-only (a list crashes with
    ``is_type<int64>(oi)``), and its separate ``dims=`` path returns a
    wrong-shaped/valued result for partial multi-axis reductions. Compute
    directly from mean/sum (which DO accept a tuple) so every axis subset
    matches torch exactly, preserving unbiased (Bessel) + keepdim semantics.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    dims = [int(d) % input.ndim for d in dims]
    mean = _owner.jt.mean(input, dims, keepdims=True)
    out = _owner.jt.sum((input - mean) ** 2, dims=dims, keepdims=keepdim)
    count = 1
    for d in dims:
        count *= input.shape[d]
    if unbiased:
        count = count - 1
    return out / count


def var(input, dim=None, unbiased=None, keepdim=False, keepdims=None,
        correction=None, axis=None, **kwargs):
    """Return Torch's variance, unbiased unless told otherwise."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    if axis is not None:
        dim = axis
    biased = _owner._correction_to_unbiased(unbiased, correction)
    kept = bool(keepdim) or bool(keepdims)
    if isinstance(dim, (list, tuple)):
        return _owner._multidim_var(input, dim, biased, kept)
    return _owner._NATIVE_VAR_METHOD(input, dim=dim, unbiased=biased, keepdims=kept)


def std(input, dim=None, unbiased=None, keepdim=False, keepdims=None,
        correction=None, axis=None, **kwargs):
    """Return Torch's standard deviation, derived from ``var``.

    jittor's native std is hardcoded unbiased AND floors at
    ``maximum(1e-6)``, which torch does not, so take the square root here.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return _owner.var(input, dim=dim, unbiased=unbiased, keepdim=keepdim,
               keepdims=keepdims, correction=correction, axis=axis).sqrt()
