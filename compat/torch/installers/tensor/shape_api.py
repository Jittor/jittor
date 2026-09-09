"""Stable Torch shape and reduction adapters using native Tensor operations."""
from ...context import get_install_context
from . import jt, np, dtype, _jittor_dtype_name, _dtype_to_str, _diff, _trapz, nn

def _torch_size(self, dim=None):
    _context = get_install_context(jt)
    _native = _context.state["tensor_shape_api"]
    _Size = _native['_Size']
    return self.shape[dim] if dim is not None else _Size(self.shape)


def _dtype_itemsize_name(ds):
    d = dtype._registry.get(ds)
    if d is not None:
        return d.itemsize
    return dtype(ds).itemsize


def _bitcast(self, dt):
    import numpy as _np
    if True:
        _np_view_of = {"bool": _np.bool_, "uint8": _np.uint8, "int8": _np.int8, "uint16": _np.uint16,
                       "int16": _np.int16, "int32": _np.int32, "int64": _np.int64,
                       "float16": _np.float16, "bfloat16": _np.uint16,
                       "float32": _np.float32, "float64": _np.float64}
    ds = getattr(dt, "name", str(dt)).replace("torch.", "")
    itemsize = getattr(dt, "itemsize", None)
    itemsize = itemsize if isinstance(itemsize, int) else _dtype_itemsize_name(ds)
    old_itemsize = getattr(getattr(self, "dtype", None), "itemsize", None)
    if old_itemsize is None:
        old_itemsize = _dtype_itemsize_name(_jittor_dtype_name(self.dtype))
    shape = list(self.shape)
    if len(shape) == 0:
        if old_itemsize != itemsize:
            raise RuntimeError("view(dtype) cannot change itemsize on a scalar tensor")
    else:
        last_bytes = int(shape[-1]) * int(old_itemsize)
        if itemsize <= 0 or last_bytes % int(itemsize) != 0:
            raise RuntimeError("view(dtype) requires the last dimension to be byte-compatible")
        shape[-1] = last_bytes // int(itemsize)
    reinterpret_view = getattr(jt, "reinterpret_view", None)
    npd = _np_view_of.get(ds, _np.uint8)
    if reinterpret_view is not None and ds in _np_view_of:
        return reinterpret_view(self, shape, ds)
    return jt.array(_np.ascontiguousarray(self.numpy()).view(npd))


def _torch_reshape(self, *shape, **_kw):
    # torch's `.view(dtype)` / `.view(dtype=...)` REINTERPRETS the bytes as
    # another dtype (bitcast), e.g. weight.view(torch.uint8) for byte-packing
    # in vLLM weight transfer. jittor has no dtype-view; bitcast via numpy.
    # (NB: 'dtype' the kwarg must not shadow the `dtype` class used below.)
    _context = get_install_context(jt)
    _native = _context.state["tensor_shape_api"]
    _orig_reshape = _native['_orig_reshape']
    _dt = _kw.get("dtype", None)
    if _dt is not None:
        return _bitcast(self, _dt)
    if not shape:
        # torch spells the target shape as a keyword too: `reshape(shape=...)`
        # (diffusers' DiT unpatchify) and `view(size=...)`. Dropping it left
        # an empty positional tuple and a "shape can't be empty" core error.
        _named = _kw.get("shape", _kw.get("size", None))
        if _named is not None:
            shape = _named if isinstance(_named, (tuple, list)) else (_named,)
            shape = (tuple(int(s) for s in shape),)
    if len(shape) == 1 and isinstance(shape[0], dtype):
        return _bitcast(self, shape[0])
    if len(shape) == 1 and isinstance(shape[0], tuple) and type(shape[0]) is not tuple:
        shape = (tuple(int(s) for s in shape[0]),)
    return _orig_reshape(self, *shape)


def _norm_reduce_kw(a, k):
    d = None
    if "axis" in k:
        d = k.pop("axis")
    if "dim" in k:
        d = k.pop("dim")
    if "dims" in k:
        d = k.pop("dims")
    if d is None and len(a) >= 1:
        if isinstance(a[0], (tuple, list)):
            d = a[0]
            a = a[1:]  # consume positional tuple-of-dims
        elif isinstance(a[0], (int, np.integer)) and not isinstance(a[0], bool):
            d = a[0]
            a = a[1:]  # consume positional scalar dim
    # torch spells it keepdim; jittor's tuple overload spells it keepdims.
    keep = k.pop("keepdim", k.pop("keepdims", None))
    if keep is None and d is not None and len(a) >= 1 and isinstance(a[0], bool):
        keep = a[0]
        a = a[1:]  # consume positional keepdim
    if d is not None:
        # jittor's scalar `dim` overload rejects keepdims, while its tuple
        # `dims` overload supports it -> always route through `dims` when a
        # keepdim was requested (wrap a scalar dim into a 1-tuple).
        if isinstance(d, (tuple, list)):
            k["dims"] = tuple(int(x) for x in d)
        elif keep is not None:
            k["dims"] = (int(d),)
        else:
            k["dim"] = int(d)
    if keep is not None:
        k["keepdims"] = bool(keep)
    return a, k


def _looks_like_dtype(x):
    return isinstance(x, dtype) or (isinstance(x, str) and x.replace("torch.", "") in dtype._registry)


def _torch_var_sum(self, *a, **k):
    _context = get_install_context(jt)
    _native = _context.state["tensor_shape_api"]
    _orig_var_sum = _native['_orig_var_sum']
    out = k.pop("out", None)
    dt = k.pop("dtype", None)
    a, k = _norm_reduce_kw(a, k)
    if dt is None and len(a) >= 1 and _looks_like_dtype(a[0]):
        dt = a[0]
        a = a[1:]
    if dt is not None:
        self = self.cast(_dtype_to_str(dt))
    elif _jittor_dtype_name(self.dtype) in ("uint8", "int8", "uint16"):
        self = self.int32()
    result = _orig_var_sum(self, *a, **k)
    if out is not None:
        out.assign(result)
        return out
    return result


def _torch_sum(input, *a, **k):
    _context = get_install_context(jt)
    Var = _context.state["Var"]
    _native = _context.state["tensor_shape_api"]
    _orig_module_sum = _native['_orig_module_sum']
    if isinstance(input, Var):
        return _torch_var_sum(input, *a, **k)
    return _orig_module_sum(input, *a, **k)


def _index_add_inplace(self, dim, index, source, *, alpha=1):
    _context = get_install_context(jt)
    _native = _context.state["tensor_shape_api"]
    _orig_index_add_inplace = _native['_orig_index_add_inplace']
    if alpha != 1:
        source = source * alpha
    _orig_index_add_inplace(self, dim, index, source)
    return self



def _plain_reduce(name, self, *a, **k):
    orig = get_install_context(jt).state["tensor_shape_api"]["reductions"][name]
    a, k = _norm_reduce_kw(a, k)
    return orig(self, *a, **k)


def _anyall_reduce(name, self, *a, **k):
    orig = get_install_context(jt).state["tensor_shape_api"]["reductions"][name]
    d = None
    if "axis" in k:
        d = k.pop("axis")
    if "dim" in k:
        d = k.pop("dim")
    if "dims" in k:
        d = k.pop("dims")
    if d is None and len(a) >= 1 and isinstance(a[0], (tuple, list)):
        d = a[0]
        a = a[1:]
    keep = k.pop("keepdim", k.pop("keepdims", None))
    if d is None:
        return orig(self, *a, **k)
    dims = [int(x) for x in d] if isinstance(d, (tuple, list)) else [int(d)]
    ndim = self.ndim
    dims = sorted((x % ndim for x in dims), reverse=True)
    out = self
    for ax in dims:
        out = orig(out, dim=ax)
        if keep:
            out = out.unsqueeze(ax)
    return out


def _axis_reduce(name, self, *a, **k):
    orig = get_install_context(jt).state["tensor_shape_api"]["reductions"][name]
    if "axis" in k:
        k["dim"] = k.pop("axis")
    return orig(self, *a, **k)


def _method_mean(self, *a, **k):
    return _plain_reduce('mean', self, *a, **k)


def _method_prod(self, *a, **k):
    return _plain_reduce('prod', self, *a, **k)


def _method_any(self, *a, **k):
    return _anyall_reduce('any', self, *a, **k)


def _method_all(self, *a, **k):
    return _anyall_reduce('all', self, *a, **k)


def _method_max(self, *a, **k):
    return _axis_reduce('max', self, *a, **k)


def _method_min(self, *a, **k):
    return _axis_reduce('min', self, *a, **k)


def _method_argmax(self, *a, **k):
    return _axis_reduce('argmax', self, *a, **k)


def _method_argmin(self, *a, **k):
    return _axis_reduce('argmin', self, *a, **k)


def _method_amax(self, *a, **k):
    return _axis_reduce('amax', self, *a, **k)


def _method_amin(self, *a, **k):
    return _axis_reduce('amin', self, *a, **k)


def _method_cumsum(self, *a, **k):
    return _axis_reduce('cumsum', self, *a, **k)


def _method_norm(self, *a, **k):
    return _axis_reduce('norm', self, *a, **k)


def _method_std(self, *a, **k):
    return _axis_reduce('std', self, *a, **k)


def _method_var(self, *a, **k):
    return _axis_reduce('var', self, *a, **k)

_SHAPE_REDUCTION_APIS = {
    'mean': _method_mean,
    'prod': _method_prod,
    'any': _method_any,
    'all': _method_all,
    'max': _method_max,
    'min': _method_min,
    'argmax': _method_argmax,
    'argmin': _method_argmin,
    'amax': _method_amax,
    'amin': _method_amin,
    'cumsum': _method_cumsum,
    'norm': _method_norm,
    'std': _method_std,
    'var': _method_var,
}


def _shape_relu(self):
    return nn.relu(self)


def _shape_relu_(self):
    return nn.relu(self)


def _shape_eq(self, other):
    return self == other


def _shape_ne(self, other):
    return self != other


def _shape_gt(self, other):
    return self > other


def _shape_ge(self, other):
    return self >= other


def _shape_lt(self, other):
    return self < other


def _shape_le(self, other):
    return self <= other


def _shape_neg(self):
    return -self


def _shape_reciprocal(self):
    return 1.0 / self


def _shape_expm1(self):
    return jt.exp(self) - 1


def _shape_log1p(self):
    return jt.log(self + 1)


def _shape_square(self):
    return self * self


def _shape_square_(self):
    return self.assign(self * self)


def _shape_clamp_min(self, v):
    return jt.maximum(self, v)


def _shape_clamp_max(self, v):
    return jt.minimum(self, v)


def _shape_bmm(self, other):
    return jt.matmul(self, other)


def _shape_mm(self, other):
    return jt.matmul(self, other)


def _shape_mv(self, vec):
    g = get_install_context(jt).target_namespace
    return g.mv(self, vec)


def _shape_fliplr(self):
    return jt.flip(self, 1)


def _shape_flipud(self):
    return jt.flip(self, 0)


def _shape_diff(self, n=1, dim=-1, prepend=None, append=None):
    return _diff(self, n, dim, prepend, append)


def _shape_trapz(self, x=None, dx=1, dim=-1):
    return _trapz(self, x=x, dx=dx, dim=dim)


def _shape_trapezoid(self, x=None, dx=1, dim=-1):
    return _trapz(self, x=x, dx=dx, dim=dim)


def _shape_fmod(self, other):
    return self - jt.trunc(self / other) * other


def _shape_remainder(self, other):
    return self - jt.floor(self / other) * other


def _shape_softplus(self, beta=1, threshold=20):
    return nn.softplus(self)
