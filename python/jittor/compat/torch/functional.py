"""Small functional operators with torch-compatible semantics."""

import jittor as jt

from .types import _dtype_to_str


def _torch_norm_impl(input, p="fro", dim=None, keepdim=False, dtype=None):
    # torch.norm / Tensor.norm with torch semantics:
    #   * dim=None  -> reduce over ALL dims to a 0-dim scalar (the key fix);
    #   * p='fro' or None -> 2-norm (Frobenius == Euclidean over the flattened
    #     reduced elements); p may be an int/float (1, 2, inf) or 'fro'/'nuc'.
    #   * dim may be an int or a tuple of ints.
    import math as _m
    if dtype is not None:
        input = input.cast(_dtype_to_str(dtype))
    # normalize the order p
    if p is None or p == "fro":
        pv = 2.0
    elif p == "nuc":
        # nuclear norm (sum of singular values) -- rare; fall back to numpy.
        import numpy as _np
        arr = input.numpy()
        return jt.array(_np.linalg.norm(arr, ord="nuc", axis=dim))
    else:
        pv = float(p)
    if dim is None:
        # full reduction over a flattened view -> 0-dim scalar
        x = input.reshape(-1)
        if pv == float("inf"):
            r = x.abs().max()
        elif pv == float("-inf"):
            r = x.abs().min()
        elif pv == 1.0:
            r = x.abs().sum()
        elif pv == 2.0:
            r = jt.sqrt((x.cast("float32") if str(x.dtype) not in ("float32", "float64") else x).sqr().sum())
        else:
            r = (x.abs() ** pv).sum() ** (1.0 / pv)
        return r
    # per-dim reduction: jittor's native norm handles a single int dim; for a
    # tuple of dims, compose manually.
    if isinstance(dim, (tuple, list)):
        if pv == float("inf"):
            r = input.abs()
            for d in sorted(dim, reverse=True):
                r = r.max(dim=d, keepdims=keepdim)
            return r
        if pv == 1.0:
            r = input.abs()
            for d in sorted(dim, reverse=True):
                r = r.sum(dim=d, keepdims=keepdim)
            return r
        if pv == 2.0:
            r = input.sqr()
            for d in sorted(dim, reverse=True):
                r = r.sum(dim=d, keepdims=keepdim)
            return jt.sqrt(r)
        r = input.abs() ** pv
        for d in sorted(dim, reverse=True):
            r = r.sum(dim=d, keepdims=keepdim)
        return r ** (1.0 / pv)
    if pv == float("inf"):
        return input.abs().max(dim=dim, keepdims=keepdim)
    if pv == float("-inf"):
        return input.abs().min(dim=dim, keepdims=keepdim)
    if pv == 1.0:
        return input.abs().sum(dim=dim, keepdims=keepdim)
    if pv == 2.0:
        return jt.sqrt(input.sqr().sum(dim=dim, keepdims=keepdim))
    return (input.abs() ** pv).sum(dim=dim, keepdims=keepdim) ** (1.0 / pv)


def _torch_where_select(condition, input, other):
    vals = []
    for x in (condition, input, other):
        vals.append(x if isinstance(x, jt.Var) else jt.array(x))
    cond, a, b = vals

    shapes = [tuple(int(d) for d in x.shape) for x in (cond, a, b)]
    out_shape = ()
    for shape in shapes:
        res = []
        for i in range(1, max(len(out_shape), len(shape)) + 1):
            da = out_shape[-i] if i <= len(out_shape) else 1
            db = shape[-i] if i <= len(shape) else 1
            if da == 1:
                res.append(db)
            elif db == 1 or da == db:
                res.append(da)
            else:
                raise RuntimeError(f"where operands could not be broadcast: {shapes}")
        out_shape = tuple(reversed(res))
    if not out_shape:
        out_shape = (1,)

    def _bcast(x):
        shape = tuple(int(d) for d in x.shape)
        if shape == out_shape:
            return x
        return x.broadcast(out_shape)

    return jt.ternary(_bcast(cond).bool(), _bcast(a), _bcast(b))


def _diff(x, n=1, dim=-1, prepend=None, append=None):
    # torch.diff(input, n=1, dim=-1, prepend=None, append=None): prepend/append are
    # concatenated along `dim` before differencing (used by transformers' packed-
    # sequence detection via torch.diff(position_ids, prepend=..., dim=-1)).
    if prepend is not None or append is not None:
        parts = []
        if prepend is not None:
            parts.append(prepend if isinstance(prepend, jt.Var) else jt.array(prepend))
        parts.append(x)
        if append is not None:
            parts.append(append if isinstance(append, jt.Var) else jt.array(append))
        x = jt.concat(parts, dim=dim)
    for _ in range(n):
        idx = [slice(None)] * x.ndim
        idx0 = list(idx); idx1 = list(idx)
        idx0[dim] = slice(1, None); idx1[dim] = slice(0, -1)
        x = x[tuple(idx0)] - x[tuple(idx1)]
    return x


def _trapz(y, x=None, dx=1, dim=-1, *, out=None):
    # torch.trapz / torch.trapezoid: composite trapezoidal integration along
    # `dim`. Torch accepts a 1-D coordinate vector or a coordinate tensor
    # broadcastable to the pairwise y slices.
    y = y if isinstance(y, jt.Var) else jt.array(y)
    ndim = y.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim
    if y.shape[dim] <= 1:
        out_shape = list(y.shape)
        out_shape.pop(dim)
        if not out_shape:
            out_shape = (1,)
        return jt.zeros(tuple(out_shape), dtype=y.dtype)
    sl0 = [slice(None)] * ndim
    sl1 = [slice(None)] * ndim
    sl0[dim] = slice(0, -1)
    sl1[dim] = slice(1, None)
    y0 = y[tuple(sl0)]
    y1 = y[tuple(sl1)]
    area = (y0 + y1) * 0.5
    if x is None:
        area = area * dx
    else:
        x = x if isinstance(x, jt.Var) else jt.array(x)
        d = _diff(x, n=1, dim=dim if x.ndim > 1 else 0)
        if x.ndim == 1 and y.ndim > 1:
            shape = [1] * y.ndim
            shape[dim] = d.shape[0]
            d = d.reshape(shape)
        area = area * d
    res = area.sum(dim=dim)
    if out is not None:
        out.assign(res)
        return out
    return res


def _repeat_interleave(x, repeats, dim=None, *, output_size=None):
    if dim is None:
        x = x.reshape(-1); dim = 0
    if hasattr(jt, "repeat_interleave"):
        try:
            return jt.repeat_interleave(x, repeats, dim=dim, output_size=output_size)
        except TypeError:
            pass
    if isinstance(repeats, int):
        idx = jt.arange(x.shape[dim]).reshape(-1, 1).broadcast([x.shape[dim], repeats]).reshape(-1)
    else:
        parts = []
        r = repeats.numpy() if hasattr(repeats, "numpy") else repeats
        for i, c in enumerate(r):
            parts += [i] * int(c)
        idx = jt.array(parts)
    return x[idx] if dim == 0 else x.transpose(0, dim)[idx].transpose(0, dim)


def _isin(elements, test_elements, **kw):
    te = test_elements.numpy() if hasattr(test_elements, "numpy") else test_elements
    import numpy as _np
    el = elements.numpy() if hasattr(elements, "numpy") else _np.asarray(elements)
    return jt.array(_np.isin(el, te))
