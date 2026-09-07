"""Torch numerical reductions operations."""

def _reduce_alias(native, input, dim=None, keepdim=False, *, axis=None,
                  keepdims=None, out=None):
    d = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    if d is None or d == ():
        return native(input)
    result = native(input, d)
    if kd:
        dims = (d,) if isinstance(d, int) else tuple(d)
        for dd in sorted(x % input.ndim for x in dims):
            result = result.unsqueeze(dd)
    return result


def all(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
    from . import (
        _native_all,
        _reduce_alias,
    )
    return _reduce_alias(_native_all, input, dim, keepdim,
                         axis=axis, keepdims=keepdims, out=out)


def any(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
    from . import (
        _native_any,
        _reduce_alias,
    )
    return _reduce_alias(_native_any, input, dim, keepdim,
                         axis=axis, keepdims=keepdims, out=out)


def _nansum_impl(input, dim=None, keepdim=False, **kwargs):
    from . import (
        jt,
    )
    values = jt.nan_to_num(input, nan=0.0)
    return (values.sum() if dim is None
            else values.sum(dim, keepdims=keepdim))


def nansum(input, dim=None, keepdim=False, **kwargs):
    """Sum tensor values while treating NaNs as zero."""
    from . import (
        _nansum_impl,
    )
    return _nansum_impl(input, dim=dim, keepdim=keepdim, **kwargs)


def _nanmean_impl(input, dim=None, keepdim=False, **kwargs):
    from . import (
        jt,
    )
    count = 1.0 - jt.isnan(input).float32()
    values = jt.nan_to_num(input, nan=0.0)
    if dim is None:
        return values.sum() / count.sum()
    return (values.sum(dim, keepdims=keepdim)
            / count.sum(dim, keepdims=keepdim))


def nanmean(input, dim=None, keepdim=False, **kwargs):
    """Mean tensor values while ignoring NaNs in the denominator."""
    from . import (
        _nanmean_impl,
    )
    return _nanmean_impl(input, dim=dim, keepdim=keepdim, **kwargs)


def _quantile_impl(input, q, dim=None, keepdim=False,
                   interpolation="linear", **kwargs):
    from . import (
        jt,
        np,
    )
    values = input.numpy()
    quantile = q.numpy() if isinstance(q, jt.Var) else q
    result = np.quantile(
        values, quantile, axis=dim, keepdims=keepdim)
    return jt.array(result.astype("float32"))


def quantile(input, q, dim=None, keepdim=False,
             interpolation="linear", **kwargs):
    """Return a NumPy-backed quantile result for CPU-compatible tensors."""
    from . import (
        _quantile_impl,
    )
    return _quantile_impl(
        input, q, dim=dim, keepdim=keepdim,
        interpolation=interpolation, **kwargs)


def _nanquantile_impl(input, q, dim=None, keepdim=False,
                      interpolation="linear", **kwargs):
    from . import (
        jt,
        np,
    )
    values = input.numpy()
    quantile = q.numpy() if isinstance(q, jt.Var) else q
    result = np.nanquantile(
        values, quantile, axis=dim, keepdims=keepdim)
    return jt.array(result.astype("float32"))


def nanquantile(input, q, dim=None, keepdim=False,
                interpolation="linear", **kwargs):
    """Return a NumPy-backed NaN-ignoring quantile for CPU tensors."""
    from . import (
        _nanquantile_impl,
    )
    return _nanquantile_impl(
        input, q, dim=dim, keepdim=keepdim,
        interpolation=interpolation, **kwargs)


def _std_mean_impl(input, dim=None, unbiased=True, keepdim=False,
                   correction=None, **kwargs):
    mean = (input.mean() if dim is None
            else input.mean(dim, keepdims=keepdim))
    std = input.std() if dim is None else input.std(dim)
    return std, mean


def std_mean(input, dim=None, unbiased=True, keepdim=False,
             correction=None, **kwargs):
    """Return standard deviation and mean using current Jittor semantics."""
    from . import (
        _std_mean_impl,
    )
    return _std_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)


def _var_mean_impl(input, dim=None, unbiased=True, keepdim=False,
                   correction=None, **kwargs):
    from . import (
        _std_mean_impl,
    )
    standard_deviation, mean = _std_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)
    return standard_deviation * standard_deviation, mean


def var_mean(input, dim=None, unbiased=True, keepdim=False,
             correction=None, **kwargs):
    """Return variance and mean using current Jittor semantics."""
    from . import (
        _var_mean_impl,
    )
    return _var_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)


def _aminmax_impl(input, dim=None, keepdim=False):
    from . import (
        _AminMax,
    )
    if dim is None:
        return _AminMax(input.min(), input.max())
    return _AminMax(
        input.min(dim, keepdims=keepdim),
        input.max(dim, keepdims=keepdim),
    )


def aminmax(input, dim=None, keepdim=False):
    """Return named minimum and maximum reductions."""
    from . import (
        _aminmax_impl,
    )
    return _aminmax_impl(input, dim=dim, keepdim=keepdim)


def _logcumsumexp_impl(input, dim):
    from . import (
        jt,
    )
    maximum = input.max(dim, keepdims=True)
    return maximum + jt.log(jt.cumsum(jt.exp(input - maximum), dim))


def logcumsumexp(input, dim):
    """Return cumulative log-sum-exp values along ``dim``."""
    from . import (
        _logcumsumexp_impl,
    )
    return _logcumsumexp_impl(input, dim)


def logsumexp(input, dim, keepdim=False):
    """Compute a numerically stable log-sum-exp reduction."""
    from . import (
        jt,
    )
    m = input.max(dim, keepdims=True)
    out = m + jt.log(jt.exp(input - m).sum(dim, keepdims=True))
    if keepdim:
        return out
    dims = [dim] if isinstance(dim, int) else list(dim)
    nd = input.ndim
    dims = [d % nd for d in dims]
    target = [s for i, s in enumerate(input.shape) if i not in dims]
    return out.reshape(target) if target else out.reshape(-1)
