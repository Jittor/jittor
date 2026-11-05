"""Torch numerical reductions operations."""

import operator
from functools import lru_cache

from jittor._core.dtypes import dtype_name


def _reduce_alias(native, input, dim=None, keepdim=False, *, axis=None,
                  keepdims=None, out=None):
    d = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    if d is None:
        dims = tuple(range(input.ndim))
    else:
        requested = d if isinstance(d, (tuple, list)) else (d,)
        rank = max(input.ndim, 1)
        dims = []
        for value in requested:
            value = operator.index(value)
            if not -rank <= value < rank:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    "[%d, %d], but got %d)" % (-rank, rank - 1, value))
            value %= rank
            if value in dims:
                raise RuntimeError("dim %d appears multiple times in the list of dims" % value)
            dims.append(value)
        dims = tuple(dims)
    # An empty explicit axis tuple performs no reduction in Torch. Native
    # Jittor uses () for all axes, so handle this and scalar inputs separately.
    if input.ndim == 0 or (d is not None and not dims):
        result = input != 0
    else:
        result = native(input, dims, keepdims=kd)
    # Torch retains the legacy byte-mask result for uint8 inputs; every other
    # input dtype returns bool. Both the truth conversion and cast stay on device.
    if dtype_name(input.dtype) == "uint8":
        result = result.cast("uint8")
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
    # `jt.nan_to_num` does not exist -- `nan_to_num` is a tensor method, which is
    # the spelling `elementwise.nan_to_num_` already uses. Reaching for it in the
    # Jittor namespace raised `AttributeError: nan_to_num` for every input, so
    # `nansum` never ran.
    values = input.nan_to_num(nan=0.0)
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
    values = input.nan_to_num(nan=0.0)
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


@lru_cache(maxsize=1)
def _fixed_row_logsumexp_class():
    """Build the Torch-only CUDA path and its first/second derivative rules."""
    from . import jt
    from jittor._runtime.dispatch import try_dispatch

    class _FixedRowLogsumexpVJP(jt.Function):
        def _run_call(self, x, seed, lse):
            self.original = jt.ops.reshape(x, x.shape)
            self.seed = jt.ops.reshape(seed, seed.shape)
            return super()._run_call(x, seed, lse)

        def execute(self, x, seed, lse):
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                probability = jt.exp(x - lse.reshape((-1, 1)).broadcast(x.shape))
                return probability * seed.broadcast(x.shape)

        def grad(self, direction):
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                probability = jt.nn.softmax(self.original, dim=-1)
                dot = (probability * direction).sum([1], keepdims=True)
                dx = self.seed.broadcast(self.original.shape) * probability * (
                    direction - dot.broadcast(self.original.shape))
                return dx, dot, None

    class _FixedRowLogsumexp(jt.Function):
        def _run_call(self, x):
            self.original = jt.ops.reshape(x, x.shape)
            return super()._run_call(x)

        def execute(self, x):
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                result = try_dispatch('nn.torch_logsumexp_last', x)
            if result is None:
                raise RuntimeError('selected CUDA logsumexp kernel became unavailable')
            self.result = result
            return result

        def grad(self, seed):
            return _FixedRowLogsumexpVJP()(self.original, seed.reshape((-1, 1)), self.result)

    return _FixedRowLogsumexp


def logsumexp(input, dim, keepdim=False):
    """Compute a numerically stable log-sum-exp reduction."""
    from . import (
        jt,
    )
    # Large FP32 last-axis CUDA reductions are used by TRL's selective
    # log-softmax. The generated reduction can use unordered atomics; select
    # the fixed-order backend only when its placement and layout contract holds.
    # All native Jittor spellings and unsupported Torch cases retain their path.
    if (input.ndim >= 2 and type(dim) is int and dim in (-1, input.ndim - 1)
            and int(input.shape[-1]) >= 32768
            and str(input.dtype).replace('torch.', '') == 'float32'):
        from jittor._runtime.dispatch import select_kernel
        row = input.reshape((-1, int(input.shape[-1])))
        if select_kernel('nn.torch_logsumexp_last', row) is not None:
            result = _fixed_row_logsumexp_class()()(row)
            shape = tuple(int(v) for v in input.shape[:-1])
            return result.reshape(shape + ((1,) if keepdim else ()))
    # The maximum is only a numerical shift. Differentiating through it
    # causes cancellation at the maximum in large rows; its derivative is
    # unnecessary because the shifted expression is invariant to the shift.
    with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
        m = input.max(dim, keepdims=True).detach()
        m = jt.where(jt.isinf(m), jt.zeros_like(m), m)
        out = m + jt.log(jt.exp(input - m).sum(dim, keepdims=True))
    # Back to the input's dtype for the two half types. The shift-exp-sum-log
    # chain is computed in float32 whatever it is handed -- `exp` is on jittor's
    # white list (`src/type/nano_string.cc`), so it widens -- and that is the
    # right width to compute it at. But torch 2.13 returns float16 for a
    # float16 input and bfloat16 for a bfloat16 one, on CPU and on CUDA, and
    # this handed back a float32: a silent widening in the middle of a half
    # model, at the op transformers reach for in every attention and every
    # cross-entropy. `jittor.nn.functional.softmax.logsumexp` carries the same
    # narrowing for the native spelling.
    name = str(input.dtype).replace("torch.", "")
    if name in ("float16", "bfloat16") and \
            str(out.dtype).replace("torch.", "") != name:
        out = out.cast(name)
    if keepdim:
        return out
    dims = [dim] if isinstance(dim, int) else list(dim)
    nd = input.ndim
    dims = [d % nd for d in dims]
    target = [s for i, s in enumerate(input.shape) if i not in dims]
    return out.reshape(target) if target else out.reshape(-1)
