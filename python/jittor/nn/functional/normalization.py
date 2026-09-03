"""Functional normalization implementations exposed through :mod:`jittor.nn`."""

from functools import lru_cache, wraps

import jittor as jt

from ... import _arg_policy


@lru_cache(maxsize=128)
def _bn_function_cls(dims, eps):
    # Normalize x with statistics supplied from OUTSIDE, so one body serves both
    # local statistics and statistics all-reduced across MPI ranks. The module
    # used to carry two: the sync branch scaled raw x by `weight/sqrt(var+eps)`
    # and let composite autodiff differentiate through it, the non-sync branch
    # called _ln_normalize -- different variance formula, different expression,
    # different backward. Whether MPI was initialised decided the numbers.
    #
    # The grads below are the closed form written per input. Chained through
    # `mean` and `var` they reproduce rstd*(g - mean(g) - xhat*mean(g*xhat))
    # exactly, and every intermediate stays O(1)-scaled -- no (var+eps)^-1.5
    # times raw x that has to catastrophically cancel. When the statistics were
    # all-reduced, the chain runs back through mpi_all_reduce, which is what
    # makes the cross-rank gradient right.
    class _BN(jt.Function):
        def execute(self, x, mean, var):
            rstd = jt.rsqrt(var + eps).broadcast(x, dims)
            xhat = (x - mean.broadcast(x, dims)) * rstd
            self.xhat = xhat
            self.rstd = rstd
            return xhat

        def grad(self, g):
            xhat, rstd = self.xhat, self.rstd
            return (
                g * rstd,                                   # d/dx
                -(g * rstd).sum(dims),                      # d/dmean
                -0.5 * (g * xhat * rstd * rstd).sum(dims),  # d/dvar
            )
    return _BN


def _bn_normalize(x, mean, var, dims, eps):
    cls = _bn_function_cls(tuple(dims), float(eps))
    return cls.apply(x, mean, var)


def _batch_statistics(x, dims, sync):
    """Mean and variance over ``dims``, optionally reduced across MPI ranks.

    Two passes on purpose. ``E[x^2] - E[x]^2`` -- what the sync branch used --
    cancels catastrophically as soon as the mean is large next to the standard
    deviation, and it was used *only* under MPI, so the error appeared when the
    job was distributed and nowhere else.

    The variance is reduced after the mean, so the all-reduced value is the true
    global two-pass variance and a world of one reproduces the single-process
    numbers exactly. That is two collectives, the same count as the old
    ``(mean, E[x^2])`` pair.
    """
    xmean = jt.mean(x, dims=dims)
    if sync:
        xmean = xmean.mpi_all_reduce("mean")
    deviation = x - xmean.broadcast(x, dims)
    xvar = jt.mean(deviation * deviation, dims=dims)
    if sync:
        xvar = xvar.mpi_all_reduce("mean")
    return xmean, xvar


def _affine(xhat, weight, bias, num_features, ndim):
    """Per-channel scale and shift, skipped entirely when it is the identity."""
    weight_is_var = isinstance(weight, jt.Var)
    bias_is_var = isinstance(bias, jt.Var)
    if not weight_is_var and not bias_is_var:
        if weight == 1 and bias == 0:
            return xhat
        return xhat * weight + bias
    shape = [1, num_features] + [1] * (ndim - 2)
    if weight_is_var:
        weight = weight.reshape(shape)
    if bias_is_var:
        bias = bias.reshape(shape)
    return xhat * weight + bias


def _batch_norm_train(x, dims, weight, bias, eps, sync=False):
    """Training-mode batch norm. Returns ``(y, mean, var)``.

    The single body behind ``nn.BatchNorm.execute`` and
    ``nn.functional.batch_norm(training=True)``; the caller decides what to do
    with the statistics (the module updates its running buffers, the functional
    updates the buffers it was handed).
    """
    xmean, xvar = _batch_statistics(x, dims, sync)
    if not sync:
        # Fused CUDA kernel for the local case. It computes its own statistics,
        # so it cannot serve the all-reduced ones; it is a backend accelerator
        # for this same function, pinned against it by
        # tests/nn/test_norm_unification.py. functional.batch_norm never
        # reached it before -- training=True went down the generic path only.
        fast = jt.nn._batch_norm_cuda(x, weight, bias, eps)
        if fast is not None:
            return fast, xmean, xvar
    xhat = _bn_normalize(x, xmean, xvar, dims, eps)
    return _affine(xhat, weight, bias, x.shape[1], x.ndim), xmean, xvar


def _batch_norm_eval(x, dims, running_mean, running_var, weight, bias, eps):
    """Eval-mode batch norm: normalize with the tracked statistics."""
    fast = jt.nn._batch_norm_eval_cuda(
        x, weight, bias, running_mean, running_var, eps)
    if fast is not None:
        return fast
    scale = weight / jt.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return x * scale.broadcast(x, dims) + shift.broadcast(x, dims)


def _unbiased(var, x, dims, world_size=1):
    """Bessel-corrected variance for the running buffer, as torch stores it."""
    count = world_size
    for dim in dims:
        count *= x.shape[dim]
    return var * (count / (count - 1)) if count > 1 else var


def batch_norm(x, running_mean, running_var, weight=1, bias=0, training=False,
               momentum=0.1, eps=1e-05):
    """Batch normalization, ``torch.nn.functional.batch_norm``'s signature.

    Shares its body with :class:`jittor.nn.BatchNorm`; the module holds the
    parameters and the running buffers and nothing else. Before that the two
    were separate transcriptions and this one never reached the fused CUDA
    kernel in training mode.
    """
    dims = [0] + list(range(2, x.ndim))
    tracking = isinstance(running_mean, jt.Var) and isinstance(running_var, jt.Var)
    if training:
        norm_x, xmean, xvar = _batch_norm_train(x, dims, weight, bias, eps)
        if tracking:
            running_mean.update(
                running_mean + (xmean.reshape((-1,)) - running_mean) * momentum)
            running_var.update(
                running_var
                + (_unbiased(xvar, x, dims).reshape((-1,)) - running_var) * momentum)
        return norm_x
    return _batch_norm_eval(x, dims, running_mean, running_var, weight, bias, eps)


def instance_norm(x,
    running_mean = None,
    running_var = None,
    weight = 1,
    bias = 0,
    momentum = 0.1,
    eps = 1e-5):
    ''' Per-sample normalisation over the spatial dims, like ``torch.nn.functional.instance_norm``.

    ``running_mean``/``running_var``/``momentum`` describe running-statistics
    tracking, which this implementation does not have; see ``jittor._arg_policy``
    and ``tests/ops/test_ignored_arguments.py``.
    '''
    if momentum != 0.1:
        # In torch momentum only ever feeds the running-stat update. There is no
        # such update here, so no value of it can change anything -- and the
        # caller who bothered to pass one is asking for tracking.
        _arg_policy.ignored(
            "jittor.nn.instance_norm", "momentum", momentum,
            "momentum only ever weights a running-statistics update, and no "
            "running statistics are tracked, so every value behaves like the "
            "default")
    if running_mean is not None or running_var is not None:
        # torch updates these from the batch and, once tracking is on, uses them
        # instead of the per-sample statistics in eval mode. Accepting them and
        # normalising per-sample anyway returns different numbers than asked for.
        _arg_policy.unsupported(
            "jittor.nn.instance_norm", "running_mean/running_var",
            "not None",
            "the running statistics are neither updated nor used, so the buffers "
            "stay at their initial values and eval-mode normalisation silently "
            "uses per-sample statistics instead of the tracked ones")
    dims = list(range(2,x.ndim))
    xhat = _ln_normalize(x, dims, eps)   # stable custom backward, see _ln_normalize
    weight = 1.0 if weight is None else weight
    bias = 0.0 if bias is None else bias
    return _affine(xhat, weight, bias, x.shape[1], x.ndim)


@lru_cache(maxsize=128)
def _ln_function_cls(dims, eps):
    # Normalize x -> (x-mean)/sqrt(var+eps) over `dims` with a numerically-STABLE
    # custom backward (the closed form torch's fused LN uses). The composite-autodiff
    # backward forms huge terms (x * d/dx[1/sqrt(var+eps)] ~ (var+eps)^-1.5) that must
    # catastrophically cancel to the true input-grad -> float32 error (~1% for small-
    # variance inputs; negligible for std~1). torch's fused LN avoids it; this matches.
    # jt.Function: invoked via .apply(); tape_together makes grad() the backward
    # (overriding the composite path inside execute).
    class _LN(jt.Function):
        def execute(self, x):
            mean = jt.mean(x, dims=dims, keepdims=1)
            var = jt.mean((x - mean) * (x - mean), dims=dims, keepdims=1)
            rstd = jt.rsqrt(var + eps)
            xhat = (x - mean) * rstd
            self.xhat = xhat
            self.rstd = rstd
            return xhat
        def grad(self, g):
            # dL/dx = rstd*(g - mean(g) - xhat*mean(g*xhat)) over the normalized dims
            xhat, rstd = self.xhat, self.rstd
            mg = jt.mean(g, dims=dims, keepdims=1)
            mgx = jt.mean(g * xhat, dims=dims, keepdims=1)
            return rstd * (g - mg - xhat * mgx)
    return _LN


def _ln_normalize(x, dims, eps):
    # Cache the immutable Function class by reduction axes and epsilon. A fresh
    # Function instance/tape is still created by apply() for every invocation.
    cls = _ln_function_cls(tuple(dims), float(eps))
    return cls.apply(x)


def group_norm(x,
    num_groups,
    weight = 1,
    bias = 0,
    eps=1e-05):
    N = x.shape[0]
    C = x.shape[1]
    # Restore the full input shape for any spatial rank (1d/2d/3d data, i.e.
    # 3d/4d/5d tensors). Only fall back to (N, C) when there is no spatial dim.
    if x.ndim >= 3:
        output_shape = x.shape
    else:
        output_shape = (N, C)
    assert C % num_groups == 0
    fast = jt.nn._group_norm_cuda(x, num_groups, weight, bias, eps)
    if fast is not None:
        return fast
    xg = x.reshape((N, num_groups, C//num_groups, -1))
    xhat = _ln_normalize(xg, [2,3], eps).reshape(output_shape)  # stable custom backward
    return _affine(xhat, weight, bias, C, len(output_shape))


def fp32_guard(func):
    @wraps(func)
    def wrapper(*args, **kw):
        if jt.flags.amp_level == 0:
            return func(*args, **kw)
        new_args = []
        need_cast = False
        dtype = None
        for arg in args:
            if isinstance(arg, jt.Var) and arg.dtype in ("float16", "bfloat16"):
                dtype = arg.dtype
                new_args.append(arg.float32())
                need_cast = True
            else:
                new_args.append(arg)
        with jt.flag_scope(amp_level=0):
            result = func(*new_args, **kw)
            if need_cast and isinstance(result, jt.Var) and result.dtype == "float32":
                result = result.cast(dtype)
        return result

    return wrapper


@fp32_guard
def layer_norm(
    x,
    normalized_shape,
    weight=1,
    bias=0,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
):
    dims = [-i for i in range(len(normalized_shape), 0, -1)]
    weight = 1.0 if weight is None else weight
    bias = 0.0 if bias is None else bias
    fast = jt.nn._layer_norm_cuda(
        x, tuple(normalized_shape), weight, bias, eps
    )
    if fast is not None:
        return fast
    fast = jt.nn._layer_norm_no_grad_cuda(
        x, tuple(normalized_shape), weight, bias, eps
    )
    if fast is not None:
        return fast
    xhat = _ln_normalize(x, dims, eps)
    return xhat * weight + bias
