"""Functional normalization implementations exposed through :mod:`jittor.nn`."""

from functools import lru_cache

import jittor as jt


def batch_norm(x, running_mean, running_var, weight=1, bias=0, training=False, momentum=0.1, eps=1e-05):
    dims = [0]+list(range(2,x.ndim))
    if training:
        # compute batch statistics (torch F.batch_norm training path; used by timm)
        xmean = x.mean(dims)
        x2mean = (x*x).mean(dims)
        xvar = (x2mean - xmean*xmean).maximum(0.0)
        w = weight / jt.sqrt(xvar + eps)
        b = bias - xmean * w
        norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
        # update running stats in-place (unbiased var), if real Vars were passed
        if isinstance(running_mean, jt.Var) and isinstance(running_var, jt.Var):
            n = x.numel() / x.shape[1]
            run_var = xvar * (n/(n-1)) if n > 1 else xvar
            running_mean.update(running_mean + (xmean.reshape((-1,)) - running_mean)*momentum)
            running_var.update(running_var + (run_var.reshape((-1,)) - running_var)*momentum)
        return norm_x
    w = weight / jt.sqrt(running_var+eps)
    b = bias - running_mean * w
    norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
    return norm_x


def instance_norm(x,
    running_mean = None,
    running_var = None,
    weight = 1,
    bias = 0,
    momentum = 0.1,
    eps = 1e-5):
    dims = list(range(2,x.ndim))
    xhat = jt.nn._ln_normalize(x, dims, eps)   # stable custom backward, see _ln_normalize
    weight = 1.0 if weight is None else weight
    bias = 0.0 if bias is None else bias
    if isinstance(weight, jt.Var):
        weight = weight.reshape([1, x.shape[1]] + [1]*len(dims))
    if isinstance(bias, jt.Var):
        bias = bias.reshape([1, x.shape[1]] + [1]*len(dims))
    return xhat * weight + bias


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
    cls = jt.nn._ln_function_cls(tuple(dims), float(eps))
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
    xg = x.reshape((N, num_groups, C//num_groups, -1))
    xhat = jt.nn._ln_normalize(xg, [2,3], eps).reshape(output_shape)  # stable custom backward
    if isinstance(weight, jt.Var):
        weight = weight.reshape([1, C] + [1]*(len(output_shape)-2))
    if isinstance(bias, jt.Var):
        bias = bias.reshape([1, C] + [1]*(len(output_shape)-2))
    return xhat * weight + bias
