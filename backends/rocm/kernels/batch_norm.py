"""MIOpen training BatchNorm with FP32 statistics and affine gradients."""

import math

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.backend_libraries import get_library_ops


def supported(x, weight, bias, eps):
    return (isinstance(weight, jt.Var) and isinstance(bias, jt.Var)
            and x.ndim == 4 and dtype_name(x.dtype) in ("float32", "float16")
            and dtype_name(weight.dtype) == dtype_name(bias.dtype) == "float32"
            and tuple(weight.shape) == tuple(bias.shape) == (x.shape[1],)
            and all(v > 0 for v in x.shape) and x.numel() // x.shape[1] > 1
            and math.isfinite(eps) and eps > 0
            and hasattr(get_library_ops("hipblas"), "miopen_bn_forward"))


class _Backward(jt.Function):
    def __init__(self, eps):
        self.eps = eps

    def execute(self, x, weight, dy, mean, invvar):
        self.saved = x, weight, dy
        with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
            return tuple(get_library_ops("hipblas").miopen_bn_backward(
                x, weight, dy.cast(x.dtype), mean, invvar, self.eps))

    def grad(self, grad_x, grad_weight, grad_bias):
        # The native SDK backward is a first derivative. Recompute its
        # differentiable closed form only when another derivative reaches it.
        x, weight, dy = self.saved
        with jt.enable_grad(), jt.flag_scope(amp_reg=jt.amp_flags.disabled):
            wide = x.float32()
            dout = dy.cast(x.dtype).float32()
            dims = (0, 2, 3)
            centered = wide - wide.mean(dims, keepdims=True)
            rstd = (centered.sqr().mean(dims, keepdims=True) + self.eps).rsqrt()
            xhat = centered * rstd
            weighted = dout * weight.reshape((1, -1, 1, 1))
            dx = (rstd * (weighted - weighted.mean(dims, keepdims=True)
                         - xhat * (weighted * xhat).mean(dims, keepdims=True))).cast(x.dtype)
            dw = (dout * xhat).sum(dims)
            db = dout.sum(dims)
            terms = [(value * seed).sum() for value, seed in
                     zip((dx, dw, db), (grad_x, grad_weight, grad_bias)) if seed is not None]
            if not terms:
                return None, None, None, None, None
            grads = jt.grad(sum(terms), [x, weight, dy], retain_graph=True)
        return grads[0], grads[1], grads[2], None, None


class _Training(jt.Function):
    def __init__(self, eps):
        self.eps = eps

    def execute(self, x, weight, bias):
        with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
            y, mean, invvar, _, running_var = get_library_ops("hipblas").miopen_bn_forward(
                x, weight, bias, self.eps)
            # MIOpen returns the unbiased running variance. The shared
            # BatchNorm caller applies that correction itself.
            count = x.numel() // x.shape[1]
            var = running_var * ((count - 1) / count)
            self.statistics = mean.stop_grad(), var.stop_grad()
            self.saved = x, weight, mean.stop_grad(), invvar.stop_grad()
            return y

    def grad(self, dy):
        x, weight, mean, invvar = self.saved
        return _Backward(self.eps)(x, weight, dy, mean, invvar)


def training(x, weight, bias, eps):
    function = _Training(float(eps))._new_call_context()
    y = function._run_call(x, weight, bias)
    return (y,) + function.statistics
