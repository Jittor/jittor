"""FP32 CUDA Softplus with a differentiable, explicitly rounded VJP.

The native scalar boundary multiplies the cotangent by exp(beta*x) before
division. Differentiating log1p(exp(beta*x))/beta reverses that order. Those
single-precision differences can survive cancellation in parameter gradients.
Other backends/dtypes and unsupported scalar configurations retain the generic
implementation; this module does not change the process-wide math policy.
"""
import math

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel


class _SoftplusCUDA(jt.Function):
    def __init__(self, beta, threshold):
        self.beta = beta
        self.threshold = threshold

    def _run_call(self, x):
        # Keep an independent holder of the original graph before Function
        # tapes its execute input, including for higher input derivatives.
        self.saved = jt.ops.reshape(x, x.shape)
        return super()._run_call(x)

    def execute(self, x):
        scaled = self.beta * x
        above = scaled > self.threshold
        safe_scaled = jt.ternary(above, self.threshold, scaled)
        nonlinear = jt.log1p(safe_scaled.exp()) / self.beta
        return jt.ternary(above, x, nonlinear)

    def grad(self, dy):
        scaled = self.beta * self.saved
        above = scaled > self.threshold
        safe_scaled = jt.ternary(above, self.threshold, scaled)
        z = safe_scaled.exp()
        # Two separately rounded arithmetic operations, not dy*(z/(z+1)).
        # Ordinary native ops preserve derivatives with respect to x AND dy.
        return jt.ternary(above, dy, (dy * z) / (z + 1.0))


def _supported(x, beta=1.0, threshold=20.0):
    return (type(beta) in (int, float) and math.isfinite(beta) and beta > 0
            and type(threshold) in (int, float) and math.isfinite(threshold))


@optional_kernel("nn.softplus", "cuda", dtypes=("float32",), supports=_supported)
def _softplus_cuda(x, beta=1.0, threshold=20.0):
    result = _SoftplusCUDA(beta, threshold)(x)
    return result if _output_requires_grad(x) else _stop_grad_outputs(result)


__all__ = []
