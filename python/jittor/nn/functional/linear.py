"""Functional linear transformation."""

import jittor as jt
from jittor._runtime.dispatch import try_dispatch

from ._amp import bias_for_compute_dtype


def linear(x, weight, bias=None):
    """Return ``x * weight.T`` with an optional bias."""
    x = jt.nn.matmul_transpose(x, weight)
    if bias is None:
        return x
    # Under the amp register the product is already the compute dtype. A bias
    # that is still wider -- a float32 Parameter, which is what torch keeps --
    # must not lift the result back: torch's autocast casts the bias along with
    # the rest of the linear operator, and a fused linear would keep one dtype
    # here. See `_amp.bias_for_compute_dtype` for why the shim needs it and
    # native jittor does not.
    bias = bias_for_compute_dtype(x, bias)
    result = try_dispatch("nn.linear_bias", x, bias)
    if result is not None:
        return result
    return x + bias


__all__ = ["linear"]
