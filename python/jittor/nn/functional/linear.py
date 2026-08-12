"""Functional linear transformation."""

import jittor as jt


def linear(x, weight, bias=None):
    """Return ``x * weight.T`` with an optional bias."""
    x = jt.nn.matmul_transpose(x, weight)
    if bias is not None:
        return x + bias
    return x


__all__ = ["linear"]
