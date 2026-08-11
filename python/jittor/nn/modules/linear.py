"""Linear transformation layers and functionals."""

import math

import jittor as jt
from jittor import Module, init


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.invariant_uniform(
            (out_features, in_features), "float32"
        )
        bound = 1.0 / math.sqrt(in_features)
        self.bias = (
            init.uniform((out_features,), "float32", -bound, bound)
            if bias else None
        )

    def execute(self, x):
        x = jt.nn.matmul_transpose(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        return x


def linear(x, weight, bias=None):
    """Return ``x * weight.T`` with an optional bias."""
    x = jt.nn.matmul_transpose(x, weight)
    if bias is not None:
        return x + bias
    return x
