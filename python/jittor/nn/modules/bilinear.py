"""Bilinear transformation module."""

import math

import jittor as jt
from jittor import Module


class Bilinear(Module):
    """bilinear transformation $out = in1^T W in2 + bias$, Example::

    m = nn.Bilinear(20, 30, 40)
    input1 = jt.randn(128, 20)
    input2 = jt.randn(128, 30)
    output = m(input1, input2)
    print(output.shape)
    # [128, 40]

    """

    def __init__(self, in1_features, in2_features, out_features, bias=True, dtype="float32"):
        bound = 1 / math.sqrt(in1_features)
        self.weight = jt.init.uniform(
            [out_features, in1_features, in2_features], dtype, -bound, bound
        )
        self.bias = bias
        if bias:
            self.bias = jt.init.uniform([out_features], dtype, -bound, bound)

    def execute(self, in1, in2):
        return jt.nn.bilinear(in1, in2, self.weight, self.bias)


__all__ = ["Bilinear"]
