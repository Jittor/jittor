"""Linear transformation layers and functionals."""

import math

import jittor as jt
from jittor import Module, init

from ..functional.linear import linear as linear


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

    def reset_parameters(self):
        self.weight.update(
            init.invariant_uniform(
                (self.out_features, self.in_features), self.weight.dtype
            )
        )
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            self.bias.update(
                init.uniform(
                    (self.out_features,), self.bias.dtype, -bound, bound
                )
            )


class Conv1d_sp(Linear):
    def __init__(self, inchannels, outchannels, kernel_size=1, bias=True):
        assert inchannels > 0, 'in_channels must be positive'
        assert outchannels > 0, 'out_channels must be positive'
        super().__init__(inchannels, outchannels, bias=bias)
        assert kernel_size == 1

    def execute(self, x):
        if x.dim() != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        x = x.transpose(0, 2, 1)
        x = super().execute(x)
        x = x.transpose(0, 2, 1)
        return x
