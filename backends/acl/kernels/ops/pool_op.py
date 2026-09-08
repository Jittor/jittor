from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable


from ._code import acl_code as pool_cmd


class PoolACL(jt.Function):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=None,
        return_indices=None,
        ceil_mode=False,
        count_include_pad=True,
        op="maximum",
    ):
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = dilation if dilation else 1
        if dilation != 1:
            raise ValueError("ACL pooling only supports dilation=1")
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        for item in self.kernel_size:
            if item <= 0:
                raise RuntimeError(f"kernel_size must be greater than zero, but got {item}")
        for item in self.stride:
            if item <= 0:
                raise RuntimeError(f"stride must be greater than zero, but got {item}")
        for item in self.padding:
            if item < 0:
                raise RuntimeError(f"padding must be non-negative, but got {item}")
        self.op = op
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, input):
        from jittor.nn.functional.pooling import _pool_output_size

        self.input = input
        attributes = {
                        "kernel_size": [self.kernel_size[0], self.kernel_size[1]],
                        "poolStrides": [self.stride[0], self.stride[1]],
                        "poolPads": [self.padding[0], self.padding[1]],
                        "poolDilations": [self.dilation[0], self.dilation[1]],
                        "poolCeil": bool(self.ceil_mode),
                        "countIncludePad": bool(self.count_include_pad),
                    }
        output_height, output_width = (
            _pool_output_size(size, kernel, stride, padding, self.ceil_mode)
            for size, kernel, stride, padding in zip(
                input.shape[-2:], self.kernel_size, self.stride, self.padding
            )
        )

        output_shape = (input.shape[0], input.shape[1], output_height, output_width)

        inputs = [input]

        if self.op == "maximum":
            result = pool_cmd(
                "Maxpool",
                inputs,
                output_dtypes=[input.dtype, "int32"],
                output_shapes=[output_shape, output_shape],
                attributes=attributes,
            )
        elif self.op == "mean":
            result = pool_cmd(
                "Avgpool",
                inputs,
                output_dtypes=[input.dtype],
                output_shapes=[output_shape],
                attributes=attributes,
            )
        else:
            raise ValueError("no this type pool")

        if self.op == "maximum":
            self.index = result[1]

        if self.return_indices:
            return result[0], result[1]
        else:
            return result[0]

    def grad(self, grad_output):
        input = self.input
        attributes = {
                        "kernel_size": [self.kernel_size[0], self.kernel_size[1]],
                        "poolStrides": [self.stride[0], self.stride[1]],
                        "poolPads": [self.padding[0], self.padding[1]],
                        "poolDilations": [self.dilation[0], self.dilation[1]],
                        "poolCeil": bool(self.ceil_mode),
                        "countIncludePad": bool(self.count_include_pad),
                    }
        output_shapes = [input.shape]
        output_dtypes = [input.dtype]
        if self.op == "maximum":
            result = pool_cmd(
                "MaxpoolBackward",
                inputs=[grad_output, input, self.index],
                output_dtypes=output_dtypes,
                output_shapes=output_shapes,
                attributes=attributes,
            )[0]
        elif self.op == "mean":
            result = pool_cmd(
                "AvgpoolBackward",
                inputs=[grad_output, input],
                output_dtypes=output_dtypes,
                output_shapes=output_shapes,
                attributes=attributes,
            )[0]
        else:
            raise ValueError("no this type pool")
        return result
