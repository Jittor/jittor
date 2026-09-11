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


from ._code import acl_emit, acl_program

#: `jittor.nn.functional.pooling.average` imports jittor.nn, so this cannot be
#: imported at module scope; resolving it once keeps the statement out of every
#: pooling call, where it walked five package objects to return a module that
#: was already in sys.modules.
_pool_output_size = None


def _output_size_fn():
    global _pool_output_size
    if _pool_output_size is None:
        from jittor.nn.functional.pooling.average import _pool_output_size as fn
        _pool_output_size = fn
    return _pool_output_size


#: Assembled programs per (runner, geometry). `acl_code` re-derives its program
#: key from a freshly built attribute mapping on every call; a pooling geometry
#: is fixed by the module that owns it, so a keyed lookup answers instead.
_POOL_PROGRAMS = {}


def _pool_program(name, input_count, output_count, kernel, stride, padding,
                  dilation, ceil_mode, count_include_pad):
    key = (name, input_count, output_count, kernel, stride, padding, dilation,
           ceil_mode, count_include_pad)
    program = _POOL_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            name,
            input_count,
            output_count,
            attributes={
                "kernel_size": [kernel[0], kernel[1]],
                "poolStrides": [stride[0], stride[1]],
                "poolPads": [padding[0], padding[1]],
                "poolDilations": [dilation[0], dilation[1]],
                "poolCeil": ceil_mode,
                "countIncludePad": count_include_pad,
            },
        )
        _POOL_PROGRAMS[key] = program
    return program


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

    def _geometry(self):
        return (self.kernel_size, self.stride, self.padding, self.dilation,
                bool(self.ceil_mode), bool(self.count_include_pad))

    def execute(self, input):
        output_size = _output_size_fn()

        self.input = input
        output_height, output_width = (
            output_size(size, kernel, stride, padding, self.ceil_mode)
            for size, kernel, stride, padding in zip(
                input.shape[-2:], self.kernel_size, self.stride, self.padding
            )
        )

        output_shape = (input.shape[0], input.shape[1], output_height, output_width)

        inputs = [input]

        if self.op == "maximum":
            result = acl_emit(
                _pool_program("Maxpool", 1, 2, *self._geometry()),
                inputs,
                [input.dtype, "int32"],
                [output_shape, output_shape],
            )
        elif self.op == "mean":
            result = acl_emit(
                _pool_program("Avgpool", 1, 1, *self._geometry()),
                inputs,
                [input.dtype],
                [output_shape],
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
        output_shapes = [input.shape]
        output_dtypes = [input.dtype]
        if self.op == "maximum":
            result = acl_emit(
                _pool_program("MaxpoolBackward", 3, 1, *self._geometry()),
                [grad_output, input, self.index],
                output_dtypes,
                output_shapes,
            )[0]
        elif self.op == "mean":
            result = acl_emit(
                _pool_program("AvgpoolBackward", 2, 1, *self._geometry()),
                [grad_output, input],
                output_dtypes,
                output_shapes,
            )[0]
        else:
            raise ValueError("no this type pool")
        return result
