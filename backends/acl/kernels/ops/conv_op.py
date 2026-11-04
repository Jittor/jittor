from ._code import acl_emit, acl_program, code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import os
import jittor_utils
from jittor_utils import env_or_try_find
import ctypes
import glob
import jittor as jt
import jittor.compiler as compiler
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable

from ._code import acl_code as conv_cmd
from .._soc import allow_reduced_precision


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple([x] * n)

    return parse


_pair = _ntuple(2)


def _conv_attr_code(stride, padding, dilation, groups, name):
    return code_program(
        [
            '\n        op.jt_name = "',
            name,
            '";\n        ',
            attribute_program(
                runner_for_alias(name),
                {
                    "convStrides": list(stride),
                    "convPads": list(padding),
                    "convDilations": list(dilation),
                    "group": groups,
                    "convOutPads": [0] * len(stride),
                    "cube_math_type": 1 if allow_reduced_precision() else 0,
                },
                variable="op",
            ),
            "\n        ",
        ]
    )


def _conv_output_shape(x, weight, stride, padding, dilation):
    dimensions = x.ndim - 2
    if dimensions not in (2, 3) or weight.ndim != x.ndim:
        raise ValueError("ACL convolution requires matching 4-D or 5-D input and weight")
    if any(len(value) != dimensions for value in (stride, padding, dilation)):
        raise ValueError("ACL convolution spatial parameters must match the input rank")
    spatial_shape = tuple(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(
            x.shape[2:], weight.shape[2:], stride, padding, dilation)
    )
    if any(size <= 0 for size in spatial_shape):
        raise ValueError("ACL convolution has a non-positive output spatial dimension")
    return (x.shape[0], weight.shape[0]) + spatial_shape


_BIASED_GRAD_SRC = "\n            // aclop\n            Conv2dBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(in2, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            \n            op.run();\n            "

_UNBIASED_GRAD_SRC = "\n            // aclop\n            Conv2dBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            \n            op.run();\n            "

#: One assembled program per convolution geometry. `cube_math_type` follows
#: `jt.acl_allow_hf32`, which a caller may flip between two convolutions, so
#: it is a key component: a program cached under the previous value would keep
#: launching the previous arithmetic.
_CONV_PROGRAMS = {}


def _conv_program(biased, stride, padding, dilation, groups, cube_math_type):
    key = (biased, stride, padding, dilation, groups, cube_math_type)
    program = _CONV_PROGRAMS.get(key)
    if program is None:
        attributes = {
            "convStrides": list(stride),
            "convPads": list(padding),
            "convDilations": list(dilation),
            "group": groups,
            "convOutPads": [0] * len(stride),
            "cube_math_type": cube_math_type,
        }
        program = acl_program(
            "Conv2d",
            3 if biased else 2,
            1,
            attributes=attributes,
            multi_grad_src=_BIASED_GRAD_SRC if biased else _UNBIASED_GRAD_SRC,
            multi_grad_attributes=attributes,
        )
        _CONV_PROGRAMS[key] = program
    return program


class _ConvACLNoBias:
    def __call__(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is not None:
            raise ValueError("Conv2d no-bias runner received a bias tensor")
        spatial_tuple = _ntuple(x.ndim - 2)
        padding = tuple(spatial_tuple(padding))
        stride = tuple(spatial_tuple(stride))
        dilation = tuple(spatial_tuple(dilation))
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        program = _conv_program(
            False,
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            groups,
            1 if allow_reduced_precision() else 0,
        )
        return acl_emit(program, [x, weight], [x.dtype], [output_shape])[0]


class ConvACL:
    def __call__(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is None:
            return _ConvACLNoBias()(x, weight, bias, stride, padding, dilation, groups)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")

        spatial_tuple = _ntuple(x.ndim - 2)
        padding = tuple(spatial_tuple(padding))
        stride = tuple(spatial_tuple(stride))
        dilation = tuple(spatial_tuple(dilation))
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        program = _conv_program(
            True,
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            groups,
            1 if allow_reduced_precision() else 0,
        )
        return acl_emit(program, [x, weight, bias], [x.dtype], [output_shape])[0]


_CONV_TRANSPOSE_PROGRAMS = {}


def _conv_transpose_program(biased, stride, padding, output_padding, dilation,
                            groups, cube_math_type):
    key = (biased, stride, padding, output_padding, dilation, groups, cube_math_type)
    program = _CONV_TRANSPOSE_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "Conv2d", 3 if biased else 2, 1,
            attributes={
                "convStrides": list(stride),
                "convPads": list(padding),
                "convDilations": list(dilation),
                "group": groups,
                "convOutPads": list(output_padding),
                "cube_math_type": cube_math_type,
                "is_transposed": True,
            },
        )
        _CONV_TRANSPOSE_PROGRAMS[key] = program
    return program


class ConvTranspose2dACL:
    """Inference-only transpose convolution; admission is owned by neural.py."""

    def __call__(self, x, weight, bias, stride, padding, output_padding,
                 dilation, groups):
        if not jt.flags.no_grad:
            raise RuntimeError("ACL ConvTranspose2d requires no_grad; gradients are not implemented")
        output_shape = (x.shape[0], weight.shape[1] * groups) + tuple(
            (size - 1) * step - 2 * pad + dil * (kernel - 1) + out_pad + 1
            for size, kernel, step, pad, out_pad, dil in zip(
                x.shape[2:], weight.shape[2:], stride, padding, output_padding, dilation)
        )
        program = _conv_transpose_program(
            bias is not None, tuple(stride), tuple(padding), tuple(output_padding),
            tuple(dilation), groups, 1 if allow_reduced_precision() else 0,
        )
        inputs = [x, weight] if bias is None else [x, weight, bias]
        return acl_emit(program, inputs, [x.dtype], [output_shape])[0]
