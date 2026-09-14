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
                    "convStrides": [stride[0], stride[1]],
                    "convPads": [padding[0], padding[1]],
                    "convDilations": [dilation[0], dilation[1]],
                    "group": groups,
                    "convOutPads": [0, 0],
                    "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0,
                },
                variable="op",
            ),
            "\n        ",
        ]
    )


def _conv_output_shape(x, weight, stride, padding, dilation):
    input_height, input_width = x.shape[-2:]
    kernel_height, kernel_width = weight.shape[-2:]
    output_height = (
        input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1
    ) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[
        1
    ] + 1
    return (x.shape[0], weight.shape[0], output_height, output_width)


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
            "convOutPads": [0, 0],
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
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        program = _conv_program(
            False,
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            groups,
            1 if getattr(jt, "acl_allow_hf32", False) else 0,
        )
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        return acl_emit(program, [x, weight], [x.dtype], [output_shape])[0]


class ConvACL:
    def __call__(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is None:
            return _ConvACLNoBias()(x, weight, bias, stride, padding, dilation, groups)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")

        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        program = _conv_program(
            True,
            tuple(stride),
            tuple(padding),
            tuple(dilation),
            groups,
            1 if getattr(jt, "acl_allow_hf32", False) else 0,
        )
        return acl_emit(program, [x, weight, bias], [x.dtype], [output_shape])[0]
