from ._code import code_with_attributes
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


class _ConvACLNoBias:
    def __call__(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is not None:
            raise ValueError("Conv2d no-bias runner received a bias tensor")
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        attributes = {"convStrides": list(stride), "convPads": list(padding), "convDilations": list(dilation), "group": groups, "convOutPads": [0, 0], "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0}
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        result = conv_cmd(
            "Conv2d",
            [x, weight],
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attributes=attributes,
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            Conv2dBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=attributes,
        )[0]
        return result


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
        attributes = {"convStrides": list(stride), "convPads": list(padding), "convDilations": list(dilation), "group": groups, "convOutPads": [0, 0], "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0}

        return conv_cmd(
            "Conv2d",
            [x, weight, bias],
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attributes=attributes,
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            Conv2dBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(in2, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=attributes,
        )[0]
