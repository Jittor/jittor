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
    return f"""
        op.jt_name = "{name}";
        ConvAttr *attr = new ConvAttr();
        attr->convStrides = {{ {stride[0]}, {stride[1]} }};
        attr->convPads = {{ {padding[0]}, {padding[1]} }};
        attr->convDilations = {{ {dilation[0]}, {dilation[1]} }};
        attr->group = {groups};
        attr->convOutPads = {{ 0, 0 }};
        op.op_attr.reset(attr);
        """


def _conv_output_shape(x, weight, stride, padding, dilation):
    input_height, input_width = x.shape[-2:]
    kernel_height, kernel_width = weight.shape[-2:]
    output_height = (input_height + 2 * padding[0] - dilation[0] *
                     (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] *
                    (kernel_width - 1) - 1) // stride[1] + 1
    return (x.shape[0], weight.shape[0], output_height, output_width)


class _ConvACLNoBias(jt.Function):

    def execute(self,
                x,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1):
        self.input = x
        self.weight = weight
        assert bias is None
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        attr_code = _conv_attr_code(stride, padding, dilation, groups, "conv2d")
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        result = conv_cmd(
            "Conv2d",
            [x, weight],
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attr_code=attr_code,
        )[0]
        return result

    def grad(self, grad_output):
        x = self.input
        weight = self.weight
        inputs = [grad_output, x, weight]
        output_shapes = [x.shape, weight.shape, [weight.shape[0]]]
        output_dtypes = [x.dtype, weight.dtype, x.dtype]
        padding = self.padding
        stride = self.stride
        dilation = self.dilation
        groups = self.groups
        attr_code = _conv_attr_code(
            stride, padding, dilation, groups, "conv2dbackward")
        results = conv_cmd("Conv2dBackward",
                           inputs,
                           output_dtypes=output_dtypes,
                           output_shapes=output_shapes,
                           attr_code=attr_code)
        return results[0], results[1]


class ConvACL:

    def __call__(self,
                 x,
                 weight,
                 bias=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        if bias is None:
            return _ConvACLNoBias()(x, weight, bias, stride, padding,
                                    dilation, groups)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")

        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        output_shape = _conv_output_shape(x, weight, stride, padding, dilation)
        attr_code = _conv_attr_code(stride, padding, dilation, groups, "conv2d")
        grad_attr_code = _conv_attr_code(
            stride, padding, dilation, groups, "conv2dbackward")

        return conv_cmd(
            "Conv2d",
            [x, weight, bias],
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attr_code=attr_code,
            multi_grad_src=f"""
            // aclop
            Conv2dBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(in1, true);
            op.add(in2, true);
            op.add(out0, false);
            op.add(out1, false);
            op.add(out2, false);
            {grad_attr_code}
            op.run();
            """,
        )[0]
