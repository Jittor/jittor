import os
import jittor_utils
from jittor_utils import env_or_try_find
import ctypes
import glob
import jittor as jt
import jittor.compiler as compiler
from jittor.extern.acl.acl_compiler import acl_cmd_forward
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple([x] * n)

    return parse


_pair = _ntuple(2)


def conv_forward(name: str,
                 inputs: list,
                 output_dtypes: list = None,
                 output_shapes: list = None,
                 attr_code: str = "",
                 attr_header: str = "",
                 outputs: list = None,
                 extra_data: dict = {}):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
    // aclop
    ConvOpRunner op;
    {input_code}
    op.add(out0, false);
    {attr_code}
    op.run();""",
                   data=extra_data)


def conv_forward(name: str,
                 inputs: list,
                 output_dtypes: list = None,
                 output_shapes: list = None,
                 attr_code: str = "",
                 attr_header: str = "",
                 outputs: list = None,
                 extra_data: dict = {}):
    # TODO: not done for now
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
    // aclop
    ConvOpRunner op;
    {input_code}
    op.add(out0, false);
    {attr_code}
    op.run();""",
                   data=extra_data)


class ConvACL(jt.Function):

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
        self.bias = bias
        padding = _pair(padding)
        stride = _pair(stride)
        dilation = _pair(dilation)
        out_channels = weight.shape[0]
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        attr_code = f"""
            op.jt_name = "conv2d";
            ConvAttr *attr = new ConvAttr();
            attr->convStrides = {{ {stride[0]}, {stride[1]} }};
            attr->convPads = {{ {padding[0]}, {padding[1]} }};
            attr->convDilations = {{ {dilation[0]}, {dilation[1]} }};
            attr->group = {groups};
            attr->convOutPads = {{1,1}};
            op.op_attr.reset(attr);
            """
        input_height, input_width = x.shape[-2:]
        kernel_height, kernel_width = weight.shape[-2:]

        output_height = (input_height + 2 * padding[0] - dilation[0] *
                         (kernel_height - 1) - 1) // stride[0] + 1
        output_width = (input_width + 2 * padding[1] - dilation[1] *
                        (kernel_width - 1) - 1) // stride[1] + 1

        output_shape = (x.shape[0], out_channels, output_height, output_width)

        inputs = [x, weight]
        if bias is not None:
            inputs.append(bias)
        result = conv_forward(
            "Conv2d",
            inputs,
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attr_code=attr_code,
        )[0]
        return result

    def grad(self, grad_output):
        x = self.input
        weight = self.weight
        bias = self.bias
        inputs = [grad_output, x, weight]
        if bias is not None:
            inputs.append(bias)
        output_shapes = [x.shape, weight.shape]
        output_dtypes = [x.dtype, weight.dtype]
        if bias is not None:
            output_shapes.append(bias.shape)
            output_dtypes.append(bias.dtype)
        else:
            output_shapes.append([1])
            output_dtypes.append(x.dtype)
        padding = self.padding
        stride = self.stride
        dilation = self.dilation
        groups = self.groups
        attr_code = f"""
            op.jt_name = "conv2dbackward";
            ConvAttr *attr = new ConvAttr();
            attr->convStrides = {{ {stride[0]}, {stride[1]} }};
            attr->convPads = {{ {padding[0]}, {padding[1]} }};
            attr->convDilations = {{ {dilation[0]}, {dilation[1]} }};
            attr->group = {groups};
            attr->convOutPads = {{ 1,1}};
            op.op_attr.reset(attr);
            """
        results = acl_cmd_forward("Conv2dBackward",
                                  inputs,
                                  output_dtypes=output_dtypes,
                                  output_shapes=output_shapes,
                                  attr_code=attr_code)
        if self.bias is None:
            return results[0], results[1]

        return results
