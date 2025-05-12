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


def pool_cmd(name: str,
             inputs: list,
             output_dtypes: list = None,
             output_shapes: list = None,
             attr_code: str = "",
             attr_header: str = "",
             outputs: list = None):
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

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
   
    // aclop
    {name}OpRunner op;
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")


class PoolACL(jt.Function):

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=None,
                 return_indices=None,
                 ceil_mode=False,
                 count_include_pad=True,
                 op='maximum'):
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding,
                                                                   padding)
        dilation = dilation if dilation else 1
        assert dilation == 1
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,
                                                                      dilation)
        for item in self.kernel_size:
            if item <= 0:
                raise RuntimeError(
                    f"kernel_size must be greater than zero, but got {item}")
        for item in self.stride:
            if item <= 0:
                raise RuntimeError(
                    f"stride must be greater than zero, but got {item}")
        for item in self.padding:
            if item < 0:
                raise RuntimeError(
                    f"padding must be non-negative, but got {item}")
        self.op = op
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, input):
        self.input = input
        attr_code = f"""
        op.jt_name  = "{"avgpool" if self.op == 'mean' else "maxpool"}";
        PoolAttr *attr = new PoolAttr();
        attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
        attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
        attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
        attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
        attr->poolCeil = {"true" if self.ceil_mode else "false"};
        attr->countIncludePad = {"true" if self.count_include_pad else "false"};
        op.op_attr.reset(attr);
        """
        input_height, input_width = input.shape[-2:]
        kernel_height, kernel_width = self.kernel_size[-2:]

        output_height = (input_height + 2 * self.padding[0] -
                         (kernel_height - 1) - 1) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] -
                        (kernel_width - 1) - 1) // self.stride[1] + 1

        output_shape = (input.shape[0], input.shape[1], output_height,
                        output_width)

        inputs = [input]

        if self.op == 'maximum':
            result = pool_cmd(
                "Maxpool",
                inputs,
                output_dtypes=[input.dtype, 'int32'],
                output_shapes=[output_shape, output_shape],
                attr_code=attr_code,
            )
        elif self.op == 'mean':
            result = pool_cmd(
                "Avgpool",
                inputs,
                output_dtypes=[input.dtype],
                output_shapes=[output_shape],
                attr_code=attr_code,
            )
        else:
            raise ValueError('no this type pool')

        if self.op == 'maximum':
            self.index = result[1]

        if self.return_indices:
            return result[0], result[1]
        else:
            return result[0]

    def grad(self, grad_output):
        input = self.input
        attr_code = f"""
        op.jt_name = "{"avgpoolbackward" if self.op == 'mean' else "maxpoolbackward"}";
        PoolAttr *attr = new PoolAttr();
        attr->kernel_size = {{ {self.kernel_size[0]}, {self.kernel_size[1]} }};
        attr->poolStrides = {{ {self.stride[0]}, {self.stride[1]} }};
        attr->poolPads = {{ {self.padding[0]}, {self.padding[1]} }};
        attr->poolDilations = {{ {self.dilation[0]}, {self.dilation[1]} }};
        attr->poolCeil = {"true" if self.ceil_mode else "false"};
        attr->countIncludePad = {"true" if self.count_include_pad else "false"};
        op.op_attr.reset(attr);
        """
        output_shapes = [input.shape]
        output_dtypes = [input.dtype]
        if self.op == 'maximum':
            result = pool_cmd("MaxpoolBackward",
                              inputs=[grad_output, input, self.index],
                              output_dtypes=output_dtypes,
                              output_shapes=output_shapes,
                              attr_code=attr_code)[0]
        elif self.op == 'mean':
            result = pool_cmd("AvgpoolBackward",
                              inputs=[grad_output, input],
                              output_dtypes=output_dtypes,
                              output_shapes=output_shapes,
                              attr_code=attr_code)[0]
        else:
            raise ValueError('no this type pool')
        return result
