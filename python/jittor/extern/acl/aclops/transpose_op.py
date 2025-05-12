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


def transpose_cmd(name: str,
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


class TransPoseACL(jt.Function):

    def __init__(self):
        super(TransPoseACL, self).__init__()

    def execute(self, x, *dim):
        self.input = x
        if len(dim) == 1 and isinstance(dim[0], Sequence):
            dim = dim[0]
        elif len(dim) == 2:
            axes = list(range(x.ndim))
            a, b = dim
            axes[a], axes[b] = axes[b], axes[a]
            dim = axes

        attr_code = f"""
        op.jt_name = "transpose";
        ReduceAttr *attr = new ReduceAttr();
        attr->axes = {{ {", ".join(map(str, dim))} }};
        op.op_attr.reset(attr);
        """
        # calculate output shape
        output_shape = [x.shape[i] for i in dim]
        output = transpose_cmd("Transpose", [x],
                               output_dtypes=[x.dtype],
                               output_shapes=[jt.empty(output_shape).shape],
                               attr_code=attr_code)[0]
        self.dim = dim
        return output

    def grad(self, grad_output):
        dim = list(range(grad_output.ndim))
        for i, p in enumerate(self.dim):
            dim[p] = i
        output_shape = [grad_output.shape[i] for i in dim]
        attr_code = f"""
        op.jt_name = "transpose";
        ReduceAttr *attr = new ReduceAttr();
        attr->axes = {{ {", ".join(map(str, dim))} }};
        op.op_attr.reset(attr);
        """
        output = transpose_cmd("Transpose", [grad_output],
                               output_dtypes=[grad_output.dtype],
                               output_shapes=[jt.empty(output_shape).shape],
                               attr_code=attr_code)[0]
        return output
