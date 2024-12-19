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

def stack_cmd(name: str,
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

class StackACL(jt.Function):

    def __init__(self):
        super(StackACL, self).__init__()

    def execute(self, input_tensors, dim):
        if type(input_tensors) is tuple:
            input_tensors = list(input_tensors)
        assert type(input_tensors) is list
        assert -1 * len(input_tensors) - 1 <= dim and dim <= len(
            input_tensors)
        for i in range(len(input_tensors)):
            if input_tensors[i].dtype != input_tensors[0].dtype:
                raise ValueError(
                    "All input tensors must have the same dtype")
            if input_tensors[i].shape != input_tensors[0].shape:
                raise ValueError(
                    "All input tensors must have the same shape")
        self.input = input_tensors
        input_shape = list(input_tensors[0].shape)
        output_shape = input_shape[:dim] + [len(input_tensors)
                                            ] + input_shape[dim:]
        attr_code = f"""
        op.jt_name = "stack";
        ConcatAttr *attr = new ConcatAttr();
        attr->tensorNum = {len(input_tensors)};
        attr->dim = {dim};
        op.op_attr.reset(attr);
        """
        self.attr_code = attr_code
        result = stack_cmd("Stack",
                            input_tensors,
                            output_dtypes=[input_tensors[0].dtype],
                            output_shapes=[output_shape],
                            attr_code=self.attr_code)[0]
        return result

    def grad(self, grad_output):
        grad_inputs = self.split_grad(grad_output, self.input, self.dim)
        return grad_inputs

    def split_grad(self, grad_output, input_tensors, axis):
        offset = []
        shapeVec = []
        dtypeVec = []
        for tensor in input_tensors:
            offset.append(tensor.shape[axis])
            dtypeVec.append(tensor.dtype)
            shapeVec.append(tensor.shape)

        attr_code = f"""
        op.jt_name = "splitwithsize";
        auto *attr = new SplitWithSizeAttr();
        attr->splitSize = {{ {", ".join(map(str, offset))} }};
        attr->dim = {axis};
        op.op_attr.reset(attr);
        """

        result = stack_cmd("SplitWithSize", [grad_output],
                            output_dtypes=dtypeVec,
                            output_shapes=shapeVec,
                            attr_code=attr_code)
        return result