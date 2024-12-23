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


def matmul_forward(name: str,
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
    MatMulOpRunner op;
    {input_code}
    op.add(out0, false);
    {attr_code}
    op.run();""",
                   data=extra_data)


class MatmulACL(jt.Function):

    def __init__(self, trans_x2=False):
        super(MatmulACL, self).__init__()
        self.trans_x2 = trans_x2

    def execute(self, x1, x2):
        self.input = [x1, x2]
        result = matmul_forward(
            "MatMul", [x1, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                x1.shape[:-1] +
                x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] +
                x2.shape[-1:]
            ],
            attr_code="op.jt_name=\"matmul_trans_1\";"
            if self.trans_x2 else "op.jt_name=\"matmul\";")[0]
        return result

    def grad(self, grad_output):
        x1, x2 = self.input
        if len(x1) != len(x2):
            reshape_grad_x2 = True
        else:
            reshape_grad_x2 = False
        grad_x1 = matmul_forward(
            "MatMul", [grad_output, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                grad_output.shape[:-1] + x2.shape[-2:-1] if not self.trans_x2
                else grad_output.shape[:-1] + x2.shape[-1:]
            ],
            attr_code="op.jt_name=\"matmul_trans_1\";"
            if not self.trans_x2 else "op.jt_name=\"matmul\";")[0]

        if self.trans_x2:
            if reshape_grad_x2:
                output_shape = grad_output.shape[1:-2] + grad_output.shape[
                    -1:] + x1.shape[-1:]
                grad_x2 = matmul_forward(
                    "MatMul", [
                        grad_output.reshape(-1, grad_output.shape[-1]),
                        x1.reshape(-1, x1.shape[-1])
                    ],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code="op.jt_name=\"matmul_trans_0\";")[0]
            else:
                output_shape = grad_output.shape[:-2] + grad_output.shape[
                    -1:] + x1.shape[-1:]
                grad_x2 = matmul_forward(
                    "MatMul", [grad_output, x1],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code="op.jt_name=\"matmul_trans_0\";")[0]
        else:
            if reshape_grad_x2:
                output_shape = x1.shape[1:-2] + x1.shape[
                    -1:] + grad_output.shape[-1:]
                grad_x2 = matmul_forward(
                    "MatMul", [
                        x1.reshape(-1, x1.shape[-1]),
                        grad_output.reshape(-1, grad_output.shape[-1])
                    ],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code="op.jt_name=\"matmul_trans_0\";")[0]
            else:
                output_shape = x1.shape[:-2] + x1.shape[
                    -1:] + grad_output.shape[-1:]
                grad_x2 = matmul_forward(
                    "MatMul", [x1, grad_output],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code="op.jt_name=\"matmul_trans_0\";")[0]
        return grad_x1, grad_x2
