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
                   extra_data: dict = {},
                   cuda_grad_src: list = None):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    if outputs is None:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    code_kwargs = dict(
        cuda_header=attr_header + cuda_header,
        cuda_grad_src=cuda_grad_src or [],
        cuda_src=f"""
    // aclop
    MatMulOpRunner op;
    {input_code}
    op.add(out0, false);
    {attr_code}
    op.cube_math_type = {1 if getattr(jt, "acl_allow_hf32", False) else 0};
    op.run();""",
        data=extra_data,
    )
    if outputs is not None:
        return jt.code(outputs=outputs, inputs=inputs, **code_kwargs)
    return jt.code(output_shapes, output_dtypes, inputs, **code_kwargs)


class MatmulACL:

    def __init__(self, trans_x2=False):
        self.trans_x2 = trans_x2

    def __call__(self, x1, x2):
        return self.execute(x1, x2)

    def execute(self, x1, x2):
        cube_math_type = 1 if getattr(jt, "acl_allow_hf32", False) else 0
        grad_x1_mode = "matmul" if self.trans_x2 else "matmul_trans_1"
        reshape_grad_x2 = len(x1) != len(x2)
        if self.trans_x2:
            grad_x2_lhs = "dout"
            grad_x2_rhs = "in0"
        else:
            grad_x2_lhs = "in0"
            grad_x2_rhs = "dout"
        reshape_code = ""
        restore_code = ""
        if reshape_grad_x2:
            reshape_code = """
auto in0_shape = in0->shape;
auto dout_shape = dout->shape;
NanoVector in0_flat_shape;
auto in0_last = in0->shape[in0->shape.size() - 1];
in0_flat_shape.push_back(in0->numel() / in0_last);
in0_flat_shape.push_back(in0_last);
in0->shape = in0_flat_shape;
NanoVector dout_flat_shape;
auto dout_last = dout->shape[dout->shape.size() - 1];
dout_flat_shape.push_back(dout->numel() / dout_last);
dout_flat_shape.push_back(dout_last);
dout->shape = dout_flat_shape;
"""
            restore_code = """
in0->shape = in0_shape;
dout->shape = dout_shape;
"""
        result = matmul_forward(
            "MatMul", [x1, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                x1.shape[:-1] +
                x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] +
                x2.shape[-1:]
            ],
            attr_code="op.jt_name=\"matmul_trans_1\";"
            if self.trans_x2 else "op.jt_name=\"matmul\";",
            cuda_grad_src=[f"""
// aclop
MatMulOpRunner op;
op.add(dout, true);
op.add(in1, true);
op.add(out0, false);
op.jt_name = "{grad_x1_mode}";
op.cube_math_type = {cube_math_type};
op.run();
""", f"""
// aclop
{reshape_code}
MatMulOpRunner op;
op.add({grad_x2_lhs}, true);
op.add({grad_x2_rhs}, true);
op.add(out0, false);
op.jt_name = "matmul_trans_0";
op.cube_math_type = {cube_math_type};
op.run();
{restore_code}
"""])[0]
        return result
