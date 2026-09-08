from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
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


from ._code import acl_code as matmul_forward


def _matmul_attributes(mode):
    return attribute_program(
        "MatMul", {"mode": mode, "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0}
    )


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
            "MatMul",
            [x1, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                x1.shape[:-1] + x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] + x2.shape[-1:]
            ],
            attr_code=_matmul_attributes(1) if self.trans_x2 else _matmul_attributes(0),
            cuda_grad_src=[
                code_program(
                    [
                        "\n// aclop\nMatMulOpRunner op;\nop.add(dout, true);\nop.add(in1, true);\nop.add(out0, false);\n",
                        attribute_program(
                            "MatMul",
                            {"mode": 0 if self.trans_x2 else 1, "cube_math_type": cube_math_type},
                            slot="matmul_grad_x1",
                        ),
                        "\nop.run();\n",
                    ]
                ),
                code_program(
                    [
                        "\n// aclop\n",
                        reshape_code,
                        "\nMatMulOpRunner op;\nop.add(",
                        grad_x2_lhs,
                        ", true);\nop.add(",
                        grad_x2_rhs,
                        ", true);\nop.add(out0, false);\n",
                        attribute_program(
                            "MatMul",
                            {"mode": 2, "cube_math_type": cube_math_type},
                            slot="matmul_grad_x2",
                        ),
                        "\nop.run();\n",
                        restore_code,
                    ]
                ),
            ],
        )[0]
        return result
