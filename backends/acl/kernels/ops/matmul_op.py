from ._code import acl_emit, acl_program, code_with_attributes
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
    return {"mode": mode, "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0}


_RESHAPE_GRAD_X2 = """
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

_RESTORE_GRAD_X2 = """
in0->shape = in0_shape;
dout->shape = dout_shape;
"""

#: One assembled program per (transpose, flattened-grad, HF32) combination.
#: `jt.acl_allow_hf32` decides `cube_math_type` and a caller may flip it
#: between two matmuls, so it is a key component and never read in the
#: builder: a program cached under the old flag would keep launching the old
#: arithmetic.
_MATMUL_PROGRAMS = {}


def _matmul_program(trans_x2, reshape_grad_x2, cube_math_type):
    key = (trans_x2, reshape_grad_x2, cube_math_type)
    program = _MATMUL_PROGRAMS.get(key)
    if program is None:
        program = _build_matmul_program(trans_x2, reshape_grad_x2, cube_math_type)
        _MATMUL_PROGRAMS[key] = program
    return program


def _build_matmul_program(trans_x2, reshape_grad_x2, cube_math_type):
    if trans_x2:
        grad_x2_lhs = "dout"
        grad_x2_rhs = "in0"
    else:
        grad_x2_lhs = "in0"
        grad_x2_rhs = "dout"
    reshape_code = _RESHAPE_GRAD_X2 if reshape_grad_x2 else ""
    restore_code = _RESTORE_GRAD_X2 if reshape_grad_x2 else ""
    return acl_program(
        "MatMul",
        2,
        1,
        attributes={"mode": 1 if trans_x2 else 0, "cube_math_type": cube_math_type},
        cuda_grad_src=[
            code_program(
                [
                    "\n// aclop\nMatMulOpRunner op;\nop.add(dout, true);\nop.add(in1, true);\nop.add(out0, false);\n",
                    attribute_program(
                        "MatMul",
                        {"mode": 0 if trans_x2 else 1, "cube_math_type": cube_math_type},
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
    )


class MatmulACL:
    def __init__(self, trans_x2=False):
        self.trans_x2 = trans_x2

    def __call__(self, x1, x2):
        return self.execute(x1, x2)

    def execute(self, x1, x2):
        trans_x2 = self.trans_x2
        program = _matmul_program(
            trans_x2,
            len(x1) != len(x2),
            1 if getattr(jt, "acl_allow_hf32", False) else 0,
        )
        return acl_emit(
            program,
            [x1, x2],
            [x1.dtype],
            [x1.shape[:-1] + x2.shape[-2:-1] if trans_x2 else x1.shape[:-1] + x2.shape[-1:]],
        )[0]
