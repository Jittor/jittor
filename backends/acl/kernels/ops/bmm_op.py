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


from ._code import acl_code as acl_cmd


def _matmul_attributes(mode):
    return attribute_program(
        "BatchMatMul",
        {"mode": mode, "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0},
    )


#: One assembled program per (mode, HF32) pair. `jt.acl_allow_hf32` decides
#: `cube_math_type` and a caller may flip it between two products, so it is a
#: key component rather than something the builder reads for itself.
_BMM_PROGRAMS = {}


def _bmm_program(mode):
    cube_math_type = 1 if getattr(jt, "acl_allow_hf32", False) else 0
    key = (mode, cube_math_type)
    program = _BMM_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "BatchMatMul",
            2,
            1,
            attr_code=attribute_program(
                "BatchMatMul", {"mode": mode, "cube_math_type": cube_math_type}
            ),
        )
        _BMM_PROGRAMS[key] = program
    return program


class BmmACL(jt.Function):
    def __init__(self, trans_x2=False):
        super(BmmACL, self).__init__()
        self.trans_x2 = trans_x2

    def execute(self, x1, x2):
        self.input = [x1, x2]
        result = acl_emit(
            _bmm_program(1 if self.trans_x2 else 0),
            [x1, x2],
            [x1.dtype],
            [
                x1.shape[:-1] + x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] + x2.shape[-1:]
            ],
        )[0]

        return result

    def grad(self, grad_output):
        x1, x2 = self.input
        if len(x1) != len(x2):
            reshape_grad_x2 = True
        else:
            reshape_grad_x2 = False
        grad_x1 = acl_emit(
            _bmm_program(0 if self.trans_x2 else 1),
            [grad_output, x2],
            [x1.dtype],
            [
                grad_output.shape[:-1] + x2.shape[-2:-1]
                if not self.trans_x2
                else grad_output.shape[:-1] + x1.shape[-1:]
            ],
        )[0]
        if self.trans_x2:
            if reshape_grad_x2:
                output_shape = grad_output.shape[1:-2] + grad_output.shape[-1:] + x1.shape[-1:]
                grad_x2 = acl_emit(
                    _bmm_program(2),
                    [grad_output.reshape(-1, grad_output.shape[-1]), x1.reshape(-1, x1.shape[-1])],
                    [x2.dtype],
                    [output_shape],
                )[0]
            else:
                output_shape = grad_output.shape[:-2] + grad_output.shape[-1:] + x1.shape[-1:]
                grad_x2 = acl_emit(
                    _bmm_program(2),
                    [grad_output, x1],
                    [x2.dtype],
                    [output_shape],
                )[0]
        else:
            if reshape_grad_x2:
                output_shape = x1.shape[1:-2] + x1.shape[-1:] + grad_output.shape[-1:]
                grad_x2 = acl_emit(
                    _bmm_program(2),
                    [x1.reshape(-1, x1.shape[-1]), grad_output.reshape(-1, grad_output.shape[-1])],
                    [x2.dtype],
                    [output_shape],
                )[0]
            else:
                output_shape = x1.shape[:-2] + x1.shape[-1:] + grad_output.shape[-1:]
                grad_x2 = acl_emit(
                    _bmm_program(2),
                    [x1, grad_output],
                    [x2.dtype],
                    [output_shape],
                )[0]
        if len(grad_x1.shape) > len(x1.shape):
            grad_x1 = grad_x1.sum(0)
        if len(grad_x2.shape) > len(x2.shape):
            grad_x2 = grad_x2.sum(0)
        return grad_x1, grad_x2
