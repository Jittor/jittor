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


from ._code import acl_code as acl_cmd


def _matmul_attributes(mode):
    return attribute_program(
        "BatchMatMul",
        {"mode": mode, "cube_math_type": 1 if getattr(jt, "acl_allow_hf32", False) else 0},
    )


class BmmACL(jt.Function):
    def __init__(self, trans_x2=False):
        super(BmmACL, self).__init__()
        self.trans_x2 = trans_x2

    def execute(self, x1, x2):
        self.input = [x1, x2]
        result = acl_cmd(
            "BatchMatMul",
            [x1, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                x1.shape[:-1] + x2.shape[-2:-1] if self.trans_x2 else x1.shape[:-1] + x2.shape[-1:]
            ],
            attr_code=_matmul_attributes(1) if self.trans_x2 else _matmul_attributes(0),
        )[0]

        return result

    def grad(self, grad_output):
        x1, x2 = self.input
        if len(x1) != len(x2):
            reshape_grad_x2 = True
        else:
            reshape_grad_x2 = False
        grad_x1 = acl_cmd(
            "BatchMatMul",
            [grad_output, x2],
            output_dtypes=[x1.dtype],
            output_shapes=[
                grad_output.shape[:-1] + x2.shape[-2:-1]
                if not self.trans_x2
                else grad_output.shape[:-1] + x1.shape[-1:]
            ],
            attr_code=_matmul_attributes(1) if not self.trans_x2 else _matmul_attributes(0),
        )[0]
        if self.trans_x2:
            if reshape_grad_x2:
                output_shape = grad_output.shape[1:-2] + grad_output.shape[-1:] + x1.shape[-1:]
                grad_x2 = acl_cmd(
                    "BatchMatMul",
                    [grad_output.reshape(-1, grad_output.shape[-1]), x1.reshape(-1, x1.shape[-1])],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code=_matmul_attributes(2),
                )[0]
            else:
                output_shape = grad_output.shape[:-2] + grad_output.shape[-1:] + x1.shape[-1:]
                grad_x2 = acl_cmd(
                    "BatchMatMul",
                    [grad_output, x1],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code=_matmul_attributes(2),
                )[0]
        else:
            if reshape_grad_x2:
                output_shape = x1.shape[1:-2] + x1.shape[-1:] + grad_output.shape[-1:]
                grad_x2 = acl_cmd(
                    "BatchMatMul",
                    [x1.reshape(-1, x1.shape[-1]), grad_output.reshape(-1, grad_output.shape[-1])],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code=_matmul_attributes(2),
                )[0]
            else:
                output_shape = x1.shape[:-2] + x1.shape[-1:] + grad_output.shape[-1:]
                grad_x2 = acl_cmd(
                    "BatchMatMul",
                    [x1, grad_output],
                    output_dtypes=[x2.dtype],
                    output_shapes=[output_shape],
                    attr_code=_matmul_attributes(2),
                )[0]
        if len(grad_x1.shape) > len(x1.shape):
            grad_x1 = grad_x1.sum(0)
        if len(grad_x2.shape) > len(x2.shape):
            grad_x2 = grad_x2.sum(0)
        return grad_x1, grad_x2
