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


def _batch_matmul(x1, x2, mode=0):
    """Normalize broadcast batches to the rank-three ACL BatchMatMul ABI."""
    shape1, shape2 = tuple(x1.shape), tuple(x2.shape)
    if len(shape1) < 2 or len(shape2) < 2:
        raise ValueError("BatchMatMul inputs must have at least two dimensions")
    batch_rank = max(len(shape1), len(shape2)) - 2
    batch1 = (1,) * (batch_rank - len(shape1) + 2) + shape1[:-2]
    batch2 = (1,) * (batch_rank - len(shape2) + 2) + shape2[:-2]
    batch = []
    for left, right in zip(batch1, batch2):
        if left != right and left != 1 and right != 1:
            raise ValueError("BatchMatMul batch dimensions cannot broadcast")
        batch.append(right if left == 1 else left)
    batch = tuple(batch)
    rows, inner1 = (shape1[-1], shape1[-2]) if mode == 2 else shape1[-2:]
    inner2, cols = (shape2[-1], shape2[-2]) if mode == 1 else shape2[-2:]
    if inner1 != inner2:
        raise ValueError("BatchMatMul contraction dimensions must match")
    count = math.prod(batch)
    inputs = []
    for value, shape in ((x1, shape1), (x2, shape2)):
        expanded = batch + shape[-2:]
        if shape != expanded:
            value = value.broadcast(expanded)
        inputs.append(value.reshape((count,) + shape[-2:]))
    result = acl_cmd(
        "BatchMatMul", inputs, output_dtypes=[x1.dtype],
        output_shapes=[(count, rows, cols)], attr_code=_matmul_attributes(mode),
    )[0]
    return result.reshape(batch + (rows, cols))


def _sum_batch_gradient(value, shape):
    shape = tuple(shape)
    padded = (1,) * (len(value.shape) - len(shape)) + shape
    axes = tuple(i for i, (actual, target) in enumerate(zip(value.shape, padded))
                 if target == 1 and actual != 1)
    if axes:
        value = value.sum(axes, keepdims=True)
    return value.reshape(shape)


class BmmACL(jt.Function):
    def __init__(self, trans_x2=False):
        super(BmmACL, self).__init__()
        self.trans_x2 = trans_x2

    def execute(self, x1, x2):
        self.input = [x1, x2]
        return _batch_matmul(x1, x2, 1 if self.trans_x2 else 0)

    def grad(self, grad_output):
        x1, x2 = self.input
        grad_x1 = _batch_matmul(grad_output, x2, 0 if self.trans_x2 else 1)
        grad_x2 = (_batch_matmul(grad_output, x1, 2) if self.trans_x2
                   else _batch_matmul(x1, grad_output, 2))
        return (_sum_batch_gradient(grad_x1, x1.shape),
                _sum_batch_gradient(grad_x2, x2.shape))
