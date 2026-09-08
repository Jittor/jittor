from ._code import code_with_attributes
from ._attributes import runner_for_alias
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


from ._code import acl_code as stack_cmd


class StackACL(jt.Function):
    def __init__(self):
        super(StackACL, self).__init__()

    def execute(self, input_tensors, dim):
        if type(input_tensors) is tuple:
            input_tensors = list(input_tensors)
        if not isinstance(input_tensors, list):
            raise TypeError("Stack expects a list or tuple of tensors")
        if not (-len(input_tensors) - 1 <= dim <= len(input_tensors)):
            raise ValueError("Stack dimension is out of range")
        for i in range(len(input_tensors)):
            if input_tensors[i].dtype != input_tensors[0].dtype:
                raise ValueError("All input tensors must have the same dtype")
            if input_tensors[i].shape != input_tensors[0].shape:
                raise ValueError("All input tensors must have the same shape")
        self.input = input_tensors
        self.dim = dim
        input_shape = list(input_tensors[0].shape)
        output_shape = input_shape[:dim] + [len(input_tensors)] + input_shape[dim:]
        result = stack_cmd(
            "Stack",
            input_tensors,
            output_dtypes=[input_tensors[0].dtype],
            output_shapes=[output_shape],
            attributes={"tensorNum": len(input_tensors), "dim": dim},
        )[0]
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

        result = stack_cmd(
            "SplitWithSize",
            [grad_output],
            output_dtypes=dtypeVec,
            output_shapes=shapeVec,
            attributes={"splitSize": list(offset), "dim": axis},
        )
        return result
