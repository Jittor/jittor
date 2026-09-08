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


from ._code import acl_code as stack_cmd


class StackACL(jt.Function):
    def __init__(self):
        super(StackACL, self).__init__()

    def execute(self, input_tensors, dim):
        if type(input_tensors) is tuple:
            input_tensors = list(input_tensors)
        assert type(input_tensors) is list
        assert -1 * len(input_tensors) - 1 <= dim and dim <= len(input_tensors)
        for i in range(len(input_tensors)):
            if input_tensors[i].dtype != input_tensors[0].dtype:
                raise ValueError("All input tensors must have the same dtype")
            if input_tensors[i].shape != input_tensors[0].shape:
                raise ValueError("All input tensors must have the same shape")
        self.input = input_tensors
        input_shape = list(input_tensors[0].shape)
        output_shape = input_shape[:dim] + [len(input_tensors)] + input_shape[dim:]
        attr_code = code_program(
            [
                '\n        op.jt_name = "stack";\n        ',
                attribute_program(
                    "Stack", {"tensorNum": len(input_tensors), "dim": dim}, variable="op"
                ),
                "\n        ",
            ]
        )
        self.attr_code = attr_code
        result = stack_cmd(
            "Stack",
            input_tensors,
            output_dtypes=[input_tensors[0].dtype],
            output_shapes=[output_shape],
            attr_code=self.attr_code,
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

        attr_code = code_program(
            [
                '\n        op.jt_name = "splitwithsize";\n        ',
                attribute_program(
                    "SplitWithSize", {"splitSize": list(offset), "dim": axis}, variable="op"
                ),
                "\n        ",
            ]
        )

        result = stack_cmd(
            "SplitWithSize",
            [grad_output],
            output_dtypes=dtypeVec,
            output_shapes=shapeVec,
            attr_code=attr_code,
        )
        return result
