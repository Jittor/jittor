from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt

from ._code import check_acl_float_dtype
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable


from ._code import acl_code as sigmoid_cmd


class SigmoidACL(jt.Function):
    def __init__(self):
        super(SigmoidACL, self).__init__()

    def execute(self, x):
        check_acl_float_dtype(x, "sigmoid")
        inputs = [x]
        outputs = [jt.empty(x.shape, x.dtype)]
        attr_code = f"""
        op.jt_name = "sigmoid";
        """
        result = sigmoid_cmd("Sigmoid", inputs=inputs, outputs=outputs, attr_code=attr_code)[0]
        self.output = result
        return result

    def grad(self, grad_output):
        attr_code = f"""
        op.jt_name = "sigmoidbackward";
        """
        inputs = [grad_output, self.output]
        outputs = [jt.empty(grad_output.shape, grad_output.dtype)]
        grad_input = sigmoid_cmd(
            "SigmoidBackward", inputs=inputs, outputs=outputs, attr_code=attr_code
        )[0]
        return grad_input
