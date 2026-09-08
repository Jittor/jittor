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


from ._code import acl_code as floor_cmd


class FloorIntACL(jt.Function):
    def __init__(self):
        super(FloorIntACL, self).__init__()

    def execute(self, input):
        self.shape = input.shape
        result = floor_cmd(
            "Floor",
            [input],
            output_dtypes=[input.dtype],
            output_shapes=[input.shape],
            attr_code='op.jt_name="floor";',
        )[0]
        return result

    def grad(self, grad_output):
        return jt.zeros(self.shape, dtype=grad_output.dtype)
