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


from ._code import acl_code as nantonum_cmd


class NanToNumACL(jt.Function):
    def __init__(self):
        super(NanToNumACL, self).__init__()

    def execute(self, input, nan_or_inf):
        attr_code = code_program(
            [
                '\n        op.jt_name = "NanToNum";\n        ',
                attribute_program(
                    "NanToNum",
                    {"nan": nan_or_inf, "posinf": -nan_or_inf, "neginf": -nan_or_inf},
                    variable="op",
                ),
                "\n        ",
            ]
        )
        self.attr_code = attr_code
        result = nantonum_cmd(
            "NanToNum",
            [input],
            output_dtypes=[input[0].dtype],
            output_shapes=[input.shape],
            attr_code=self.attr_code,
        )[0]
        return result
