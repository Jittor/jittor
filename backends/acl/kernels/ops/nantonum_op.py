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
from .._soc import kernel_available


class NanToNumACL(jt.Function):
    def __init__(self):
        super(NanToNumACL, self).__init__()

    def execute(self, input, nan_or_inf):
        if not kernel_available("NanToNum"):
            # Same attribute mapping as the aclnn path: nan->nan_or_inf,
            # +-inf -> -nan_or_inf. Ascend310P comparison kernels are not IEEE
            # for non-finite operands (isnan is always False, isinf counts
            # nan, masked where/fill silently no-ops), but `x == +inf` matches
            # {+inf, nan} and `x == -inf` matches {-inf, nan} -- nan is
            # exactly the intersection. Writes go through nonzero + index
            # assignment (IndexPutImpl), which is verified on this SoC.
            eq_pos = (input == float("inf")).reshape(-1)
            eq_neg = (input == float("-inf")).reshape(-1)
            not_pos = eq_pos.logical_not()
            not_neg = eq_neg.logical_not()
            out = input + 0.0
            flat = out.reshape(-1)
            flat[(eq_pos & eq_neg).nonzero().reshape(-1)] = nan_or_inf
            flat[(eq_pos & not_neg).nonzero().reshape(-1)] = -nan_or_inf
            flat[(eq_neg & not_pos).nonzero().reshape(-1)] = -nan_or_inf
            return out
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
            output_dtypes=[input.dtype],
            output_shapes=[input.shape],
            attr_code=self.attr_code,
        )[0]
        return result
