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


from ._code import acl_code as range_forward


class IndexACL(jt.Function):
    def __init__(self):
        super(IndexACL, self).__init__()

    def execute(self, inshape: list, dim=None, dtype="int32"):
        # zeros a tensor, shape is inshape, dtype is dtype
        dim_input = dim
        if dim == None:
            dim = [i for i in range(len(inshape))]
        elif type(dim) == int:
            dim = [dim]
        results = []

        for i, d in enumerate(dim):
            max_len = inshape[d]

            tmp = jt.zeros(max_len, dtype=dtype)
            range_attr_code = attribute_program("Range", {"start": 0, "end": max_len, "step": 1})
            result = range_forward(
                "Range",
                [],
                output_dtypes=[tmp.dtype],
                output_shapes=[tmp.shape],
                attr_code=range_attr_code,
            )[0]
            broadcast_dims = list(range(len(inshape)))
            broadcast_dims.remove(d)
            result = jt.broadcast(result, shape=inshape, dims=broadcast_dims)
            results.append(result)

        if len(results) != 1 or dim_input == None:
            return tuple(results)
        elif len(results) == 1 and dim_input != None:
            return results[0]
        else:
            return results

    def grad(self, grad_output):
        return grad_output
