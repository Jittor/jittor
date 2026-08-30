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


from ._code import acl_code as transpose_cmd


class TransPoseACL:

    def __call__(self, x, *dim):
        return self.execute(x, *dim)

    def execute(self, x, *dim):
        if len(dim) == 0:
            # no-arg transpose reverses all dims (numpy/jittor .T semantics). Without this
            # dim stayed () -> output_shape [] -> aclnnPermute "output shape [1]" crash.
            dim = list(range(x.ndim))[::-1]
        elif len(dim) == 1 and isinstance(dim[0], Sequence):
            dim = dim[0]
        elif len(dim) == 2:
            axes = list(range(x.ndim))
            a, b = dim
            axes[a], axes[b] = axes[b], axes[a]
            dim = axes
        dim = list(dim)

        inverse_dim = list(range(x.ndim))
        for index, axis in enumerate(dim):
            inverse_dim[axis] = index

        attr_code = f"""
        op.jt_name = "transpose";
        ReduceAttr *attr = new ReduceAttr();
        attr->axes = {{ {", ".join(map(str, dim))} }};
        op.op_attr.reset(attr);
        """
        # calculate output shape
        output_shape = [x.shape[i] for i in dim]
        output = transpose_cmd("Transpose", [x],
                               output_dtypes=[x.dtype],
                               output_shapes=[output_shape],
                               attr_code=attr_code,
                               cuda_grad_src=[f"""
// aclop
TransposeOpRunner op;
op.add(dout, true);
op.add(out0, false);
op.jt_name = "transpose";
ReduceAttr *attr = new ReduceAttr();
attr->axes = {{ {", ".join(map(str, inverse_dim))} }};
op.op_attr.reset(attr);
op.run();
"""])[0]
        return output
