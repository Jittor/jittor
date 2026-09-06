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


from ._code import acl_code as silu_cmd
from ._code import check_acl_float_dtype


class SiLUACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        check_acl_float_dtype(x, "silu")
        attr_code = """
        op.jt_name = "silu";
        """
        result = silu_cmd("SiLU",
                          inputs=[x],
                          output_dtypes=[x.dtype],
                          output_shapes=[x.shape],
                          attr_code=attr_code,
                          cuda_grad_src=['''
// aclop
SiLUBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "silubackward";
op.run();
'''])[0]
        return result


class SwishACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        result = silu_cmd(
            "Swish",
            inputs=[x],
            output_dtypes=[x.dtype],
            output_shapes=[x.shape],
            attr_code='op.jt_name = "swish";',
            cuda_grad_src=['''
// aclop
SwishBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "swishbackward";
op.run();
'''],
        )[0]
        return result


class SwiGluACL:

    def __call__(self, x, dim=-1):
        return self.execute(x, dim)

    def execute(self, x, dim=-1):
        axis = int(dim) % int(x.ndim)
        output_shape = list(x.shape)
        output_shape[axis] //= 2
        return silu_cmd(
            "SwiGlu",
            inputs=[x],
            output_dtypes=[x.dtype],
            output_shapes=[output_shape],
            attr_code=f'''
op.jt_name = "swiglu";
op.dim = {axis};
''',
        )[0]
