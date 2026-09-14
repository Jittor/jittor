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


from ._code import acl_emit, acl_program
from ._code import check_acl_float_dtype
from ._attributes import attribute_program

_SILU_ATTR_CODE = """
        op.jt_name = "silu";
        """

_SILU_GRAD_SRC = '''
// aclop
SiLUBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "silubackward";
op.run();
'''

_SWISH_GRAD_SRC = '''
// aclop
SwishBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "swishbackward";
op.run();
'''

#: These three programs are fixed by their runner alone -- SwiGlu by its axis
#: as well -- so they are assembled once instead of having `acl_code` rebuild
#: the same cache key from the same strings on every activation.
_PROGRAMS = {}


def _program(name, key, **kwargs):
    program = _PROGRAMS.get(key)
    if program is None:
        program = acl_program(name, 1, 1, **kwargs)
        _PROGRAMS[key] = program
    return program


class SiLUACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        check_acl_float_dtype(x, "silu")
        program = _program("SiLU", "silu", attr_code=_SILU_ATTR_CODE,
                           cuda_grad_src=[_SILU_GRAD_SRC])
        return acl_emit(program, [x], [x.dtype], [x.shape])[0]


class SwishACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        program = _program("Swish", "swish", attr_code='op.jt_name = "swish";',
                           cuda_grad_src=[_SWISH_GRAD_SRC])
        return acl_emit(program, [x], [x.dtype], [x.shape])[0]


class SwiGluACL:

    def __call__(self, x, dim=-1):
        return self.execute(x, dim)

    def execute(self, x, dim=-1):
        axis = int(dim) % int(x.ndim)
        output_shape = list(x.shape)
        output_shape[axis] //= 2
        program = _program("SwiGlu", ("swiglu", axis), attributes={"dim": axis})
        return acl_emit(program, [x], [x.dtype], [output_shape])[0]
