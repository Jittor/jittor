"""Exact (erf) GELU through CANN's own kernel.

Jittor's portable definition is five elementwise nodes, and the ACL fused path
issues one aclnn launch per node, so every activation cost five launches on the
forward and more on the backward. aclnnGelu is the same function in one launch.
"""

import jittor as jt

from ._code import acl_emit, acl_program
from ._code import check_acl_float_dtype

_ATTR_CODE = """
        op.jt_name = "gelu";
        """

_GRAD_SRC = '''
// aclop
GeluBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "gelubackward";
op.run();
'''

#: Nothing about this program varies with the call, so it is assembled once
#: instead of having `acl_code` re-derive its cache key from the same strings
#: on every activation.
_PROGRAM = None


def _gelu_program():
    global _PROGRAM
    if _PROGRAM is None:
        _PROGRAM = acl_program("Gelu", 1, 1, attr_code=_ATTR_CODE,
                               cuda_grad_src=[_GRAD_SRC])
    return _PROGRAM


class GeluACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        check_acl_float_dtype(x, "gelu")
        return acl_emit(_gelu_program(), [x], [x.dtype], [x.shape])[0]
