"""Exact and tanh GELU through CANN's V2 kernels.

Jittor's portable definition is five elementwise nodes, and the ACL fused path
issues one aclnn launch per node, so every activation cost five launches on the
forward and more on the backward. V2 implements either requested algorithm in
one launch, preserving the algorithm choice in its backward program as well.
"""

import jittor as jt

from ._code import acl_emit, acl_program
from ._code import check_acl_float_dtype
from ._attributes import attribute_program, code_program

_GRAD_PREFIX = '''
// aclop
GeluBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
'''

#: An immutable program per algorithm. Forward and gradient attribute payloads
#: occupy different namespaces so an alternating none/tanh call cannot mix them.
_PROGRAMS = {}


def _gelu_program(approximate="none"):
    if approximate not in ("none", "tanh"):
        raise ValueError("GELU approximation must be 'none' or 'tanh'")
    program = _PROGRAMS.get(approximate)
    if program is None:
        attributes = {"approximate": int(approximate == "tanh")}
        backward = code_program([
            _GRAD_PREFIX,
            attribute_program("GeluBackward", attributes),
            "\nop.run();\n",
        ])
        program = acl_program("Gelu", 1, 1, attributes=attributes,
                              cuda_grad_src=[backward])
        _PROGRAMS[approximate] = program
    return program


class GeluACL:

    def __call__(self, x, approximate="none"):
        return self.execute(x, approximate)

    def execute(self, x, approximate="none"):
        check_acl_float_dtype(x, "gelu")
        return acl_emit(_gelu_program(approximate), [x], [x.dtype], [x.shape])[0]
