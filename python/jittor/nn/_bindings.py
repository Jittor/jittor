"""Install neural-network convenience methods on :class:`jittor.Var`."""

import jittor as jt

from .functional.activation import hardsigmoid, hardswish, prelu, rrelu
from .functional.autograd import backward
from .functional.complex import _var_angle, _var_imag, _var_real
from .functional.matrix import matmul
from .functional.softmax import log_sigmoid, log_softmax, logsumexp, softmax


def _imatmul(left, right):
    return left.assign(matmul(left, right))


_REAL_PROPERTY = property(_var_real)
_IMAG_PROPERTY = property(_var_imag)


def install_var_bindings():
    """Install the stable public method bindings; repeated calls are harmless."""
    jt.Var.matmul = matmul
    jt.Var.__matmul__ = matmul
    jt.Var.__imatmul__ = _imatmul
    jt.Var.prelu = prelu
    jt.Var.hardswish = hardswish
    jt.Var.hardsigmoid = hardsigmoid
    jt.Var.rrelu = rrelu
    jt.Var.softmax = softmax
    jt.Var.log_softmax = log_softmax
    jt.Var.log_sigmoid = log_sigmoid
    jt.Var.logsumexp = logsumexp
    jt.Var.backward = backward
    jt.Var.real = _REAL_PROPERTY
    jt.Var.imag = _IMAG_PROPERTY
    jt.Var.angle = _var_angle
