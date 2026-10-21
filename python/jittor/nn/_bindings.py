"""Install neural-network convenience methods on :class:`jittor.Var`."""

import jittor as jt
import numpy as np

from .functional.activation import hardsigmoid, hardswish, prelu, rrelu
from .functional.autograd import backward
from .functional.complex import _var_angle, _var_imag, _var_real
from .functional.matrix import matmul
from .functional.softmax import log_sigmoid, log_softmax, logsumexp, softmax


def _matmul_operand(value):
    # The right operand of ``Var @ value``, as torch reads it: a Var as is, a
    # NumPy array converted (``Tensor @ ndarray`` is a Tensor), anything else
    # None -- handed back to Python so its ``__rmatmul__`` runs, or the usual
    # TypeError is raised. `matmul` used to be called with it regardless and
    # failed inside on ``b.shape``/``b.broadcast`` with an AttributeError.
    # ``ndarray @ Var`` stays a TypeError, as it is for a torch Tensor.
    if isinstance(value, jt.Var):
        return value
    if isinstance(value, np.ndarray):
        return jt.array(value)
    return None


def _var_matmul(left, right):
    right = _matmul_operand(right)
    if right is None:
        return NotImplemented
    return matmul(left, right)


def _imatmul(left, right):
    right = _matmul_operand(right)
    if right is None:
        return NotImplemented
    return left.assign(matmul(left, right))


def _requires_grad_(value, requires_grad=True):
    value.requires_grad = bool(requires_grad)
    return value


#: Complex scalar operands are converted by the native argument converter
#: (``ArrayArgs::from_py_object`` in ``src/bindings/pyjt/py_converter.h``),
#: not here. They used to go through a python wrapper installed over the
#: eight arithmetic operators, which meant a python frame on every ``+``,
#: ``-``, ``*`` and ``/`` in every model -- 88 of them per decode step of an
#: 8-layer transformer, 0.160 ms, 5.9% of the step -- to serve an operand
#: shape almost no model ever passes.


_REAL_PROPERTY = property(_var_real)
_IMAG_PROPERTY = property(_var_imag)


def install_var_bindings():
    """Install the stable public method bindings; repeated calls are harmless."""
    jt.Var.matmul = matmul
    jt.Var.__matmul__ = _var_matmul
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
    jt.Var.requires_grad_ = _requires_grad_
    jt.Var.real = _REAL_PROPERTY
    jt.Var.imag = _IMAG_PROPERTY
    jt.Var.angle = _var_angle
