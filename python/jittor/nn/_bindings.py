"""Install neural-network convenience methods on :class:`jittor.Var`."""

import jittor as jt
import numpy as np

from .functional.activation import hardsigmoid, hardswish, prelu, rrelu
from .functional.autograd import backward
from .functional.complex import _var_angle, _var_imag, _var_real
from .functional.matrix import matmul
from .functional.softmax import log_sigmoid, log_softmax, logsumexp, softmax


def _imatmul(left, right):
    return left.assign(matmul(left, right))


def _requires_grad_(value, requires_grad=True):
    value.requires_grad = bool(requires_grad)
    return value


def _install_complex_scalar_binary_bindings():
    if getattr(jt.Var, "_native_complex_scalar_binary", False):
        return

    def wrap(name):
        native = getattr(jt.Var, name)

        def binary(self, other):
            if isinstance(other, (complex, np.complexfloating)):
                other = jt.array(np.asarray([other], dtype=np.complex64))
            return native(self, other)

        binary.__name__ = name
        setattr(jt.Var, name, binary)

    for name in (
        "__add__", "__radd__", "__sub__", "__rsub__",
        "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    ):
        wrap(name)
    jt.Var._native_complex_scalar_binary = True


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
    jt.Var.requires_grad_ = _requires_grad_
    jt.Var.real = _REAL_PROPERTY
    jt.Var.imag = _IMAG_PROPERTY
    jt.Var.angle = _var_angle
    _install_complex_scalar_binary_bindings()
