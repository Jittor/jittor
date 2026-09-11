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


#: A python ``complex`` and a numpy complex scalar are the only operands the
#: wrapper below has to convert. Naming the pair once keeps a global lookup, an
#: attribute lookup and a tuple build out of every ``+``, ``-``, ``*`` and
#: ``/`` whose right operand is not a Var -- a float or an int scalar reaches
#: the check too, and there is no cheaper way to keep the conversion: a python
#: complex may meet a *float32* Var (``1j * x``), so nothing about the Vars in
#: play says in advance that the wrapper is unnecessary.
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)


def _install_complex_scalar_binary_bindings():
    if getattr(jt.Var, "_native_complex_scalar_binary", False):
        return

    var_type = jt.Var

    def wrap(name):
        native = getattr(jt.Var, name)

        def binary(self, other):
            # Var-with-Var is the overwhelmingly common case and it sits on the
            # hot path of every model: this wrapper runs for each +, -, * and /
            # in the graph, so settle it before the complex-scalar check.
            if other.__class__ is not var_type and isinstance(
                    other, _COMPLEX_SCALAR_TYPES):
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
