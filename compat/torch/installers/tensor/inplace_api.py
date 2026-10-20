"""Augmented Tensor assignment through the compatibility write owner."""

import operator

from jittor._core.dtypes import dtype_name

from ...context import get_install_context
from ..numerical.sparse import _SparseCOO
from .method_api import _ip


def _inplace_binary(self, other, operation):
    backend = type(self)._frontend_backend
    if not backend.flags.no_grad and self.requires_grad and self.is_leaf:
        raise RuntimeError(
            "a leaf Variable that requires grad is being used in an in-place operation")
    if isinstance(other, _SparseCOO):
        other = other.to_dense()
    result = operation(self, other)
    if tuple(result.shape) != tuple(self.shape):
        raise RuntimeError(
            "output with shape %s doesn't match the broadcast shape %s"
            % (tuple(self.shape), tuple(result.shape)))
    namespace = get_install_context(backend).target_namespace
    if not namespace.can_cast(result.dtype, self.dtype):
        raise RuntimeError(
            "result type %s can't be cast to the desired output type %s"
            % (result.dtype, self.dtype))
    if result.dtype != self.dtype:
        result = result.cast(dtype_name(self.dtype))
    return _ip(self, result)


def _tensor_iadd(self, other):
    return _inplace_binary(self, other, operator.add)


def _tensor_isub(self, other):
    return _inplace_binary(self, other, operator.sub)


def _tensor_imul(self, other):
    return _inplace_binary(self, other, operator.mul)


def _tensor_itruediv(self, other):
    return _inplace_binary(self, other, operator.truediv)


_INPLACE_BINARY_APIS = {
    "__iadd__": _tensor_iadd,
    "__isub__": _tensor_isub,
    "__imul__": _tensor_imul,
    "__itruediv__": _tensor_itruediv,
}
