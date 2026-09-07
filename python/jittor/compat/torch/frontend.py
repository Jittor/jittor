"""Python frontend types sharing the native VarHolder payload and graph."""

from contextlib import contextmanager
from functools import wraps


@contextmanager
def tensor_frontend(tensor_type):
    backend = vars(tensor_type).get("_frontend_backend")
    if backend is None:
        yield
        return
    token = backend.core._set_tensor_frontend_type(tensor_type)
    try:
        with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
            yield
    finally:
        backend.core._reset_tensor_frontend_type(token)


def frontend_factory(function, tensor_type):
    if "_frontend_backend" not in vars(tensor_type):
        return function

    @wraps(function)
    def scoped(*args, **kwargs):
        with tensor_frontend(tensor_type):
            return function(*args, **kwargs)

    return scoped


class _TensorMeta(type):
    def __call__(cls, *args, **kwargs):
        backend = vars(cls).get("_frontend_backend")
        if backend is None:
            return super().__call__(*args, **kwargs)
        if kwargs:
            raise TypeError("Tensor constructor does not accept keyword arguments")
        from .nested import _TorchSize
        with tensor_frontend(cls):
            if not args:
                result = backend.empty((0,), dtype="float32")
            elif all(isinstance(arg, int) for arg in args):
                result = backend.empty(tuple(args), dtype="float32")
            elif len(args) != 1:
                raise TypeError("Tensor expects data or integer dimensions")
            elif isinstance(args[0], (backend.NanoVector, _TorchSize)):
                result = backend.empty(tuple(args[0]), dtype="float32")
            else:
                result = backend.array(args[0]).float32()
            result.requires_grad = False
            return result


def make_tensor_type(backend):
    """Create the installation's real type without writing to native Var."""
    return _TensorMeta("Tensor", (backend.Var,), {
        "__module__": "torch",
        "__slots__": ("__weakref__",),
        "_frontend_backend": backend,
        "_frontend_autograd_policy": 3,
    })
