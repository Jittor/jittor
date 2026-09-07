"""Python frontend types sharing the native VarHolder payload and graph."""

from contextlib import contextmanager
from functools import wraps


@contextmanager
def tensor_frontend(tensor_type):
    backend = getattr(tensor_type, "_frontend_backend", None)
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


def make_parameter_type(backend, tensor_type):
    """Create a real tensor subtype; construction makes a new graph leaf."""
    class Parameter(tensor_type):
        __slots__ = ()
        _frontend_result_type = tensor_type
        _torch_compat_type = True

        def __new__(cls, data=None, requires_grad=True):
            with tensor_frontend(cls):
                if data is None:
                    value = backend.empty((0,), dtype="float32")
                else:
                    source = data if isinstance(data, backend.Var) else backend.array(data)
                    value = backend.Var.detach(source)
            value._is_torch_parameter = True
            value.requires_grad = bool(requires_grad)
            return value

        def __init__(self, data=None, requires_grad=True):
            # The native holder was already initialized by the allocation
            # converter. Python subclasses still receive their real __init__.
            pass

    Parameter.__module__ = "torch.nn.parameter"
    Parameter.__qualname__ = "Parameter"
    return Parameter


def reduce_tensor(value):
    return (
        rebuild_tensor,
        (type(value), value.numpy(), str(value.dtype), value.requires_grad),
        value.__dict__.copy(),
    )


def rebuild_tensor(tensor_type, array, dtype, requires_grad):
    # Rebuild the holder without rerunning an application subclass's __init__;
    # pickle restores its Python state after this object has entered the memo.
    backend = tensor_type._frontend_backend
    with tensor_frontend(tensor_type):
        value = backend.array(array, dtype=dtype)
        value.requires_grad = requires_grad
    return value


def deepcopy_tensor(value, memo):
    from copy import deepcopy
    result = rebuild_tensor(type(value), value.numpy(), str(value.dtype), value.requires_grad)
    memo[id(value)] = result
    result.__dict__.update(deepcopy(value.__dict__, memo))
    return result
