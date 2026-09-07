"""Python frontend types sharing the native VarHolder payload and graph."""

from contextlib import contextmanager
from functools import wraps


def _default_tensor_dtype(backend):
    from .tensor_state import compatibility_owner
    from .types import _dtype_to_str
    owner = compatibility_owner(backend)
    getter = getattr(owner, "get_default_dtype", None)
    # Types are created before core.extended publishes the default-dtype API.
    return _dtype_to_str(getter()) if getter is not None else "float32"


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
        dtype = _default_tensor_dtype(backend)
        with tensor_frontend(cls):
            if not args:
                result = backend.empty((0,), dtype=dtype)
            elif all(isinstance(arg, int) for arg in args):
                result = backend.empty(tuple(args), dtype=dtype)
            elif len(args) != 1:
                raise TypeError("Tensor expects data or integer dimensions")
            elif isinstance(args[0], (backend.NanoVector, _TorchSize)):
                result = backend.empty(tuple(args[0]), dtype=dtype)
            elif isinstance(args[0], backend.Var):
                result = backend.Var.clone(args[0])
                result._set_view_of(args[0], Ellipsis)
                return result
            else:
                result = backend.array(args[0], dtype=dtype)
            result.requires_grad = False
            return result


def make_tensor_type(backend):
    """Create the installation's real type without writing to native Var."""
    return _TensorMeta("Tensor", (backend.Var,), {
        "__module__": "torch",
        "__slots__": ("__weakref__",),
        "_frontend_backend": backend,
        "_frontend_autograd_policy": 3,
        "clone": clone,
    })


def clone(input, *, memory_format=None):
    """Copy Tensor storage through the native copy op, preserving gradients."""
    import jittor as backend
    from .tensor_state import compatibility_owner
    from .types import _var_is_cpu_resident
    if not isinstance(input, backend.Var):
        raise TypeError("clone expects a tensor")
    if memory_format not in (None, "preserve_format", "contiguous_format"):
        raise NotImplementedError("clone supports preserve_format and contiguous_format")
    target = compatibility_owner(backend)
    with tensor_frontend(target.Var):
        if backend.flags.use_cuda and _var_is_cpu_resident(input):
            with backend.flag_scope(use_cuda=0):
                result = backend.Var.copy(input)
                result.sync()
            result._jittor_torch_force_cpu = True
            return result
        return backend.Var.copy(input)


def make_parameter_type(backend, tensor_type):
    """Create a real tensor subtype; construction makes a new graph leaf."""
    class Parameter(tensor_type):
        __slots__ = ()
        _frontend_result_type = tensor_type
        _torch_compat_type = True

        def __new__(cls, data=None, requires_grad=True):
            with tensor_frontend(cls):
                if data is None:
                    value = backend.empty((0,), dtype=_default_tensor_dtype(backend))
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
