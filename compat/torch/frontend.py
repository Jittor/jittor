"""Python frontend types sharing the native VarHolder payload and graph."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from contextlib import contextmanager
from functools import update_wrapper
from types import MethodType


def _frontend_precision_policy(cls):
    """Read this frontend's two native accumulation tiers without flag writes."""
    from .context import get_install_context
    state = get_install_context(cls._frontend_backend).state.get("cuda_runtime")
    if state is None:
        # Type creation precedes CUDA facade publication during installation.
        return (0, 1)
    tiers = {"highest": 0, "high": 1, "medium": 2}
    return tiers[state.matmul_precision], tiers[state.cudnn_precision]


def _default_tensor_dtype(backend):
    from .tensor_state import compatibility_owner
    from .types import _dtype_to_str
    owner = compatibility_owner(backend)
    getter = getattr(owner, "get_default_dtype", None)
    # Types are created before core.extended publishes the default-dtype API.
    return _dtype_to_str(getter()) if getter is not None else "float32"


def _placement_request(backend, device, like=None):
    """Resolve native placement without changing Runtime flags or materializing."""
    if device is None:
        if isinstance(like, backend.Var) and like.placement_backend >= 0:
            return int(like.placement_backend), max(int(like.device_id), 0)
        return None
    numeric_index = isinstance(device, int) and not isinstance(device, bool)
    name = "cuda" if numeric_index else (getattr(device, "type", None) or str(device).split(":", 1)[0])
    if name == "cpu":
        return 0, 0
    if name not in ("cuda", "npu"):
        raise NotImplementedError("native Tensor placement does not support device %r" % str(device))
    registered = set(backend.core.registered_backends()) - {"cpu"}
    selected = "acl" if name == "npu" else next(iter(registered), "cuda")
    if selected not in registered:
        raise RuntimeError("Tensor placement requires an available %s backend" % name)
    index = int(device) if numeric_index else (None if isinstance(device, str) else getattr(device, "index", None))
    if index is None and ":" in str(device):
        index = int(str(device).split(":", 1)[1])
    if index is None:
        index = max(int(backend.core.current_device()), 0)
    count = backend.core.backend_device_count(selected)
    if not 0 <= int(index) < count:
        raise RuntimeError("Invalid %s device index %s; visible device count is %s" %
                           (name, index, count))
    return {"cuda": 1, "acl": 2, "acl_legacy": 2, "rocm": 3, "corex": 4}[selected], int(index)


@contextmanager
def tensor_frontend(tensor_type, *, device=None, like=None):
    backend = getattr(tensor_type, "_frontend_backend", None)
    if backend is None:
        yield
        return
    token = backend.core._set_tensor_frontend_type(tensor_type)
    placement_token = None
    precision_token = None
    try:
        precision_token = backend.core._set_float32_precision(*tensor_type._frontend_precision_policy())
        placement = _placement_request(backend, device, like)
        if placement is not None:
            placement_token = backend.core._set_tensor_placement(*placement)
        with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
            yield
    finally:
        if precision_token is not None:
            backend.core._reset_float32_precision(precision_token)
        if placement_token is not None:
            backend.core._reset_tensor_placement(placement_token)
        backend.core._reset_tensor_frontend_type(token)


class FrontendFactory:
    """Explicit callable/descriptor holding a native factory and frontend type."""

    def __init__(self, function, tensor_type):
        self.function = function
        self.tensor_type = tensor_type
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        with tensor_frontend(self.tensor_type, device=kwargs.get("device")):
            return self.function(*args, **kwargs)

    def __get__(self, instance, owner=None):
        return self if instance is None else MethodType(self, instance)


def frontend_factory(function, tensor_type):
    if "_frontend_backend" not in vars(tensor_type):
        return function
    return FrontendFactory(function, tensor_type)


class _TensorMeta(type):
    def __call__(cls, *args, **kwargs):
        backend = vars(cls).get("_frontend_backend")
        if backend is None:
            return super().__call__(*args, **kwargs)
        if kwargs:
            raise TypeError("Tensor constructor does not accept keyword arguments")
        from .nested import _TorchSize
        dtype = _default_tensor_dtype(backend)
        with tensor_frontend(cls, like=args[0] if len(args) == 1 else None):
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
    from .tensor_object_state import tensor_object_properties
    return _TensorMeta("Tensor", (backend.Var,), {
        "__module__": "torch",
        "__slots__": ("__weakref__",),
        "_frontend_backend": backend,
        "_frontend_autograd_policy": 3,
        "_frontend_precision_policy": classmethod(_frontend_precision_policy),
        "clone": clone,
        **tensor_object_properties(),
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
    with tensor_frontend(target.Var, like=input):
        if input.placement_backend < 0 and backend.flags.use_cuda and _var_is_cpu_resident(input):
            with backend.flag_scope(use_cuda=0):
                result = backend.Var.copy(input)
                result.sync()
                setattr(result, "_jittor_torch_force_cpu", True)
            return result
        return backend.Var.copy(input)


def parameter_new(cls, data=None, requires_grad=True):
    backend = cls._parameter_backend
    with tensor_frontend(cls, like=data):
        if data is None:
            value = backend.empty((0,), dtype=_default_tensor_dtype(backend))
        else:
            source = data if isinstance(data, backend.Var) else backend.array(data)
            value = backend.Var.detach(source)
    value.requires_grad = bool(requires_grad)
    return value


def parameter_init(self, data=None, requires_grad=True):
    # The conversion boundary already initialized the native holder. Python
    # subclasses still receive their real __init__ through normal dispatch.
    return None


def make_parameter_type(backend, tensor_type):
    """Bind the native backend and Tensor subtype to stable parameter methods."""
    return type("Parameter", (tensor_type,), {
        "__module__": "torch.nn.parameter", "__slots__": (),
        "_parameter_backend": backend, "_frontend_result_type": tensor_type,
        "_torch_compat_type": True,
        "__new__": parameter_new, "__init__": parameter_init,
    })


def reduce_tensor(value):
    return (
        rebuild_tensor,
        (type(value), value.numpy(), _jittor_dtype_name(value.dtype), value.requires_grad,
         str(value.device)),
        value.__dict__.copy(),
    )


def rebuild_tensor(tensor_type, array, dtype, requires_grad, device=None):
    # Rebuild the holder without rerunning an application subclass's __init__;
    # pickle restores its Python state after this object has entered the memo.
    backend = tensor_type._frontend_backend
    with tensor_frontend(tensor_type, device=device):
        value = backend.array(array, dtype=dtype)
        value.requires_grad = requires_grad
    return value


def deepcopy_tensor(value, memo):
    from copy import deepcopy
    result = rebuild_tensor(type(value), value.numpy(), _jittor_dtype_name(value.dtype),
                            value.requires_grad, str(value.device))
    memo[id(value)] = result
    result.__dict__.update(deepcopy(value.__dict__, memo))
    return result
