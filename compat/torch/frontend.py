"""Python frontend types sharing the native VarHolder payload and graph."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor_core import ops as _native_ops

from functools import update_wrapper
import numbers as _numbers
import operator as _operator
from types import MethodType
import weakref
from typing import Any, Callable, Optional


def _is_size_argument(value):
    """An integer dimension as torch reads it: Python or NumPy integers, not bool or tensors."""
    return isinstance(value, _numbers.Integral) and not isinstance(value, bool)

#: Resolved on first use, then reused. These two helpers sit on the per-op path
#: (`torch.cat` alone re-imported them 91 times, ~110 us of its 340 us), and a
#: function-local `import` pays the import machinery on every call. The import
#: stays lazy so this module is still importable before the install context
#: exists.
_get_install_context: Optional[Callable[..., Any]] = None

# Resolve this helper once; an import in _placement_request otherwise runs for
# every factory and module scope. Its result must stay live: device contexts
# are thread-local and set_default_device can change between calls.
_active_device_context: Optional[Callable[..., Any]] = None

#: The two accumulation tiers, as the state object spells them.
_TIERS = {"highest": 0, "high": 1, "medium": 2}

#: The `cuda_runtime` state object per frontend type. Only the *lookup* is
#: memoized; the two tier names are read from it on every call, because they are
#: settable at runtime (`torch.backends.cuda.matmul.allow_tf32`, the H3 VAE's
#: determinism scope) and a cached tuple would answer with a stale policy.
#: The type is created by the installer, so a reinstallation makes a new type and
#: a new entry rather than reusing an old one. Weak: subclasses made at run time
#: (a Parameter subclass) ask too and must stay collectable.
_precision_state: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def _frontend_precision_policy(cls):
    """Read this frontend's two native accumulation tiers without flag writes."""
    global _get_install_context
    state = _precision_state.get(cls)
    if state is None:
        get_install_context = _get_install_context
        if get_install_context is None:
            from .context import get_install_context as _resolve
            get_install_context = _get_install_context = _resolve
        state = get_install_context(cls._frontend_backend).state.get("cuda_runtime")
        if state is None:
            # Type creation precedes CUDA facade publication during installation.
            return (0, 1)
        _precision_state[cls] = state
    return _TIERS[state.matmul_precision], _TIERS[state.cudnn_precision]


def _default_tensor_dtype(backend):
    from .tensor_state import compatibility_owner
    from .types import _dtype_to_str
    owner = compatibility_owner(backend)
    getter = getattr(owner, "get_default_dtype", None)
    # Types are created before core.extended publishes the default-dtype API.
    return _dtype_to_str(getter()) if getter is not None else "float32"


#: What `torch.set_default_device` was last told, or None meaning CPU.
#: Deliberately *not* jittor's `use_cuda`: that flag answers "is the
#: accelerator enabled", which torch treats as a different question from
#: "where does a tensor with no device go".
_DEFAULT_DEVICE = None


def default_device():
    """The device a factory with no `device=` should use."""
    return _DEFAULT_DEVICE or "cpu"


def set_default_device_spelling(device):
    """Record what `torch.set_default_device` chose; None clears it to CPU."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def _placement_request(backend, device, like=None, default_placement=True):
    """Resolve native placement without changing Runtime flags or materializing.

    ``default_placement=False`` is for a scope that *runs* ops rather than
    constructs tensors -- a module's forward. torch's default device answers
    "where does a new tensor with no ``device=`` go"; it does not relocate the
    buffers an op allocates for its result, which follow the op's inputs.
    Returning the default there put the whole forward under an ambient CPU
    placement: every native allocation inside it (the destination
    ``jt.concat`` fills) landed on the host while the inputs were on the GPU,
    and the first ``setitem`` died in dispatch_context. An explicit
    ``with torch.device(...)`` still applies.
    """
    if device is None:
        if isinstance(like, backend.Var) and like.placement_backend >= 0:
            return int(like.placement_backend), max(int(like.device_id), 0)
        # `with torch.device(d):` -- torch builds new tensors on `d`. `like`
        # keeps priority above because `torch.empty_like(x)` inherits x's
        # device rather than the ambient context's.
        global _active_device_context
        active_device_context = _active_device_context
        if active_device_context is None:
            from .types import active_device_context
            _active_device_context = active_device_context
        device = active_device_context()
        if device is None and not default_placement:
            return None
        if device is None:
            # torch's default device is CPU, and stays CPU until somebody calls
            # `set_default_device`. This used to return None -- "no placement,
            # let jittor decide" -- which means jittor's global `use_cuda`, and
            # that flag says *the accelerator is enabled*, not *the accelerator
            # is the default device*. torch keeps those apart.
            #
            # The cost of conflating them: MiniMax-H3's VAE does
            # `latent = latent.float().cpu()` and then builds its normalisation
            # constants with a plain `torch.tensor(...)`. Under torch both are
            # on the CPU; here the constants landed on cuda:0 and the subtract
            # died in device_copy_op with "Expected all tensor inputs on the
            # same backend and device".
            device = default_device()
    numeric_index = isinstance(device, int) and not isinstance(device, bool)
    if numeric_index:
        name = "cuda"
    elif isinstance(device, str):
        # ``str, Enum`` device values (DeepSpeed's OffloadDeviceEnum) compare
        # and split like their public value even though ``str(value)`` spells
        # the enum member name on modern Python.
        name = device.split(":", 1)[0]
    else:
        name = getattr(device, "type", None) or str(device).split(":", 1)[0]
    # Jittor has no meta storage backend. Transformers uses ``device='meta'``
    # while inspecting checkpoint shapes; materialize those temporary tensors
    # on host so the subsequent real-weight load can proceed.
    if name == "meta":
        return 0, 0
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


class tensor_frontend:
    """Build/run under the frontend's tensor type, precision and placement.

    See `_placement_request` for ``default_placement``: construction scopes
    keep it, a scope that executes ops (a module's forward) turns it off.

    A class rather than a generator: every module call and every tensor
    factory enters one, and a `@contextmanager` generator plus the nested
    autograd `policy_scope` cost more than the native calls they wrap.
    """

    __slots__ = ("_type", "_device", "_like", "_default_placement", "_backend",
                 "_token", "_placement_token", "_precision_token", "_policy_bits")

    def __init__(self, tensor_type, *, device=None, like=None, default_placement=True):
        self._type = tensor_type
        self._device = device
        self._like = like
        self._default_placement = default_placement
        self._backend = None

    def __enter__(self):
        backend = getattr(self._type, "_frontend_backend", None)
        self._backend = backend
        if backend is None:
            return None
        core = backend.core
        self._token = core._set_tensor_frontend_type(self._type)
        self._placement_token = self._precision_token = self._policy_bits = None
        try:
            self._precision_token = core._set_float32_precision(
                *self._type._frontend_precision_policy())
            placement = _placement_request(backend, self._device, self._like,
                                           self._default_placement)
            if placement is not None:
                self._placement_token = core._set_tensor_placement(*placement)
            # backend.autograd.policy_scope(EXPLICIT_REQUIRES_GRAD), inline.
            policy = backend.autograd.EXPLICIT_REQUIRES_GRAD
            self._policy_bits = core._get_autograd_policy()
            core._set_autograd_policy(policy.stop_outputs_when_inputs_stopped,
                                      policy.preserve_requires_grad_on_assignment)
        except BaseException:
            self._restore()
            raise
        return None

    def __exit__(self, *exc):
        if self._backend is not None:
            self._restore()
        return False

    def _restore(self):
        backend = self._backend
        if backend is None:
            return
        core = backend.core
        bits = self._policy_bits
        if bits is not None:
            core._set_autograd_policy(bool(bits & 1), bool(bits & 2))
        if self._precision_token is not None:
            core._reset_float32_precision(self._precision_token)
        if self._placement_token is not None:
            core._reset_tensor_placement(self._placement_token)
        core._reset_tensor_frontend_type(self._token)


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
            # A Tensor subtype may be rebuilt from another holder's __dict__.
            # Keep its public constructor metadata, but never pass our holder's
            # private autograd/residency state into the new subtype constructor.
            from .tensor_object_state import TensorObjectState
            if isinstance(kwargs.get("_torch_object_state"), TensorObjectState):
                kwargs = dict(kwargs)
                kwargs.pop("_torch_object_state")
            return super().__call__(*args, **kwargs)
        # torch's Tensor constructor spells dtype/device/requires_grad/pin_memory
        # as keywords, and downstream code builds tensors that way -- `accelerate`
        # moves a parameter with `param_cls(value, requires_grad=...)`. Rejecting
        # every keyword made those calls fail; anything outside the documented
        # four still raises.
        requested_dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        requested_requires_grad = kwargs.pop("requires_grad", None)
        requires_grad = bool(requested_requires_grad)
        kwargs.pop("pin_memory", None)
        if kwargs:
            raise TypeError(
                "Tensor constructor does not accept keyword arguments: %s"
                % ", ".join(sorted(kwargs))
            )
        from .nested import _TorchSize
        dtype = requested_dtype if requested_dtype is not None else _default_tensor_dtype(backend)
        with tensor_frontend(cls, like=args[0] if len(args) == 1 else None, device=device):
            if not args:
                result = backend.empty((0,), dtype=dtype)
            elif all(_is_size_argument(arg) for arg in args):
                # torch reads any integer (NumPy included) as a dimension.
                result = backend.empty(tuple(_operator.index(arg) for arg in args), dtype=dtype)
            elif len(args) != 1:
                raise TypeError("Tensor expects data or integer dimensions")
            elif isinstance(args[0], (backend.NanoVector, _TorchSize)):
                result = backend.empty(tuple(args[0]), dtype=dtype)
            elif isinstance(args[0], backend.Var):
                # torch.Tensor(t) aliases t: the native clone op is the view;
                # `Var.clone()` copies, as torch.Tensor.clone does.
                result = _native_ops.clone(args[0])
                # The native view owner preserves both the storage and the
                # creation-mode-dependent gradient edge. An inherited flag
                # under no_grad does not explicitly activate an accumulator.
                result._set_view_of(args[0], Ellipsis, requested_requires_grad is None)
                if requested_requires_grad is not None:
                    result.requires_grad = requires_grad
                return result
            else:
                result = backend.array(args[0], dtype=dtype)
            result.requires_grad = requires_grad
            return result


def make_tensor_type(backend):
    """Create the installation's real type without writing to native Var."""
    from .tensor_object_state import tensor_object_properties
    return _TensorMeta("Tensor", (backend.Var,), {
        "__module__": "torch",
        "__slots__": ("__weakref__",),
        "_frontend_backend": backend,
        "_frontend_autograd_policy": 3,
        "_frontend_view_base_ownership": True,
        "_frontend_precision_policy": classmethod(_frontend_precision_policy),
        "_make_subclass": staticmethod(tensor_make_subclass),
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
                result = backend.Var.copy(input, memory_format != "contiguous_format")
                result.sync()
                setattr(result, "_jittor_torch_force_cpu", True)
            return result
        return backend.Var.copy(input, memory_format != "contiguous_format")


def tensor_make_subclass(cls, data, requires_grad=False):
    """Make an independent autograd leaf of ``cls`` over existing storage."""
    backend = getattr(cls, "_frontend_backend", None)
    if backend is None or not issubclass(cls, backend.Var):
        raise TypeError("_make_subclass expects a Torch Tensor subclass")
    if not isinstance(data, backend.Var):
        raise TypeError("_make_subclass expects Tensor data")
    with tensor_frontend(cls, like=data):
        value = backend.Var.detach(data)
    value.requires_grad = bool(requires_grad)
    return value


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


def parameter_init(self, *args, **kwargs):
    # The conversion boundary already initialized the native holder. Python
    # subclasses still receive their real __init__ through normal dispatch.
    # A subtype may consume more __new__ keywords than Parameter itself; its
    # inherited no-op initializer must accept those same constructor inputs.
    return None


def make_parameter_type(backend, tensor_type):
    """Bind the native backend and Tensor subtype to stable parameter methods."""
    return type("Parameter", (tensor_type,), {
        "__module__": "torch.nn.parameter", "__slots__": (),
        "_parameter_backend": backend, "_frontend_result_type": tensor_type,
        "_torch_compat_type": True,
        "__new__": parameter_new, "__init__": parameter_init,
    })


#: Attributes a runtime notes on a tensor -- the weight version a convolution
#: has seen, see `jittor/nn/backends/cudnn.py` -- which a pickle or a copy of
#: the tensor must not carry.
_RUNTIME_CACHES = ("_jittor_conv_filter",)

# A shared-memory tensor keeps the OS mapping alive on its Python holder.  The
# handle itself is deliberately not pickled; a spawned process reopens the
# mapping by name in rebuild_shared_tensor below.
_SHARED_MEMORY_HANDLE = "_torch_shared_memory_handle"
_SHARED_MEMORY_SPEC = "_torch_shared_memory_spec"


def _python_state(value):
    state = value.__dict__.copy()
    for name in _RUNTIME_CACHES:
        state.pop(name, None)
    state.pop(_SHARED_MEMORY_HANDLE, None)
    return state


def shared_memory_owner(value):
    """Find the mapping retained by a shared tensor or a derived view."""
    seen = set()
    while value is not None and id(value) not in seen:
        seen.add(id(value))
        if getattr(value, _SHARED_MEMORY_SPEC, None) is not None:
            return value
        owner = getattr(value, "_torch_shared_memory_view_owner", None)
        value = owner if owner is not None else value._view_base_object()
    return None


def reduce_tensor(value):
    shared = getattr(value, _SHARED_MEMORY_SPEC, None)
    if shared is not None:
        # .data aliases carry their parameter owner in Python state.  The
        # shared allocation already contains all information needed to rebuild
        # the alias, and serializing the owner would copy the whole parameter
        # graph into the optimizer process.
        state = _python_state(value)
        state.pop("_torch_data_owner", None)
        state.pop("_torch_data_path", None)
        return (
            rebuild_shared_tensor,
            (type(value), *shared, value.requires_grad),
            state,
        )
    if shared_memory_owner(value) is not None:
        raise NotImplementedError(
            "multiprocessing shared-memory views are not supported; "
            "send the shared base tensor and create the view in the child")
    return (
        rebuild_tensor,
        (type(value), value.numpy(), _jittor_dtype_name(value.dtype), value.requires_grad,
         str(value.device)),
        _python_state(value),
    )


def _shared_numpy_dtype(dtype):
    import numpy as np
    return {
        "bool": np.bool_, "uint8": np.uint8, "int8": np.int8,
        "uint16": np.uint16, "int16": np.int16,
        "uint32": np.uint32, "int32": np.int32,
        "uint64": np.uint64, "int64": np.int64,
        "float16": np.float16, "float32": np.float32,
        "float64": np.float64, "complex64": np.complex64,
        "complex128": np.complex128,
    }.get(dtype)


def _tensor_from_shared_memory(tensor_type, name, shape, dtype):
    """Open name and make a CPU Var that aliases its bytes without a copy."""
    import numpy as np
    from multiprocessing import shared_memory

    backend = tensor_type._frontend_backend
    np_dtype = _shared_numpy_dtype(dtype)
    raw_dtype = np.uint16 if dtype == "bfloat16" else np_dtype
    if raw_dtype is None:
        raise TypeError("share_memory_ does not support dtype %s" % dtype)
    handle = shared_memory.SharedMemory(name=name)
    array = np.ndarray(tuple(shape), dtype=raw_dtype, buffer=handle.buf)
    # reuse_array makes ArrayOp retain the ndarray allocation instead of
    # copying it.  Disable the usual 64->32 narrowing and force host placement:
    # both are required for the reopened Var to address the exact shared bytes.
    with backend.flag_scope(use_cuda=0, reuse_array=1,
                            auto_convert_64_to_32=0):
        value = backend.array(array)
        if dtype == "bfloat16":
            value = backend.reinterpret_view(value, tuple(shape), "bfloat16")
    value = tensor_type(value).detach()
    setattr(value, _SHARED_MEMORY_HANDLE, handle)
    setattr(value, _SHARED_MEMORY_SPEC,
            (name, tuple(int(s) for s in shape), dtype))
    return value


def share_tensor_memory(value):
    """Share an independent CPU tensor before creating views or aliases.

    Shared base tensors and full detached aliases support spawn transport.
    Derived views must be recreated in the child from the shared base.
    """
    if getattr(value, _SHARED_MEMORY_SPEC, None) is not None:
        return value
    if bool(getattr(value, "is_cuda", False)):
        # Matches Torch's no-op for CUDA; this does not provide CUDA IPC.
        return value

    if value._is_view() or value._has_storage_aliases():
        raise NotImplementedError(
            "share_memory_ requires an independent CPU tensor; "
            "call it before creating views or detached aliases")

    import numpy as np
    from multiprocessing import shared_memory

    dtype = _jittor_dtype_name(value.dtype)
    np_dtype = _shared_numpy_dtype(dtype)
    raw_dtype = np.uint16 if dtype == "bfloat16" else np_dtype
    if raw_dtype is None:
        raise TypeError("share_memory_ does not support dtype %s" % dtype)
    shape = tuple(int(s) for s in value.shape)
    nbytes = int(value.numel()) * np.dtype(raw_dtype).itemsize
    # SharedMemory refuses size=0.  A one-byte mapping still correctly backs a
    # zero-element ndarray and has no observable tensor element.
    handle = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
    shared_array = np.ndarray(shape, dtype=raw_dtype, buffer=handle.buf)
    source = np.asarray(value.numpy())
    if dtype == "bfloat16":
        # Jittor exports bfloat16 through float32.  Preserve the high 16 bits
        # that form the actual bfloat16 storage representation.
        shared_array[...] = (source.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    else:
        shared_array[...] = source.astype(raw_dtype, copy=False)

    shared = _tensor_from_shared_memory(type(value), handle.name, shape, dtype)
    # Keep the creating handle as the owner.  The temporary opener is no longer
    # needed once its ndarray-backed Var has retained the mapping.
    opener = getattr(shared, _SHARED_MEMORY_HANDLE)
    handles = (handle, opener)
    setattr(shared, _SHARED_MEMORY_HANDLE, handles)

    target = getattr(value, "_torch_data_owner", None)
    if target is None:
        target = value
    target._update(shared, logical_write=False)
    spec = (handle.name, shape, dtype)
    setattr(target, _SHARED_MEMORY_HANDLE, handles)
    setattr(target, _SHARED_MEMORY_SPEC, spec)
    setattr(value, _SHARED_MEMORY_HANDLE, handles)
    setattr(value, _SHARED_MEMORY_SPEC, spec)
    return value


def rebuild_shared_tensor(tensor_type, name, shape, dtype, requires_grad):
    value = _tensor_from_shared_memory(tensor_type, name, shape, dtype)
    if requires_grad:
        value.requires_grad_(True)
    return value


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
    from .tensor_object_state import TensorObjectState, get_tensor_object_state
    result = rebuild_tensor(type(value), value.numpy(), _jittor_dtype_name(value.dtype),
                            value.requires_grad, str(value.device))
    memo[id(value)] = result
    state = _python_state(value)
    original_state = get_tensor_object_state(value)
    object_state = state.pop("_torch_object_state", None)
    result.__dict__.update(deepcopy(state, memo))
    if original_state is None:
        return result
    # The copy owns new storage, as torch's does. A `.data` alias must not
    # survive pointing at the *original* holder -- `weight = deepcopy(
    # base.weight.data); weight += delta` then wrote through `base` and dropped
    # delta's gradient edge, which is how PEFT's SHiRA adapter builds its
    # dense weight -- while an alias that named this tensor itself follows into
    # the copy, which is the state contract the frontend publishes.
    copied = TensorObjectState()
    self_owner = original_state.data_owner is value
    copied.data_owner = result if self_owner else None
    copied.data_path = (deepcopy(original_state.data_path, memo)
                        if self_owner else ())
    copied.grad = deepcopy(original_state.grad, memo)
    copied.scalar_marker = original_state.scalar_marker
    copied.force_cpu = original_state.force_cpu
    copied.force_cuda = original_state.force_cuda
    copied.retains_grad = original_state.retains_grad
    copied.rms_norm_unit_weight = (
        result if original_state.rms_norm_unit_weight is value
        else deepcopy(original_state.rms_norm_unit_weight, memo))
    result.__dict__["_torch_object_state"] = copied
    return result
