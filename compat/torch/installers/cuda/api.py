"""Stable CUDA facade APIs with installation-owned mutable runtime state.

Installation publishes these objects; it does not manufacture implementations.
Logical streams, sampled memory peaks and unsupported placeholders retain their
existing behavior and have explicit fidelity records.
"""

import contextlib
import itertools
import threading
import types as _types

import jittor as jt

from ...context import InstallContext, get_install_context, registry_for
from ...fidelity import Fidelity, register_fidelity
from ....stub_policy import unimplemented as _unimplemented

from ...grad import (
    _amp_passthrough_decorator, _AutocastContext,
    _GradScaler,
)
from ...types import (
    device, dtype, _cuda_index_of, _device_is_cpu,
)
from ....diagnostics import EXPECTED, swallowed

_cuda_props_cache = {}


def _cuda_driver():
    try:
        import ctypes
        for n in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(n)
                lib.cuInit(0)
                return lib, ctypes
            except OSError as exc:
                swallowed("torch/installers/cuda/api.py _cuda_driver: lib = ctypes.CDLL(n)", exc)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_driver: import ctypes", exc)
    return None, None


def _cuda_device_index(device=None):
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py _cuda_device_index: return int(device.split(':', 1)[1])", exc)
            return 0
    if isinstance(device, int):
        return device
    idx = getattr(device, "index", None)
    return int(idx) if idx is not None else 0


def _cuda_device_name(device=None):
    name = _cuda_props_cache.get("name")
    if name is not None:
        return name
    name = "CUDA"
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), _cuda_device_index(device))
            buf = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(buf, len(buf), dev)
            got = buf.value.decode("utf-8", "ignore")
            if got:
                name = got
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_device_name: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache["name"] = name
    return name


def _cuda_capability():
    """(major, minor) compute capability of the active CUDA device.

    Queried once from the CUDA driver (compute-capability of device 0); falls
    back to (8, 0) when the driver query is unavailable (e.g. Ascend NPU).
    """
    cc = _cuda_props_cache.get("cap")
    if cc is not None:
        return cc
    cc = (8, 0)
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            maj = ctypes.c_int(0); mino = ctypes.c_int(0)
            lib.cuDeviceComputeCapability(ctypes.byref(maj), ctypes.byref(mino), dev)
            if maj.value > 0:
                cc = (maj.value, mino.value)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_capability: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache["cap"] = cc
    return cc


def _cuda_sm_count():
    """SM (multiprocessor) count of CUDA device 0, queried via the driver.

    Triton-based libraries (e.g. flex_gemm's autotuner) size their grids by
    ``get_device_properties(...).multi_processor_count``; a wrong value only
    affects performance/occupancy, not correctness, so we default to 132 (an
    H100-class count) when the driver can't be queried.
    """
    n = _cuda_props_cache.get("sm")
    if n is not None:
        return n
    n = 132
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            val = ctypes.c_int(0)
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            lib.cuDeviceGetAttribute(ctypes.byref(val), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
            if val.value > 0:
                n = val.value
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_sm_count: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache["sm"] = n
    return n


def _cuda_total_memory():
    total = _cuda_props_cache.get("total_memory")
    if total is not None:
        return total
    total = 64 * 1024 ** 3
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            val = ctypes.c_size_t(0)
            fn = getattr(lib, "cuDeviceTotalMem_v2", None) or getattr(lib, "cuDeviceTotalMem", None)
            if fn is not None:
                fn(ctypes.byref(val), dev)
            if val.value > 0:
                total = int(val.value)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_total_memory: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache["total_memory"] = total
    return total


class _DeviceProps:
    """torch.cuda.get_device_properties(...) result.

    Exposes the attributes real-torch device props carry that libraries read:
    ``name``, ``major``/``minor``, ``total_memory``, ``multi_processor_count``
    (alias ``multiprocessor_count``), ``warp_size``, ``max_threads_per_*``.
    """
    def __init__(self):
        cap = _cuda_capability()
        self.name = "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name()
        self.major, self.minor = cap
        self.total_memory = _cuda_total_memory()
        self.multi_processor_count = _cuda_sm_count()
        self.multiprocessor_count = self.multi_processor_count
        self.warp_size = 32
        self.max_threads_per_multi_processor = 2048
        self.max_threads_per_block = 1024
        self.is_integrated = 0
        self.is_multi_gpu_board = 0
        self.regs_per_multiprocessor = 65536
        self.shared_memory_per_block = 49152
        self.shared_memory_per_multiprocessor = 102400

    def __repr__(self):
        return (f"_DeviceProps(name='{self.name}', major={self.major}, "
                f"minor={self.minor}, total_memory={self.total_memory}, "
                f"multi_processor_count={self.multi_processor_count})")


#: ``torch.backends``' TF32 switches, and the single Jittor flag behind each.
#:
#: torch spells "may fp32 math use reduced-precision tensor cores" three ways
#: per domain, and Jittor keeps one flag per domain (matmul and cuDNN are
#: genuinely independent in torch too):
#:
#: =========================================== =============================
#: torch spelling                              Jittor flag
#: =========================================== =============================
#: ``backends.cuda.matmul.allow_tf32``         ``flags.cuda_allow_tf32``
#: ``backends.cuda.matmul.fp32_precision``     (same)
#: ``get/set_float32_matmul_precision()``      (same)
#: ``backends.cudnn.allow_tf32``               ``flags.cuda_allow_cudnn_tf32``
#: ``backends.cudnn.conv.fp32_precision``      (same)
#: ``backends.cudnn.rnn.fp32_precision``       (same)
#: =========================================== =============================
#:
#: Every spelling is a *view* of its flag. It did not use to be: each kept its
#: own state, so one semantic had three answers that disagreed the moment a
#: write went through a spelling other than the one holding the state.
#: ``fp32_precision`` was the literal string ``"ieee"`` on all four objects --
#: it never reflected tf32 being on, and assigning to it did nothing at all --
#: and ``get_float32_matmul_precision()`` read a string that
#: ``matmul.allow_tf32 = True`` never touched.
#:
#: tests/compat/torch/test_torch_backends_tf32.py drives this table.
_PRECISION_FIELDS = {
    "matmul": "matmul_precision",
    "cudnn": "cudnn_precision",
}

#: The two ``fp32_precision`` values this layer can actually deliver. torch
#: also accepts "bf16" and "none"; Jittor has no separate bf16-accumulate mode
#: and no per-op override, so accepting them would be inventing a semantics.
_FP32_PRECISIONS = ("ieee", "tf32")


def _tf32_get(domain):
    """Whether reduced-precision fp32 math is enabled for ``domain``."""
    return getattr(_cuda_runtime(), _PRECISION_FIELDS[domain]) != "highest"


def _tf32_set(domain, value):
    """Point every spelling of ``domain``'s switch at ``value``."""
    enabled = bool(value)
    tier = _cuda_runtime().matmul_refinement if domain == "matmul" else "high"
    _set_precision_tier(domain, tier if enabled else "highest")
    return enabled


def _set_precision_tier(domain, tier):
    from ....transaction import current_transaction
    state = _cuda_runtime()
    field = _PRECISION_FIELDS[domain]
    transaction = current_transaction()
    if transaction is None:
        setattr(state, field, tier)
    else:
        transaction.mutate_attr(state, field, tier)
    if domain == "matmul" and tier != "highest":
        if transaction is None:
            state.matmul_refinement = tier
        else:
            transaction.mutate_attr(state, "matmul_refinement", tier)


def _tf32_to_precision(enabled):
    return "tf32" if enabled else "ieee"


def _precision_to_tf32(value, where):
    if not isinstance(value, str):
        raise TypeError("%s.fp32_precision must be a string, not %s"
                        % (where, type(value).__name__))
    text = value.lower()
    if text not in _FP32_PRECISIONS:
        raise ValueError(
            "%s.fp32_precision does not support %r on Jittor; supported "
            "values are %s. torch also accepts 'bf16' and 'none', which would "
            "need a separate bf16-accumulate mode and a per-op override that "
            "Jittor does not have -- accepting them here would silently mean "
            "something else."
            % (where, value, " and ".join(repr(p) for p in _FP32_PRECISIONS)))
    return text == "tf32"


class _PrecisionBackend(object):
    """``<backend>.fp32_precision`` as a view of the domain's flag.

    ``torch.backends.cudnn.conv`` and ``.rnn`` are two of these. They used to
    be instances of a class whose ``fp32_precision`` was the class attribute
    ``"ieee"``: reading it never reported tf32 being on, and writing it only
    shadowed the attribute on the instance.
    """

    __slots__ = ("_domain", "_label")

    def __init__(self, domain, label):
        object.__setattr__(self, "_domain", domain)
        object.__setattr__(self, "_label", label)

    @property
    def fp32_precision(self):
        return _tf32_to_precision(_tf32_get(self._domain))

    @fp32_precision.setter
    def fp32_precision(self, value):
        _tf32_set(self._domain, _precision_to_tf32(value, self._label))

    def __repr__(self):
        return "<%s fp32_precision=%r>" % (self._label, self.fp32_precision)


def _cuda_target():
    return get_install_context(jt).target_namespace


def _cuda_runtime():
    return get_install_context(jt).state["cuda_runtime"]


class CudaRuntimeState:
    """Mutable CUDA facade state owned by one frontend installation."""

    def __init__(self):
        import os
        self.stream_state = threading.local()
        self.default_stream = _Stream()
        self.nvtx_state = threading.local()
        self.nvtx_handles = itertools.count(1)
        self.native_nvtx = [None]
        self.mem_peak = [0]
        self.memgetinfo = [None]
        self.matmul_precision = "highest"
        self.cudnn_precision = "high"
        self.matmul_refinement = "high"
        self.empty_cache_mode = str(os.environ.get(
            "JITTOR_TORCH_CUDA_EMPTY_CACHE", "0")).strip().lower()


def _identity(value):
    return value


class CUDAGraph:
    """Annotation placeholder; graph capture and replay are unsupported."""


class OutOfMemoryError(RuntimeError):
    pass

def _cuda_visible_devices_empty():
    import os as _os_cuda
    _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
    return _cvd is not None and _cvd.strip() == ""


def is_available():
    try:
        if _cuda_visible_devices_empty():
            return False
        return bool(getattr(jt, "has_cuda", 0)) or bool(getattr(jt.compiler, "has_cuda", 0)) \
            or bool(getattr(jt.compiler, "has_acl", 0))
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py is_available: if _cuda_visible_devices_empty():", exc)
        return False


def device_count():
    if not is_available():
        return 0
    # Every visible device is usable from this process now, so the
    # runtime's own count is the answer; it already honours
    # CUDA_VISIBLE_DEVICES.
    try:
        n = int(jt.get_device_count())
        if n > 0:
            return n
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py device_count: "
                  "n = int(jt.get_device_count())", exc,
                  "falling back to counting CUDA_VISIBLE_DEVICES")
    try:
        import os as _os_cuda
        _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
        if _cvd is not None:
            return len([_d for _d in _cvd.split(",") if _d.strip()])
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py device_count: import os as _os_cuda", exc)
    return 1


def current_device():
    try:
        d = int(jt.current_device())
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py current_device: "
                  "d = int(jt.current_device())", exc,
                  "reporting device 0, which is wrong on any other device")
        d = -1
    return d if d >= 0 else 0


def set_device(device=None, *a, **k):
    """torch.cuda.set_device: make a device current, in place."""
    if device is None or _device_is_cpu(device):
        return None
    index = _cuda_index_of(device)
    if index is None:
        # A bare "cuda"/torch.device("cuda") names no particular device.
        return None
    try:
        jt.set_device(int(index))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "torch.cuda.set_device(%r): %s" % (device, error))
    return None


class _CudaDeviceContext:
    """``with torch.cuda.device(i):`` -- i is current inside the block."""
    def __init__(self, device=None):
        self.device = device
        self.idx = None if _device_is_cpu(device) else _cuda_index_of(device)
        self.prev_idx = -1
    def __enter__(self):
        if self.idx is not None and self.idx >= 0:
            self.prev_idx = current_device()
            if self.prev_idx != self.idx:
                set_device(self.idx)
        return self
    def __exit__(self, *exc):
        if self.prev_idx >= 0 and self.prev_idx != current_device():
            set_device(self.prev_idx)
        self.prev_idx = -1
        return False


class _CudaDeviceOf(_CudaDeviceContext):
    """``with torch.cuda.device_of(tensor):`` -- the tensor's own device."""
    def __init__(self, tensor):
        idx = None
        if isinstance(tensor, jt.Var):
            try:
                got = int(tensor.device_id)
            except EXPECTED as exc:
                swallowed("torch/installers/cuda/api.py device_of: "
                          "got = int(tensor.device_id)", exc,
                          "the context will not switch device")
                got = -1
            if got >= 0:
                idx = got
        super().__init__(idx)


def _empty_cache():
    if _cuda_runtime().empty_cache_mode in ("0", "false", "no", "off", "none", "noop"):
        return
    if _cuda_runtime().empty_cache_mode in ("", "1", "true", "yes", "on", "gc"):
        try:
            jt.gc()
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py _empty_cache: jt.gc()", exc)
    elif _cuda_runtime().empty_cache_mode in ("sync", "full"):
        try:
            jt.sync_all(True)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py _empty_cache: jt.sync_all(True)", exc)
        try:
            jt.gc()
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py _empty_cache: jt.gc()", exc)


def _device_name(*a, **k):
    try:
        return "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(a[0] if a else None)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _device_name: return 'Ascend910B/NPU' if getattr(jt.compiler, 'has_ac...", exc)
        return "CUDA"


class _amp:
    @staticmethod
    def autocast(device_type="cuda", *a, **k):
        return _AutocastContext(device_type, *a, **k)
    GradScaler = _GradScaler
    custom_fwd = staticmethod(_amp_passthrough_decorator)
    custom_bwd = staticmethod(_amp_passthrough_decorator)


class _CudaTypedTensorMeta(type):
    def __instancecheck__(cls, obj):
        base_type = getattr(_cuda_target(), cls._tensor_name)
        return isinstance(obj, base_type) and bool(getattr(obj, "is_cuda", False))

    def __call__(cls, *args, **kwargs):
        if not _cuda_target().cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return getattr(_cuda_target(), cls._tensor_name)(*args, **kwargs).cuda()


_CUDA_TENSOR_TYPES = {
    name: _CudaTypedTensorMeta(name, (), {
        "_tensor_name": name, "__module__": __name__,
    })
    for name in (
        "FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
        "LongTensor", "IntTensor", "ShortTensor", "CharTensor",
        "ByteTensor", "BoolTensor",
    )
}
globals().update(_CUDA_TENSOR_TYPES)


device_type = device


class _Stream:
    def __init__(self, device=None, priority=0, *a, **k):
        self.cuda_stream = 0
        if device is None:
            self.device = device_type("cuda", current_device())
        elif isinstance(device, int):
            self.device = device_type("cuda", device)
        else:
            self.device = device_type(device)
        self.priority = int(priority)
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): jt.sync_all(True)
    # Jittor serialises every logical stream onto one physical stream, so
    # "wait for that other stream" is already satisfied by program order.
    # These are honest no-ops, not missing implementations.
    def wait_stream(self, *a, **k): return None
    def wait_event(self, *a, **k): return None
    def record_event(self, event=None, *a, **k):
        if event is not None and hasattr(event, "record"):
            event.record(self)
        return event
    def query(self): return True


class _Event:
    """torch.cuda.Event.

    Timing used to be a lie: elapsed_time() returned 0.0 unconditionally,
    so every `start.record(); ...; end.record(); start.elapsed_time(end)`
    benchmark reported 0 ms and any code dividing by it produced inf/NaN.
    Jittor has no CUDA event objects exposed, so record() takes a host
    timestamp after a device synchronisation, which measures the same
    wall-clock interval for the single physical stream used here.
    """

    def __init__(self, enable_timing=False, blocking=False,
                 interprocess=False, *a, **k):
        self.enable_timing = bool(enable_timing)
        self._time = None

    def record(self, stream=None, *a, **k):
        import time as _time_event
        try:
            jt.sync_all(True)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py record: jt.sync_all(True)", exc)
        self._time = _time_event.perf_counter()
        return None

    def synchronize(self):
        try:
            jt.sync_all(True)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda/api.py synchronize: jt.sync_all(True)", exc)

    def query(self):
        return self._time is not None

    def wait(self, stream=None):
        return None

    def elapsed_time(self, end_event):
        if not self.enable_timing or not getattr(end_event, "enable_timing", False):
            raise RuntimeError(
                "Both events must be created with enable_timing=True to "
                "call elapsed_time()")
        if self._time is None or getattr(end_event, "_time", None) is None:
            raise RuntimeError(
                "elapsed_time() needs both events to have been recorded")
        return (end_event._time - self._time) * 1000.0


def _current_stream(*a, **k):
    return getattr(_cuda_runtime().stream_state, "current", _cuda_runtime().default_stream)


def _set_stream(stream):
    if stream is None:
        return None
    if not isinstance(stream, _Stream):
        raise TypeError("set_stream expects a torch.cuda.Stream or None")
    _cuda_runtime().stream_state.current = stream
    return None


class _StreamContext:
    def __init__(self, stream):
        if stream is not None and not isinstance(stream, _Stream):
            raise TypeError("stream expects a torch.cuda.Stream or None")
        self.stream = stream
        self.previous = None
    def __enter__(self):
        if self.stream is not None:
            self.previous = _current_stream()
            _set_stream(self.stream)
        return self
    def __exit__(self, *exc):
        if self.stream is not None:
            _set_stream(self.previous)
        return False


def _load_native_nvtx():
    if _cuda_runtime().native_nvtx[0] is False:
        return None
    if _cuda_runtime().native_nvtx[0] is None:
        try:
            from jittor.tools import nvtx as native_nvtx
            _cuda_runtime().native_nvtx[0] = native_nvtx
        except (ImportError, OSError, RuntimeError):
            _cuda_runtime().native_nvtx[0] = False
    return _cuda_runtime().native_nvtx[0] or None


def _nvtx_stack():
    stack = getattr(_cuda_runtime().nvtx_state, "stack", None)
    if stack is None:
        stack = []
        _cuda_runtime().nvtx_state.stack = stack
    return stack


def _nvtx_range_push(message):
    stack = _nvtx_stack()
    stack.append(str(message))
    native_nvtx = _load_native_nvtx()
    if native_nvtx is not None:
        native_nvtx.nvtxRangePushA(str(message).encode("utf-8"))
    return len(stack) - 1


def _nvtx_range_pop():
    stack = _nvtx_stack()
    depth = len(stack) - 1
    if stack:
        stack.pop()
    native_nvtx = _load_native_nvtx()
    if native_nvtx is not None:
        native_nvtx.nvtxRangePop()
    return depth


def _nvtx_mark(message):
    str(message)
    return None


def _nvtx_range_start(message):
    str(message)
    return next(_cuda_runtime().nvtx_handles)


def _nvtx_range_end(range_id):
    int(range_id)
    return None


@contextlib.contextmanager
def _nvtx_range(message, *args, **kwargs):
    _nvtx_range_push(str(message).format(*args, **kwargs))
    try:
        yield
    finally:
        _nvtx_range_pop()


def _mem_used(*a, **k):
    try:
        mi = jt.get_mem_info()
        used = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _mem_used: mi = jt.get_mem_info()", exc)
        used = 0
    if used > _cuda_runtime().mem_peak[0]:
        _cuda_runtime().mem_peak[0] = used
    return used


def _mem_max(*a, **k):
    _mem_used()
    return _cuda_runtime().mem_peak[0]


def _reset_peak(*a, **k):
    try:
        mi = jt.get_mem_info()
        _cuda_runtime().mem_peak[0] = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _reset_peak: mi = jt.get_mem_info()", exc)
        _cuda_runtime().mem_peak[0] = 0


def _cuda_mem_get_info_fn():
    if _cuda_runtime().memgetinfo[0] is not None:
        return _cuda_runtime().memgetinfo[0]
    fn = False
    try:
        import ctypes as _ct
        lib = None
        for _n in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                lib = _ct.CDLL(_n)
                break
            except OSError:
                lib = None
        if lib is not None:
            lib.cudaMemGetInfo.argtypes = [_ct.POINTER(_ct.c_size_t),
                                           _ct.POINTER(_ct.c_size_t)]
            lib.cudaMemGetInfo.restype = _ct.c_int
            def fn(_lib=lib, _ct=_ct):
                free, total = _ct.c_size_t(0), _ct.c_size_t(0)
                if _lib.cudaMemGetInfo(_ct.byref(free), _ct.byref(total)) != 0:
                    return None
                return (int(free.value), int(total.value))
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_mem_get_info_fn: import ctypes as _ct", exc)
        fn = False
    _cuda_runtime().memgetinfo[0] = fn
    return fn


def _mem_get_info(*a, **k):
    fn = _cuda_mem_get_info_fn()
    if fn:
        got = fn()
        if got and got[1] > 0:
            return got
    # No cudart to ask: fall back to jittor's own accounting. This knows the
    # device total exactly and jittor's live bytes, but not the context's or
    # another process's, so it reads slightly optimistic rather than fictional.
    try:
        mi = jt.get_mem_info()
        total = int(mi.total_cuda_ram)
        return (max(0, total - int(mi.total_cuda_used)), total)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _mem_get_info: mi = jt.get_mem_info()", exc)
        return (0, 0)


class CUDAPluggableAllocator:
    def __init__(self, path_to_so_file, alloc_fn_name, free_fn_name):
        raise NotImplementedError(
            "Jittor does not support PyTorch CUDA pluggable allocators"
        )


class TorchFunctionMode:
    """torch.overrides.TorchFunctionMode -- refused, not faked.

    A mode is supposed to intercept EVERY torch function called inside
    it. Jittor's ops go straight to the C++ core and consult no such
    hook, so entering this used to change nothing at all: a device mode
    or a tracing mode silently observed and rewrote nothing.
    """

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        _unimplemented(
            "torch.overrides.TorchFunctionMode",
            "intercept no torch call at all, so a device/tracing/"
            "logging mode silently observes and rewrites nothing",
            "Jittor ops dispatch in C++ and consult no "
            "__torch_function__ hook.",
            stub_result=self)
        return self

    def __exit__(self, *a):
        return False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def _has_torch_function(*relevant_args):
    """Truthfully report whether any argument overrides __torch_function__.

    Was `lambda *a, **k: False`, so every torch-function protocol check
    in downstream code took the "plain tensor" branch and a tensor
    subclass's override was silently skipped.
    """
    for group in relevant_args:
        items = group if isinstance(group, (tuple, list, set)) else (group,)
        for item in items:
            tp = type(item)
            if tp is jt.Var:
                continue
            if getattr(tp, "__torch_function__", None) is not None:
                return True
    return False


def _handle_torch_function(public_api, relevant_args, *args, **kwargs):
    for item in (relevant_args or ()):
        override = getattr(type(item), "__torch_function__", None)
        if override is None or type(item) is jt.Var:
            continue
        types_tuple = tuple(type(a) for a in relevant_args)
        return override(public_api, types_tuple, args, kwargs)
    return public_api(*args, **kwargs)


def _parse_to(*args, **kwargs):
    dev = kwargs.get("device", None)
    dtype_arg = kwargs.get("dtype", None)
    non_blocking = kwargs.get("non_blocking", False)
    for arg in args:
        if isinstance(arg, jt.Var):
            dev = getattr(arg, "device", dev)
            dtype_arg = getattr(arg, "dtype", dtype_arg)
            continue
        if isinstance(arg, dtype) or str(arg).replace("torch.", "") in dtype._registry:
            if dtype_arg is None:
                dtype_arg = arg
            continue
        if isinstance(arg, str) or hasattr(arg, "type"):
            if dev is None:
                dev = arg
            continue
        if arg in getattr(dtype, "_registry", {}).values():
            if dtype_arg is None:
                dtype_arg = arg
    if isinstance(dev, str):
        dev = device(dev)
    elif dev is not None and not isinstance(dev, device):
        dev = device(getattr(dev, "type", dev), getattr(dev, "index", None))
    return dev, dtype_arg, non_blocking, kwargs.get("memory_format", None)


class _CudnnBackendModule(_types.ModuleType):
    # `allow_tf32` is a *view* of the jittor flag, not a stored
    # attribute, so it cannot disagree with the other two spellings of
    # the same switch (`cudnn.conv.fp32_precision`, `cudnn.rnn.
    # fp32_precision`). Reading goes through __getattribute__ because a
    # module cannot carry a property.
    def __getattribute__(self, name):
        if name == "allow_tf32":
            return _tf32_get("cudnn")
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        # `benchmark` is the one setting whose write has a side effect
        # on the runtime rather than on a flag, so the install-time
        # default assignment below must not fire it: writing the
        # default back would call set_benchmark(0) and clobber whatever
        # the process had already chosen. `allow_tf32` needs no such
        # gate -- it reads and writes the same flag, so assigning its
        # own current value at install time is a no-op by construction.
        if name == "benchmark" and not getattr(self, "_jittor_cudnn_init", False):
            try:
                if getattr(jt, "cudnn", None) is not None and hasattr(jt.cudnn, "set_benchmark"):
                    jt.cudnn.set_benchmark(int(bool(value)))
            except EXPECTED as exc:
                swallowed("torch/installers/cuda/api.py cudnn.__setattr__: "
                          "jt.cudnn.set_benchmark(%r)" % (value,), exc,
                          "cuDNN autotuning stays as it was")
        if name == "allow_tf32":
            _tf32_set("cudnn", value)
            return None
        return super().__setattr__(name, value)


class _SDPKernel:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _MatmulBackend:
    def __init__(self):
        self.allow_fp16_reduced_precision_reduction = True
        self.allow_bf16_reduced_precision_reduction = True

    @property
    def allow_tf32(self):
        return _tf32_get("matmul")

    @allow_tf32.setter
    def allow_tf32(self, value):
        _tf32_set("matmul", value)

    @property
    def fp32_precision(self):
        return _tf32_to_precision(_tf32_get("matmul"))

    @fp32_precision.setter
    def fp32_precision(self, value):
        _tf32_set("matmul", _precision_to_tf32(
            value, "torch.backends.cuda.matmul"))


def _preferred_blas_library(backend=None):
    previous = _cuda_target().backends.cuda._preferred_blas_library
    if backend is not None:
        _cuda_target().backends.cuda._preferred_blas_library = str(backend)
    return previous


def _get_float32_matmul_precision():
    return _cuda_runtime().matmul_precision


def _set_float32_matmul_precision(precision):
    if not isinstance(precision, str):
        raise TypeError("precision must be a string")
    precision = precision.lower()
    if precision not in ("highest", "high", "medium"):
        raise ValueError("precision must be one of 'highest', 'high', or 'medium'")
    _set_precision_tier("matmul", precision)


def _api_cuda_is_initialized(*a, **k):
    return bool(is_available() and getattr(jt.flags, 'use_cuda', 0))


def _api_cuda__is_in_bad_fork(*a, **k):
    return False


def _api_cuda_synchronize(*a, **k):
    return jt.sync_all(True)


def _api_cuda_manual_seed(s):
    return jt.set_global_seed(int(s))


def _api_cuda_manual_seed_all(s):
    return jt.set_global_seed(int(s))


def _api_cuda_is_bf16_supported():
    return True


def _api_cuda_get_device_capability(*a, **k):
    return _cuda_capability()


def _api_cuda_get_device_properties(*a, **k):
    return _DeviceProps()


def _api_cuda_default_stream(*a, **k):
    return _cuda_runtime().default_stream


def _api_cuda_memory_stats(*a, **k):
    return {'allocated_bytes.all.current': _mem_used(), 'allocated_bytes.all.peak': _cuda_runtime().mem_peak[0]}


def _api_cuda_ipc_collect(*a, **k):
    return None


def _api_cuda_memory__set_allocator_settings(*a, **k):
    return None


def _api_cuda_get_rng_state(*a, **k):
    return jt.array([0], dtype='uint8')


def _api_cuda_get_rng_state_all(*a, **k):
    return [jt.array([0], dtype='uint8')]


def _api_cuda_set_rng_state(*a, **k):
    return None


def _api_cuda_set_rng_state_all(*a, **k):
    return None


def _api_cuda_initial_seed(*a, **k):
    return 0


def _api_cuda_seed(*a, **k):
    return None


def _api_cuda_seed_all(*a, **k):
    return None


def _api_mp_reductions_reduce_tensor(tensor):
    return (_identity, (tensor,))


def _api_mp_reductions_rebuild_cuda_tensor(*a, **k):
    return None


def _api_mp_reductions_rebuild_tensor(*a, **k):
    return a[0] if a else None


def _api_g__C__autograd__push_saved_tensors_default_hooks(*a, **k):
    return None


def _api_g__C__autograd__pop_saved_tensors_default_hooks(*a, **k):
    return None


def _api_cudnn_version():
    return None


def _api_cuda_backend_sdp_kernel(*a, **k):
    return _SDPKernel()


def _api_cuda_backend_enable_flash_sdp(*a, **k):
    return None


def _api_cuda_backend_enable_mem_efficient_sdp(*a, **k):
    return None


def _api_cuda_backend_enable_math_sdp(*a, **k):
    return None


def _api_cuda_backend_enable_cudnn_sdp(*a, **k):
    return None


def _api_mps_is_available():
    return False


def _api_cpu_get_cpu_capability():
    return 'DEFAULT'


def _api_mkldnn_is_available():
    return False


def _api_mod_is_available(*a, **k):
    return False


def _api_mod_is_initialized(*a, **k):
    return False


def _api_mod_device_count(*a, **k):
    return 0


def _api_mod_current_device(*a, **k):
    return 0


def _api_mod_set_device(*a, **k):
    return None


def _api_mod_empty_cache(*a, **k):
    return None


def _api_mod_synchronize(*a, **k):
    return None


def _api_mod_ipc_collect(*a, **k):
    return None


def _api_mod_manual_seed(*a, **k):
    return None


def _api_mod_manual_seed_all(*a, **k):
    return None


def _api_mod_seed(*a, **k):
    return None


def _api_mod_reset_peak_memory_stats(*a, **k):
    return None


def _api_mod_reset_max_memory_allocated(*a, **k):
    return None


def _api_mod_memory_allocated(*a, **k):
    return 0


def _api_mod_max_memory_allocated(*a, **k):
    return 0


def _api_overrides_get_default_nowrap_functions():
    return set()


def _api_c_mod__get_tracing_state():
    return None


def _api_c_mod__log_api_usage_once(*a, **k):
    return None


def _api_c_mod__cuda_clearCublasWorkspaces(*a, **k):
    return None


def _api_c_mod__disabled_torch_function_impl(*a, **k):
    return NotImplemented


def _api_functorch_c_get_unwrapped(x):
    return x


def _api_functorch_c_is_batchedtensor(*a, **k):
    return False


def _api_functorch_c__add_batch_dim(x, *a, **k):
    return x


def _api_functorch_c__remove_batch_dim(x, *a, **k):
    return x


def _api_c_mod__accelerator_setAllocatorSettings(*args, **kwargs):
    return None


def _api_c_mod__cuda_setAllocatorSettings(*args, **kwargs):
    return None


def _api_accelerator_is_available(*a, **k):
    return True


def _api_accelerator_current_device_index(*a, **k):
    return _cuda_target().cuda.current_device()


def _api_accelerator_set_device_index(d, *a, **k):
    return _cuda_target().cuda.set_device(d)


def _api_accelerator_set_stream(*a, **k):
    return None


def _api_accelerator_current_accelerator(*a, **k):
    return _cuda_target().device('cuda')


_CUDA_FIDELITY_DETAILS = {
    _Stream: "Logical streams share native execution; no independent CUDA stream handle.",
    _StreamContext: "Thread-local logical stream selection; native execution remains serialized.",
    _current_stream: "Thread-local logical stream identity; device argument is not implemented.",
    _set_stream: "Selects a logical stream; does not rebind the native backend stream.",
    _api_cuda_default_stream: "One installation-owned logical default stream, not one per device.",
    _Event: "Host timestamps after synchronization; not native CUDA event timing.",
    _mem_used: "Native live-byte accounting; reserved/cached aliases do not report pool reservation.",
    _mem_max: "High-water mark sampled at memory queries, not every allocation.",
    _reset_peak: "Resets the sampled live-byte high-water mark.",
    _api_cuda_memory_stats: "Current and sampled peak live bytes only; other PyTorch counters absent.",
    _mem_get_info: "cudaMemGetInfo when available; native live-byte fallback excludes other processes.",
    _empty_cache: "Environment-selected no-op, GC or synchronized GC; default is a memory hint.",
    _get_float32_matmul_precision: "Frontend-owned CUDA matmul policy; native Jittor and cuDNN policies are independent.",
    _set_float32_matmul_precision: "Frontend-owned highest/high/medium CUDA matmul accumulation; does not change cuDNN or native Jittor.",
}

_CUDA_PLACEHOLDERS = frozenset((
    CUDAGraph, CUDAPluggableAllocator, TorchFunctionMode,
    _api_cuda_get_rng_state, _api_cuda_get_rng_state_all,
    _api_cuda_set_rng_state, _api_cuda_set_rng_state_all,
    _api_cuda_initial_seed, _api_cuda_seed, _api_cuda_seed_all,
    _api_cuda_ipc_collect, _api_cuda_memory__set_allocator_settings,
    _api_mp_reductions_rebuild_cuda_tensor,
    _api_g__C__autograd__push_saved_tensors_default_hooks,
    _api_g__C__autograd__pop_saved_tensors_default_hooks,
    _api_cudnn_version, _api_cuda_backend_sdp_kernel,
    _api_cuda_backend_enable_flash_sdp, _api_cuda_backend_enable_mem_efficient_sdp,
    _api_cuda_backend_enable_math_sdp, _api_cuda_backend_enable_cudnn_sdp,
    _api_c_mod__get_tracing_state, _api_c_mod__cuda_clearCublasWorkspaces,
    _api_functorch_c_get_unwrapped, _api_functorch_c_is_batchedtensor,
    _api_functorch_c__add_batch_dim, _api_functorch_c__remove_batch_dim,
    _api_c_mod__accelerator_setAllocatorSettings, _api_c_mod__cuda_setAllocatorSettings,
    _api_accelerator_is_available, _api_accelerator_set_stream,
    _nvtx_mark, _nvtx_range_start, _nvtx_range_end,
))


def _register_cuda_fidelity(ctx):
    """Describe the APIs this owner actually published, including old stubs."""
    modules = ctx.registry.module_map
    namespaces = [("torch", ctx.target_namespace)]
    prefixes = ("torch.cuda", "torch.backends", "torch.accelerator", "torch._C")
    ancillary = {"torch.overrides", "torch.multiprocessing.reductions",
                 "torch.mps", "torch.cpu", "torch.npu", "torch.xpu", "torch.mtia"}
    namespaces.extend((name, module) for name, module in modules.items()
                      if name in ancillary or any(name == prefix or name.startswith(prefix + ".")
                                                  for prefix in prefixes))
    for namespace, module in namespaces:
        for name, implementation in tuple(vars(module).items()):
            if not callable(implementation) or getattr(implementation, "__module__", None) != __name__:
                continue
            placeholder = implementation in _CUDA_PLACEHOLDERS
            if implementation.__name__.startswith("_api_mod_"):
                placeholder = True
            detail = _CUDA_FIDELITY_DETAILS.get(implementation,
                "Compatibility facade with existing signature/device limitations; full PyTorch equivalence is not claimed.")
            if placeholder:
                detail = "Unsupported compatibility placeholder; existing stub or refusal retained, not a working implementation."
            register_fidelity(namespace + "." + name, implementation,
                              Fidelity.UNIMPLEMENTED if placeholder else Fidelity.APPROXIMATE, detail)
