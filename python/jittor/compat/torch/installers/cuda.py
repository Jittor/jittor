"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt

from ..context import registry_for

from ..grad import (
    _amp_passthrough_decorator, _AutocastContext,
    _GradScaler,
)
from ..types import (
    device, dtype, _cuda_index_of, _device_is_cpu,
)
from ...diagnostics import EXPECTED, swallowed

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
                swallowed("torch/installers/cuda.py _cuda_driver: lib = ctypes.CDLL(n)", exc)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda.py _cuda_driver: import ctypes", exc)
    return None, None


def _cuda_device_index(device=None):
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py _cuda_device_index: return int(device.split(':', 1)[1])", exc)
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
        swallowed("torch/installers/cuda.py _cuda_device_name: lib, ctypes = _cuda_driver()", exc)
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
        swallowed("torch/installers/cuda.py _cuda_capability: lib, ctypes = _cuda_driver()", exc)
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
        swallowed("torch/installers/cuda.py _cuda_sm_count: lib, ctypes = _cuda_driver()", exc)
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
        swallowed("torch/installers/cuda.py _cuda_total_memory: lib, ctypes = _cuda_driver()", exc)
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
_TF32_FLAGS = {
    "matmul": "cuda_allow_tf32",
    "cudnn": "cuda_allow_cudnn_tf32",
}

#: The two ``fp32_precision`` values this layer can actually deliver. torch
#: also accepts "bf16" and "none"; Jittor has no separate bf16-accumulate mode
#: and no per-op override, so accepting them would be inventing a semantics.
_FP32_PRECISIONS = ("ieee", "tf32")


#: Where a domain's setting lives on a build that has no such ``jt.flags``
#: entry -- a CPU-only or pre-CUDA Jittor. Without it, ``cudnn.allow_tf32 =
#: True`` there was a silent no-op that read back ``False``: the caller asked
#: for something, got no error, and got the opposite answer. Real torch on a
#: CPU-only build round-trips the setting too (inert, but honest), and keeping
#: it here is what lets the six spellings agree on *every* build rather than
#: only where the flags happen to exist.
_TF32_FALLBACK = {}


def _tf32_get(domain):
    """Whether reduced-precision fp32 math is enabled for ``domain``."""
    flag = _TF32_FLAGS[domain]
    if hasattr(jt.flags, flag):
        enabled = bool(getattr(jt.flags, flag))
    else:
        enabled = bool(_TF32_FALLBACK.get(domain, False))
    if domain == "matmul":
        # Ascend spells the same idea hf32, and it is not a jt.flags entry.
        enabled = enabled or bool(getattr(jt, "acl_allow_hf32", False))
    return enabled


def _tf32_set(domain, value):
    """Point every spelling of ``domain``'s switch at ``value``."""
    enabled = bool(value)
    flag = _TF32_FLAGS[domain]
    if hasattr(jt.flags, flag):
        setattr(jt.flags, flag, int(enabled))
    else:
        _TF32_FALLBACK[domain] = enabled
    if domain == "matmul":
        jt.acl_allow_hf32 = enabled
    return enabled


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


def _install_cuda(g, registry=None):
    _modules = registry_for(g, registry).module_map
    import contextlib
    import itertools
    import threading
    import types as _types
    cuda = _types.ModuleType("torch.cuda")
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
            swallowed("torch/installers/cuda.py is_available: if _cuda_visible_devices_empty():", exc)
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
            swallowed("torch/installers/cuda.py device_count: "
                      "n = int(jt.get_device_count())", exc,
                      "falling back to counting CUDA_VISIBLE_DEVICES")
        try:
            import os as _os_cuda
            _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
            if _cvd is not None:
                return len([_d for _d in _cvd.split(",") if _d.strip()])
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py device_count: import os as _os_cuda", exc)
        return 1
    cuda.is_available = is_available
    cuda.device_count = device_count
    # Whether torch itself was built with CUDA, as distinct from whether a
    # device is present. Here the two are the same question.
    cuda._is_compiled = is_available
    # torch counts devices twice: once through the driver and once through
    # NVML, so that it can answer before CUDA is initialised. Here both
    # questions go to the same place.
    cuda._device_count_nvml = device_count
    # Devices are real: every Var carries the CUDA device it lives on and
    # jittor keeps a current device that new tensors are placed on -- torch's
    # model exactly. This used to be `lambda: 0` next to a set_device that
    # refused anything but 0.
    def current_device():
        try:
            d = int(jt.current_device())
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py current_device: "
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

    cuda.current_device = current_device
    cuda.set_device = set_device

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
    cuda.device = _CudaDeviceContext

    class _CudaDeviceOf(_CudaDeviceContext):
        """``with torch.cuda.device_of(tensor):`` -- the tensor's own device."""
        def __init__(self, tensor):
            idx = None
            if isinstance(tensor, jt.Var):
                try:
                    got = int(tensor.device_id)
                except EXPECTED as exc:
                    swallowed("torch/installers/cuda.py device_of: "
                              "got = int(tensor.device_id)", exc,
                              "the context will not switch device")
                    got = -1
                if got >= 0:
                    idx = got
            super().__init__(idx)
    cuda.device_of = _CudaDeviceOf
    cuda.is_initialized = lambda *a, **k: bool(is_available() and getattr(jt.flags, "use_cuda", 0))
    cuda._is_in_bad_fork = lambda *a, **k: False
    # Match PyTorch's empty_cache() as a memory hint instead of a forced
    # synchronization point. TRELLIS calls it inside the inference path before
    # decode; running jt.gc() there costs several seconds. Users that need
    # explicit release can opt in with JITTOR_TORCH_CUDA_EMPTY_CACHE=gc or sync.
    try:
        import os as _os_empty_cache
        _empty_cache_mode = str(_os_empty_cache.environ.get(
            "JITTOR_TORCH_CUDA_EMPTY_CACHE", "0")).strip().lower()
    except EXPECTED as exc:
        swallowed("torch/installers/cuda.py _install_cuda: import os as _os_empty_cache", exc)
        _empty_cache_mode = "0"

    def _empty_cache():
        if _empty_cache_mode in ("0", "false", "no", "off", "none", "noop"):
            return
        if _empty_cache_mode in ("", "1", "true", "yes", "on", "gc"):
            try:
                jt.gc()
            except EXPECTED as exc:
                swallowed("torch/installers/cuda.py _empty_cache: jt.gc()", exc)
        elif _empty_cache_mode in ("sync", "full"):
            try:
                jt.sync_all(True)
            except EXPECTED as exc:
                swallowed("torch/installers/cuda.py _empty_cache: jt.sync_all(True)", exc)
            try:
                jt.gc()
            except EXPECTED as exc:
                swallowed("torch/installers/cuda.py _empty_cache: jt.gc()", exc)
    cuda.empty_cache = _empty_cache
    cuda.synchronize = lambda *a, **k: jt.sync_all(True)
    cuda.manual_seed = lambda s: jt.set_global_seed(int(s))
    cuda.manual_seed_all = lambda s: jt.set_global_seed(int(s))
    cuda.is_bf16_supported = lambda: True
    cuda.get_device_capability = lambda *a, **k: _cuda_capability()
    def _device_name(*a, **k):
        try:
            return "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(a[0] if a else None)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py _device_name: return 'Ascend910B/NPU' if getattr(jt.compiler, 'has_ac...", exc)
            return "CUDA"
    cuda.get_device_name = _device_name
    cuda.get_device_properties = lambda *a, **k: _DeviceProps()
    class _amp:
        @staticmethod
        def autocast(device_type="cuda", *a, **k):
            return _AutocastContext(device_type, *a, **k)
        GradScaler = _GradScaler
        custom_fwd = staticmethod(_amp_passthrough_decorator)
        custom_bwd = staticmethod(_amp_passthrough_decorator)
    cuda.amp = _amp
    # OpenMMLab imports these legacy CUDA tensor classes in type annotations.
    # Keep them distinct from the top-level CPU classes: a direct alias would
    # make a host tensor pass ``isinstance(x, torch.cuda.LongTensor)``.
    class _CudaTypedTensorMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, cls._base_type) and bool(getattr(obj, "is_cuda", False))

        def __call__(cls, *args, **kwargs):
            if not cuda.is_available():
                raise RuntimeError("CUDA is not available")
            return cls._base_type(*args, **kwargs).cuda()

    for _tensor_name in (
        "FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
        "LongTensor", "IntTensor", "ShortTensor", "CharTensor",
        "ByteTensor", "BoolTensor",
    ):
        _tensor_type = getattr(g, _tensor_name, None)
        if _tensor_type is not None:
            setattr(cuda, _tensor_name, _CudaTypedTensorMeta(
                _tensor_name,
                (),
                {"_base_type": _tensor_type, "__module__": "torch.cuda"},
            ))
    # stub classes referenced in annotations / guarded paths
    cuda.CUDAGraph = type("CUDAGraph", (), {})
    class _Stream:
        def __init__(self, device=None, priority=0, *a, **k):
            self.cuda_stream = 0
            if device is None:
                self.device = g.device("cuda", cuda.current_device())
            elif isinstance(device, int):
                self.device = g.device("cuda", device)
            else:
                self.device = g.device(device)
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
    cuda.Stream = _Stream

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
                swallowed("torch/installers/cuda.py record: jt.sync_all(True)", exc)
            self._time = _time_event.perf_counter()
            return None

        def synchronize(self):
            try:
                jt.sync_all(True)
            except EXPECTED as exc:
                swallowed("torch/installers/cuda.py synchronize: jt.sync_all(True)", exc)

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

    cuda.Event = _Event
    g.Stream = cuda.Stream
    g.Event = cuda.Event
    g.CUDAGraph = cuda.CUDAGraph
    # Jittor currently launches CUDA/ACL work on one physical backend stream.
    # Keep the Python-visible current-stream identity coherent while all logical
    # streams remain serialized on that physical stream.
    _stream_state = threading.local()
    _default_stream = _Stream()
    def _current_stream(*a, **k):
        return getattr(_stream_state, "current", _default_stream)
    def _set_stream(stream):
        if stream is None:
            return None
        if not isinstance(stream, _Stream):
            raise TypeError("set_stream expects a torch.cuda.Stream or None")
        _stream_state.current = stream
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
    cuda.stream = _StreamContext
    cuda.set_stream = _set_stream
    cuda.current_stream = _current_stream
    cuda.default_stream = lambda *a, **k: _default_stream
    nvtx = _types.ModuleType("torch.cuda.nvtx")
    _nvtx_state = threading.local()
    _nvtx_handles = itertools.count(1)
    _native_nvtx = [None]

    def _load_native_nvtx():
        if _native_nvtx[0] is False:
            return None
        if _native_nvtx[0] is None:
            previous_utils = getattr(g, "utils", None)
            try:
                from jittor.tools import nvtx as native_nvtx
                _native_nvtx[0] = native_nvtx
            except (ImportError, OSError, RuntimeError):
                _native_nvtx[0] = False
            finally:
                # Importing ``jittor.utils`` binds it on the shared jittor/torch
                # root and would replace the published ``torch.utils`` module.
                if previous_utils is not None:
                    g.utils = previous_utils
        return _native_nvtx[0] or None

    def _nvtx_stack():
        stack = getattr(_nvtx_state, "stack", None)
        if stack is None:
            stack = []
            _nvtx_state.stack = stack
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
        return next(_nvtx_handles)

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

    nvtx.range_push = _nvtx_range_push
    nvtx.range_pop = _nvtx_range_pop
    nvtx.range_start = _nvtx_range_start
    nvtx.range_end = _nvtx_range_end
    nvtx.mark = _nvtx_mark
    nvtx.range = _nvtx_range
    nvtx.__all__ = [
        "range_push", "range_pop", "range_start", "range_end", "mark", "range"
    ]
    cuda.nvtx = nvtx
    _modules["torch.cuda.nvtx"] = nvtx
    # report REAL device memory from jittor's MemInfo (was a 0-stub, so training-code
    # memory logging printed 0). total_cuda_used on an accelerator, else total_cpu_used.
    # jittor doesn't expose a per-reset peak, so max_* track a process-lifetime high-water
    # mark we maintain here (still real, monotone -- better than a flat 0).
    _mem_peak = [0]
    def _mem_used(*a, **k):
        try:
            mi = jt.get_mem_info()
            used = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py _mem_used: mi = jt.get_mem_info()", exc)
            used = 0
        if used > _mem_peak[0]:
            _mem_peak[0] = used
        return used
    def _mem_max(*a, **k):
        _mem_used()
        return _mem_peak[0]
    cuda.memory_allocated = _mem_used
    cuda.max_memory_allocated = _mem_max
    cuda.memory_reserved = _mem_used
    cuda.max_memory_reserved = _mem_max
    cuda.memory_cached = _mem_used
    cuda.max_memory_cached = _mem_max
    def _reset_peak(*a, **k):
        try:
            mi = jt.get_mem_info()
            _mem_peak[0] = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
        except EXPECTED as exc:
            swallowed("torch/installers/cuda.py _reset_peak: mi = jt.get_mem_info()", exc)
            _mem_peak[0] = 0
    cuda.reset_peak_memory_stats = _reset_peak
    cuda.reset_max_memory_allocated = _reset_peak
    cuda.memory_stats = lambda *a, **k: {"allocated_bytes.all.current": _mem_used(),
                                         "allocated_bytes.all.peak": _mem_peak[0]}
    # torch's mem_get_info is cudaMemGetInfo: the DRIVER's free/total for the whole
    # device, counting other processes, the CUDA context and every byte jittor's
    # pool holds -- not just the bytes currently live in Vars. Serving stacks size
    # their weight/activation/KV budget from it (vLLM plans the KV cache as
    # `total*util - (total-free) - peak_activation`), so the flat 64GiB stub that
    # used to stand here made them plan against a device that does not exist.
    _memgetinfo = [None]
    def _cuda_mem_get_info_fn():
        if _memgetinfo[0] is not None:
            return _memgetinfo[0]
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
            swallowed("torch/installers/cuda.py _cuda_mem_get_info_fn: import ctypes as _ct", exc)
            fn = False
        _memgetinfo[0] = fn
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
            swallowed("torch/installers/cuda.py _mem_get_info: mi = jt.get_mem_info()", exc)
            return (0, 0)
    cuda.mem_get_info = _mem_get_info
    cuda.ipc_collect = lambda *a, **k: None
    cuda.memory = _types.ModuleType("torch.cuda.memory")
    cuda.memory._set_allocator_settings = lambda *a, **k: None
    cuda.memory.empty_cache = cuda.empty_cache
    cuda.memory.memory_allocated = cuda.memory_allocated
    cuda.memory.max_memory_allocated = cuda.max_memory_allocated
    cuda.memory.memory_reserved = cuda.memory_reserved
    cuda.memory.max_memory_reserved = cuda.max_memory_reserved
    class CUDAPluggableAllocator:
        def __init__(self, path_to_so_file, alloc_fn_name, free_fn_name):
            raise NotImplementedError(
                "Jittor does not support PyTorch CUDA pluggable allocators"
            )
    cuda.memory.CUDAPluggableAllocator = CUDAPluggableAllocator
    cuda.CUDAPluggableAllocator = CUDAPluggableAllocator
    # rng state (trainer checkpoints save/restore it). jittor has no portable
    # CUDA rng-state handle, so use a small placeholder Var round-trip.
    cuda.get_rng_state = lambda *a, **k: jt.array([0], dtype="uint8")
    cuda.get_rng_state_all = lambda *a, **k: [jt.array([0], dtype="uint8")]
    cuda.set_rng_state = lambda *a, **k: None
    cuda.set_rng_state_all = lambda *a, **k: None
    cuda.initial_seed = lambda *a, **k: 0
    cuda.seed = lambda *a, **k: None
    cuda.seed_all = lambda *a, **k: None
    import types as _types_cuda
    _curandom = _types_cuda.ModuleType("torch.cuda.random")
    _curandom.get_rng_state = cuda.get_rng_state
    _curandom.get_rng_state_all = cuda.get_rng_state_all
    _curandom.set_rng_state = cuda.set_rng_state
    _curandom.set_rng_state_all = cuda.set_rng_state_all
    _curandom.manual_seed = cuda.manual_seed
    _curandom.manual_seed_all = cuda.manual_seed_all
    _curandom.initial_seed = cuda.initial_seed
    cuda.random = _curandom
    _modules["torch.cuda.random"] = _curandom
    g.cuda = cuda
    _modules["torch.cuda"] = cuda
    _modules["torch.cuda.memory"] = cuda.memory
    if hasattr(cuda, "amp"):
        _modules["torch.cuda.amp"] = cuda.amp

    for _dev_ns in ("mps", "cpu", "npu", "xpu", "mtia"):
        _mod = _modules.get("torch." + _dev_ns)
        if _mod is None:
            _mod = _types.ModuleType("torch." + _dev_ns)
            _modules["torch." + _dev_ns] = _mod
        _mod.is_available = getattr(_mod, "is_available", lambda *a, **k: False)
        _mod.is_initialized = getattr(_mod, "is_initialized", lambda *a, **k: False)
        _mod.device_count = getattr(_mod, "device_count", lambda *a, **k: 0)
        _mod.current_device = getattr(_mod, "current_device", lambda *a, **k: 0)
        _mod.set_device = getattr(_mod, "set_device", lambda *a, **k: None)
        _mod.empty_cache = getattr(_mod, "empty_cache", lambda *a, **k: None)
        _mod.synchronize = getattr(_mod, "synchronize", lambda *a, **k: None)
        _mod.ipc_collect = getattr(_mod, "ipc_collect", lambda *a, **k: None)
        _mod.manual_seed = getattr(_mod, "manual_seed", lambda *a, **k: None)
        _mod.manual_seed_all = getattr(_mod, "manual_seed_all", lambda *a, **k: None)
        _mod.seed = getattr(_mod, "seed", lambda *a, **k: None)
        _mod.reset_peak_memory_stats = getattr(_mod, "reset_peak_memory_stats", lambda *a, **k: None)
        _mod.reset_max_memory_allocated = getattr(_mod, "reset_max_memory_allocated", lambda *a, **k: None)
        _mod.memory_allocated = getattr(_mod, "memory_allocated", lambda *a, **k: 0)
        _mod.max_memory_allocated = getattr(_mod, "max_memory_allocated", lambda *a, **k: 0)
        setattr(g, _dev_ns, _mod)

    if "torch.multiprocessing" not in _modules:
        import multiprocessing as _mp
        _modules["torch.multiprocessing"] = _mp
    g.multiprocessing = _modules["torch.multiprocessing"]
    _mp_reductions = _types.ModuleType("torch.multiprocessing.reductions")
    _mp_reductions.reduce_tensor = lambda tensor: (lambda x: x, (tensor,))
    _mp_reductions.rebuild_cuda_tensor = lambda *a, **k: None
    _mp_reductions.rebuild_tensor = lambda *a, **k: a[0] if a else None
    _modules["torch.multiprocessing.reductions"] = _mp_reductions
    try:
        g.multiprocessing.reductions = _mp_reductions
    except (AttributeError, TypeError) as exc:
        swallowed("torch/installers/cuda.py _install_cuda: g.multiprocessing.reductions = _mp_reductions", exc)

    if "torch.overrides" not in _modules:
        from ...stub_policy import unimplemented as _unimplemented
        overrides = _types.ModuleType("torch.overrides")

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

        overrides.TorchFunctionMode = TorchFunctionMode
        overrides.BaseTorchFunctionMode = TorchFunctionMode
        overrides.get_default_nowrap_functions = lambda: set()
        overrides.has_torch_function = _has_torch_function
        overrides.has_torch_function_unary = _has_torch_function
        overrides.has_torch_function_variadic = _has_torch_function
        overrides.handle_torch_function = _handle_torch_function
        _modules["torch.overrides"] = overrides
    g.overrides = _modules["torch.overrides"]

    if "torch._C" not in _modules:
        c_mod = _types.ModuleType("torch._C")
        c_mod._TensorMeta = type(getattr(g, "Tensor", jt.Var))
        c_mod._get_tracing_state = lambda: None
        c_mod._log_api_usage_once = lambda *a, **k: None
        c_mod._cuda_clearCublasWorkspaces = lambda *a, **k: None
        c_mod._disabled_torch_function_impl = lambda *a, **k: NotImplemented
        functorch_c = _types.ModuleType("torch._C._functorch")
        functorch_c.get_unwrapped = lambda x: x
        functorch_c.is_batchedtensor = lambda *a, **k: False
        functorch_c._add_batch_dim = lambda x, *a, **k: x
        functorch_c._remove_batch_dim = lambda x, *a, **k: x
        c_mod._distributed_c10d = _types.SimpleNamespace(Reducer=type("Reducer", (), {}))
        nn_c = _types.ModuleType("torch._C._nn")
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
        nn_c._parse_to = _parse_to
        c_mod._nn = nn_c
        c_mod._functorch = functorch_c
        # Allocator tuning arrives as a settings string that the caching
        # allocator parses. Jittor manages its own pool, so there is nothing to
        # tune -- but the call has to exist, because the caller makes it before
        # asking whether it could have had any effect.
        c_mod._accelerator_setAllocatorSettings = lambda *args, **kwargs: None
        c_mod._cuda_setAllocatorSettings = getattr(
            c_mod, "_cuda_setAllocatorSettings", lambda *args, **kwargs: None)
        _modules["torch._C"] = c_mod
        _modules["torch._C._nn"] = nn_c
        _modules["torch._C._functorch"] = functorch_c
    g._C = _modules["torch._C"]
    if not hasattr(g._C, "_autograd"):
        g._C._autograd = _types.SimpleNamespace()
    g._C._autograd._push_saved_tensors_default_hooks = lambda *a, **k: None
    g._C._autograd._pop_saved_tensors_default_hooks = lambda *a, **k: None
    _modules["torch._C._autograd"] = g._C._autograd

    backends = _modules.get("torch.backends")
    if backends is None:
        backends = _types.ModuleType("torch.backends")
        _modules["torch.backends"] = backends
    cudnn = _modules.get("torch.backends.cudnn")
    if cudnn is None:
        cudnn = _types.ModuleType("torch.backends.cudnn")
        _modules["torch.backends.cudnn"] = cudnn
    if type(cudnn).__name__ != "_CudnnBackendModule":
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
                        swallowed("torch/installers/cuda.py cudnn.__setattr__: "
                                  "jt.cudnn.set_benchmark(%r)" % (value,), exc,
                                  "cuDNN autotuning stays as it was")
                if name == "allow_tf32":
                    _tf32_set("cudnn", value)
                    return None
                return super().__setattr__(name, value)
        cudnn.__class__ = _CudnnBackendModule
    cudnn._jittor_cudnn_init = True
    cudnn.enabled = getattr(cudnn, "enabled", True)
    cudnn.benchmark = getattr(cudnn, "benchmark", False)
    cudnn.deterministic = getattr(cudnn, "deterministic", False)
    cudnn.version = getattr(cudnn, "version", lambda: None)
    if not isinstance(getattr(cudnn, "conv", None), _PrecisionBackend):
        cudnn.conv = _PrecisionBackend("cudnn", "torch.backends.cudnn.conv")
    if not isinstance(getattr(cudnn, "rnn", None), _PrecisionBackend):
        cudnn.rnn = _PrecisionBackend("cudnn", "torch.backends.cudnn.rnn")
    cudnn._jittor_cudnn_init = False
    cuda_backend = _modules.get("torch.backends.cuda")
    if cuda_backend is None:
        cuda_backend = _types.ModuleType("torch.backends.cuda")
        _modules["torch.backends.cuda"] = cuda_backend
    class _SDPKernel:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    cuda_backend.sdp_kernel = getattr(cuda_backend, "sdp_kernel", lambda *a, **k: _SDPKernel())
    cuda_backend.enable_flash_sdp = getattr(cuda_backend, "enable_flash_sdp", lambda *a, **k: None)
    cuda_backend.enable_mem_efficient_sdp = getattr(cuda_backend, "enable_mem_efficient_sdp", lambda *a, **k: None)
    cuda_backend.enable_math_sdp = getattr(cuda_backend, "enable_math_sdp", lambda *a, **k: None)
    # The fourth of torch's attention-backend switches. Attention here picks
    # its own path, so all four are settings nothing acts on -- but a serving
    # stack turns cuDNN's off during platform detection, and an AttributeError
    # there is swallowed into "no platform detected" rather than reported.
    cuda_backend.enable_cudnn_sdp = getattr(cuda_backend, "enable_cudnn_sdp", lambda *a, **k: None)
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
    if not hasattr(cuda_backend, "matmul") or not isinstance(cuda_backend.matmul, _MatmulBackend):
        cuda_backend.matmul = _MatmulBackend()
    cuda_backend._preferred_blas_library = getattr(
        cuda_backend, "_preferred_blas_library", "cublas")
    def _preferred_blas_library(backend=None):
        previous = cuda_backend._preferred_blas_library
        if backend is not None:
            cuda_backend._preferred_blas_library = str(backend)
        return previous
    cuda_backend.preferred_blas_library = _preferred_blas_library
    mps = _modules.get("torch.backends.mps")
    if mps is None:
        mps = _types.ModuleType("torch.backends.mps")
        _modules["torch.backends.mps"] = mps
    mps.is_available = getattr(mps, "is_available", lambda: False)
    cpu = _modules.get("torch.backends.cpu")
    if cpu is None:
        cpu = _types.ModuleType("torch.backends.cpu")
        _modules["torch.backends.cpu"] = cpu
    cpu.get_cpu_capability = getattr(cpu, "get_cpu_capability", lambda: "DEFAULT")
    mkldnn = _modules.get("torch.backends.mkldnn")
    if mkldnn is None:
        mkldnn = _types.ModuleType("torch.backends.mkldnn")
        _modules["torch.backends.mkldnn"] = mkldnn
    mkldnn.is_available = getattr(mkldnn, "is_available", lambda: False)
    mkldnn.enabled = getattr(mkldnn, "enabled", False)
    backends.cudnn = cudnn
    backends.cuda = cuda_backend
    backends.mps = mps
    backends.cpu = cpu
    backends.mkldnn = mkldnn
    g.backends = backends
    # The third spelling of the matmul switch. torch offers a three-step
    # ladder; jittor has one tf32 flag, so "high" and "medium" both mean it is
    # on. Only that refinement is remembered -- the on/off half is *derived*
    # from the flag, which is what keeps this from drifting away from
    # `torch.backends.cuda.matmul.allow_tf32`. It used to be an independent
    # string, so after `matmul.allow_tf32 = True` this still answered
    # "highest", and after `set_float32_matmul_precision("high")` a reader of
    # `cudnn.conv.fp32_precision` still saw "ieee".
    def _get_float32_matmul_precision():
        if not _tf32_get("matmul"):
            return "highest"
        return getattr(g, "_torch_float32_matmul_refinement", "high")
    def _set_float32_matmul_precision(precision):
        if not isinstance(precision, str):
            raise TypeError("precision must be a string")
        precision = precision.lower()
        if precision not in ("highest", "high", "medium"):
            raise ValueError("precision must be one of 'highest', 'high', or 'medium'")
        if precision != "highest":
            g._torch_float32_matmul_refinement = precision
        _tf32_set("matmul", precision != "highest")
    g.get_float32_matmul_precision = _get_float32_matmul_precision
    g.set_float32_matmul_precision = _set_float32_matmul_precision


def _install_version(g, registry=None):
    """Install torch.version for libraries that probe torch.cuda/hip versions."""
    _modules = registry_for(g, registry).module_map
    import types as _types
    torch_api_version = "2.11.0"
    jittor_version = getattr(g, "__jittor_version__", getattr(g, "__version__", getattr(jt, "__version__", None)))
    g.__jittor_version__ = jittor_version
    g.__torch_version__ = torch_api_version
    version = _types.ModuleType("torch.version")
    version.__version__ = torch_api_version
    version.jittor = jittor_version
    try:
        nv = getattr(getattr(jt, "compiler", None), "nvcc_version", None)
        version.cuda = ".".join(map(str, nv[:2])) if nv else None
    except EXPECTED as exc:
        swallowed("torch/installers/cuda.py _install_version: nv = getattr(getattr(jt, 'compiler', None), 'nvcc_versi...", exc)
        version.cuda = None
    version.hip = None
    version.git_version = "jittor"
    _modules["torch.version"] = version
    g.version = version


def _install_accelerator(g, registry=None):
    """torch.accelerator: the device-agnostic surface newer torch code uses.

    A serving stack that has moved off the torch.cuda names (vLLM's V1 worker,
    for one) reaches for these instead. They are the same handles the cuda
    module already exposes, so this is a rename layer rather than a second
    implementation -- and torch.OutOfMemoryError, which allocation-failure
    handlers catch by name.
    """
    import types as _types_acc

    cuda = getattr(g, "cuda", None)
    if cuda is None:
        return

    if not hasattr(g, "OutOfMemoryError"):
        g.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    if not hasattr(cuda, "OutOfMemoryError"):
        cuda.OutOfMemoryError = g.OutOfMemoryError

    # Build once, but publish on every install: a failed install restores the
    # module table while this attribute survives on the jittor module, so an
    # early return would leave torch.accelerator out of the registry the second
    # time through.
    accelerator = getattr(g, "accelerator", None)
    if accelerator is not None:
        if registry is not None:
            registry.publish("torch.accelerator", accelerator)
        return
    accelerator = _types_acc.ModuleType("torch.accelerator")
    accelerator.is_available = lambda *a, **k: True
    accelerator.device_count = cuda.device_count
    accelerator.current_device_index = lambda *a, **k: cuda.current_device()
    accelerator.set_device_index = lambda d, *a, **k: cuda.set_device(d)
    accelerator.device_index = getattr(cuda, "device", None)
    accelerator.current_stream = cuda.current_stream
    accelerator.set_stream = getattr(cuda, "set_stream", lambda *a, **k: None)
    accelerator.synchronize = cuda.synchronize
    accelerator.empty_cache = cuda.empty_cache
    accelerator.memory_allocated = cuda.memory_allocated
    accelerator.memory_reserved = cuda.memory_reserved
    accelerator.max_memory_allocated = cuda.max_memory_allocated
    accelerator.memory_stats = cuda.memory_stats
    accelerator.reset_peak_memory_stats = cuda.reset_peak_memory_stats
    accelerator.current_accelerator = lambda *a, **k: g.device("cuda")
    g.accelerator = accelerator
    if registry is not None:
        registry.publish("torch.accelerator", accelerator)


def install(ctx):
    g = ctx.jittor_module
    _install_cuda(g, ctx.registry)
    _install_version(g, ctx.registry)
    _install_accelerator(g, ctx.registry)
