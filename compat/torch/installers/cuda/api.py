"""Stable CUDA facade APIs with installation-owned mutable runtime state.

Installation publishes these objects; it does not manufacture implementations.
Logical streams, pool-recorded memory peaks and unsupported placeholders retain their
existing behavior and have explicit fidelity records.
"""

import contextlib
import itertools
import threading
import types as _types

import jittor as jt

from ...context import get_install_context
from ...fidelity import Fidelity, register_fidelity
from ....stub_policy import unimplemented as _unimplemented

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
    """The ordinal a ``torch.cuda`` device argument names.

    ``None``/a bare "cuda" means the *current* device, as in torch -- not
    device 0. Returning 0 for them made every per-device query answer for
    device 0 on a rank whose current device was something else.
    """
    if device is None:
        return current_device()
    index = _cuda_index_of(device)
    if index is not None:
        return index
    if _device_is_cpu(device):
        raise ValueError("Expected a cuda device, but got: %s" % (device,))
    # A bare "cuda" / torch.device("cuda") names no particular device.
    return current_device()


def _cuda_device_name(device=None):
    """The marketing name of one device, queried per ordinal.

    The whole props family used to cache a single answer under one key and
    hand it back for every index, so ``get_device_name(1)`` reported device
    0's name -- indistinguishable on a uniform box and simply wrong on a
    mixed one. Each entry is now keyed by the ordinal it was read from.
    """
    index = _cuda_device_index(device)
    key = ("name", index)
    name = _cuda_props_cache.get(key)
    if name is not None:
        return name
    name = "CUDA"
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), index)
            buf = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(buf, len(buf), dev)
            got = buf.value.decode("utf-8", "ignore")
            if got:
                name = got
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_device_name: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache[key] = name
    return name


def _cuda_capability(device=None):
    """(major, minor) compute capability of one CUDA device.

    Queried from the CUDA driver for the ordinal asked for -- it used to
    always read device 0 and cache that one answer -- and falls back to
    (8, 0) when the driver query is unavailable (e.g. Ascend NPU).
    """
    index = _cuda_device_index(device)
    key = ("cap", index)
    cc = _cuda_props_cache.get(key)
    if cc is not None:
        return cc
    cc = (8, 0)
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), index)
            maj = ctypes.c_int(0); mino = ctypes.c_int(0)
            lib.cuDeviceComputeCapability(ctypes.byref(maj), ctypes.byref(mino), dev)
            if maj.value > 0:
                cc = (maj.value, mino.value)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_capability: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache[key] = cc
    return cc


def _cuda_sm_count(device=None):
    """SM (multiprocessor) count of CUDA device 0, queried via the driver.

    Triton-based libraries (e.g. flex_gemm's autotuner) size their grids by
    ``get_device_properties(...).multi_processor_count``; a wrong value only
    affects performance/occupancy, not correctness, so we default to 132 (an
    H100-class count) when the driver can't be queried.
    """
    index = _cuda_device_index(device)
    key = ("sm", index)
    n = _cuda_props_cache.get(key)
    if n is not None:
        return n
    n = 132
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), index)
            val = ctypes.c_int(0)
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            lib.cuDeviceGetAttribute(ctypes.byref(val), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
            if val.value > 0:
                n = val.value
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_sm_count: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache[key] = n
    return n


def _cuda_total_memory(device=None):
    index = _cuda_device_index(device)
    key = ("total_memory", index)
    total = _cuda_props_cache.get(key)
    if total is not None:
        return total
    total = 64 * 1024 ** 3
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), index)
            val = ctypes.c_size_t(0)
            fn = getattr(lib, "cuDeviceTotalMem_v2", None) or getattr(lib, "cuDeviceTotalMem", None)
            if fn is not None:
                fn(ctypes.byref(val), dev)
            if val.value > 0:
                total = int(val.value)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _cuda_total_memory: lib, ctypes = _cuda_driver()", exc)
    _cuda_props_cache[key] = total
    return total


class _DeviceProps:
    """torch.cuda.get_device_properties(...) result.

    Exposes the attributes real-torch device props carry that libraries read:
    ``name``, ``major``/``minor``, ``total_memory``, ``multi_processor_count``
    (alias ``multiprocessor_count``), ``warp_size``, ``max_threads_per_*``.
    """
    def __init__(self, device=None):
        # The ordinal is resolved once and every field read from *it*: these
        # used to be four independent device-0 queries, so
        # `get_device_properties(1)` described device 0.
        index = _cuda_device_index(device)
        cap = _cuda_capability(index)
        self.index = index
        self.name = "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(index)
        self.major, self.minor = cap
        self.total_memory = _cuda_total_memory(index)
        self.multi_processor_count = _cuda_sm_count(index)
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


#: ``torch.backends``' TF32 switches, and the single precision tier behind each.
#:
#: torch spells "may fp32 math use reduced-precision tensor cores" three ways
#: per domain, and this installation keeps one precision tier per domain
#: (matmul and cuDNN are genuinely independent in torch too):
#:
#: =========================================== ==============================
#: torch spelling                              ``CudaRuntimeState`` field
#: =========================================== ==============================
#: ``backends.cuda.matmul.allow_tf32``         ``matmul_precision``
#: ``backends.cuda.matmul.fp32_precision``     (same)
#: ``get/set_float32_matmul_precision()``      (same)
#: ``backends.cudnn.allow_tf32``               ``cudnn_precision``
#: ``backends.cudnn.conv.fp32_precision``      (same)
#: ``backends.cudnn.rnn.fp32_precision``       (same)
#: =========================================== ==============================
#:
#: Every spelling is a *view* of its domain's tier. It did not use to be: each
#: kept its own state, so one semantic had three answers that disagreed the
#: moment a write went through a spelling other than the one holding the state.
#: ``fp32_precision`` was the literal string ``"ieee"`` on all four objects --
#: it never reflected tf32 being on, and assigning to it did nothing at all --
#: and ``get_float32_matmul_precision()`` read a string that
#: ``matmul.allow_tf32 = True`` never touched.
#:
#: These fields are the *frontend's own* policy, and they are what the ops this
#: frontend builds actually execute with: ``frontend.py`` resolves the pair
#: from this same state into a thread-local native scope, each Op captures it
#: at construction, and execution restores it. They are deliberately not views
#: of ``cuda_allow_tf32`` / ``cuda_allow_cudnn_tf32``: those are deprecated
#: native overrides that can only raise the tier of a native
#: Runtime-following call, and an independent frontend does not reach into the
#: native policy in either direction (7.19/7.20; before that these two names
#: were where a torch write landed). Because the tiers are stored rather than
#: pushed to a flag, "the write was accepted and reads back" is not evidence
#: that it took effect -- what pins that is the compute type the library call
#: logs. See docs/notes/float32-precision-policy.md and
#: refactor-wip/results/2026-09-08-frontend-precision-isolation.md.
#:
#: compat/tests/torch/test_torch_backends_tf32.py drives this table;
#: compat/tests/torch/test_torch_compat_cuda_tf32.py pins each spelling to the
#: cuBLAS/cuDNN call it selects on a real device.
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
        #: One logical default stream per device ordinal (see _device_stream).
        self.device_default_streams = {}
        #: Sampled live-byte high-water mark per device ordinal.
        self.device_mem_peak = {}
        self.nvtx_state = threading.local()
        self.nvtx_handles = itertools.count(1)
        self.native_nvtx = [None]
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
    """torch.cuda.set_device: make a device current, in place.

    A device that is not a CUDA one is refused, as in torch
    (``ValueError: Expected a cuda device, but got: cpu``). It used to return
    ``None`` for it: ``torch.cuda.set_device("cpu")`` reported success and
    changed nothing, so a caller that meant to leave CUDA carried on issuing
    work to whatever device was current.
    """
    if device is None:
        # torch resolves an omitted index to the current device, i.e. a no-op.
        return None
    if _device_is_cpu(device) or (
            getattr(device, "type", None) not in (None, "cuda", "npu")
            and not isinstance(device, str)):
        raise ValueError("Expected a cuda device, but got: %s" % (device,))
    if isinstance(device, str) and device.split(":")[0] not in ("cuda", "npu"):
        raise ValueError("Expected a cuda device, but got: %s" % (device,))
    index = _cuda_index_of(device)
    if index is None:
        # A bare "cuda"/torch.device("cuda") names no particular device.
        return None
    # An ordinal beyond the visible devices is refused rather than delegated:
    # `jt.set_device` accepts one without a word on a build without an
    # accelerator (the core's own refusal is only asserted where a device
    # exists -- tests/runtime/test_runtime_device_state.py returns early
    # otherwise), so `torch.cuda.set_device(device_count() + 8)` reported
    # success and moved nothing. Device 0 stays accepted because it is the
    # ambient default even with no accelerator visible, which is the other half
    # of the same contract (`test_set_device_zero_is_accepted`).
    visible = device_count()
    if index >= max(1, visible):
        raise RuntimeError(
            "Invalid device ordinal: torch.cuda.set_device(%r) names device %d, "
            "but this build sees %d device(s)" % (device, index, visible))
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


def _device_name(device=None, *a, **k):
    try:
        return "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(device)
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _device_name: return 'Ascend910B/NPU' if getattr(jt.compiler, 'has_ac...", exc)
        return "CUDA"


class _CudaTypedTensorMeta(type):
    def __instancecheck__(cls, obj):
        base_type = getattr(_cuda_target(), cls._tensor_name)
        return isinstance(obj, base_type) and bool(getattr(obj, "is_cuda", False))

    def __call__(cls, *args, **kwargs):
        if not _cuda_target().cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return getattr(_cuda_target(), cls._tensor_name)(*args, **kwargs).cuda()


_CUDA_TENSOR_TYPES = {
    # ``__module__`` is "torch.cuda", where these are published and where
    # torch reports them from -- not this file, which is an implementation
    # detail nothing outside should have to know the name of.
    name: _CudaTypedTensorMeta(name, (), {
        "_tensor_name": name, "__module__": "torch.cuda",
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


def _device_stream(device, attribute):
    """One logical stream per device, created on first use.

    ``current_stream(1)``/``default_stream(1)`` used to ignore the argument and
    hand back the one process-wide stream object, whose ``.device`` read
    ``cuda:0``. A caller that records an event on it, or reads
    ``stream.device`` to decide where to launch, was then told a device the
    stream does not belong to. The streams are still logical -- jittor
    serialises them onto one physical backend stream -- but their identity and
    their device are now the ones asked for.
    """
    index = _cuda_device_index(device)
    table = getattr(_cuda_runtime(), attribute)
    stream = table.get(index)
    if stream is None:
        stream = _Stream(device=index)
        table[index] = stream
    return stream


def _current_stream(device=None, *a, **k):
    current = getattr(_cuda_runtime().stream_state, "current", None)
    if current is not None and (
            device is None or _cuda_device_index(device) == current.device.index):
        return current
    return _device_stream(device, "device_default_streams")


def _set_stream(stream):
    if stream is None:
        return None
    if not isinstance(stream, _Stream):
        raise TypeError("set_stream expects a torch.cuda.Stream or None")
    _cuda_runtime().stream_state.current = stream
    return None


class _StreamContext:
    """``with torch.cuda.stream(s):`` -- s is current, and so is s's device.

    torch's stream context is also a device context: entering a stream that
    belongs to cuda:1 makes cuda:1 current for the block and restores the
    caller's device on the way out. This used to change only the logical
    stream, so work issued inside ``with torch.cuda.stream(side_stream_on_1)``
    was still placed on whatever device was current outside it.
    """
    def __init__(self, stream):
        if stream is not None and not isinstance(stream, _Stream):
            raise TypeError("stream expects a torch.cuda.Stream or None")
        self.stream = stream
        self.previous = None
        self.previous_index = -1
    def __enter__(self):
        if self.stream is not None:
            self.previous = _current_stream()
            index = getattr(self.stream.device, "index", None)
            if index is not None and index >= 0:
                self.previous_index = current_device()
                if self.previous_index != index:
                    set_device(index)
            _set_stream(self.stream)
        return self
    def __exit__(self, *exc):
        if self.stream is not None:
            _set_stream(self.previous)
            if self.previous_index >= 0 and self.previous_index != current_device():
                set_device(self.previous_index)
            self.previous_index = -1
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


def _mem_device_key(device=None):
    """The ordinal a memory query is about, or -1 for the host pools."""
    if not jt.flags.use_cuda:
        return -1
    if device is not None and _device_is_cpu(device):
        return -1
    return _cuda_device_index(device)


def _mem_bytes(index, reserved=False):
    """Live (or pool-held) bytes on one device, from jittor's own accounting.

    ``MemInfo.total_cuda_used`` sums *every* device's pool, so the whole family
    answered the same process-wide number whatever ordinal it was handed:
    with 256 MiB allocated on cuda:1, ``memory_allocated(0)`` also said
    256 MiB. ``jt.core.device_memory_used``/``_reserved`` are per ordinal.
    """
    try:
        reader = jt.core.device_memory_reserved if reserved else jt.core.device_memory_used
        return int(reader(int(index)))
    except EXPECTED as exc:
        swallowed("torch/installers/cuda/api.py _mem_bytes: jt.core.device_memory_*", exc,
                  "falling back to the process-wide total, which is not per device")
        try:
            mi = jt.get_mem_info()
            return int(mi.total_cuda_used if index >= 0 else mi.total_cpu_used)
        except EXPECTED as inner:
            swallowed("torch/installers/cuda/api.py _mem_bytes: mi = jt.get_mem_info()", inner)
            return 0


def _mem_sample(index, value):
    """Record ``value`` against ``index``'s high-water mark and return it."""
    peaks = _cuda_runtime().device_mem_peak
    if value > peaks.get(index, 0):
        peaks[index] = value
    return value


def _mem_used(device=None, *a, **k):
    index = _mem_device_key(device)
    return _mem_sample(index, _mem_bytes(index))


def _mem_reserved(device=None, *a, **k):
    index = _mem_device_key(device)
    _mem_sample(index, _mem_bytes(index))
    return _mem_bytes(index, reserved=True)


def _native_peak(index):
    """The pools' own high-water mark for one device, or 0 if they keep none.

    The Python-side mark only moves when some memory API is *called*, so on its
    own it misses every peak reached inside an execution batch: a training step
    that filled 22 GB read 0.1-1.3 GB. The pools record the peak on every
    allocation; the sampled mark stays as the fallback for a process that
    turned them off.
    """
    if index < 0:
        return 0
    return int(jt.core.device_memory_peak(int(index)))


def _mem_max(device=None, *a, **k):
    index = _mem_device_key(device)
    sampled = _mem_sample(index, _mem_bytes(index))
    return max(_native_peak(index), _cuda_runtime().device_mem_peak.get(index, sampled))


def _reset_peak(device=None, *a, **k):
    index = _mem_device_key(device)
    if index >= 0:
        jt.core.reset_device_memory_peak(int(index))
    _cuda_runtime().device_mem_peak[index] = _mem_bytes(index)


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


def _mem_get_info(device=None, *a, **k):
    """torch.cuda.mem_get_info(device): the driver's free/total for *that* card.

    ``cudaMemGetInfo`` reports the *current* device, so asking about another
    one has to make it current for the call -- it used to ignore the argument
    and report whichever device happened to be current, i.e. device 0's free
    memory presented as device N's.
    """
    fn = _cuda_mem_get_info_fn()
    index = _cuda_device_index(device)
    if fn:
        previous = current_device()
        try:
            if index != previous:
                set_device(index)
            got = fn()
        finally:
            if index != previous:
                set_device(previous)
        if got and got[1] > 0:
            return got
    # No cudart to ask: fall back to jittor's own accounting. This knows the
    # device total exactly and jittor's live bytes, but not the context's or
    # another process's, so it reads slightly optimistic rather than fictional.
    total = _cuda_total_memory(index)
    return (max(0, total - _mem_bytes(index, reserved=True)), total)


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


def _api_cuda_can_device_access_peer(device, peer_device):
    """torch.cuda.can_device_access_peer: whether device can read peer's memory.

    Asked of the CUDA driver (``cuDeviceCanAccessPeer``), so it answers for the
    pair actually named. It was simply absent, and an ``AttributeError`` at the
    point where a serving stack decides between a peer copy and a host bounce
    is not a graceful degradation -- it aborts the run.
    """
    first, second = _cuda_device_index(device), _cuda_device_index(peer_device)
    if first == second:
        return False
    lib, ctypes = _cuda_driver()
    if lib is None:
        raise RuntimeError(
            "torch.cuda.can_device_access_peer(%r, %r): the CUDA driver is not "
            "loadable here, so peer access cannot be determined"
            % (device, peer_device))
    src, dst, out = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    lib.cuDeviceGet(ctypes.byref(src), first)
    lib.cuDeviceGet(ctypes.byref(dst), second)
    if lib.cuDeviceCanAccessPeer(ctypes.byref(out), src, dst) != 0:
        raise RuntimeError(
            "torch.cuda.can_device_access_peer(%r, %r): the driver refused the "
            "query" % (device, peer_device))
    return bool(out.value)


def _api_cuda_is_initialized(*a, **k):
    return bool(is_available() and getattr(jt.flags, 'use_cuda', 0))


def _api_cuda__is_in_bad_fork(*a, **k):
    return False


def _api_cuda_synchronize(device=None, *a, **k):
    """torch.cuda.synchronize(device).

    ``jt.sync_all(True)`` waits for every device this run touched, which is a
    superset of what torch's per-device synchronize promises, so the argument
    is validated (a non-CUDA device is refused, as in torch) and then the
    stronger wait is performed. Reported as APPROXIMATE for that reason.
    """
    if device is not None:
        _cuda_device_index(device)
    return jt.sync_all(True)


def _api_cuda_manual_seed(s):
    return jt.set_global_seed(int(s))


def _api_cuda_manual_seed_all(s):
    return jt.set_global_seed(int(s))


def _api_cuda_is_bf16_supported():
    return True


def _api_cuda_get_device_capability(device=None, *a, **k):
    return _cuda_capability(device)


def _api_cuda_get_device_properties(device=None, *a, **k):
    return _DeviceProps(device)


def _api_cuda_default_stream(device=None, *a, **k):
    return _device_stream(device, "device_default_streams")


def _api_cuda_memory_stats(device=None, *a, **k):
    index = _mem_device_key(device)
    current = _mem_used(device)
    return {'allocated_bytes.all.current': current,
            'allocated_bytes.all.peak': _mem_max(device),
            'reserved_bytes.all.current': _mem_bytes(index, reserved=True)}


def _api_cuda_ipc_collect(*a, **k):
    return None


def _api_cuda_memory__set_allocator_settings(*a, **k):
    return None


#: The CUDA RNG state, as a seed and a position.
#:
#: `get_rng_state` used to return the constant `[0]` and `set_rng_state` used
#: to do nothing, so `accelerator.save_state()` wrote a byte that meant
#: nothing, `load_state()` restored nothing, and the resumed run drew a
#: different sequence than the one it was continuing -- with no error anywhere.
#:
#: What makes a real state possible is that cuRAND's offset turns out to
#: describe the position completely. Measured against
#: CURAND_RNG_PSEUDO_DEFAULT on this box, by drawing a history, drawing a
#: continuation, then reseeding, setting the summed offset and drawing again:
#:
#:   uniform float32/float64   n elements advance the generator by n
#:   normal  float32/float64   n elements advance it by n/2
#:
#: a mixed history advances by the sum of its parts, and after a restore every
#: one of the four kinds continues exactly. So a seed plus one integer is the
#: whole state -- `backends/cuda/libraries/curand/` counts it, this packs it.
#:
#: The bytes are jittor's own format, not torch's CUDA state bytes: save them,
#: hand them back, do not parse them, and do not feed torch's to this.
_RNG_STATE_MAGIC = b"JTCURAND"
_RNG_STATE_VERSION = 1


def _rng_state_pack(seed, offset):
    import struct
    import numpy as _np
    blob = _RNG_STATE_MAGIC + struct.pack("<Iqq", _RNG_STATE_VERSION,
                                          int(seed), int(offset))
    return jt.array(_np.frombuffer(blob, dtype=_np.uint8).copy())


def _rng_state_unpack(state):
    import struct
    import numpy as _np
    raw = state
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    raw = _np.asarray(raw, dtype=_np.uint8).reshape(-1).tobytes()
    if not raw.startswith(_RNG_STATE_MAGIC):
        raise ValueError(
            "not a jittor CUDA RNG state. torch's own state bytes are a "
            "different format and a different algorithm; they cannot be "
            "restored into jittor's cuRAND generator.")
    version, seed, offset = struct.unpack("<Iqq", raw[len(_RNG_STATE_MAGIC):])
    if version != _RNG_STATE_VERSION:
        raise ValueError("unsupported jittor CUDA RNG state version %d" % version)
    return int(seed), int(offset)


#: What to say when the native half is not there.
#:
#: The seed/offset counting lives in `backends/cuda/libraries/curand/`. When
#: that is absent -- an older build, a backend compiled without it, or the
#: native change not present -- there is no position to save, and the honest
#: answer is to say so rather than to call a function that is not there or,
#: worse, to go back to answering with a constant.
_NO_NATIVE_RNG_STATE = (
    "this jittor build cannot express the CUDA RNG state: its cuRAND wrapper "
    "does not count how far the generator has advanced, so there is no "
    "position to save and a restore could only reseed. "
    "torch.cuda.manual_seed(seed) starts a reproducible sequence; resuming one "
    "from a checkpoint needs curand_generator_offset/curand_restore_state in "
    "the native cuRAND backend."
)


def _curand(required=True):
    backend = getattr(jt.compile_extern, "curand", None)
    if backend is None or not hasattr(backend, "curand_generator_offset") \
            or not hasattr(backend, "curand_restore_state"):
        if required:
            raise NotImplementedError(_NO_NATIVE_RNG_STATE)
        return None
    return backend


def cuda_rng_state_is_supported():
    """Whether this build can save and restore the CUDA RNG position."""
    return _curand(required=False) is not None


def _rng_device_index(device):
    """Which device's generator, as a real index.

    `jt.flags.device_id` is -1 until something sets it, meaning "whichever is
    current" rather than device -1, and passing that through reached the
    native restore as `device >= 0` failing. `jt.current_device()` is the one
    the cuRAND wrapper itself indexes by.
    """
    if device is None:
        index = -1
    elif isinstance(device, bool):
        raise TypeError("device index must not be a bool")
    elif isinstance(device, int):
        index = int(device)
    elif isinstance(device, str):
        # Before the `.index` probe below, because `str` *has* an `.index`
        # method: `getattr("cuda:0", "index", None)` hands back a bound method,
        # which is not None, and `int()` on it raised "int() argument must be
        # ... not 'builtin_function_or_method'". torch accepts this spelling.
        index = int(device.split(":")[1]) if ":" in device else -1
    else:
        attr = getattr(device, "index", None)
        # An int, not merely present: the same trap one line up, for any object
        # that happens to carry a callable `.index`.
        index = int(attr) if isinstance(attr, int) else -1
    if index < 0:
        index = int(jt.current_device())
    return max(index, 0)


def _api_cuda_get_rng_state(device=None, *a, **k):
    # Everything queued has to have happened, or the offset describes a
    # position the device has not reached: the saved state would then be ahead
    # of the data the checkpoint was taken with.
    jt.sync_all(True)
    backend = _curand()
    index = _rng_device_index(device)
    return _rng_state_pack(backend.curand_generator_seed(),
                           backend.curand_generator_offset(index))


def _api_cuda_get_rng_state_all(*a, **k):
    jt.sync_all(True)
    backend = _curand()
    seed = backend.curand_generator_seed()
    return [_rng_state_pack(seed, backend.curand_generator_offset(i))
            for i in range(int(jt.get_device_count()))]


def _api_cuda_set_rng_state(state, device=None, *a, **k):
    seed, offset = _rng_state_unpack(state)
    # Parsed before anything is touched, so a malformed state leaves the
    # generator where it was rather than half-restored.
    jt.sync_all(True)
    _curand().curand_restore_state(_rng_device_index(device), seed, offset)


def _api_cuda_set_rng_state_all(states, *a, **k):
    parsed = [_rng_state_unpack(state) for state in states]
    jt.sync_all(True)
    backend = _curand()
    for index, (seed, offset) in enumerate(parsed):
        backend.curand_restore_state(index, seed, offset)


def _api_cuda_initial_seed(*a, **k):
    # The seed jittor is actually running on, not 0. `set_seed` records it and
    # the cuRAND wrapper replays it onto every device generator, so this is the
    # one part of the state that *is* expressible.
    return int(jt.get_seed())


def _api_cuda_seed(*a, **k):
    # torch reseeds from a fresh nondeterministic value and returns None; doing
    # nothing instead left the caller on the old sequence while looking
    # reseeded.
    import random as _random
    jt.set_seed(_random.SystemRandom().randrange(1 << 31))


def _api_cuda_seed_all(*a, **k):
    # One generator per device, all replayed from the same seed by the cuRAND
    # wrapper's set_seed callback, so seeding "all" is seeding.
    _api_cuda_seed()


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


class _CudnnFlags:
    """`torch.backends.cudnn.flags(...)`: set the switches, then put them back.

    A context manager, which is how callers use it::

        with torch.backends.cudnn.flags(enabled=False, benchmark=True):
            ...

    **It restores the attributes; it does not change which kernels run.** The
    module's four switches are already settings nothing acts on -- convolution
    here picks its own path -- and this does not make them act. What it buys is
    that the attribute a caller reads back inside the block is the one it set,
    and that the block is not an AttributeError. MiniMax-H3's reference path
    wraps work in it, and a missing `flags` killed the request outright.

    A class rather than `@contextlib.contextmanager` so the old values are read
    on `__enter__`, not when the object is built.
    """

    _NAMES = ("enabled", "benchmark", "deterministic", "allow_tf32")

    def __init__(self, module, values):
        self._module = module
        self._values = values
        self._saved = {}

    def __enter__(self):
        for name in self._NAMES:
            if name in self._values:
                self._saved[name] = getattr(self._module, name, None)
                setattr(self._module, name, self._values[name])
        return self

    def __exit__(self, exc_type, exc, traceback):
        for name, previous in self._saved.items():
            setattr(self._module, name, previous)
        return False


def _api_cudnn_flags(module, enabled=False, benchmark=False, benchmark_limit=10,
                     deterministic=False, allow_tf32=True):
    """The module comes in bound, it is not looked up.

    `bindings.py` binds this to the cudnn module it just built. An installer
    reaching into the interpreter's module registry is what
    `test_torch_compat_structure::test_canonical_module_line_budgets` forbids
    -- "it fails on growth, not on a boundary violation" -- and the first
    version of this did exactly that.
    """
    del benchmark_limit          # accepted for signature parity; nothing reads it
    return _CudnnFlags(module, {"enabled": enabled, "benchmark": benchmark,
                                "deterministic": deterministic,
                                "allow_tf32": allow_tf32})


def _api_cudnn_version():
    return None


def _api_cuda_backend_sdp_kernel(*a, **k):
    return _SDPKernel()


#: torch's four SDPA backend switches. Attention here picks its own kernel, so
#: these only record the requested state -- but torch pairs every setter with a
#: getter, and callers read it back (MiniMax-H3's encoder saves
#: `cudnn_sdp_enabled()`, forces cuDNN SDPA on, and restores the saved value).
#: There is no cuDNN fused SDPA here, so `cudnn_sdp_enabled()` defaults to False;
#: the other three match torch's defaults.
_SDP_BACKEND_FLAGS = {
    "flash": True,
    "mem_efficient": True,
    "math": True,
    "cudnn": False,
}


def _enable_sdp_backend(name, enabled=True):
    _SDP_BACKEND_FLAGS[name] = bool(enabled)
    return None


def _api_cuda_backend_enable_flash_sdp(enabled=True):
    return _enable_sdp_backend("flash", enabled)


def _api_cuda_backend_flash_sdp_enabled():
    return _SDP_BACKEND_FLAGS["flash"]


def _api_cuda_backend_enable_mem_efficient_sdp(enabled=True):
    return _enable_sdp_backend("mem_efficient", enabled)


def _api_cuda_backend_mem_efficient_sdp_enabled():
    return _SDP_BACKEND_FLAGS["mem_efficient"]


def _api_cuda_backend_enable_math_sdp(enabled=True):
    return _enable_sdp_backend("math", enabled)


def _api_cuda_backend_math_sdp_enabled():
    return _SDP_BACKEND_FLAGS["math"]


def _api_cuda_backend_enable_cudnn_sdp(enabled=True):
    return _enable_sdp_backend("cudnn", enabled)


def _api_cuda_backend_cudnn_sdp_enabled():
    return _SDP_BACKEND_FLAGS["cudnn"]


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
    _current_stream: "Thread-local logical stream identity, one logical stream per device; native execution remains serialized.",
    _set_stream: "Selects a logical stream; does not rebind the native backend stream.",
    _api_cuda_default_stream: "One logical default stream per device ordinal; native execution remains serialized.",
    _Event: "Host timestamps after synchronization; not native CUDA event timing.",
    _mem_used: "Per-device live bytes from jittor's own pools; excludes the CUDA context and other processes.",
    _mem_reserved: "Per-device bytes held by jittor's pools (live plus cached); not the driver's view of the card.",
    _mem_max: "Per-device high-water mark the pools record at every allocation; sampled at memory queries only when the caching pools are disabled.",
    _reset_peak: "Restarts one device's live-byte high-water mark at the current live bytes.",
    _api_cuda_memory_stats: "Current and peak live bytes plus pool reservation, per device; other PyTorch counters absent.",
    _mem_get_info: "cudaMemGetInfo on the device asked about; the fallback reports jittor's pools only, so it excludes other processes.",
    _api_cuda_synchronize: "Waits for every device this run touched, which is stronger than torch's per-device synchronize.",
    _api_cuda_can_device_access_peer: "Answered by the CUDA driver for the named pair; raises where the driver cannot be loaded rather than guessing.",
    _device_name: "Queried per device ordinal from the CUDA driver, cached per ordinal.",
    _api_cuda_get_device_capability: "Queried per device ordinal from the CUDA driver; falls back to (8, 0) where the driver query is unavailable.",
    _api_cuda_get_device_properties: "Name, compute capability, total memory and SM count are real and per ordinal; the remaining fields are class defaults.",
    _empty_cache: "Environment-selected no-op, GC or synchronized GC; default is a memory hint.",
    _get_float32_matmul_precision: "Frontend-owned CUDA matmul policy; native Jittor and cuDNN policies are independent.",
    _set_float32_matmul_precision: "Frontend-owned highest/high/medium CUDA matmul accumulation; does not change cuDNN or native Jittor.",
}

_CUDA_PLACEHOLDERS = frozenset((
    CUDAGraph, CUDAPluggableAllocator, TorchFunctionMode,
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
    # ``__module__`` is how this sweep recognises what it published -- except
    # for the legacy CUDA tensor classes, which report ``torch.cuda`` because
    # that is where torch reports them from and what mmcv reads. Name them.
    published_types = set(_CUDA_TENSOR_TYPES.values())
    for namespace, module in namespaces:
        for name, implementation in tuple(vars(module).items()):
            if not callable(implementation):
                continue
            if (getattr(implementation, "__module__", None) != __name__
                    and implementation not in published_types):
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
