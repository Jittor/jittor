"""Device activity from CUPTI, loaded with ctypes when a profile asks for it.

This is the in-process version of what ``nsys`` measured externally: every
kernel, copy and memset the device ran, with its own start/end timestamps,
and -- through CUPTI's external correlation -- the jittor operator (or device
graph launch) that issued it. ``src/runtime/profiler/step_trace.cc`` pushes
the operator's trace sequence number as the external correlation id around
each launch; here the id is read back from the activity records.

CUPTI is not a jittor dependency. It is found, in order, at
``$JITTOR_CUPTI_LIB``, in the ``nvidia-cuda-cupti-cu12`` wheel, next to the
CUDA toolkit jittor compiles with, and on the loader path. When none is
found :func:`load` says why, and the profile reports device time as
unavailable instead of inventing it.

Record layouts are CUPTI's packed ``CUpti_ActivityKernel*`` /
``CUpti_ActivityMemcpy*`` / ``CUpti_ActivityMemset*`` /
``CUpti_ActivityExternalCorrelation`` structs. Only fields whose offsets have
been stable since CUPTI 11 are read.
"""

import ctypes
import glob
import os
import sys
import threading

KIND_MEMCPY = 1
KIND_MEMSET = 2
KIND_DRIVER = 4
KIND_RUNTIME = 5
KIND_CONCURRENT_KERNEL = 10
KIND_EXTERNAL_CORRELATION = 39
EXTERNAL_KIND_CUSTOM0 = 3
# RUNTIME is on only because CUPTI emits external-correlation records for API
# records; without it no kernel can be attributed. DRIVER is left off: every
# launch measured here (jittor kernels, cuBLAS, cuDNN) goes through the
# runtime, and the report's "attributed" share shows any launch that did not.
_ENABLED_KINDS = (KIND_CONCURRENT_KERNEL, KIND_MEMCPY, KIND_MEMSET,
                  KIND_RUNTIME, KIND_EXTERNAL_CORRELATION)
_BUFFER_BYTES = 8 << 20

_MEMCPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA",
                 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}


def _candidates():
    explicit = os.environ.get("JITTOR_CUPTI_LIB")
    if explicit:
        yield explicit
        return
    for entry in sys.path:
        for path in sorted(glob.glob(os.path.join(entry, "nvidia", "cuda_cupti", "lib", "libcupti.so*"))):
            yield path
    try:
        from jittor.build import compiler
        roots = [getattr(compiler, "cuda_home", None), os.path.dirname(getattr(compiler, "cuda_lib", "") or "")]
    except ImportError:
        roots = []
    roots.append(os.environ.get("CUDA_HOME"))
    for root in filter(None, roots):
        for pattern in ("extras/CUPTI/lib64/libcupti.so*", "lib64/libcupti.so*", "lib/libcupti.so*"):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                yield path
    yield "libcupti.so"


class _Cupti:
    """One per process: CUPTI's activity callbacks are process-global."""

    def __init__(self, lib, path):
        self.lib = lib
        self.path = path
        self.lock = threading.Lock()
        self.kernels = []
        self.copies = []
        self.correlation = {}
        self._buffers = {}
        # Buffers are handed to CUPTI from inside CUDA calls on the profiled
        # thread; making them there would put Python allocation work inside
        # the measured launch. A few are made ahead, and refilled at start.
        self._spare = []
        req = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        done = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_size_t)
        self._req = req(self._request)
        self._done = done(self._complete)
        lib.cuptiActivityGetNextRecord.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                   ctypes.POINTER(ctypes.c_void_p)]
        lib.cuptiGetTimestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self._check(lib.cuptiActivityRegisterCallbacks(self._req, self._done),
                    "cuptiActivityRegisterCallbacks")
        self.push = ctypes.cast(lib.cuptiActivityPushExternalCorrelationId, ctypes.c_void_p).value
        self.pop = ctypes.cast(lib.cuptiActivityPopExternalCorrelationId, ctypes.c_void_p).value
        self.active = False

    def _check(self, result, what):
        if result != 0:
            raise RuntimeError("%s failed with CUPTI error %d (library %s)" % (what, result, self.path))

    def _request(self, pbuf, psize, pmax):
        raw = self._spare.pop() if self._spare else ctypes.create_string_buffer(_BUFFER_BYTES + 8)
        address = (ctypes.addressof(raw) + 7) & ~7
        self._buffers[address] = raw
        pbuf[0] = address
        psize[0] = _BUFFER_BYTES
        pmax[0] = 0

    def _complete(self, context, stream, buffer, size, valid):
        record = ctypes.c_void_p()
        u32 = ctypes.c_uint32.from_address
        u64 = ctypes.c_uint64.from_address
        kernels, copies, correlation = [], [], {}
        while self.lib.cuptiActivityGetNextRecord(buffer, valid, ctypes.byref(record)) == 0:
            a = record.value
            kind = u32(a).value
            if kind == KIND_CONCURRENT_KERNEL:
                name = ctypes.c_char_p.from_address(a + 104).value
                kernels.append((u64(a + 16).value, u64(a + 24).value, u32(a + 40).value,
                                u32(a + 48).value, u32(a + 92).value,
                                name.decode("utf-8", "replace") if name else "?",
                                u32(a + 156).value))
            elif kind == KIND_MEMCPY:
                copies.append((u64(a + 16).value, u64(a + 24).value, u32(a + 32).value,
                               u32(a + 40).value, u32(a + 44).value,
                               "memcpy " + _MEMCPY_KINDS.get(ctypes.c_uint8.from_address(a + 4).value, "?"),
                               u64(a + 8).value))
            elif kind == KIND_MEMSET:
                copies.append((u64(a + 16).value, u64(a + 24).value, u32(a + 32).value,
                               u32(a + 40).value, u32(a + 44).value, "memset", u64(a + 8).value))
            elif kind == KIND_EXTERNAL_CORRELATION:
                if u32(a + 4).value == EXTERNAL_KIND_CUSTOM0:
                    correlation[u32(a + 16).value] = u64(a + 8).value
        with self.lock:
            self.kernels.extend(kernels)
            self.copies.extend(copies)
            self.correlation.update(correlation)
        self._buffers.pop(buffer, None)

    def timestamp(self):
        value = ctypes.c_uint64()
        self._check(self.lib.cuptiGetTimestamp(ctypes.byref(value)), "cuptiGetTimestamp")
        return value.value

    def start(self):
        self.flush()
        while len(self._spare) < 4:
            self._spare.append(ctypes.create_string_buffer(_BUFFER_BYTES + 8))
        with self.lock:
            self.kernels, self.copies, self.correlation = [], [], {}
        for kind in _ENABLED_KINDS:
            self._check(self.lib.cuptiActivityEnable(kind), "cuptiActivityEnable(%d)" % kind)
        self.active = True

    def stop(self):
        self.flush()
        for kind in _ENABLED_KINDS:
            self._check(self.lib.cuptiActivityDisable(kind), "cuptiActivityDisable(%d)" % kind)
        self.active = False
        with self.lock:
            out = self.kernels, self.copies, self.correlation
            self.kernels, self.copies, self.correlation = [], [], {}
        return out

    def flush(self):
        self._check(self.lib.cuptiActivityFlushAll(1), "cuptiActivityFlushAll")


_instance = None
_failure = None


def load():
    """The process CUPTI instance, or None with :func:`why_unavailable` set."""
    global _instance, _failure
    if _instance is not None or _failure is not None:
        return _instance
    tried = []
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            tried.append("%s (%s)" % (path, exc))
            continue
        _instance = _Cupti(lib, path)
        return _instance
    _failure = ("CUPTI was not found; pip install \"jittor[profile]\" (the "
                "nvidia-cuda-cupti-cu12 wheel for Jittor's CUDA 12.2), or set "
                "JITTOR_CUPTI_LIB to a libcupti matching your CUDA runtime. Tried: "
                + "; ".join(tried[-4:]))
    return None


def why_unavailable():
    return _failure
