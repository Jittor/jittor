"""Device activity from CUPTI, loaded with ctypes when a profile asks for it.

This is the in-process version of what ``nsys`` measured externally: every
kernel, copy and memset the device ran, with its own start/end timestamps,
and -- through CUPTI's external correlation -- the jittor operator (or device
graph launch) that issued it. ``src/runtime/profiler/step_trace.cc`` pushes
the operator's trace sequence number as the external correlation id around
each launch; here the id is read back from the activity records.

Buffer request and completion callbacks run directly in C++
(``step_trace_cupti_init`` in ``src/runtime/profiler/step_trace.cc``) so CUPTI's
background delivery thread never acquires the Python GIL (eliminating GIL vs.
CUPTI deadlocks on large steps) and record parsing runs at native speed.

CUPTI is not a jittor dependency. It is found, in order, at
``$JITTOR_CUPTI_LIB``, in the ``nvidia-cuda-cupti-cu12`` wheel, next to the
CUDA toolkit jittor compiles with, and on the loader path. When none is
found :func:`load` says why, and the profile reports device time as
unavailable instead of inventing it.
"""

import ctypes
import glob
import os
import sys

import jittor_core as _core

KIND_MEMCPY = 1
KIND_MEMSET = 2
KIND_DRIVER = 4
KIND_RUNTIME = 5
KIND_CONCURRENT_KERNEL = 10
KIND_MEMCPY2 = 22
KIND_EXTERNAL_CORRELATION = 39
EXTERNAL_KIND_CUSTOM0 = 3
# CUPTI emits external correlation for enabled API activity records. Libraries
# such as cuBLAS can launch through the driver directly, bypassing the runtime;
# collect both domains so those kernels retain their issuing operator's id.
_ENABLED_KINDS = (KIND_CONCURRENT_KERNEL, KIND_MEMCPY, KIND_MEMSET,
                  KIND_RUNTIME, KIND_DRIVER, KIND_EXTERNAL_CORRELATION)
_OPTIONAL_KINDS = (KIND_MEMCPY2,)

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
    seen = set()
    for root in filter(None, roots):
        for pattern in ("extras/CUPTI/lib64/libcupti.so*", "targets/x86_64-linux/lib/libcupti.so*",
                        "lib64/libcupti.so*", "lib/libcupti.so*"):
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                if path not in seen:
                    seen.add(path)
                    yield path
    yield "libcupti.so"


class _Cupti:
    """One per process: CUPTI's activity callbacks are process-global."""

    def __init__(self, lib, path):
        self.lib = lib
        self.path = path
        lib.cuptiActivityRegisterCallbacks.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        lib.cuptiGetTimestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.cuptiGetTimestamp.restype = ctypes.c_int
        lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
        lib.cuptiActivityEnable.restype = ctypes.c_int
        lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
        lib.cuptiActivityDisable.restype = ctypes.c_int
        lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        lib.cuptiActivityFlushAll.restype = ctypes.c_int

        get_next_addr = ctypes.cast(lib.cuptiActivityGetNextRecord, ctypes.c_void_p).value
        get_dropped_fn = getattr(lib, "cuptiActivityGetNumDroppedRecords", None)
        get_dropped_addr = ctypes.cast(get_dropped_fn, ctypes.c_void_p).value if get_dropped_fn else 0
        req_cb, done_cb = _core.step_trace_cupti_init(get_next_addr, get_dropped_addr)
        self._check(lib.cuptiActivityRegisterCallbacks(ctypes.c_void_p(req_cb),
                                                       ctypes.c_void_p(done_cb)),
                    "cuptiActivityRegisterCallbacks")
        self.push = ctypes.cast(lib.cuptiActivityPushExternalCorrelationId, ctypes.c_void_p).value
        self.pop = ctypes.cast(lib.cuptiActivityPopExternalCorrelationId, ctypes.c_void_p).value
        self._active_kinds = []
        self.active = False
        self.last_dropped = 0

    def _check(self, result, what):
        if result != 0:
            raise RuntimeError("%s failed with CUPTI error %d (library %s)" % (what, result, self.path))

    def timestamp(self):
        value = ctypes.c_uint64()
        self._check(self.lib.cuptiGetTimestamp(ctypes.byref(value)), "cuptiGetTimestamp")
        return value.value

    def start(self):
        self.flush()
        _core.step_trace_cupti_reset()
        self.last_dropped = 0
        enabled = []
        try:
            for kind in _ENABLED_KINDS:
                self._check(self.lib.cuptiActivityEnable(kind), "cuptiActivityEnable(%d)" % kind)
                enabled.append(kind)
            for kind in _OPTIONAL_KINDS:
                if self.lib.cuptiActivityEnable(kind) == 0:
                    enabled.append(kind)
        except Exception:
            for kind in reversed(enabled):
                self.lib.cuptiActivityDisable(kind)
            raise
        self._active_kinds = enabled
        self.active = True

    def stop(self):
        first_error = None
        try:
            self.flush()
        except Exception as exc:
            first_error = exc
        kinds, self._active_kinds = self._active_kinds, []
        for kind in reversed(kinds):
            rc = self.lib.cuptiActivityDisable(kind)
            if rc != 0 and first_error is None:
                first_error = RuntimeError(
                    "cuptiActivityDisable(%d) failed with CUPTI error %d (library %s)"
                    % (kind, rc, self.path))
        try:
            self.flush()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        self.active = False
        names = list(_core.step_trace_cupti_take_names())
        raw_kernels = _core.step_trace_cupti_take_kernels()
        raw_copies = _core.step_trace_cupti_take_copies()
        raw_corrs = _core.step_trace_cupti_take_correlations()
        self.last_dropped = int(_core.step_trace_cupti_dropped())
        if first_error is not None:
            raise first_error

        kernels = [
            (raw_kernels[i], raw_kernels[i + 1], raw_kernels[i + 2], raw_kernels[i + 3],
             raw_kernels[i + 4], names[raw_kernels[i + 5]], raw_kernels[i + 6],
             ((raw_kernels[i + 7], raw_kernels[i + 8], raw_kernels[i + 9]),
              (raw_kernels[i + 10], raw_kernels[i + 11], raw_kernels[i + 12]),
              raw_kernels[i + 13], raw_kernels[i + 14]))
            for i in range(0, len(raw_kernels), 15)
        ]
        copies = [
            (raw_copies[i], raw_copies[i + 1], raw_copies[i + 2], raw_copies[i + 3],
             raw_copies[i + 4],
             "memset" if raw_copies[i + 5] == 0 else ("memcpy " + _MEMCPY_KINDS.get(raw_copies[i + 5], "?")),
             raw_copies[i + 6])
            for i in range(0, len(raw_copies), 7)
        ]
        correlation = {raw_corrs[i]: raw_corrs[i + 1] for i in range(0, len(raw_corrs), 2)}
        return kernels, copies, correlation

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
            _instance = _Cupti(lib, path)
            return _instance
        except (OSError, RuntimeError, AttributeError) as exc:
            tried.append("%s (%s)" % (path, exc))
            continue
    _failure = ("CUPTI was not found or failed to initialize; install "
                "nvidia-cuda-cupti-cu12 matching your CUDA runtime (or "
                "`pip install \"jittor[profile]\"`), or set JITTOR_CUPTI_LIB "
                "to a compatible libcupti.so. Tried: " + "; ".join(tried[-4:]))
    return None


def why_unavailable():
    return _failure
