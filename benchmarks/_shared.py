"""Shared runtime contracts for the ASV benchmark suite."""

from __future__ import annotations

import gc
import importlib
import os
import pathlib
import sys
from typing import Any, Tuple

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore


ASV_CACHE_PREFIX = "asv-"


def require_isolated_cache() -> pathlib.Path:
    """Validate the cache contract before importing Jittor.

    Cache mistakes are configuration failures, not unsupported benchmark cases,
    so this deliberately raises RuntimeError instead of NotImplementedError.
    """

    cache_name = os.environ.get("cache_name", "")
    if not cache_name.startswith(ASV_CACHE_PREFIX):
        raise RuntimeError(
            "ASV benchmarks require a dedicated cache_name beginning with "
            "'asv-'; refusing to share a unittest or default JIT cache"
        )

    raw_home = os.environ.get("JITTOR_HOME", "")
    if not raw_home:
        raise RuntimeError("ASV benchmarks require an explicit dedicated JITTOR_HOME")

    home = pathlib.Path(raw_home).expanduser()
    if not home.is_absolute():
        conf_dir = os.environ.get("ASV_CONF_DIR")
        if not conf_dir:
            raise RuntimeError("relative JITTOR_HOME requires ASV_CONF_DIR")
        home = pathlib.Path(conf_dir) / home
    home = home.resolve()

    if not any("asv" in part.lower() for part in home.parts):
        raise RuntimeError(
            "JITTOR_HOME must contain an 'asv' path component so benchmark "
            "and unittest caches cannot be confused"
        )
    if home == pathlib.Path.home().resolve():
        raise RuntimeError("JITTOR_HOME must not be the user home directory")

    home.mkdir(parents=True, exist_ok=True)
    os.environ["JITTOR_HOME"] = str(home)
    return home


def load_backend(name: str, device: str) -> Any:
    """Import and configure a benchmark backend, or explicitly skip it."""

    require_isolated_cache()
    if name == "jittor":
        try:
            backend = importlib.import_module("jittor")
        except Exception as exc:
            raise RuntimeError("the mandatory Jittor backend failed to import") from exc
        backend.flags.use_cuda = 0
        if device == "cuda":
            if not bool(getattr(backend.compiler, "has_cuda", False)):
                raise NotImplementedError("Jittor was built without CUDA support")
            try:
                backend.flags.use_cuda = 1
                backend.flags.cuda_allow_tf32 = 1
                backend.ones((1,), dtype="float32").sync()
            except Exception as exc:
                backend.flags.use_cuda = 0
                raise NotImplementedError("Jittor CUDA execution is unavailable: %s" % exc)
        return backend

    if name == "torch":
        try:
            backend = importlib.import_module("torch")
        except Exception as exc:
            raise NotImplementedError(
                "the optional real-PyTorch baseline is not installed"
            ) from exc
        module_file = str(getattr(backend, "__file__", ""))
        if "jittor" in module_file.lower():
            raise RuntimeError("the PyTorch baseline resolved to the Jittor torch shim")
        if device == "cuda" and not backend.cuda.is_available():
            raise NotImplementedError("real PyTorch CUDA execution is unavailable")
        if device == "cuda":
            backend.backends.cuda.matmul.allow_tf32 = True
            backend.backends.cudnn.allow_tf32 = True
            backend.set_float32_matmul_precision("high")
        return backend

    raise ValueError("unknown benchmark backend: %s" % name)


def backend_tensor(backend_name: str, backend: Any, array: np.ndarray, device: str) -> Any:
    if backend_name == "jittor":
        return backend.array(array)
    return backend.tensor(array, device=backend.device(device))


def synchronize(backend_name: str, backend: Any, device: str) -> None:
    if backend_name == "jittor":
        backend.sync_all(True)
    elif device == "cuda":
        backend.cuda.synchronize()


def as_numpy(backend_name: str, value: Any) -> np.ndarray:
    if backend_name == "jittor":
        return np.asarray(value.float32().numpy(), dtype="float32")
    return np.asarray(value.detach().float().cpu().numpy(), dtype="float32")


def reset_memory_stats(backend_name: str, backend: Any, device: str) -> None:
    if backend_name == "torch" and device == "cuda":
        backend.cuda.reset_peak_memory_stats()


def working_set_bytes(backend_name: str, backend: Any, device: str) -> int:
    """Return backend allocation on CUDA, or process peak RSS on CPU."""

    if device == "cuda":
        if backend_name == "torch":
            return int(backend.cuda.memory_allocated())
        info = backend.get_mem_info()
        return int(info.total_cuda_used)

    if resource is not None:
        # ru_maxrss is KiB on Linux and bytes on macOS.
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            rss *= 1024
        return rss

    if os.name == "nt":  # pragma: no cover - Windows
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    raise NotImplementedError("CPU working-set tracking is unavailable on this platform")


def reset_peak_device_bytes(backend_name: str, backend: Any, device: str) -> None:
    """Start a peak measurement from a known floor.

    Torch keeps a real high-water counter. Jittor does not expose one, so the
    proxy below reads its caching pool, which only grows within a process --
    that is only a peak if the pool starts empty, which is what the collection
    here is for.
    """

    if device != "cuda":
        return
    if backend_name == "torch":
        backend.cuda.reset_peak_memory_stats()
        return
    backend.clean()


def peak_device_bytes(backend_name: str, backend: Any, device: str) -> int:
    """The high-water device allocation since the last reset.

    Not the same measurement as :func:`working_set_bytes`, and deliberately a
    second metric rather than a redefinition of it: that one reads the
    allocation still standing *after* an operation, so a transient spike that is
    freed again is invisible to it. A device-to-host copy that allocated its
    destination on the device held twice the tensor for the length of the
    transfer and then released it -- a real defect, and one the existing metric
    structurally could not see.

    The two backends measure this differently and the difference is worth
    stating rather than smoothing over. Torch reports its own high-water mark
    exactly. Jittor has no such counter, so this reads the size of its caching
    pool, which never shrinks inside a process: after
    :func:`reset_peak_device_bytes` empties it, growth is the peak. That proxy
    is exact only while nothing else in the process allocates, which holds
    inside one benchmark and would not hold in a shared session.
    """

    if device != "cuda":
        return working_set_bytes(backend_name, backend, device)
    if backend_name == "torch":
        return int(backend.cuda.max_memory_allocated())
    return int(backend.get_mem_info().total_cuda_used)


def cleanup_backend(backend_name: str, backend: Any) -> None:
    gc.collect()
    if backend_name == "jittor":
        backend.clean()
    elif hasattr(backend, "cuda") and backend.cuda.is_available():
        backend.cuda.empty_cache()


def import_transformers(backend_name: str, backend: Any) -> Tuple[Any, Any]:
    """Load Transformers against one unambiguous torch-compatible backend."""

    if backend_name == "jittor":
        existing = sys.modules.get("torch")
        if existing is not None and existing is not backend:
            raise RuntimeError("real torch was imported before the Jittor Transformers benchmark")
        sys.modules["torch"] = backend
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        raise NotImplementedError("Tiny Llama requires transformers==4.56.2") from exc
    if str(getattr(transformers, "__version__", "")) != "4.56.2":
        raise RuntimeError(
            "Tiny Llama comparisons require transformers==4.56.2, got %s"
            % getattr(transformers, "__version__", "unknown")
        )
    return backend, transformers
