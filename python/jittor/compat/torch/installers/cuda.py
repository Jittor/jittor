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
    device, dtype,
)

_cuda_props_cache = {}


def _cuda_driver():
    try:
        import ctypes
        for n in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(n)
                lib.cuInit(0)
                return lib, ctypes
            except OSError:
                pass
    except Exception:
        pass
    return None, None


def _cuda_device_index(device=None):
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except Exception:
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
    except Exception:
        pass
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
    except Exception:
        pass
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
    except Exception:
        pass
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
    except Exception:
        pass
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


def _install_cuda(g, registry=None):
    _modules = registry_for(g, registry).module_map
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
        except Exception:
            return False
    def device_count():
        if not is_available():
            return 0
        try:
            import os as _os_cuda
            _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
            if _cvd is not None:
                return len([_d for _d in _cvd.split(",") if _d.strip()])
        except Exception:
            pass
        return 1
    cuda.is_available = is_available
    cuda.device_count = device_count
    cuda.current_device = lambda: 0
    cuda.set_device = lambda *a, **k: None
    class _CudaDeviceContext:
        def __init__(self, device=None):
            self.device = device
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    cuda.device = _CudaDeviceContext
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
    except Exception:
        _empty_cache_mode = "0"

    def _empty_cache():
        if _empty_cache_mode in ("0", "false", "no", "off", "none", "noop"):
            return
        if _empty_cache_mode in ("", "1", "true", "yes", "on", "gc"):
            try:
                jt.gc()
            except Exception:
                pass
        elif _empty_cache_mode in ("sync", "full"):
            try:
                jt.sync_all(True)
            except Exception:
                pass
            try:
                jt.gc()
            except Exception:
                pass
    cuda.empty_cache = _empty_cache
    cuda.synchronize = lambda *a, **k: jt.sync_all(True)
    cuda.manual_seed = lambda s: jt.set_global_seed(int(s))
    cuda.manual_seed_all = lambda s: jt.set_global_seed(int(s))
    cuda.is_bf16_supported = lambda: True
    cuda.get_device_capability = lambda *a, **k: _cuda_capability()
    def _device_name(*a, **k):
        try:
            return "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(a[0] if a else None)
        except Exception:
            return "CUDA"
    cuda.get_device_name = _device_name
    cuda.get_device_properties = lambda *a, **k: _DeviceProps()
    class _amp:
        @staticmethod
        def autocast(*a, **k):
            return _AutocastContext()
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
        def wait_stream(self, *a, **k): return None
        def wait_event(self, *a, **k): return None
        def record_event(self, *a, **k): return None
        def query(self): return True
    cuda.Stream = _Stream
    cuda.Event = type("Event", (), {"__init__": lambda self, *a, **k: None,
                                      "record": lambda self, *a, **k: None,
                                      "synchronize": lambda self: None,
                                      "elapsed_time": lambda self, o: 0.0})
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
    # report REAL device memory from jittor's MemInfo (was a 0-stub, so training-code
    # memory logging printed 0). total_cuda_used on an accelerator, else total_cpu_used.
    # jittor doesn't expose a per-reset peak, so max_* track a process-lifetime high-water
    # mark we maintain here (still real, monotone -- better than a flat 0).
    _mem_peak = [0]
    def _mem_used(*a, **k):
        try:
            mi = jt.get_mem_info()
            used = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
    except Exception:
        pass

    if "torch.overrides" not in _modules:
        overrides = _types.ModuleType("torch.overrides")
        class TorchFunctionMode:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        overrides.TorchFunctionMode = TorchFunctionMode
        overrides.BaseTorchFunctionMode = TorchFunctionMode
        overrides.get_default_nowrap_functions = lambda: set()
        overrides.has_torch_function = lambda *a, **k: False
        overrides.handle_torch_function = lambda func, types, *a, **k: func(*a, **k)
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
            def __getattribute__(self, name):
                if name == "allow_tf32":
                    return bool(getattr(
                        jt.flags, "cuda_allow_cudnn_tf32", 0
                    ))
                return super().__getattribute__(name)

            def __setattr__(self, name, value):
                if name == "benchmark" and not getattr(self, "_jittor_cudnn_init", False):
                    try:
                        if getattr(jt, "cudnn", None) is not None and hasattr(jt.cudnn, "set_benchmark"):
                            jt.cudnn.set_benchmark(int(bool(value)))
                    except Exception:
                        pass
                if name == "allow_tf32" and not getattr(
                        self, "_jittor_cudnn_init", False):
                    if hasattr(jt.flags, "cuda_allow_cudnn_tf32"):
                        jt.flags.cuda_allow_cudnn_tf32 = int(bool(value))
                return super().__setattr__(name, value)
        cudnn.__class__ = _CudnnBackendModule
    cudnn._jittor_cudnn_init = True
    cudnn.enabled = getattr(cudnn, "enabled", True)
    cudnn.benchmark = getattr(cudnn, "benchmark", False)
    cudnn.deterministic = getattr(cudnn, "deterministic", False)
    cudnn.allow_tf32 = getattr(cudnn, "allow_tf32", False)
    cudnn.version = getattr(cudnn, "version", lambda: None)
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
    class _MatmulBackend:
        @property
        def allow_tf32(self):
            cuda_tf32 = bool(getattr(jt.flags, "cuda_allow_tf32", 0))
            acl_hf32 = bool(getattr(jt, "acl_allow_hf32", False))
            return cuda_tf32 or acl_hf32

        @allow_tf32.setter
        def allow_tf32(self, value):
            enabled = bool(value)
            if hasattr(jt.flags, "cuda_allow_tf32"):
                jt.flags.cuda_allow_tf32 = int(enabled)
            jt.acl_allow_hf32 = enabled
    if not hasattr(cuda_backend, "matmul") or not isinstance(cuda_backend.matmul, _MatmulBackend):
        cuda_backend.matmul = _MatmulBackend()
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
    if not hasattr(g, "_torch_float32_matmul_precision"):
        g._torch_float32_matmul_precision = "highest"
    def _get_float32_matmul_precision():
        return getattr(g, "_torch_float32_matmul_precision", "highest")
    def _set_float32_matmul_precision(precision):
        if not isinstance(precision, str):
            raise TypeError("precision must be a string")
        precision = precision.lower()
        if precision not in ("highest", "high", "medium"):
            raise ValueError("precision must be one of 'highest', 'high', or 'medium'")
        g._torch_float32_matmul_precision = precision
        try:
            cuda_backend.matmul.allow_tf32 = precision in ("high", "medium")
        except Exception:
            pass
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
    except Exception:
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
    accelerator.current_device_index = lambda *a, **k: 0
    accelerator.set_device_index = lambda *a, **k: None
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
