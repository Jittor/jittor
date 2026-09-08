"""Publish the stable CUDA API objects on an explicit installation target."""

from .api import (
    CUDAGraph,
    CUDAPluggableAllocator,
    CudaRuntimeState,
    EXPECTED,
    InstallContext,
    OutOfMemoryError,
    TorchFunctionMode,
    _CUDA_TENSOR_TYPES,
    _CudaDeviceContext,
    _CudaDeviceOf,
    _CudnnBackendModule,
    _Event,
    _MatmulBackend,
    _PrecisionBackend,
    _Stream,
    _StreamContext,
    _amp,
    _api_accelerator_current_accelerator,
    _api_accelerator_current_device_index,
    _api_accelerator_is_available,
    _api_accelerator_set_device_index,
    _api_accelerator_set_stream,
    _api_c_mod__accelerator_setAllocatorSettings,
    _api_c_mod__cuda_clearCublasWorkspaces,
    _api_c_mod__cuda_setAllocatorSettings,
    _api_c_mod__disabled_torch_function_impl,
    _api_c_mod__get_tracing_state,
    _api_c_mod__log_api_usage_once,
    _api_cpu_get_cpu_capability,
    _api_cuda__is_in_bad_fork,
    _api_cuda_backend_enable_cudnn_sdp,
    _api_cuda_backend_enable_flash_sdp,
    _api_cuda_backend_enable_math_sdp,
    _api_cuda_backend_enable_mem_efficient_sdp,
    _api_cuda_backend_sdp_kernel,
    _api_cuda_default_stream,
    _api_cuda_get_device_capability,
    _api_cuda_get_device_properties,
    _api_cuda_get_rng_state,
    _api_cuda_get_rng_state_all,
    _api_cuda_initial_seed,
    _api_cuda_ipc_collect,
    _api_cuda_is_bf16_supported,
    _api_cuda_is_initialized,
    _api_cuda_manual_seed,
    _api_cuda_manual_seed_all,
    _api_cuda_memory__set_allocator_settings,
    _api_cuda_memory_stats,
    _api_cuda_seed,
    _api_cuda_seed_all,
    _api_cuda_set_rng_state,
    _api_cuda_set_rng_state_all,
    _api_cuda_synchronize,
    _api_cudnn_version,
    _api_functorch_c__add_batch_dim,
    _api_functorch_c__remove_batch_dim,
    _api_functorch_c_get_unwrapped,
    _api_functorch_c_is_batchedtensor,
    _api_g__C__autograd__pop_saved_tensors_default_hooks,
    _api_g__C__autograd__push_saved_tensors_default_hooks,
    _api_mkldnn_is_available,
    _api_mod_current_device,
    _api_mod_device_count,
    _api_mod_empty_cache,
    _api_mod_ipc_collect,
    _api_mod_is_available,
    _api_mod_is_initialized,
    _api_mod_manual_seed,
    _api_mod_manual_seed_all,
    _api_mod_max_memory_allocated,
    _api_mod_memory_allocated,
    _api_mod_reset_max_memory_allocated,
    _api_mod_reset_peak_memory_stats,
    _api_mod_seed,
    _api_mod_set_device,
    _api_mod_synchronize,
    _api_mp_reductions_rebuild_cuda_tensor,
    _api_mp_reductions_rebuild_tensor,
    _api_mp_reductions_reduce_tensor,
    _api_mps_is_available,
    _api_overrides_get_default_nowrap_functions,
    _current_stream,
    _device_name,
    _empty_cache,
    _get_float32_matmul_precision,
    _handle_torch_function,
    _has_torch_function,
    _mem_get_info,
    _mem_max,
    _mem_used,
    _nvtx_mark,
    _nvtx_range,
    _nvtx_range_end,
    _nvtx_range_pop,
    _nvtx_range_push,
    _nvtx_range_start,
    _parse_to,
    _preferred_blas_library,
    _register_cuda_fidelity,
    _reset_peak,
    _set_float32_matmul_precision,
    _set_stream,
    _types,
    current_device,
    device_count,
    get_install_context,
    is_available,
    jt,
    registry_for,
    set_device,
    swallowed,
)


def _install_cuda(g, registry=None):
    context = get_install_context(g, required=False)
    if context is None:
        context = InstallContext.for_module(g)
    if "cuda_runtime" not in context.state:
        context.state["cuda_runtime"] = CudaRuntimeState()
    _modules = registry_for(g, registry).module_map
    cuda = _types.ModuleType("torch.cuda")

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


    cuda.current_device = current_device
    cuda.set_device = set_device

    cuda.device = _CudaDeviceContext

    cuda.device_of = _CudaDeviceOf
    cuda.is_initialized = _api_cuda_is_initialized
    cuda._is_in_bad_fork = _api_cuda__is_in_bad_fork
    # Match PyTorch's empty_cache() as a memory hint instead of a forced
    # synchronization point. TRELLIS calls it inside the inference path before
    # decode; running jt.gc() there costs several seconds. Users that need
    # explicit release can opt in with JITTOR_TORCH_CUDA_EMPTY_CACHE=gc or sync.

    cuda.empty_cache = _empty_cache
    cuda.synchronize = _api_cuda_synchronize
    cuda.manual_seed = _api_cuda_manual_seed
    cuda.manual_seed_all = _api_cuda_manual_seed_all
    cuda.is_bf16_supported = _api_cuda_is_bf16_supported
    cuda.get_device_capability = _api_cuda_get_device_capability
    cuda.get_device_name = _device_name
    cuda.get_device_properties = _api_cuda_get_device_properties
    cuda.amp = _amp
    # OpenMMLab imports these legacy CUDA tensor classes in type annotations.
    # Keep them distinct from the top-level CPU classes: a direct alias would
    # make a host tensor pass ``isinstance(x, torch.cuda.LongTensor)``.

    for _tensor_name, _cuda_tensor_type in _CUDA_TENSOR_TYPES.items():
        if getattr(g, _tensor_name, None) is not None:
            setattr(cuda, _tensor_name, _cuda_tensor_type)
    # stub classes referenced in annotations / guarded paths
    cuda.CUDAGraph = CUDAGraph
    cuda.Stream = _Stream


    cuda.Event = _Event
    g.Stream = cuda.Stream
    g.Event = cuda.Event
    g.CUDAGraph = cuda.CUDAGraph
    # Jittor currently launches CUDA/ACL work on one physical backend stream.
    # Keep the Python-visible current-stream identity coherent while all logical
    # streams remain serialized on that physical stream.
    cuda.stream = _StreamContext
    cuda.set_stream = _set_stream
    cuda.current_stream = _current_stream
    cuda.default_stream = _api_cuda_default_stream
    nvtx = _types.ModuleType("torch.cuda.nvtx")


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
    cuda.memory_allocated = _mem_used
    cuda.max_memory_allocated = _mem_max
    cuda.memory_reserved = _mem_used
    cuda.max_memory_reserved = _mem_max
    cuda.memory_cached = _mem_used
    cuda.max_memory_cached = _mem_max
    cuda.reset_peak_memory_stats = _reset_peak
    cuda.reset_max_memory_allocated = _reset_peak
    cuda.memory_stats = _api_cuda_memory_stats
    # torch's mem_get_info is cudaMemGetInfo: the DRIVER's free/total for the whole
    # device, counting other processes, the CUDA context and every byte jittor's
    # pool holds -- not just the bytes currently live in Vars. Serving stacks size
    # their weight/activation/KV budget from it (vLLM plans the KV cache as
    # `total*util - (total-free) - peak_activation`), so the flat 64GiB stub that
    # used to stand here made them plan against a device that does not exist.

    cuda.mem_get_info = _mem_get_info
    cuda.ipc_collect = _api_cuda_ipc_collect
    cuda.memory = _types.ModuleType("torch.cuda.memory")
    cuda.memory._set_allocator_settings = _api_cuda_memory__set_allocator_settings
    cuda.memory.empty_cache = cuda.empty_cache
    cuda.memory.memory_allocated = cuda.memory_allocated
    cuda.memory.max_memory_allocated = cuda.max_memory_allocated
    cuda.memory.memory_reserved = cuda.memory_reserved
    cuda.memory.max_memory_reserved = cuda.max_memory_reserved
    cuda.memory.CUDAPluggableAllocator = CUDAPluggableAllocator
    cuda.CUDAPluggableAllocator = CUDAPluggableAllocator
    # rng state (trainer checkpoints save/restore it). jittor has no portable
    # CUDA rng-state handle, so use a small placeholder Var round-trip.
    cuda.get_rng_state = _api_cuda_get_rng_state
    cuda.get_rng_state_all = _api_cuda_get_rng_state_all
    cuda.set_rng_state = _api_cuda_set_rng_state
    cuda.set_rng_state_all = _api_cuda_set_rng_state_all
    cuda.initial_seed = _api_cuda_initial_seed
    cuda.seed = _api_cuda_seed
    cuda.seed_all = _api_cuda_seed_all
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
        _mod.is_available = getattr(_mod, "is_available", _api_mod_is_available)
        _mod.is_initialized = getattr(_mod, "is_initialized", _api_mod_is_initialized)
        _mod.device_count = getattr(_mod, "device_count", _api_mod_device_count)
        _mod.current_device = getattr(_mod, "current_device", _api_mod_current_device)
        _mod.set_device = getattr(_mod, "set_device", _api_mod_set_device)
        _mod.empty_cache = getattr(_mod, "empty_cache", _api_mod_empty_cache)
        _mod.synchronize = getattr(_mod, "synchronize", _api_mod_synchronize)
        _mod.ipc_collect = getattr(_mod, "ipc_collect", _api_mod_ipc_collect)
        _mod.manual_seed = getattr(_mod, "manual_seed", _api_mod_manual_seed)
        _mod.manual_seed_all = getattr(_mod, "manual_seed_all", _api_mod_manual_seed_all)
        _mod.seed = getattr(_mod, "seed", _api_mod_seed)
        _mod.reset_peak_memory_stats = getattr(_mod, "reset_peak_memory_stats", _api_mod_reset_peak_memory_stats)
        _mod.reset_max_memory_allocated = getattr(_mod, "reset_max_memory_allocated", _api_mod_reset_max_memory_allocated)
        _mod.memory_allocated = getattr(_mod, "memory_allocated", _api_mod_memory_allocated)
        _mod.max_memory_allocated = getattr(_mod, "max_memory_allocated", _api_mod_max_memory_allocated)
        setattr(g, _dev_ns, _mod)

    if "torch.multiprocessing" not in _modules:
        import multiprocessing as _mp
        _modules["torch.multiprocessing"] = _mp
    g.multiprocessing = _modules["torch.multiprocessing"]
    _mp_reductions = _types.ModuleType("torch.multiprocessing.reductions")
    _mp_reductions.reduce_tensor = _api_mp_reductions_reduce_tensor
    _mp_reductions.rebuild_cuda_tensor = _api_mp_reductions_rebuild_cuda_tensor
    _mp_reductions.rebuild_tensor = _api_mp_reductions_rebuild_tensor
    _modules["torch.multiprocessing.reductions"] = _mp_reductions
    try:
        g.multiprocessing.reductions = _mp_reductions
    except (AttributeError, TypeError) as exc:
        swallowed("torch/installers/cuda/bindings.py _install_cuda: g.multiprocessing.reductions = _mp_reductions", exc)

    if "torch.overrides" not in _modules:
        overrides = _types.ModuleType("torch.overrides")


        overrides.TorchFunctionMode = TorchFunctionMode
        overrides.BaseTorchFunctionMode = TorchFunctionMode
        overrides.get_default_nowrap_functions = _api_overrides_get_default_nowrap_functions
        overrides.has_torch_function = _has_torch_function
        overrides.has_torch_function_unary = _has_torch_function
        overrides.has_torch_function_variadic = _has_torch_function
        overrides.handle_torch_function = _handle_torch_function
        _modules["torch.overrides"] = overrides
    g.overrides = _modules["torch.overrides"]

    if "torch._C" not in _modules:
        c_mod = _types.ModuleType("torch._C")
        c_mod._TensorMeta = type(getattr(g, "Tensor", jt.Var))
        c_mod._get_tracing_state = _api_c_mod__get_tracing_state
        c_mod._log_api_usage_once = _api_c_mod__log_api_usage_once
        c_mod._cuda_clearCublasWorkspaces = _api_c_mod__cuda_clearCublasWorkspaces
        c_mod._disabled_torch_function_impl = _api_c_mod__disabled_torch_function_impl
        functorch_c = _types.ModuleType("torch._C._functorch")
        functorch_c.get_unwrapped = _api_functorch_c_get_unwrapped
        functorch_c.is_batchedtensor = _api_functorch_c_is_batchedtensor
        functorch_c._add_batch_dim = _api_functorch_c__add_batch_dim
        functorch_c._remove_batch_dim = _api_functorch_c__remove_batch_dim
        c_mod._distributed_c10d = _types.SimpleNamespace(Reducer=type("Reducer", (), {}))
        nn_c = _types.ModuleType("torch._C._nn")
        nn_c._parse_to = _parse_to
        c_mod._nn = nn_c
        c_mod._functorch = functorch_c
        # Allocator tuning arrives as a settings string that the caching
        # allocator parses. Jittor manages its own pool, so there is nothing to
        # tune -- but the call has to exist, because the caller makes it before
        # asking whether it could have had any effect.
        c_mod._accelerator_setAllocatorSettings = _api_c_mod__accelerator_setAllocatorSettings
        c_mod._cuda_setAllocatorSettings = getattr(
            c_mod, "_cuda_setAllocatorSettings", _api_c_mod__cuda_setAllocatorSettings)
        _modules["torch._C"] = c_mod
        _modules["torch._C._nn"] = nn_c
        _modules["torch._C._functorch"] = functorch_c
    g._C = _modules["torch._C"]
    if not hasattr(g._C, "_autograd"):
        g._C._autograd = _types.SimpleNamespace()
    g._C._autograd._push_saved_tensors_default_hooks = _api_g__C__autograd__push_saved_tensors_default_hooks
    g._C._autograd._pop_saved_tensors_default_hooks = _api_g__C__autograd__pop_saved_tensors_default_hooks
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
        cudnn.__class__ = _CudnnBackendModule
    cudnn._jittor_cudnn_init = True
    cudnn.enabled = getattr(cudnn, "enabled", True)
    cudnn.benchmark = getattr(cudnn, "benchmark", False)
    cudnn.deterministic = getattr(cudnn, "deterministic", False)
    cudnn.version = getattr(cudnn, "version", _api_cudnn_version)
    if not isinstance(getattr(cudnn, "conv", None), _PrecisionBackend):
        cudnn.conv = _PrecisionBackend("cudnn", "torch.backends.cudnn.conv")
    if not isinstance(getattr(cudnn, "rnn", None), _PrecisionBackend):
        cudnn.rnn = _PrecisionBackend("cudnn", "torch.backends.cudnn.rnn")
    cudnn._jittor_cudnn_init = False
    cuda_backend = _modules.get("torch.backends.cuda")
    if cuda_backend is None:
        cuda_backend = _types.ModuleType("torch.backends.cuda")
        _modules["torch.backends.cuda"] = cuda_backend
    cuda_backend.sdp_kernel = getattr(cuda_backend, "sdp_kernel", _api_cuda_backend_sdp_kernel)
    cuda_backend.enable_flash_sdp = getattr(cuda_backend, "enable_flash_sdp", _api_cuda_backend_enable_flash_sdp)
    cuda_backend.enable_mem_efficient_sdp = getattr(cuda_backend, "enable_mem_efficient_sdp", _api_cuda_backend_enable_mem_efficient_sdp)
    cuda_backend.enable_math_sdp = getattr(cuda_backend, "enable_math_sdp", _api_cuda_backend_enable_math_sdp)
    # The fourth of torch's attention-backend switches. Attention here picks
    # its own path, so all four are settings nothing acts on -- but a serving
    # stack turns cuDNN's off during platform detection, and an AttributeError
    # there is swallowed into "no platform detected" rather than reported.
    cuda_backend.enable_cudnn_sdp = getattr(cuda_backend, "enable_cudnn_sdp", _api_cuda_backend_enable_cudnn_sdp)
    if not hasattr(cuda_backend, "matmul") or not isinstance(cuda_backend.matmul, _MatmulBackend):
        cuda_backend.matmul = _MatmulBackend()
    cuda_backend._preferred_blas_library = getattr(
        cuda_backend, "_preferred_blas_library", "cublas")
    cuda_backend.preferred_blas_library = _preferred_blas_library
    mps = _modules.get("torch.backends.mps")
    if mps is None:
        mps = _types.ModuleType("torch.backends.mps")
        _modules["torch.backends.mps"] = mps
    mps.is_available = getattr(mps, "is_available", _api_mps_is_available)
    cpu = _modules.get("torch.backends.cpu")
    if cpu is None:
        cpu = _types.ModuleType("torch.backends.cpu")
        _modules["torch.backends.cpu"] = cpu
    cpu.get_cpu_capability = getattr(cpu, "get_cpu_capability", _api_cpu_get_cpu_capability)
    mkldnn = _modules.get("torch.backends.mkldnn")
    if mkldnn is None:
        mkldnn = _types.ModuleType("torch.backends.mkldnn")
        _modules["torch.backends.mkldnn"] = mkldnn
    mkldnn.is_available = getattr(mkldnn, "is_available", _api_mkldnn_is_available)
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
        swallowed("torch/installers/cuda/bindings.py _install_version: nv = getattr(getattr(jt, 'compiler', None), 'nvcc_versi...", exc)
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
        g.OutOfMemoryError = OutOfMemoryError
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
    accelerator.is_available = _api_accelerator_is_available
    accelerator.device_count = cuda.device_count
    accelerator.current_device_index = _api_accelerator_current_device_index
    accelerator.set_device_index = _api_accelerator_set_device_index
    accelerator.device_index = getattr(cuda, "device", None)
    accelerator.current_stream = cuda.current_stream
    accelerator.set_stream = getattr(cuda, "set_stream", _api_accelerator_set_stream)
    accelerator.synchronize = cuda.synchronize
    accelerator.empty_cache = cuda.empty_cache
    accelerator.memory_allocated = cuda.memory_allocated
    accelerator.memory_reserved = cuda.memory_reserved
    accelerator.max_memory_allocated = cuda.max_memory_allocated
    accelerator.memory_stats = cuda.memory_stats
    accelerator.reset_peak_memory_stats = cuda.reset_peak_memory_stats
    accelerator.current_accelerator = _api_accelerator_current_accelerator
    g.accelerator = accelerator
    if registry is not None:
        registry.publish("torch.accelerator", accelerator)


def install(ctx):
    g = ctx.jittor_module
    _install_cuda(g, ctx.registry)
    _install_version(g, ctx.registry)
    _install_accelerator(g, ctx.registry)
    _register_cuda_fidelity(ctx)
