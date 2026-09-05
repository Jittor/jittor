"""The native flag partition shared by binding generation and Python runtime."""

STARTUP_FLAGS = frozenset((
    "jittor_path", "cc_path", "cc_type", "cc_flags", "nvcc_path", "nvcc_flags",
    "python_path", "cache_path", "cuda_archs", "disable_lock",
))

READONLY_FLAGS = frozenset((
    "exec_called", "stat_allocator_total_alloc_call", "stat_allocator_total_alloc_byte",
    "stat_allocator_total_free_call", "stat_allocator_total_free_byte",
))

RUNTIME_FLAGS = frozenset((
    "use_cuda", "device_id", "sync_run", "compile_options", "no_grad", "no_fuse",
    "node_order", "amp_reg", "auto_mixed_precision_level", "auto_convert_64_to_32",
    "reuse_array", "missing_grad_error", "try_use_32bit_index", "check_graph",
    "lazy_execution", "auto_flush_ops", "gopt_disable", "use_threading",
    "use_parallel_op_compiler", "float32_matmul_precision", "use_tensorcore",
    "cuda_allow_tf32", "cuda_allow_cudnn_tf32", "cuda_kernel_math",
    "cpu_mem_limit", "device_mem_limit", "use_cuda_host_allocator",
    "use_cuda_managed_allocator", "use_nfef_allocator", "use_stat_allocator",
    "use_temp_allocator", "use_sfrl_allocator", "sfrl_large_block_size_device",
    "cuda_device_allocator_managed_fallback", "profile_memory_enable",
    "profiler_enable", "profiler_warmup", "profiler_rerun", "profiler_record_peek",
    "profiler_record_shape", "profiler_hide_relay", "enable_tuner", "exclude_pass",
    "log_op_hash", "para_opt_level", "l1_cache_size", "rewrite_op",
    "jit_search_kernel", "jit_search_warmup", "jit_search_rerun",
    "jit_search_timeout", "jit_search_max_candidates", "gdb_path", "addr2line_path",
    "extra_gdb_cmd", "has_pybt", "trace_depth", "gdb_trace_timeout", "gdb_attach",
    "trace_py_var", "trace_var_data", "log_sync", "log_silent", "log_v",
    "log_vprefix", "log_file",
))

FLAG_ALIASES = {
    "use_device": "use_cuda", "use_acl": "use_cuda", "use_rocm": "use_cuda",
    "use_corex": "use_cuda", "amp_level": "auto_mixed_precision_level",
}


def flag_category(name):
    name = FLAG_ALIASES.get(name, name)
    if name in STARTUP_FLAGS:
        return "startup"
    if name in READONLY_FLAGS:
        return "counter"
    if name in RUNTIME_FLAGS:
        return "runtime"
    raise ValueError("Native flag has no lifecycle classification: " + name)
