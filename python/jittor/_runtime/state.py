"""Runtime state views independent of native bootstrap and tensor operations."""

class RuntimeContext:
    """Read-only access to execution state backed by an injected native Flags object.

    The context deliberately holds the native ``Flags`` object instead of
    copying individual values.  This gives the runtime a single migration
    seam while preserving the existing flag_scope and native setter semantics.
    """

    __slots__ = ("_flags",)

    def __init__(self, native_flags):
        self._flags = native_flags

    @property
    def sync_run(self):
        """Whether backend operators should synchronize after each launch."""
        return self._flags.sync_run

    @property
    def device_id(self):
        """Current device selected by the native runtime, or ``-1`` on CPU."""
        return getattr(self._flags, "device_id", -1)

    @property
    def use_cuda(self):
        """Whether the native runtime is configured to use CUDA."""
        return self._flags.use_cuda

    @property
    def cpu_mem_limit(self):
        """Maximum CPU allocation budget in bytes, or ``-1`` when unlimited."""
        return self._flags.cpu_mem_limit

    @property
    def device_mem_limit(self):
        """Maximum device allocation budget in bytes, or ``-1`` when unlimited."""
        return self._flags.device_mem_limit

    @property
    def node_order(self):
        """Ordering policy used when assigning graph node priorities."""
        return self._flags.node_order

    @property
    def lazy_execution(self):
        """Whether graph execution is deferred until an explicit flush."""
        return self._flags.lazy_execution

    @property
    def auto_flush_ops(self):
        """CUDA pipeline threshold for automatically submitting pending ops."""
        return self._flags.auto_flush_ops

    @property
    def auto_convert_64_to_32(self):
        """Whether NumPy 64-bit scalar arrays are narrowed on import."""
        return self._flags.auto_convert_64_to_32

    @property
    def reuse_array(self):
        """Whether CPU NumPy array storage may be reused by ``jt.array``."""
        return self._flags.reuse_array

    @property
    def no_grad(self):
        """Whether newly created operations are excluded from autograd."""
        return self._flags.no_grad

    @property
    def amp_reg(self):
        """Auto-mixed-precision register used while inferring op dtypes."""
        return self._flags.amp_reg

    @property
    def float32_matmul_precision(self):
        """Accumulation policy shared by float32 matmul and convolution."""
        return self._flags.float32_matmul_precision

    @property
    def use_tensorcore(self):
        """Legacy tensor-core enable flag retained for compatibility."""
        return self._flags.use_tensorcore

    @property
    def cuda_allow_tf32(self):
        """Whether CUDA matmul may use TF32 accumulation."""
        return self._flags.cuda_allow_tf32

    @property
    def auto_mixed_precision_level(self):
        """Convenience AMP policy level reflected by the native flag."""
        return self._flags.auto_mixed_precision_level

    @property
    def try_use_32bit_index(self):
        """Whether operators may use 32-bit indices when shapes fit."""
        return self._flags.try_use_32bit_index

    @property
    def no_fuse(self):
        """Whether fusion optimization is disabled for new operations."""
        return self._flags.no_fuse

    @property
    def gopt_disable(self):
        """Whether graph optimization is disabled for execution."""
        return self._flags.gopt_disable

    @property
    def enable_tuner(self):
        """Whether the compiler's operator tuners may select candidates."""
        return self._flags.enable_tuner

    @property
    def exec_called(self):
        """Number of executor synchronizations started by the runtime."""
        return self._flags.exec_called

    @property
    def use_threading(self):
        """Whether executor synchronization may use Python threading."""
        return self._flags.use_threading

    @property
    def use_parallel_op_compiler(self):
        """Number of workers used by the parallel operator compiler."""
        return self._flags.use_parallel_op_compiler

    @property
    def profile_memory_enable(self):
        """Whether execution records memory-profiler state."""
        return self._flags.profile_memory_enable

    @property
    def profiler_warmup(self):
        """Number of profiler warmup rounds configured for execution."""
        return self._flags.profiler_warmup

    @property
    def profiler_enable(self):
        """Whether the runtime profiler is collecting execution records."""
        return self._flags.profiler_enable

    @property
    def profiler_rerun(self):
        """Number of additional profiler reruns configured for execution."""
        return self._flags.profiler_rerun

    @property
    def profiler_record_peek(self):
        """Whether the profiler records memory-bandwidth peek data."""
        return self._flags.profiler_record_peek

    @property
    def profiler_record_shape(self):
        """Whether the profiler records per-operator shape metadata."""
        return self._flags.profiler_record_shape

    @property
    def profiler_hide_relay(self):
        """Whether relayed profiler operators are hidden from reports."""
        return self._flags.profiler_hide_relay

    @property
    def check_graph(self):
        """Whether graph liveness checks are enabled for execution."""
        return self._flags.check_graph

    @property
    def missing_grad_error(self):
        """Whether a missing target gradient is reported as an error."""
        return self._flags.missing_grad_error

    @property
    def disable_lock(self):
        """Whether the build cache file lock is disabled."""
        return self._flags.disable_lock

    @property
    def trace_var_data(self):
        """Whether variable data tracing is enabled for Python debugging."""
        return self._flags.trace_var_data

    @property
    def trace_py_var(self):
        """Whether Python variable stack tracing is enabled."""
        return self._flags.trace_py_var

    @property
    def trace_depth(self):
        """Maximum native debugger backtrace depth."""
        return self._flags.trace_depth

    @property
    def rewrite_op(self):
        """Whether generated JIT operator sources may be rewritten."""
        return self._flags.rewrite_op

    @property
    def log_silent(self):
        """Whether informational and warning logs are suppressed."""
        return self._flags.log_silent

    @property
    def log_sync(self):
        """Whether log records are emitted synchronously."""
        return self._flags.log_sync

    @property
    def log_v(self):
        """Current verbose logging level."""
        return self._flags.log_v

    @property
    def use_stat_allocator(self):
        """Allocator policy that records allocation statistics."""
        return self._flags.use_stat_allocator

    @property
    def use_nfef_allocator(self):
        """Whether the never-free exact-fit allocator is enabled."""
        return self._flags.use_nfef_allocator

    @property
    def use_temp_allocator(self):
        """Whether the temporary allocator is enabled."""
        return self._flags.use_temp_allocator

    @property
    def use_sfrl_allocator(self):
        """Whether the SFRL allocator is enabled."""
        return self._flags.use_sfrl_allocator

    @property
    def use_cuda_host_allocator(self):
        """Whether CUDA host allocations use the pinned host allocator."""
        return self._flags.use_cuda_host_allocator

    def snapshot(self):
        """Return a detached dictionary snapshot of the exposed native fields."""
        return {
            "sync_run": int(self.sync_run),
            "device_id": int(self.device_id),
            "use_cuda": int(self.use_cuda),
            "cpu_mem_limit": int(self.cpu_mem_limit),
            "device_mem_limit": int(self.device_mem_limit),
            "node_order": int(self.node_order),
            "lazy_execution": int(self.lazy_execution),
            "auto_flush_ops": int(self.auto_flush_ops),
            "auto_convert_64_to_32": int(self.auto_convert_64_to_32),
            "reuse_array": int(self.reuse_array),
            "no_grad": int(self.no_grad),
            "amp_reg": int(self.amp_reg),
            "float32_matmul_precision": self.float32_matmul_precision,
            "use_tensorcore": int(self.use_tensorcore),
            "cuda_allow_tf32": int(self.cuda_allow_tf32),
            "auto_mixed_precision_level": int(self.auto_mixed_precision_level),
            "try_use_32bit_index": int(self.try_use_32bit_index),
            "no_fuse": int(self.no_fuse),
            "gopt_disable": int(self.gopt_disable),
            "enable_tuner": int(self.enable_tuner),
            "exec_called": int(self.exec_called),
            "use_threading": int(self.use_threading),
            "use_parallel_op_compiler": int(self.use_parallel_op_compiler),
            "profile_memory_enable": int(self.profile_memory_enable),
            "profiler_warmup": int(self.profiler_warmup),
            "profiler_enable": int(self.profiler_enable),
            "profiler_rerun": int(self.profiler_rerun),
            "profiler_record_peek": int(self.profiler_record_peek),
            "profiler_record_shape": int(self.profiler_record_shape),
            "profiler_hide_relay": int(self.profiler_hide_relay),
            "check_graph": int(self.check_graph),
            "missing_grad_error": int(self.missing_grad_error),
            "disable_lock": int(self.disable_lock),
            "rewrite_op": int(self.rewrite_op),
            "trace_var_data": int(self.trace_var_data),
            "trace_py_var": int(self.trace_py_var),
            "trace_depth": int(self.trace_depth),
            "log_silent": int(self.log_silent),
            "log_sync": int(self.log_sync),
            "log_v": int(self.log_v),
            "use_stat_allocator": int(self.use_stat_allocator),
            "use_nfef_allocator": int(self.use_nfef_allocator),
            "use_temp_allocator": int(self.use_temp_allocator),
            "use_sfrl_allocator": int(self.use_sfrl_allocator),
            "use_cuda_host_allocator": int(self.use_cuda_host_allocator),
        }


class RuntimeState:
    """Read-only Python view of :class:`RuntimeContext`.

    The view stores no state of its own.  In particular, ``flag_scope`` and
    direct native flag writes remain immediately visible through this object.
    """

    __slots__ = ("_context",)

    def __init__(self, context):
        self._context = context

    @property
    def sync_run(self):
        return self._context.sync_run

    @property
    def device_id(self):
        return self._context.device_id

    @property
    def use_cuda(self):
        return self._context.use_cuda

    @property
    def cpu_mem_limit(self):
        return self._context.cpu_mem_limit

    @property
    def device_mem_limit(self):
        return self._context.device_mem_limit

    @property
    def node_order(self):
        return self._context.node_order

    @property
    def lazy_execution(self):
        return self._context.lazy_execution

    @property
    def auto_flush_ops(self):
        return self._context.auto_flush_ops

    @property
    def auto_convert_64_to_32(self):
        return self._context.auto_convert_64_to_32

    @property
    def reuse_array(self):
        return self._context.reuse_array

    @property
    def no_grad(self):
        return self._context.no_grad

    @property
    def amp_reg(self):
        return self._context.amp_reg

    @property
    def float32_matmul_precision(self):
        return self._context.float32_matmul_precision

    @property
    def use_tensorcore(self):
        return self._context.use_tensorcore

    @property
    def cuda_allow_tf32(self):
        return self._context.cuda_allow_tf32

    @property
    def auto_mixed_precision_level(self):
        return self._context.auto_mixed_precision_level

    @property
    def try_use_32bit_index(self):
        return self._context.try_use_32bit_index

    @property
    def no_fuse(self):
        return self._context.no_fuse

    @property
    def gopt_disable(self):
        return self._context.gopt_disable

    @property
    def enable_tuner(self):
        return self._context.enable_tuner

    @property
    def exec_called(self):
        return self._context.exec_called

    @property
    def use_threading(self):
        return self._context.use_threading

    @property
    def use_parallel_op_compiler(self):
        return self._context.use_parallel_op_compiler

    @property
    def profile_memory_enable(self):
        return self._context.profile_memory_enable

    @property
    def profiler_warmup(self):
        return self._context.profiler_warmup

    @property
    def profiler_enable(self):
        return self._context.profiler_enable

    @property
    def profiler_rerun(self):
        return self._context.profiler_rerun

    @property
    def profiler_record_peek(self):
        return self._context.profiler_record_peek

    @property
    def profiler_record_shape(self):
        return self._context.profiler_record_shape

    @property
    def profiler_hide_relay(self):
        return self._context.profiler_hide_relay

    @property
    def check_graph(self):
        return self._context.check_graph

    @property
    def missing_grad_error(self):
        return self._context.missing_grad_error

    @property
    def disable_lock(self):
        return self._context.disable_lock

    @property
    def trace_var_data(self):
        return self._context.trace_var_data

    @property
    def trace_py_var(self):
        return self._context.trace_py_var

    @property
    def trace_depth(self):
        return self._context.trace_depth

    @property
    def rewrite_op(self):
        return self._context.rewrite_op

    @property
    def log_silent(self):
        return self._context.log_silent

    @property
    def log_sync(self):
        return self._context.log_sync

    @property
    def log_v(self):
        return self._context.log_v

    @property
    def use_stat_allocator(self):
        return self._context.use_stat_allocator

    @property
    def use_nfef_allocator(self):
        return self._context.use_nfef_allocator

    @property
    def use_temp_allocator(self):
        return self._context.use_temp_allocator

    @property
    def use_sfrl_allocator(self):
        return self._context.use_sfrl_allocator

    @property
    def use_cuda_host_allocator(self):
        return self._context.use_cuda_host_allocator

    @property
    def context(self):
        """The state owner, exposed for diagnostics but not replacement."""
        return self._context

__all__ = ["RuntimeContext", "RuntimeState"]

