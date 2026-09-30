"""Restricted DeepSpeed accelerator using public framework device APIs.

The explicit adapter restricts NPU execution to the verified FP32 eager
Stage 0/1/2/3 configurations with explicit torch.optim.AdamW and no offload.
Checkpoint RNG, streams, mixed precision, extension ops and memory profiling
remain unsupported. Unsupported calls raise rather than pretending to execute.
"""
import torch



class UnsupportedStream:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Explicit accelerator streams are outside this experiment")


class UnsupportedEvent:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Accelerator events are outside this experiment")


def _unsupported(*args, **kwargs):
    raise NotImplementedError(
        "This API is outside the verified FP32 eager provider scope (explicit AdamW, no offload); "
        "no stream, RNG, memory-statistic or extension result is fabricated")


class EagerOnly:
    """Shared restrictions override both stock CPU and NPU accelerators."""

    def op_builder_dir(self):
        return __package__ + ".builders"

    def get_op_builder(self, class_name):
        from .builders.unsupported import UnsupportedBuilder
        return UnsupportedBuilder

    def create_op_builder(self, class_name):
        return self.get_op_builder(class_name)()

    build_extension = _unsupported

    @property
    def Stream(self):
        return UnsupportedStream

    @property
    def Event(self):
        # DeepSpeed evaluates Event and Stream properties in type annotations.
        # Return a stable type; actual use still fails explicitly.
        return UnsupportedEvent

    def is_synchronized_device(self):
        return True

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None
    random = _unsupported
    get_rng_state = _unsupported
    set_rng_state = _unsupported
    manual_seed = _unsupported
    manual_seed_all = _unsupported
    initial_seed = _unsupported
    default_generator = _unsupported
    lazy_call = _unsupported
    def empty_cache(self):
        import jittor
        return jittor.gc()
    def memory_allocated(self, device_index=None):
        return torch.accelerator.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.accelerator.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.accelerator.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return torch.accelerator.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return torch.cuda.max_memory_reserved(device_index)

    def memory_cached(self, device_index=None):
        return self.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        return self.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.accelerator.reset_peak_memory_stats(device_index)

    def memory_stats(self, device_index=None):
        return torch.accelerator.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return torch.accelerator.reset_peak_memory_stats(device_index)
    total_memory = _unsupported
    available_memory = _unsupported
    amp = _unsupported
    create_graph = _unsupported
    capture_to_graph = _unsupported
    replay_graph = _unsupported
    pin_memory = _unsupported
    is_pinned = _unsupported

    def is_fp16_supported(self):
        return False

    def is_bf16_supported(self):
        return False

    def supported_dtypes(self):
        return [torch.float32]

    def get_compile_backend(self):
        # Imported function defaults need a value; this is intentionally not
        # an advertised registered compiler. compile itself is out of scope.
        return "jittor_eager_only_unsupported_compile"

    set_compile_backend = _unsupported

    def _tensor_constructor(self, dtype):
        def construct(*args, **kwargs):
            if kwargs:
                raise TypeError("Experimental accelerator tensor factories take positional arguments only")
            if args and all(isinstance(arg, int) for arg in args):
                return torch.empty(tuple(args), dtype=dtype, device=self.current_device_name())
            if len(args) == 1:
                return torch.tensor(args[0], dtype=dtype, device=self.current_device_name())
            raise TypeError("Expected tensor data or integer dimensions")
        return construct

    @property
    def FloatTensor(self):
        return self._tensor_constructor(torch.float32)

    @property
    def ByteTensor(self):
        return self._tensor_constructor(torch.uint8)

    @property
    def IntTensor(self):
        return self._tensor_constructor(torch.int32)

    @property
    def LongTensor(self):
        return self._tensor_constructor(torch.int64)

    @property
    def HalfTensor(self):
        return _unsupported

    @property
    def BFloat16Tensor(self):
        return _unsupported

    @property
    def DoubleTensor(self):
        return _unsupported


def create_accelerator(target):
    """Use only public framework APIs; this adapter is activated in shim only."""
    from .._common import require_version
    from . import SUPPORTED_VERSIONS
    require_version("deepspeed", SUPPORTED_VERSIONS)
    if target == "cpu":
        from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator

        class EagerCPU(EagerOnly, CPU_Accelerator):
            def __init__(self):
                super().__init__()
                self._communication_backend_name = "gloo"

            def device(self, device_index=None):
                return torch.device("cpu")

            def current_device(self):
                return 0

            def device_count(self):
                return 1

            def synchronize(self, device_index=None):
                import jittor
                return jittor.sync_all(True)

        return EagerCPU()
    if target != "npu":
        raise ValueError("Expected cpu or npu")
    from deepspeed.accelerator.npu_accelerator import NPU_Accelerator

    class EagerNPU(EagerOnly, NPU_Accelerator):
        def __init__(self):
            super().__init__()
            self._communication_backend_name = "hccl"
            if not torch.npu.is_available():
                raise RuntimeError("A real available NPU backend is required")

        def use_host_timers(self):
            return False

    return EagerNPU()
