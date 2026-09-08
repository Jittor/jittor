"""Stable FSDP spelling adapters; sharding and collective math stay in owners."""
import contextlib
import enum

from . import api, common, compat_types, shard
from ..torch.fidelity import Fidelity, register_fidelity


@contextlib.contextmanager
def enable_wrap(*args, **kwargs):
    """Legacy scope spelling; does not install implicit wrapping defaults."""
    yield


def wrap(module, *args, **kwargs):
    return api.fully_shard(module, **{
        key: value for key, value in kwargs.items()
        if key in ("mesh", "reshard_after_forward", "mp_policy", "offload_policy")
    })


def always_wrap_policy(*args, **kwargs):
    return True


def size_based_auto_wrap_policy(module, recurse, nonwrapped_numel,
                                min_num_params=1e8, *args, **kwargs):
    return bool(nonwrapped_numel >= min_num_params)


def transformer_auto_wrap_policy(*args, **kwargs):
    return False


def lambda_auto_wrap_policy(module, recurse, nonwrapped_numel,
                            lambda_fn=None, *args, **kwargs):
    return bool(lambda_fn(module) if callable(lambda_fn) else False)


def _or_policy(module, recurse, nonwrapped_numel, policies=None, *args, **kwargs):
    return any(policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
               for policy in (policies or ()) if callable(policy))


def _get_fsdp_states(module):
    return [getattr(child, "_fsdp_state")
            for child in shard._iter_fsdp_modules(module, True)
            if hasattr(child, "_fsdp_state")]


def _get_fsdp_handles(module):
    """The local FSDP2 implementation has no FSDP1 FlatParamHandle objects."""
    return []


def all_gather(tensor, *args, **kwargs):
    return common._all_gather_shards(tensor) if common._in_true_distributed() else tensor


def reduce_scatter(tensor, *args, **kwargs):
    return common._reduce_scatter_padded(tensor) if common._in_true_distributed() else tensor


def is_available(*args, **kwargs):
    """Compatibility namespace availability, not communicator readiness."""
    return True


class AsyncCollectiveTensor:
    """Legacy completed-value wrapper; this does not schedule asynchronous work."""
    def __init__(self, tensor=None):
        self.tensor = tensor

    def wait(self):
        return self.tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def offload_wrapper(module, *args, **kwargs):
    return module


class CheckpointImpl(enum.Enum):
    NO_REENTRANT = "no_reentrant"
    REENTRANT = "reentrant"


class ColwiseParallel(compat_types.ParallelStyle):
    pass


class RowwiseParallel(compat_types.ParallelStyle):
    pass


class SequenceParallel(compat_types.ParallelStyle):
    pass


class PrepareModuleInput(compat_types.ParallelStyle):
    pass


class PrepareModuleOutput(compat_types.ParallelStyle):
    pass


class PrepareModuleInputOutput(compat_types.ParallelStyle):
    pass


WRAP_HELPERS = (
    "enable_wrap", "wrap", "always_wrap_policy", "size_based_auto_wrap_policy",
    "transformer_auto_wrap_policy", "lambda_auto_wrap_policy", "_or_policy",
)
PARALLEL_STYLES = (
    "ColwiseParallel", "RowwiseParallel", "SequenceParallel", "PrepareModuleInput",
    "PrepareModuleOutput", "PrepareModuleInputOutput",
)


def register_helper_fidelity():
    """Publish fidelity with the same transaction that installs these helpers."""
    wrap_prefix = "torch.distributed.fsdp.wrap."
    for name in WRAP_HELPERS:
        level = Fidelity.APPROXIMATE
        detail = "Legacy wrapping predicate/adapter; not the complete Torch auto-wrap traversal contract"
        if name in ("enable_wrap", "transformer_auto_wrap_policy"):
            level = Fidelity.UNIMPLEMENTED
            detail = ("Legacy no-op wrapping scope" if name == "enable_wrap" else
                      "Legacy policy always declines wrapping; transformer class matching is absent")
        register_fidelity(wrap_prefix + name, globals()[name], level, detail)
    for name in PARALLEL_STYLES:
        register_fidelity("torch.distributed.tensor.parallel." + name, globals()[name],
                          Fidelity.UNIMPLEMENTED,
                          "Configuration holder only; does not implement tensor-parallel transformation")
    for name, function, level, detail in (
        ("torch.distributed.is_available", is_available, Fidelity.APPROXIMATE,
         "Reports installed namespace availability, not backend or communicator readiness"),
        ("torch.distributed._functional_collectives.AsyncCollectiveTensor", AsyncCollectiveTensor,
         Fidelity.APPROXIMATE, "Wraps an already returned tensor; wait does not schedule or synchronize new work"),
        ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.offload_wrapper", offload_wrapper,
         Fidelity.UNIMPLEMENTED, "Legacy identity wrapper; no activation offload is performed"),
        ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointImpl", CheckpointImpl,
         Fidelity.APPROXIMATE, "Mode spelling only; the local checkpoint path does not recompute activations"),
        ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper",
         compat_types._checkpoint_wrapper, Fidelity.UNIMPLEMENTED, "Legacy identity wrapper; no activation recomputation"),
        ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing",
         compat_types._apply_activation_checkpointing, Fidelity.UNIMPLEMENTED, "Legacy identity operation; model is not rewritten"),
        ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint",
         compat_types._checkpoint, Fidelity.UNIMPLEMENTED, "Calls the function directly; no backward recomputation"),
        ("torch.distributed.fsdp._fully_shard._fsdp_collectives.all_gather", all_gather,
         Fidelity.APPROXIMATE, "Delegates to existing shard gather in distributed mode, otherwise returns input"),
        ("torch.distributed.fsdp._fully_shard._fsdp_collectives.reduce_scatter", reduce_scatter,
         Fidelity.APPROXIMATE, "Delegates to existing padded reduce-scatter in distributed mode, otherwise returns input"),
    ):
        register_fidelity(name, function, level, detail)
