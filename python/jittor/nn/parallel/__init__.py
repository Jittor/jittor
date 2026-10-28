"""Native neural-network parallel wrappers."""

from .distributed_data_parallel import DistributedDataParallel

__all__ = ["DistributedDataParallel", "ColumnParallelLinear", "RowParallelLinear",
           "FullyShardedDataParallel"]


def __getattr__(name):
    # Importing nn is part of core initialization. Keep distributed owners out of
    # that import chain until their public wrappers are actually requested.
    if name in ("ColumnParallelLinear", "RowParallelLinear"):
        from jittor.distributed import tensor_parallel
        return getattr(tensor_parallel, name)
    if name == "FullyShardedDataParallel":
        from jittor.distributed.fsdp import FullyShardedDataParallel
        return FullyShardedDataParallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
