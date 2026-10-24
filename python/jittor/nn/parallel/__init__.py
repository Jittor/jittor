"""Native neural-network parallel wrappers."""

from .distributed_data_parallel import DistributedDataParallel

__all__ = ["DistributedDataParallel", "FullyShardedDataParallel"]


def __getattr__(name):
    if name == "FullyShardedDataParallel":
        from jittor.distributed.fsdp import FullyShardedDataParallel

        return FullyShardedDataParallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
