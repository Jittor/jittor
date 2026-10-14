"""Native neural-network parallel wrappers."""

from .distributed_data_parallel import DistributedDataParallel

__all__ = ["DistributedDataParallel"]
