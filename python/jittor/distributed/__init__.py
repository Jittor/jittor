"""Jittor distributed launching and cross-process rendezvous primitives.

See ``jittor.distributed.launch`` (run as ``python -m jittor.distributed.launch``).
"""

from .bucket import bucket_scope, comm_wait, join_pending
from .store import FileStore, PrefixStore, Store, TCPStore, rendezvous
from .process_group import (
    ProcessGroup,
    Work,
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    destroy_process_group,
    get_backend,
    get_device,
    get_default_group,
    get_default_store,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_process_group_ranks,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
    reduce_scatter,
)


__all__ = ["FileStore", "PrefixStore", "Store", "TCPStore", "rendezvous",
           "bucket_scope", "comm_wait", "join_pending", "ProcessGroup", "Work",
           "init_process_group", "destroy_process_group", "is_initialized",
           "get_rank", "get_world_size", "get_backend", "get_default_group",
           "get_default_store", "get_local_rank", "get_local_world_size",
           "get_device", "all_reduce", "broadcast", "all_gather",
           "reduce_scatter", "barrier", "new_group", "get_global_rank",
           "get_process_group_ranks"]
