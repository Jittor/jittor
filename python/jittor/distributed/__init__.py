"""Jittor distributed launching and cross-process rendezvous primitives.

See ``jittor.distributed.launch`` (run as ``python -m jittor.distributed.launch``).
"""

from .bucket import bucket_scope, comm_wait, join_pending
from .store import FileStore, PrefixStore, Store, TCPStore, rendezvous
from .process_group import ProcessGroup, Work


__all__ = ["FileStore", "PrefixStore", "Store", "TCPStore", "rendezvous",
           "bucket_scope", "comm_wait", "join_pending", "ProcessGroup", "Work"]
