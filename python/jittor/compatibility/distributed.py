import datetime
from enum import Enum
import jittor as jt


class DistributedDataParallel:
    def __new__(cls, model):
        return model

def is_initialized():
    return True

def get_rank(group=None):
    return 0

def get_world_size(group=None):
    return 1

def get_backend(group=None):
    return "nccl"

def new_group(ranks=None, timeout=datetime.timedelta(seconds=1800), backend=None, pg_options=None):
    return 1

def barrier():
    pass

def is_available():
    return True 

def is_built():
    return True

class ReduceOp:
    SUM = 0

class GroupMember:
    WORLD = 0 

class ProcessGroup:
    pass

class Join:
    pass

dist_backend = Enum("dist_backend", ("GLOO", "MPI", "NCCL"))
_backend = dist_backend.NCCL

def is_mpi_available():
    return jt.in_mpi

def DistributedDataParallel(model, *args, **kw):
    return model
