"""Native process-group identity and backend communicator ownership."""

import os

import jittor as jt


def is_initialized():
    """Whether the native runtime owns an active multi-rank communicator."""
    return int(getattr(jt, "world_size", 1)) > 1 and (
        os.environ.get("JT_NCCL_WORLD_SIZE") is not None
        or os.environ.get("OMPI_COMM_WORLD_SIZE") is not None
        or bool(getattr(jt, "in_mpi", False))
    )


def get_rank():
    return int(getattr(jt, "rank", 0)) if is_initialized() else 0


def get_world_size():
    return int(getattr(jt, "world_size", 1)) if is_initialized() else 1


class Work:
    def __init__(self, value=None):
        self._value = value

    def wait(self, *args, **kwargs):
        return self._value

    def is_completed(self):
        return True


class ProcessGroup:
    def __init__(self, ranks=None, name="default"):
        self.ranks = None if ranks is None else tuple(int(rank) for rank in ranks)
        self._name = name
        self.group_name = name
        self.bound_device_id = 0
        self._backend_kind = None
        self._backend_handle = 0 if ranks is None else None

    def _create_backend_communicator(self):
        compile_extern = jt.compile_extern
        choices = (
            ("nccl", getattr(compile_extern, "nccl", None),
             getattr(compile_extern, "nccl_ops", None)),
            ("hccl", getattr(compile_extern, "hccl_mod", None),
             getattr(compile_extern, "hccl_ops", None)),
        )
        for kind, module, ops in choices:
            create = getattr(
                module, "{}_create_process_group".format(kind), None
            )
            if module is None or ops is None or not callable(create):
                continue
            from jittor_utils import lock as _jit_lock
            with _jit_lock.unlock_scope():
                handle = create(list(self.ranks))
            self._backend_kind = kind
            self._backend_handle = int(handle)
            return
        if self.size() > 1:
            raise NotImplementedError(
                "Jittor process-group subgroups require NCCL or HCCL"
            )

    def _all_reduce(self, tensor, reduce_name):
        if self.rank() < 0:
            return tensor
        if self._backend_handle is None:
            if self.size() == 1:
                return tensor
            raise RuntimeError("process group has no backend communicator")
        if self._backend_kind == "nccl":
            if reduce_name not in ("sum", "mean"):
                raise NotImplementedError(
                    "NCCL process-group all_reduce supports sum and mean only"
                )
            result = jt.compile_extern.nccl_ops.nccl_all_reduce(
                tensor, self._backend_handle
            )
            return result / self.size() if reduce_name == "mean" else result
        if self._backend_kind == "hccl":
            op = "sum" if reduce_name == "mean" else reduce_name
            result = jt.compile_extern.hccl_ops.hccl_all_reduce(
                tensor, op, self._backend_handle
            )
            return result / self.size() if reduce_name == "mean" else result
        if self.ranks is None:
            return tensor.mpi_all_reduce(reduce_name)
        raise RuntimeError("process group has no collective backend")

    def rank(self):
        rank = get_rank()
        if self.ranks is None:
            return rank
        try:
            return self.ranks.index(rank)
        except ValueError:
            return -1

    def size(self):
        return get_world_size() if self.ranks is None else len(self.ranks)

    def name(self):
        return self._name

    def _get_backend_name(self):
        if self._backend_kind is not None:
            return self._backend_kind
        if os.environ.get("JT_NCCL_WORLD_SIZE") is not None:
            return "nccl"
        if os.environ.get("JT_HCCL_WORLD_SIZE") is not None:
            return "hccl"
        if (getattr(jt.compile_extern, "nccl_ops", None) is not None
                and bool(getattr(jt.flags, "use_cuda", 0))):
            return "nccl"
        if getattr(jt.compile_extern, "hccl_ops", None) is not None:
            return "hccl"
        return "mpi"

    def _get_backend(self, device=None):
        return self
