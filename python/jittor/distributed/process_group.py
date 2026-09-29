"""Native process-group state and collective backend ownership."""

import os
import re

import jittor as jt


_STATE = {
    "initialized": False,
    "destroyed": False,
    "backend": None,
    "rank": 0,
    "world_size": 1,
    "local_rank": 0,
    "local_world_size": 1,
    "device": None,
    "store": None,
    "default_group": None,
    "pg_map": {},
}


def _ensure_nccl_rootinfo_env():
    """Restore the env/file rendezvous path for later subgroup creation."""
    if os.environ.get("JT_NCCL_ROOTINFO_FILE", "").strip():
        return
    if os.environ.get("OMPI_COMM_WORLD_SIZE", "").strip():
        return
    rendezvous_dir = os.environ.get("JITTOR_DIST_RENDEZVOUS_DIR", "").strip()
    if not rendezvous_dir:
        local_world_size = int(os.environ.get(
            "LOCAL_WORLD_SIZE", os.environ.get("RAY_LOCAL_WORLD_SIZE", "1")))
        world_size = int(os.environ.get(
            "JT_NCCL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
        if local_world_size != world_size:
            raise RuntimeError(
                "multi-node NCCL process groups require "
                "JITTOR_DIST_RENDEZVOUS_DIR or JT_NCCL_ROOTINFO_FILE")
        rendezvous_dir = "/tmp"
    address = os.environ.get("MASTER_ADDR", "localhost")
    port = os.environ.get("MASTER_PORT", "default")
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", "{}-{}".format(address, port))
    os.environ["JT_NCCL_ROOTINFO_FILE"] = os.path.join(
        rendezvous_dir, "jittor-nccl-{}.bin".format(key))


def _runtime_world_size():
    return int(getattr(jt, "world_size", 1))


def _runtime_rank():
    return int(getattr(jt, "rank", 0))


def _mpi_local_identity(name, fallback):
    getter = getattr(getattr(jt, "compile_extern", None), "get_library", None)
    if callable(getter):
        mpi = getter("mpi")
        query = getattr(mpi, name, None)
        if callable(query):
            return int(query())
    return int(fallback)


def _runtime_distributed():
    if _runtime_world_size() <= 1:
        return False
    return any(os.environ.get(name) is not None for name in (
        "JT_NCCL_WORLD_SIZE", "JT_HCCL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS",
        "JT_MPI",
    )) or bool(getattr(jt, "in_mpi", False))


def _runtime_backend():
    if os.environ.get("JT_NCCL_WORLD_SIZE") is not None:
        return "nccl"
    if os.environ.get("JT_HCCL_WORLD_SIZE") is not None:
        return "hccl"
    compile_extern = getattr(jt, "compile_extern", None)
    if (getattr(compile_extern, "nccl_ops", None) is not None
            and bool(getattr(getattr(jt, "flags", None), "use_cuda", 0))):
        return "nccl"
    if getattr(compile_extern, "hccl_ops", None) is not None:
        return "hccl"
    if bool(getattr(jt, "in_mpi", False)):
        return "mpi"
    return None


def _default_backend_name():
    return _STATE["backend"] or _runtime_backend()


def is_initialized():
    """Return whether Jittor owns an active distributed process group."""
    if _STATE["destroyed"]:
        return False
    return bool(_STATE["initialized"] or _runtime_distributed())


def get_rank(group=None):
    """Return this process's rank, optionally translated to a subgroup rank."""
    if group is None or getattr(group, "ranks", None) is None:
        if _runtime_distributed():
            return _runtime_rank()
        return int(_STATE["rank"]) if _STATE["initialized"] else 0
    rank = _runtime_rank() if _runtime_distributed() else (
        int(_STATE["rank"]) if _STATE["initialized"] else 0)
    try:
        return tuple(group.ranks).index(rank)
    except ValueError:
        return -1


def get_world_size(group=None):
    """Return the process-group size, defaulting to WORLD."""
    if group is not None and getattr(group, "ranks", None) is not None:
        return len(group.ranks)
    if _runtime_distributed():
        return _runtime_world_size()
    return int(_STATE["world_size"]) if _STATE["initialized"] else 1


def get_local_rank():
    if _STATE["initialized"]:
        return int(_STATE["local_rank"])
    for name in ("LOCAL_RANK", "JT_NCCL_LOCAL_RANK", "JT_HCCL_LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    if _runtime_distributed() and bool(getattr(jt, "in_mpi", False)):
        return _mpi_local_identity("local_rank", _runtime_rank())
    return 0


def get_local_world_size():
    if _STATE["initialized"]:
        return int(_STATE["local_world_size"])
    for name in ("LOCAL_WORLD_SIZE", "JT_NCCL_LOCAL_WORLD_SIZE",
                 "JT_HCCL_LOCAL_WORLD_SIZE", "RAY_LOCAL_WORLD_SIZE"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    if _runtime_distributed() and bool(getattr(jt, "in_mpi", False)):
        return _mpi_local_identity("local_size", _runtime_world_size())
    return 1


def get_device():
    if _STATE["initialized"]:
        return _STATE["device"]
    current_device = getattr(jt, "current_device", None)
    return current_device() if callable(current_device) else None


def get_backend(group=None):
    """Return the active backend name, or fail before backend initialization."""
    if not is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    if group is not None:
        backend = getattr(group, "_backend_kind", None) or _default_backend_name()
    else:
        backend = _default_backend_name()
    if backend is None:
        raise RuntimeError("distributed process group has no active backend")
    return str(backend).lower()


class Work:
    """Synchronous completion handle; asynchronous launch is not implemented.

    ``async_op=True`` currently wraps an already completed value. Its ``wait``
    arguments are accepted for API compatibility, but timeout and cancellation
    are not supported.
    """

    def __init__(self, value=None):
        self._value = value

    def wait(self, *args, **kwargs):
        return self._value

    def is_completed(self):
        return True


class ProcessGroup:
    """A WORLD or subgroup view over Jittor's collective communicator."""

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
            if kind == "nccl":
                _ensure_nccl_rootinfo_env()
            with _jit_lock.unlock_scope():
                handle = create(list(self.ranks))
            self._backend_kind = kind
            self._backend_handle = int(handle)
            return
        if self.size() > 1:
            raise NotImplementedError(
                "Jittor process-group subgroups require NCCL or HCCL"
            )

    def _collective_backend(self):
        return self._backend_kind or _default_backend_name()

    def _collective_handle(self):
        if _STATE["destroyed"]:
            raise RuntimeError(
                "distributed process group was destroyed; its backend "
                "communicator remains owned by the worker until process exit")
        if self.rank() < 0:
            return None
        if self.size() <= 1:
            return self._backend_handle
        if self._backend_handle is None:
            raise RuntimeError("process group has no backend communicator")
        return self._backend_handle

    def all_reduce(self, tensor, op="mean"):
        """Return ``tensor`` reduced across this group, preserving autograd."""
        operation = str(op).lower()
        if operation == "avg":
            operation = "mean"
        if operation not in ("sum", "mean", "max", "min", "product", "prod"):
            raise ValueError("unsupported all_reduce operation {!r}".format(op))
        if operation == "prod":
            operation = "product"
        if self.rank() < 0 or self.size() <= 1:
            return tensor
        backend = self._collective_backend()
        handle = self._collective_handle()
        if backend == "nccl":
            if operation not in ("sum", "mean"):
                raise NotImplementedError(
                    "NCCL process-group all_reduce supports sum and mean only")
            ops = getattr(jt.compile_extern, "nccl_ops", None)
            if ops is None:
                raise RuntimeError("NCCL process-group backend is unavailable")
            result = ops.nccl_all_reduce(tensor, handle or 0)
            return result / self.size() if operation == "mean" else result
        if backend == "hccl":
            ops = getattr(jt.compile_extern, "hccl_ops", None)
            if ops is None:
                raise RuntimeError("HCCL process-group backend is unavailable")
            reduce_op = {"mean": "sum", "product": "prod"}.get(
                operation, operation)
            result = ops.hccl_all_reduce(tensor, reduce_op, handle or 0)
            return result / self.size() if operation == "mean" else result
        if backend == "mpi" and self.ranks is None:
            if operation not in ("sum", "mean"):
                raise NotImplementedError(
                    "MPI process-group all_reduce supports sum and mean only")
            reducer = getattr(tensor, "mpi_all_reduce", None)
            if not callable(reducer):
                raise RuntimeError("MPI all_reduce is unavailable for this tensor")
            return reducer(operation)
        raise RuntimeError(
            "process group has no {} all_reduce implementation".format(
                backend or "active backend"))

    def _all_reduce(self, tensor, reduce_name):
        """Backward-compatible private spelling used by the Torch installer."""
        return self.all_reduce(tensor, reduce_name)

    def broadcast(self, tensor, src=0):
        """Return the source rank's tensor, preserving the collective graph."""
        if self.rank() < 0 or self.size() <= 1:
            return tensor
        source = int(src)
        ranks = (tuple(range(get_world_size())) if self.ranks is None
                 else self.ranks)
        try:
            root = ranks.index(source)
        except ValueError as error:
            raise ValueError(
                "broadcast source rank is outside the process group") from error
        backend = self._collective_backend()
        handle = self._collective_handle()
        if backend == "nccl":
            ops = getattr(jt.compile_extern, "nccl_ops", None)
            if ops is None:
                raise RuntimeError("NCCL process-group backend is unavailable")
            return ops.nccl_broadcast(tensor, root, handle or 0)
        if backend == "hccl":
            ops = getattr(jt.compile_extern, "hccl_ops", None)
            if ops is None:
                raise RuntimeError("HCCL process-group backend is unavailable")
            return ops.hccl_broadcast(tensor, root, handle or 0)
        if backend == "mpi" and self.ranks is None:
            broadcaster = getattr(tensor, "mpi_broadcast", None)
            if not callable(broadcaster):
                raise RuntimeError("MPI broadcast is unavailable for this tensor")
            return broadcaster(source)
        raise RuntimeError(
            "process group has no {} broadcast implementation".format(
                backend or "active backend"))

    def all_gather(self, tensor):
        """Concatenate equal-shaped tensor shards along their leading axis."""
        if self.rank() < 0 or self.size() <= 1:
            return tensor
        if not getattr(tensor, "shape", ()):
            raise ValueError("all_gather requires a tensor with at least one dimension")
        backend = self._collective_backend()
        handle = self._collective_handle()
        if backend == "nccl":
            ops = getattr(jt.compile_extern, "nccl_ops", None)
            gather = getattr(ops, "nccl_all_gather", None)
            if callable(gather):
                return gather(tensor, handle or 0)
        elif backend == "hccl":
            ops = getattr(jt.compile_extern, "hccl_ops", None)
            gather = getattr(ops, "hccl_all_gather", None)
            if callable(gather):
                return gather(tensor, handle or 0)
        elif backend == "mpi" and self.ranks is None:
            gather = getattr(tensor, "mpi_all_gather", None)
            if callable(gather):
                return gather()
        raise NotImplementedError(
            "{} process-group all_gather is unavailable".format(
                backend or "active backend"))

    def reduce_scatter(self, tensor):
        """Sum and scatter equal leading-axis chunks across this group."""
        if self.rank() < 0 or self.size() <= 1:
            return tensor
        shape = tuple(int(dim) for dim in tensor.shape)
        if not shape or shape[0] % self.size():
            raise ValueError(
                "reduce_scatter expects dim0 divisible by process-group size")
        backend = self._collective_backend()
        handle = self._collective_handle()
        if backend == "nccl":
            ops = getattr(jt.compile_extern, "nccl_ops", None)
            scatter = getattr(ops, "nccl_reduce_scatter", None)
            if callable(scatter):
                return scatter(tensor, handle or 0)
        elif backend == "hccl":
            ops = getattr(jt.compile_extern, "hccl_ops", None)
            scatter = getattr(ops, "hccl_reduce_scatter", None)
            if callable(scatter):
                return scatter(tensor, handle or 0)
        if backend in ("mpi", "hccl"):
            reduced = self.all_reduce(tensor, "sum")
            chunk = shape[0] // self.size()
            start = self.rank() * chunk
            return reduced[start:start + chunk]
        raise NotImplementedError(
            "{} process-group reduce_scatter is unavailable".format(
                backend or "active backend"))

    def barrier(self):
        if self.rank() < 0 or self.size() <= 1:
            return None
        marker = jt.array([self.rank()], dtype="int32")
        self.all_reduce(marker, "sum").sync(device_sync=True)
        return None

    def new_group(self, ranks, name="subgroup"):
        return new_group(ranks=ranks, name=name)

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
        return self._backend_kind or _default_backend_name() or "mpi"

    def _get_backend(self, device=None):
        return self


_WORLD = ProcessGroup(name="world")
_STATE["default_group"] = _WORLD
_STATE["pg_map"][_WORLD] = ("undefined",)


def get_default_group():
    return _STATE["default_group"]


def get_default_store():
    return _STATE["store"]


def new_group(ranks=None, name="subgroup"):
    world_size = get_world_size()
    normalized = tuple(range(world_size)) if ranks is None else tuple(
        int(rank) for rank in ranks)
    if not normalized:
        raise ValueError("process group ranks cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("process group ranks must be unique")
    if any(rank < 0 or rank >= world_size for rank in normalized):
        raise ValueError("process group rank is outside WORLD")
    group = ProcessGroup(normalized, name)
    group._create_backend_communicator()
    _STATE["pg_map"][group] = (group._get_backend_name(),)
    return group


def get_global_rank(group, group_rank):
    local_rank = int(group_rank)
    ranks = getattr(group, "ranks", None)
    if ranks is None:
        if not 0 <= local_rank < get_world_size():
            raise ValueError("process-group rank is outside WORLD")
        return local_rank
    if not 0 <= local_rank < len(ranks):
        raise ValueError("group rank is outside process group")
    return int(ranks[local_rank])


def get_process_group_ranks(group=None):
    group = get_default_group() if group is None else group
    ranks = getattr(group, "ranks", None)
    return list(range(get_world_size())) if ranks is None else list(ranks)


def _backend_matches(requested, active):
    requested = str(requested).strip().lower()
    active = str(active).strip().lower()
    if requested == active:
        return True
    device = {"nccl": "cuda", "hccl": "npu", "mpi": "cpu"}.get(active)
    if not device:
        return False
    return any(item.strip() == "{}:{}".format(device, active)
               for item in requested.split(","))


def init_process_group(backend=None, init_method=None, timeout=None,
                       world_size=-1, rank=-1, store=None, **kwargs):
    """Register the active native communicator as the default WORLD group.

    Backend initialization itself remains owned by the launcher/bootstrap and
    ``compile_extern``. This function validates that communicator and publishes
    its state through one native owner.
    """
    if _STATE["initialized"] and not _STATE["destroyed"]:
        raise RuntimeError("distributed process group is already initialized")
    if store is not None and init_method is not None:
        raise ValueError("init_process_group accepts store or init_method, not both")
    if init_method is not None:
        from .store import rendezvous
        store, requested_rank, requested_world = next(rendezvous(
            init_method, rank=rank, world_size=world_size, timeout=timeout))
    else:
        requested_world = int(
            os.environ.get("WORLD_SIZE", _runtime_world_size())
            if world_size is None or int(world_size) < 0 else world_size)
        requested_rank = int(
            os.environ.get("RANK", _runtime_rank())
            if rank is None or int(rank) < 0 else rank)
    if requested_world < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= requested_rank < requested_world:
        raise ValueError("rank {} is outside world size {}".format(
            requested_rank, requested_world))
    if requested_world > 1 and not _runtime_distributed():
        raise RuntimeError(
            "multi-rank initialization requires an active Jittor MPI, NCCL, "
            "or HCCL communicator; refusing to fall back to a singleton")
    active_backend = _runtime_backend()
    backend_name = str(backend).lower() if backend is not None else active_backend
    if backend_name is None and requested_world == 1:
        backend_name = "mpi"
    if requested_world > 1:
        if active_backend is None:
            raise RuntimeError(
                "distributed was requested but no collective backend is active")
        if backend_name is None:
            raise RuntimeError("distributed process group has no selected backend")
        ops_name = {"nccl": "nccl_ops", "hccl": "hccl_ops", "mpi": "mpi_ops"}.get(
            active_backend)
        if ops_name is None or getattr(jt.compile_extern, ops_name, None) is None:
            raise RuntimeError(
                "distributed was requested with {} but its collective operators "
                "are unavailable; refusing to fall back to a singleton".format(
                    active_backend))
    if (requested_world > 1 and active_backend is not None and backend_name is not None
            and not _backend_matches(backend_name, active_backend)):
        raise RuntimeError(
            "requested backend {} does not match active Jittor backend {}".format(
                backend_name, active_backend))
    runtime_rank = _runtime_rank() if _runtime_distributed() else requested_rank
    runtime_world = _runtime_world_size() if _runtime_distributed() else requested_world
    if (runtime_world != requested_world or runtime_rank != requested_rank):
        raise RuntimeError(
            "requested rank/world {}/{} does not match active Jittor rank/world "
            "{}/{}".format(requested_rank, requested_world,
                           runtime_rank, runtime_world))
    local_rank = int(os.environ.get(
        "LOCAL_RANK", os.environ.get(
            "JT_NCCL_LOCAL_RANK", os.environ.get(
                "JT_HCCL_LOCAL_RANK", get_local_rank()))))
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE", os.environ.get(
            "RAY_LOCAL_WORLD_SIZE", os.environ.get(
                "JT_NCCL_LOCAL_WORLD_SIZE", os.environ.get(
                    "JT_HCCL_LOCAL_WORLD_SIZE", requested_world)))))
    if local_world_size < 1 or not 0 <= local_rank < local_world_size:
        raise ValueError("invalid local rank/world-size configuration")

    _STATE.update({
        "initialized": True,
        "destroyed": False,
        "backend": backend_name,
        "rank": requested_rank,
        "world_size": requested_world,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "store": store,
        "device": (jt.current_device() if callable(
            getattr(jt, "current_device", None)) else None),
    })
    _STATE["pg_map"][_WORLD] = ((backend_name or "undefined"),)
    return None


def destroy_process_group(group=None):
    """Release Python registration and close its Store.

    NCCL/HCCL communicator destruction remains process-owned: the current
    backend wrappers expose teardown only through their process finalizer.
    They are released when the worker exits, not by this function.
    """
    if group is not None and group is not _WORLD:
        _STATE["pg_map"].pop(group, None)
        return None
    store = _STATE.get("store")
    close = getattr(store, "close", None)
    if callable(close):
        close()
    _STATE.update({
        "initialized": False,
        "destroyed": True,
        "backend": None,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "local_world_size": 1,
        "store": None,
    })
    _STATE["pg_map"].clear()
    _STATE["pg_map"][_WORLD] = ("undefined",)
    return None


def all_reduce(tensor, op="mean", group=None, async_op=False):
    group = get_default_group() if group is None else group
    value = group.all_reduce(tensor, op)
    return Work(value) if async_op else value


def broadcast(tensor, src=0, group=None, async_op=False):
    group = get_default_group() if group is None else group
    value = group.broadcast(tensor, src)
    return Work(value) if async_op else value


def all_gather(tensor, group=None, async_op=False):
    group = get_default_group() if group is None else group
    value = group.all_gather(tensor)
    return Work(value) if async_op else value


def reduce_scatter(tensor, group=None, async_op=False):
    group = get_default_group() if group is None else group
    value = group.reduce_scatter(tensor)
    return Work(value) if async_op else value


def barrier(group=None, async_op=False):
    group = get_default_group() if group is None else group
    value = group.barrier()
    return Work(value) if async_op else None


def _get_state():
    """Private shared state hook used by compatibility installer integration."""
    return _STATE
