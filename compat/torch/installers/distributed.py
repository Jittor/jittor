"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import os
import pickle

import numpy as np

import jittor as jt

from ..context import registry_for
from ..fidelity import Fidelity, register_api_bindings
from ...diagnostics import EXPECTED, swallowed
from ...transaction import set_env, set_flag
from ... import collectives as _collectives
from ... import fsdp_hooks as _fsdp_hooks
from jittor.distributed.store import (
    FileStore,
    PrefixStore,
    Store,
    TCPStore,
    rendezvous as _store_rendezvous,
)


from jittor.distributed import process_group as _process_group
from jittor.distributed.process_group import (
    ProcessGroup as _JittorProcessGroup,
    Work as _JittorWork,
)


def _native_distributed_active():
    return _process_group.is_initialized()


def _is_truthy(value):
    return str(value or "").strip().lower() not in ("", "0", "false", "no", "off")


def _backend_matches_active(requested_backend, active_backend):
    requested = str(requested_backend).strip().lower()
    active = str(active_backend).strip().lower()
    if requested == active:
        return True

    device_backends = {}
    for item in requested.split(","):
        device, separator, backend = item.strip().partition(":")
        if not separator or not device.strip() or not backend.strip():
            return False
        device_backends[device.strip()] = backend.strip()

    active_device = {
        "nccl": "cuda",
        "hccl": "npu",
        "gloo": "cpu",
        "mpi": "cpu",
    }.get(active)
    return active_device is not None and device_backends.get(active_device) == active


def _bootstrap_var_all_reduce(self, op="mean"):
    ops = jt.compile_extern.nccl_ops
    world_size = jt.compile_extern.world_size
    if op not in ("sum", "mean"):
        raise NotImplementedError(
            "Jittor NCCL Var.mpi_all_reduce supports sum and mean only")
    result = ops.nccl_all_reduce(self)
    return result / world_size if op == "mean" else result


def _bootstrap_var_broadcast(self, root=0):
    ops = jt.compile_extern.nccl_ops
    return ops.nccl_broadcast(self, int(root))


def _bootstrap_native_distributed(rank, world_size, backend=None, store=None):
    if not _is_truthy(os.environ.get("JITTOR_TORCH_DISTRIBUTED_AUTO_INIT")):
        return False
    backend_name = str(backend or "nccl").lower()
    if "nccl" not in backend_name:
        raise NotImplementedError(
            "Jittor dynamic torch.distributed bootstrap currently supports NCCL only"
        )
    if not getattr(jt, "has_cuda", False):
        raise RuntimeError("Jittor dynamic NCCL bootstrap requires CUDA")

    rank = int(rank)
    world_size = int(world_size)
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE", os.environ.get("RAY_LOCAL_WORLD_SIZE", world_size)))
    rootinfo = os.environ.get("JT_NCCL_ROOTINFO_FILE", "").strip()
    if not rootinfo and store is None:
        explicit_rendezvous_dir = os.environ.get(
            "JITTOR_DIST_RENDEZVOUS_DIR", "").strip()
        if local_world_size != world_size and not explicit_rendezvous_dir:
            raise RuntimeError(
                "multi-node Jittor NCCL requires JT_NCCL_ROOTINFO_FILE or "
                "JITTOR_DIST_RENDEZVOUS_DIR on shared storage"
            )
        rendezvous_dir = explicit_rendezvous_dir or "/tmp"
        os.makedirs(rendezvous_dir, exist_ok=True)
        address = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "default")
        key = "{}-{}".format(address, port)
        key = "".join(char if char.isalnum() or char in "_.-" else "_"
                      for char in key)
        rootinfo = os.path.join(
            rendezvous_dir, "jittor-nccl-{}.bin".format(key))

    visible = [item for item in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    local_rank = 0 if len(visible) == 1 else int(os.environ.get("LOCAL_RANK", rank))
    set_env("JT_NCCL_WORLD_SIZE", world_size)
    set_env("JT_NCCL_RANK", rank)
    set_env("JT_NCCL_LOCAL_RANK", local_rank)
    if rootinfo:
        set_env("JT_NCCL_ROOTINFO_FILE", rootinfo)
    set_env("use_nccl", "1")
    set_env("use_mpi", "0")

    set_flag(jt.flags, "use_cuda", 1)
    jt.compile_extern.setup_nccl(store=store)
    ops = getattr(jt.compile_extern, "nccl_ops", None)
    if ops is None:
        raise RuntimeError("Jittor NCCL setup did not publish collective ops")

    # compile_extern is the single owner; jt.rank / jt.world_size / jt.in_mpi
    # read through to it. Assigning them here as well used to create a second
    # copy that then had to be kept in sync by hand. 6.B15.
    jt.compile_extern.rank = rank
    jt.compile_extern.world_size = world_size
    jt.compile_extern.in_mpi = True



    jt.core.Var.mpi_all_reduce = _bootstrap_var_all_reduce
    jt.core.Var.mpi_broadcast = _bootstrap_var_broadcast
    return True


def _distributed_rank():
    return _process_group.get_rank()


def _distributed_world_size():
    return _process_group.get_world_size()


def _group_size(group):
    if group is None:
        return _distributed_world_size()
    size = getattr(group, "size", None)
    return int(size() if callable(size) else size or 1)


def _require_supported_group(group, allow_subgroup=False):
    size = _group_size(group)
    if not allow_subgroup and size not in (1, _distributed_world_size()):
        raise NotImplementedError(
            "Jittor torch.distributed currently supports only WORLD and singleton groups"
        )
    return size


def _copy_tensor(dst, src):
    if hasattr(dst, "update"):
        dst.update(src)
    elif hasattr(dst, "assign"):
        dst.assign(src)
    else:
        dst[:] = src


def _collective_result(value, async_op):
    return _JittorWork(value) if async_op else None


def _reduce_name(op, reduce_op):
    if op is None or op == reduce_op.SUM:
        return "sum"
    if op in (reduce_op.MEAN, reduce_op.AVG):
        return "mean"
    if op == reduce_op.MAX:
        return "max"
    if op == reduce_op.MIN:
        return "min"
    if op == reduce_op.PRODUCT:
        return "product"
    raise NotImplementedError(
        "unsupported Jittor torch.distributed all_reduce operation"
    )


def _native_all_gather_flat(tensor):
    return _collectives._all_gather_shards(tensor.reshape((-1,)))


def _native_all_gather_object(object_list, obj, group=None):
    size = _require_supported_group(group)
    if len(object_list) < size:
        raise ValueError(
            "all_gather_object output list is shorter than group size")
    if size == 1:
        object_list[0] = obj
        return None

    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    local_length = jt.array(np.asarray([len(payload)], dtype=np.int32))
    lengths = np.asarray(
        _native_all_gather_flat(local_length).numpy(), dtype=np.int32
    ).reshape(-1)
    max_length = int(lengths.max()) if lengths.size else 0
    encoded = np.zeros((max_length,), dtype=np.int32)
    if payload:
        encoded[:len(payload)] = np.frombuffer(payload, dtype=np.uint8).astype(np.int32)
    gathered = np.asarray(
        _native_all_gather_flat(jt.array(encoded)).numpy(), dtype=np.int32
    ).reshape(size, max_length)
    for rank, length in enumerate(lengths):
        raw = gathered[rank, :int(length)].astype(np.uint8).tobytes()
        object_list[rank] = pickle.loads(raw)
    return None


def _install_fsdp2_distributed(dist, torch_module=None, registry=None):
    """Install the FSDP2/DTensor compatibility surface.

    THE one place below fsdp2 that may name it. Everything else that needs
    FSDP-aware behaviour -- Tensor.backward, optimizer step/zero_grad,
    Module.__call__, the distributed state dict -- goes through
    ``jittor.compat.fsdp_hooks`` instead, so that the dependency runs one way
    (``core -> tensor -> nn/optim -> distributed -> fsdp``).

    This edge stays because it is composition, not use: installing
    ``torch.distributed`` is precisely when the FSDP2 surface has to be hung
    off it, and the objects it needs (``dist``, the torch module, the install
    registry) exist only here. It runs once, at install time, off any hot path.
    ``tests/structure/test_compat_layering.py`` allows exactly this one and
    fails on a second.
    """
    from jittor.compat.fsdp2 import installer as _fsdp2_installer
    return _fsdp2_installer.install_with_registry(
        dist, torch_module, registry=registry
    )

from ..context import get_install_context
from types import MappingProxyType
import types as _types
import datetime as _datetime_c10d

def _init_process_group(*args, **kwargs):
    _context = get_install_context(jt).state["distributed_api"]
    state = _context['state']
    world_group = _context['world_group']
    rank_arg = kwargs.get("rank", -1)
    world_arg = kwargs.get("world_size", -1)
    init_method = kwargs.get(
        "init_method", args[1] if len(args) > 1 else None
    )
    store = kwargs.get("store")
    if store is not None and init_method is not None:
        raise ValueError("init_process_group accepts store or init_method, not both")
    if init_method is not None:
        store, requested_rank, requested_world_size = next(
            _store_rendezvous(
                init_method, rank=rank_arg, world_size=world_arg,
                timeout=kwargs.get("timeout"),
            )
        )
    else:
        requested_world_size = int(
            os.environ.get("WORLD_SIZE", 1)
            if int(world_arg) < 0 else world_arg
        )
        requested_rank = int(
            os.environ.get("RANK", 0) if int(rank_arg) < 0 else rank_arg
        )
    backend = kwargs.get("backend", args[0] if args else None)
    backend_name = str(backend).lower() if backend is not None else None
    if requested_world_size > 1 and not _native_distributed_active():
        _bootstrap_native_distributed(
            requested_rank, requested_world_size, backend=backend,
            store=store,
        )
    if requested_world_size > 1 and not _native_distributed_active():
        raise RuntimeError(
            "multi-rank torch.distributed requires launching Jittor with "
            "jittor.distributed.launch or explicit dynamic bootstrap"
        )
    if _native_distributed_active():
        active_backend = world_group._get_backend_name()
        if backend_name is not None and not _backend_matches_active(
            backend_name, active_backend
        ):
            raise RuntimeError(
                "requested torch.distributed backend {} does not match "
                "active Jittor backend {}".format(
                    backend_name, active_backend))
        if requested_world_size not in (1, _distributed_world_size()):
            raise RuntimeError("torch/Jittor distributed world-size mismatch")
        if "rank" in kwargs and requested_rank != _distributed_rank():
            raise RuntimeError("torch/Jittor distributed rank mismatch")
    state["initialized"] = True
    state["store"] = store
    return None


def _destroy_process_group(*args, **kwargs):
    _context = get_install_context(jt).state["distributed_api"]
    state = _context['state']
    store = state.get("store")
    close = getattr(store, "close", None)
    if callable(close):
        close()
    state["store"] = None
    state["initialized"] = False
    return None


def _get_rank(group=None):
    if group is None:
        return _distributed_rank()
    rank = getattr(group, "rank", None)
    return int(rank() if callable(rank) else rank or 0)


def _get_world_size(group=None):
    return _group_size(group)


class _ReduceOp:
    SUM = 0
    MEAN = 1
    AVG = 1
    MAX = 2
    MIN = 3
    PRODUCT = 4


_ReduceOp.RedOpType = _ReduceOp


def _all_reduce(tensor, op=None, group=None, async_op=False):
    size = _require_supported_group(group, allow_subgroup=True)
    if group is not None and getattr(group, "rank", lambda: 0)() < 0:
        return _collective_result(tensor, async_op)
    reduce_name = _reduce_name(op, _ReduceOp)
    if group is not None and hasattr(group, "_all_reduce"):
        result = group._all_reduce(tensor, reduce_name)
        _copy_tensor(tensor, result)
        return _collective_result(tensor, async_op)
    if size > 1:
        if reduce_name in ("sum", "mean"):
            result = tensor.mpi_all_reduce(reduce_name)
        else:
            gathered = _native_all_gather_flat(tensor).reshape(
                (size,) + tuple(tensor.shape))
            result = gathered[0]
            for rank in range(1, size):
                if reduce_name == "max":
                    result = jt.maximum(result, gathered[rank])
                elif reduce_name == "min":
                    result = jt.minimum(result, gathered[rank])
                else:
                    result = result * gathered[rank]
        _copy_tensor(tensor, result)
    return _collective_result(tensor, async_op)


def _all_gather(tensor_list, tensor, group=None, async_op=False):
    size = _require_supported_group(group)
    if len(tensor_list) < size:
        raise ValueError("all_gather output list is shorter than group size")
    if size == 1:
        _copy_tensor(tensor_list[0], tensor)
    else:
        gathered = _native_all_gather_flat(tensor)
        numel = int(np.prod(tuple(int(dim) for dim in tensor.shape)))
        for rank in range(size):
            part = gathered[rank * numel:(rank + 1) * numel].reshape(
                tensor.shape)
            _copy_tensor(tensor_list[rank], part)
    return _collective_result(tensor_list, async_op)


def _all_gather_into_tensor(output_tensor, input_tensor, group=None,
                            async_op=False):
    size = _require_supported_group(group)
    gathered = (
        input_tensor.reshape((-1,)) if size == 1
        else _native_all_gather_flat(input_tensor)
    )
    _copy_tensor(output_tensor, gathered.reshape(output_tensor.shape))
    return _collective_result(output_tensor, async_op)


def _broadcast(tensor, src=0, group=None, async_op=False, group_src=None):
    size = _require_supported_group(group)
    root = int(src if group_src is None else group_src)
    if size > 1:
        _copy_tensor(tensor, tensor.mpi_broadcast(root))
    return _collective_result(tensor, async_op)


def _barrier(group=None, async_op=False, device_ids=None):
    size = _require_supported_group(group)
    marker = None
    if size > 1:
        marker = jt.array(np.asarray([_distributed_rank()], dtype=np.int32))
        marker = marker.mpi_all_reduce("sum")
        marker.sync()
    return _collective_result(marker, async_op)


def _broadcast_object_list(object_list, src=0, group=None, device=None):
    gathered = [None] * _group_size(group)
    local = object_list if _get_rank(group) == int(src) else None
    _native_all_gather_object(gathered, local, group)
    object_list[:] = gathered[int(src)]
    return None


def _gather_object(obj, object_gather_list=None, dst=0, group=None,
                   group_dst=None):
    size = _require_supported_group(group)
    if group_dst is not None:
        destination = int(group_dst)
    elif group is not None and getattr(group, "ranks", None) is not None:
        try:
            destination = group.ranks.index(int(dst))
        except ValueError as error:
            raise ValueError(
                "gather_object destination rank is outside the group"
            ) from error
    else:
        destination = int(dst)
    if not 0 <= destination < size:
        raise ValueError("gather_object destination rank is outside the group")
    is_destination = _get_rank(group) == destination
    if is_destination:
        if object_gather_list is None:
            raise ValueError(
                "gather_object requires an output list on the destination rank")
        if len(object_gather_list) < size:
            raise ValueError(
                "gather_object output list is shorter than group size")
    elif object_gather_list is not None:
        raise ValueError(
            "gather_object output list must be None on non-destination ranks")

    gathered = [None] * size
    _native_all_gather_object(gathered, obj, group)
    if is_destination:
        object_gather_list[:size] = gathered
    return None


def _new_group(ranks=None, *args, **kwargs):
    _context = get_install_context(jt).state["distributed_api"]
    pg_map = _context['pg_map']
    ranks = (
        tuple(range(_distributed_world_size()))
        if ranks is None else tuple(int(rank) for rank in ranks)
    )
    if not ranks:
        raise ValueError("process group ranks cannot be empty")
    if len(set(ranks)) != len(ranks):
        raise ValueError("process group ranks must be unique")
    if any(int(rank) < 0 or int(rank) >= _distributed_world_size()
           for rank in ranks):
        raise ValueError("process group rank is outside WORLD")
    group = _JittorProcessGroup(ranks, "subgroup")
    group._create_backend_communicator()
    pg_map[group] = (group._get_backend_name(),)
    return group


class Backend(str):
    GLOO = "gloo"
    NCCL = "nccl"
    MPI = "mpi"
    UCC = "ucc"
    UNDEFINED = "undefined"


class P2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = int(peer)
        self.group = group
        self.tag = int(tag)


def _unsupported_p2p(*args, **kwargs):
    raise NotImplementedError(
        "Jittor torch.distributed point-to-point communication is unavailable"
    )


def _batch_isend_irecv(p2p_ops):
    if p2p_ops:
        _unsupported_p2p()
    return []


def _new_subgroups_by_enumeration(ranks_per_subgroup_list=None, *a, **k):
    _context = get_install_context(jt).state["distributed_api"]
    dist = _context['dist']
    groups = list(ranks_per_subgroup_list or [])
    world = _distributed_world_size()
    if world <= 1 or (len(groups) <= 1
                      and (not groups or len(groups[0]) == world)):
        return ([dist.group.WORLD], dist.group.WORLD)
    try:
        process_groups = [_new_group(ranks) for ranks in groups]
    except NotImplementedError:
        from ...stub_policy import unimplemented
        return unimplemented(
            "torch.distributed.new_subgroups_by_enumeration",
            "hand back the WORLD group for every requested subgroup, so "
            "a subgroup collective silently reduces across all %d ranks"
            % world,
            "This runtime has no NCCL/HCCL process-group backend.",
            stub_result=([dist.group.WORLD], dist.group.WORLD))
    rank = _distributed_rank()
    current = next(
        (group for group in process_groups if rank in group.ranks), -100
    )
    return process_groups, current


class JoinHook:
    def main_hook(self):
        return None
    def post_hook(self, is_last_joiner):
        return None


class Joinable:
    def __init__(self, *a, **k):
        pass
    @property
    def join_hook(self):
        return JoinHook()
    @property
    def join_device(self):
        return None
    @property
    def join_process_group(self):
        _context = get_install_context(jt).state["distributed_api"]
        dist = _context['dist']
        return dist.group.WORLD


class Join:
    def __init__(self, joinables, enable=True, throw_on_early_termination=False, **kwargs):
        self.joinables = list(joinables) if joinables is not None else []
        self.enable = enable
        self.throw_on_early_termination = throw_on_early_termination
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    @staticmethod
    def notify_join_context(joinable):
        return None
    @staticmethod
    def notify_join_context_enabled(joinable):
        return False


class _DeviceMesh:
    def __init__(self, device_type=None, mesh=None, *, mesh_dim_names=None, **k):
        self.device_type = device_type
        self.mesh = mesh
        self.mesh_dim_names = mesh_dim_names
    def __getitem__(self, *a, **k): return self
    def size(self, *a, **k): return 1
    def get_rank(self, *a, **k): return 0
    def get_group(self, *a, **k): return None
    def get_local_rank(self, *a, **k): return 0


def _init_device_mesh(device_type=None, mesh_shape=None, *, mesh_dim_names=None, **k):
    return _DeviceMesh(device_type=device_type, mesh=mesh_shape,
                       mesh_dim_names=mesh_dim_names)


def _autograd_all_reduce(input, op=None, group=None, *a, **k):
    """Differentiable all-reduce -- real on >1 rank, identity on 1 rank.

    Was `lambda input, *a, **k: input`: on N ranks both the value AND the
    gradient were wrong, with no error. On a single rank an all-reduce IS
    the identity, so that case stays exact.
    """
    if group is not None and getattr(group, "rank", lambda: 0)() < 0:
        return input
    reduce_name = _reduce_name(op, _ReduceOp)
    if group is not None and hasattr(group, "_all_reduce"):
        return group._all_reduce(input, reduce_name)
    if _group_size(group) <= 1:
        return input
    if reduce_name in ("sum", "mean"):
        # nccl_all_reduce is a real jittor op, so this stays differentiable.
        return input.mpi_all_reduce(reduce_name)
    from ...stub_policy import unimplemented
    return unimplemented(
        "torch.distributed.nn.all_reduce(op=%s)" % reduce_name,
        "return the local tensor unchanged, so both the value and its "
        "gradient are wrong on every rank",
        "Only sum and mean are differentiable here.",
        stub_result=input)


def _symmetric_memory_unavailable(*args, **kwargs):
    raise RuntimeError("Jittor symmetric memory is unavailable")


class Future:
    def __init__(self, devices=None):
        self._value = None
    def set_result(self, value):
        self._value = value
        return self
    def value(self):
        return self._value
    def wait(self):
        return self._value
    def then(self, callback):
        return callback(self)


class FileSystemReader:
    def __init__(self, path, *a, **k):
        self.path = path


class FileSystemWriter:
    def __init__(self, path, *a, **k):
        self.path = path


class StateDictOptions:
    def __init__(self, *, full_state_dict=False, cpu_offload=False,
                 ignore_frozen_params=False, keep_submodule_prefixes=True,
                 strict=True, broadcast_from_rank0=False, flatten_optimizer_state_dict=False):
        self.full_state_dict = bool(full_state_dict)
        self.cpu_offload = bool(cpu_offload)
        self.ignore_frozen_params = bool(ignore_frozen_params)
        self.keep_submodule_prefixes = bool(keep_submodule_prefixes)
        self.strict = bool(strict)
        self.broadcast_from_rank0 = bool(broadcast_from_rank0)
        self.flatten_optimizer_state_dict = bool(flatten_optimizer_state_dict)


def _get_model_state_dict(model, *a, options=None, **k):
    return model.state_dict(*a, **k) if hasattr(model, "state_dict") else {}


def _set_model_state_dict(model, state_dict, *a, options=None, **k):
    # `_is_fsdp_module` is set only by fsdp2, so a model carrying it proves
    # fsdp2 was imported and has registered -- see jittor/compat/
    # fsdp_hooks.py for why this file must not import fsdp2 directly.
    if getattr(model, "_is_fsdp_module", False):
        _fsdp = _fsdp_hooks.provider()
        if _fsdp is not None:
            _fsdp._load_full_state_dict(model, state_dict)
            return None
    if hasattr(model, "load_state_dict"):
        return model.load_state_dict(state_dict, strict=getattr(options, "strict", True))
    return None


class ShardedTensor:
    pass


class _RendezvousModule(_types.ModuleType):
    def __call__(self, *args, **kwargs):
        return self.rendezvous(*args, **kwargs)


def _rendezvous(url, rank=-1, world_size=-1, **kwargs):
    yield from _store_rendezvous(
        url, rank=rank, world_size=world_size,
        timeout=kwargs.get("timeout"),
    )


class _Accel:
    type = "cuda" if getattr(jt.compiler, "has_cuda", 0) else "cpu"


class _CNS:
    @staticmethod
    def _get_accelerator():
        return _Accel()


def _api_dist_is_available(*a, **k):
    return True


def _api_dist_is_backend_available(backend):
    return bool(getattr(jt, 'has_cuda', False)) if str(backend).lower() == 'nccl' else bool(getattr(jt.compile_extern, 'has_mpi', False)) if str(backend).lower() == 'mpi' else False


def _api_dist_is_nccl_available():
    _context = get_install_context(jt).state["distributed_api"]
    dist = _context['dist']
    return dist.is_backend_available('nccl')


def _api_dist_is_gloo_available():
    _context = get_install_context(jt).state["distributed_api"]
    dist = _context['dist']
    return dist.is_backend_available('gloo')


def _api_dist_is_mpi_available():
    _context = get_install_context(jt).state["distributed_api"]
    dist = _context['dist']
    return dist.is_backend_available('mpi')


def _api_dist_is_ucc_available():
    return False


def _api_dist_is_initialized(*a, **k):
    _context = get_install_context(jt).state["distributed_api"]
    state = _context['state']
    return bool(state['initialized'] or _native_distributed_active())


def _api_dist_get_backend(group=None):
    _context = get_install_context(jt).state["distributed_api"]
    world_group = _context['world_group']
    return group._get_backend_name() if group is not None and hasattr(group, '_get_backend_name') else world_group._get_backend_name()


def _api_dist_get_global_rank(group=None, group_rank=0):
    return int(group_rank) if group is None or getattr(group, 'ranks', None) is None else int(group.ranks[int(group_rank)])


def _api_dist_get_process_group_ranks(group=None):
    return list(range(_distributed_world_size())) if group is None or getattr(group, 'ranks', None) is None else list(group.ranks)


def _api_dist_is_torchelastic_launched(*a, **k):
    return False


def _api_c10d_is_xccl_available(*a, **k):
    return False


def _api_c10d_get_default_group(*a, **k):
    _context = get_install_context(jt).state["distributed_api"]
    world_group = _context['world_group']
    return world_group


def _api_c10d_get_default_store(*a, **k):
    _context = get_install_context(jt).state["distributed_api"]
    state = _context['state']
    return state['store']


def _api_c10d_get_default_timeout(*a, **k):
    return _datetime_c10d.timedelta(minutes=10)


def _api_c10d_unregister_process_group(*a, **k):
    return None


def _api_c10d_register_process_group(*a, **k):
    return None


def _api_c10d_resolve_process_group(name='world', *a, **k):
    _context = get_install_context(jt).state["distributed_api"]
    world_group = _context['world_group']
    return world_group


def _api_rpc_is_available(*a, **k):
    return False


def _api_rpc_init_rpc(*a, **k):
    return None


def _api_rpc_shutdown(*a, **k):
    return None


def _api_symmetric_memory_is_symm_mem_enabled_for_group(*a, **k):
    return False


def _api_checkpoint_sd_get_state_dict(model, optimizers=None, *a, **k):
    return (_get_model_state_dict(model, *a, **k), optimizers.state_dict() if hasattr(optimizers, 'state_dict') else {})


def _api_checkpoint_sd_set_state_dict(model, optimizers=None, model_state_dict=None, optim_state_dict=None, *a, **k):
    return _set_model_state_dict(model, model_state_dict or {}, *a, **k)


def _api_sharded_tensor_init_from_local_shards(shards, *a, **k):
    return shards[0] if shards else None


def _api_sharded_tensor_empty(*a, **k):
    return jt.empty(*a, **{kk: vv for kk, vv in k.items() if kk == 'dtype'})


from ...stub_policy import unimplemented as _stub_unimplemented, record_unimplemented as _record_unimplemented

class SerializationFormat:
    TORCH_SAVE = "torch_save"
    SAFETENSORS = "safetensors"

_dcp_save_effect = ("write no bytes at all while reporting a successful "
                    "save, silently discarding the checkpoint")


_dcp_load_effect = ("return the state dict unchanged without reading the "
                    "checkpoint, silently leaving the model at its current "
                    "weights")


_dcp_hint = ("Use torch.save / Module.state_dict for a single-rank "
             "checkpoint; sharded dcp is task 8.18.")


def _api_checkpoint_load_state_dict(*args, **kwargs):
    return _stub_unimplemented('torch.distributed.checkpoint.load_state_dict', _dcp_load_effect, _dcp_hint, stub_result=None)


def _api_checkpoint_save_state_dict(*args, **kwargs):
    return _stub_unimplemented('torch.distributed.checkpoint.save_state_dict', _dcp_save_effect, _dcp_hint, stub_result=None)


def _api_checkpoint_load(*args, **kwargs):
    return _stub_unimplemented('torch.distributed.checkpoint.load', _dcp_load_effect, _dcp_hint, stub_result=None)


def _api_checkpoint_save(*args, **kwargs):
    return _stub_unimplemented('torch.distributed.checkpoint.save', _dcp_save_effect, _dcp_hint, stub_result=None)


def _api_checkpoint_fs_write_item(*args, **kwargs):
    return _stub_unimplemented('torch.distributed.checkpoint.filesystem._write_item', _dcp_save_effect, _dcp_hint, stub_result=None)


_api_checkpoint_load_state_dict._jittor_unimplemented = 'torch.distributed.checkpoint.load_state_dict'
_api_checkpoint_save_state_dict._jittor_unimplemented = 'torch.distributed.checkpoint.save_state_dict'
_api_checkpoint_load._jittor_unimplemented = 'torch.distributed.checkpoint.load'
_api_checkpoint_save._jittor_unimplemented = 'torch.distributed.checkpoint.save'
_api_checkpoint_fs_write_item._jittor_unimplemented = 'torch.distributed.checkpoint.filesystem._write_item'

def _install_distributed(g, registry=None):
    """Install Torch distributed compatibility over Jittor collectives.

    Transformers 5 imports torch.distributed at module import time for tensor
    parallel helpers even when no distributed execution is requested. The jittor
    torch shim keeps identity semantics by default. When Jittor was started by
    its distributed launcher, WORLD maps to the active NCCL/MPI communicator.
    """
    _modules = registry_for(g, registry).module_map
    import types as _types

    dist = _modules.get("torch.distributed")
    if dist is None:
        dist = _types.ModuleType("torch.distributed")
        _modules["torch.distributed"] = dist
    state = {"initialized": _native_distributed_active(), "store": None}
    world_group = _JittorProcessGroup(name="world")
    pg_map = {world_group: (world_group._get_backend_name(),)}
    get_install_context(g).state["distributed_api"] = MappingProxyType({
        "state": state, "world_group": world_group, "pg_map": pg_map, "dist": dist,
    })

    dist.is_available = _api_dist_is_available
    dist.is_backend_available = _api_dist_is_backend_available
    dist.is_nccl_available = _api_dist_is_nccl_available
    dist.is_gloo_available = _api_dist_is_gloo_available
    dist.is_mpi_available = _api_dist_is_mpi_available
    dist.is_ucc_available = _api_dist_is_ucc_available
    dist.is_initialized = _api_dist_is_initialized
    dist.get_rank = _get_rank
    dist.get_world_size = _get_world_size
    dist.init_process_group = _init_process_group
    dist.destroy_process_group = _destroy_process_group
    dist.Backend = Backend
    dist.P2POp = P2POp
    dist.isend = _unsupported_p2p
    dist.irecv = _unsupported_p2p
    dist.send = _unsupported_p2p
    dist.recv = _unsupported_p2p
    dist.batch_isend_irecv = _batch_isend_irecv
    dist.get_backend = _api_dist_get_backend
    dist.barrier = _barrier
    dist.all_reduce = _all_reduce
    dist.all_gather = _all_gather
    dist.all_gather_into_tensor = _all_gather_into_tensor
    dist.broadcast = _broadcast
    dist.all_gather_object = _native_all_gather_object
    dist.broadcast_object_list = _broadcast_object_list
    dist.gather_object = _gather_object
    dist.new_group = _new_group

    dist.new_subgroups_by_enumeration = _new_subgroups_by_enumeration
    dist.get_global_rank = _api_dist_get_global_rank
    dist.get_process_group_ranks = _api_dist_get_process_group_ranks
    dist.is_torchelastic_launched = _api_dist_is_torchelastic_launched
    dist.ReduceOp = getattr(dist, "ReduceOp", _ReduceOp)
    if not hasattr(dist.ReduceOp, "RedOpType"):
        dist.ReduceOp.RedOpType = dist.ReduceOp
    dist.GroupMember = type(
        "GroupMember", (), {"WORLD": world_group, "NON_GROUP_MEMBER": -100})
    dist.group = type("group", (), {"WORLD": world_group})

    for sub in ("tensor", "fsdp", "device_mesh", "algorithms", "_composable",
                "checkpoint", "_shard", "nn"):
        name = "torch.distributed." + sub
        mod = _modules.get(name)
        if mod is None:
            mod = _types.ModuleType(name)
            _modules[name] = mod
        setattr(dist, sub, mod)

    dist.algorithms.__path__ = getattr(dist.algorithms, "__path__", [])
    const_mod = _modules.get("torch.distributed.constants")
    if const_mod is None:
        const_mod = _types.ModuleType("torch.distributed.constants")
        _modules["torch.distributed.constants"] = const_mod
    try:
        import datetime as _datetime_dist
        const_mod.default_pg_timeout = getattr(
            const_mod, "default_pg_timeout", _datetime_dist.timedelta(minutes=30)
        )
    except EXPECTED as exc:
        swallowed("torch/installers/distributed.py _install_distributed: import datetime as _datetime_dist", exc)
        const_mod.default_pg_timeout = getattr(const_mod, "default_pg_timeout", None)
    dist.constants = const_mod
    join_mod = _modules.get("torch.distributed.algorithms.join")
    if join_mod is None:
        join_mod = _types.ModuleType("torch.distributed.algorithms.join")
        _modules["torch.distributed.algorithms.join"] = join_mod
    join_mod.Join = Join
    join_mod.Joinable = Joinable
    join_mod.JoinHook = JoinHook
    dist.algorithms.join = join_mod

    dist.device_mesh.DeviceMesh = getattr(dist.device_mesh, "DeviceMesh", _DeviceMesh)
    dist.device_mesh.init_device_mesh = getattr(dist.device_mesh, "init_device_mesh", _init_device_mesh)
    dist.DeviceMesh = getattr(dist, "DeviceMesh", _DeviceMesh)
    dist.init_device_mesh = getattr(dist, "init_device_mesh", _init_device_mesh)
    dist.ProcessGroup = _JittorProcessGroup

    c10d = _modules.get("torch.distributed.distributed_c10d")
    if c10d is None:
        c10d = _types.ModuleType("torch.distributed.distributed_c10d")
        _modules["torch.distributed.distributed_c10d"] = c10d
    for name in dir(dist):
        if not name.startswith("__"):
            setattr(c10d, name, getattr(dist, name))
    c10d.is_xccl_available = _api_c10d_is_xccl_available
    c10d.is_nccl_available = dist.is_nccl_available
    c10d.is_gloo_available = dist.is_gloo_available
    c10d.is_mpi_available = dist.is_mpi_available
    c10d.is_ucc_available = dist.is_ucc_available
    c10d.ProcessGroup = dist.ProcessGroup
    c10d._get_default_group = _api_c10d_get_default_group
    c10d._get_default_store = _api_c10d_get_default_store
    c10d.Work = _JittorWork
    c10d.default_pg_timeout = getattr(c10d, "default_pg_timeout", None)
    c10d._get_default_timeout = _api_c10d_get_default_timeout
    c10d._unregister_process_group = _api_c10d_unregister_process_group
    c10d._register_process_group = _api_c10d_register_process_group
    c10d._resolve_process_group = _api_c10d_resolve_process_group
    c10d.ProcessGroupGloo = _JittorProcessGroup
    c10d.ProcessGroupNCCL = _JittorProcessGroup
    c10d._world = _types.SimpleNamespace(
        default_pg=world_group,
        pg_map=pg_map,
        pg_names={},
        pg_group_ranks={},
    )
    dist.distributed_c10d = c10d

    rpc = _modules.get("torch.distributed.rpc")
    if rpc is None:
        rpc = _types.ModuleType("torch.distributed.rpc")
        _modules["torch.distributed.rpc"] = rpc
    rpc.is_available = _api_rpc_is_available
    rpc.init_rpc = _api_rpc_init_rpc
    rpc.shutdown = _api_rpc_shutdown
    dist.rpc = rpc

    optim = _modules.get("torch.distributed.optim")
    if optim is None:
        optim = _types.ModuleType("torch.distributed.optim")
        _modules["torch.distributed.optim"] = optim
    dist.optim = optim


    dist.nn.all_reduce = _autograd_all_reduce
    _modules["torch.distributed.nn"] = dist.nn

    symmetric_memory = _modules.get("torch.distributed._symmetric_memory")
    if symmetric_memory is None:
        symmetric_memory = _types.ModuleType(
            "torch.distributed._symmetric_memory")
        _modules[symmetric_memory.__name__] = symmetric_memory

    symmetric_memory.enable_symm_mem_for_group = _symmetric_memory_unavailable
    symmetric_memory.is_symm_mem_enabled_for_group = _api_symmetric_memory_is_symm_mem_enabled_for_group
    symmetric_memory.rendezvous = _symmetric_memory_unavailable
    symmetric_memory.empty = _symmetric_memory_unavailable
    dist._symmetric_memory = symmetric_memory

    futures = _modules.get("torch.futures")
    if futures is None:
        futures = _types.ModuleType("torch.futures")
        _modules["torch.futures"] = futures
    futures.Future = Future
    g.futures = futures

    checkpoint = dist.checkpoint
    checkpoint.__path__ = getattr(checkpoint, "__path__", [])
    checkpoint.FileSystemReader = FileSystemReader
    checkpoint.FileSystemWriter = FileSystemWriter
    # torch.distributed.checkpoint save/load were the identity: dcp.save()
    # wrote nothing and returned successfully, so a sharded checkpoint was
    # silently never persisted and dcp.load() silently left the model at its
    # current weights. Refuse instead of losing the run.
    checkpoint.load_state_dict = _api_checkpoint_load_state_dict
    _record_unimplemented('torch.distributed.checkpoint.load_state_dict', _dcp_load_effect, _dcp_hint)
    checkpoint.save_state_dict = _api_checkpoint_save_state_dict
    _record_unimplemented('torch.distributed.checkpoint.save_state_dict', _dcp_save_effect, _dcp_hint)
    checkpoint.load = _api_checkpoint_load
    _record_unimplemented('torch.distributed.checkpoint.load', _dcp_load_effect, _dcp_hint)
    checkpoint.save = _api_checkpoint_save
    _record_unimplemented('torch.distributed.checkpoint.save', _dcp_save_effect, _dcp_hint)
    checkpoint_sd = _types.ModuleType("torch.distributed.checkpoint.state_dict")
    checkpoint_sd.StateDictOptions = StateDictOptions
    checkpoint_sd.get_model_state_dict = _get_model_state_dict
    checkpoint_sd.set_model_state_dict = _set_model_state_dict
    checkpoint_sd.get_state_dict = _api_checkpoint_sd_get_state_dict
    checkpoint_sd.set_state_dict = _api_checkpoint_sd_set_state_dict
    checkpoint_fs = _types.ModuleType("torch.distributed.checkpoint.filesystem")
    checkpoint_fs.FileSystemReader = FileSystemReader
    checkpoint_fs.FileSystemWriter = FileSystemWriter
    checkpoint_fs.SerializationFormat = SerializationFormat
    checkpoint_fs._write_item = _api_checkpoint_fs_write_item
    _record_unimplemented('torch.distributed.checkpoint.filesystem._write_item', _dcp_save_effect, _dcp_hint)
    _modules["torch.distributed.checkpoint"] = checkpoint
    _modules["torch.distributed.checkpoint.state_dict"] = checkpoint_sd
    _modules["torch.distributed.checkpoint.filesystem"] = checkpoint_fs
    checkpoint.state_dict = checkpoint_sd
    checkpoint.filesystem = checkpoint_fs

    shard = dist._shard
    shard.__path__ = getattr(shard, "__path__", [])
    sharded_tensor = _types.ModuleType("torch.distributed._shard.sharded_tensor")
    sharded_tensor.ShardedTensor = ShardedTensor
    sharded_tensor.init_from_local_shards = _api_sharded_tensor_init_from_local_shards
    sharded_tensor.empty = _api_sharded_tensor_empty
    shard.sharded_tensor = sharded_tensor
    _modules["torch.distributed._shard"] = shard
    _modules["torch.distributed._shard.sharded_tensor"] = sharded_tensor
    for _sub in ("api", "metadata", "reshard", "shard"):
        _m = _types.ModuleType("torch.distributed._shard.sharded_tensor." + _sub)
        _m.ShardedTensor = ShardedTensor
        _modules[_m.__name__] = _m

    dist.Store = Store
    dist.TCPStore = TCPStore
    dist.FileStore = FileStore
    dist.PrefixStore = PrefixStore
    for name in ("Store", "TCPStore", "FileStore", "PrefixStore"):
        setattr(c10d, name, getattr(dist, name))
        setattr(g._C._distributed_c10d, name, getattr(dist, name))


    rendezvous_mod = _modules.get("torch.distributed.rendezvous")
    if rendezvous_mod is None:
        rendezvous_mod = _RendezvousModule("torch.distributed.rendezvous")
        _modules[rendezvous_mod.__name__] = rendezvous_mod
    elif not isinstance(rendezvous_mod, _RendezvousModule):
        rendezvous_mod.__class__ = _RendezvousModule


    rendezvous_mod.rendezvous = _rendezvous
    dist.rendezvous = rendezvous_mod

    if not hasattr(g, "_C"):
        g._C = _CNS()
    _install_fsdp2_distributed(dist, g, registry=registry)
    g.distributed = dist
    register_api_bindings(dist, "torch.distributed",
        ("is_available", "is_backend_available", "is_nccl_available", "is_gloo_available",
         "is_mpi_available", "is_ucc_available", "is_initialized", "get_rank",
         "get_world_size", "init_process_group", "destroy_process_group", "get_backend",
         "Backend", "P2POp", "ReduceOp", "ProcessGroup", "barrier", "all_reduce",
         "all_gather", "all_gather_into_tensor", "broadcast", "all_gather_object",
         "broadcast_object_list", "gather_object", "new_group",
         "new_subgroups_by_enumeration", "get_global_rank", "get_process_group_ranks",
         "Store", "TCPStore", "FileStore", "PrefixStore"),
        Fidelity.APPROXIMATE,
        "Reuses native ProcessGroup, store and collective owners; group, reduction "
        "and backend capability limits and existing Work semantics apply")
    register_api_bindings(dist, "torch.distributed",
        ("send", "recv", "isend", "irecv", "batch_isend_irecv"),
        Fidelity.UNIMPLEMENTED, "Point-to-point communication explicitly rejects nonempty work")
    register_api_bindings(dist.nn, "torch.distributed.nn", ("all_reduce",),
        Fidelity.APPROXIMATE, "Native differentiable sum/mean or selected group's native reduction")
    register_api_bindings(futures, "torch.futures", ("Future",),
        Fidelity.APPROXIMATE, "Synchronous result holder; callbacks run immediately")
    register_api_bindings(join_mod, "torch.distributed.algorithms.join",
        ("Join", "Joinable", "JoinHook"), Fidelity.UNIMPLEMENTED,
        "Import-compatible context and hooks do not implement uneven-input communication")
    register_api_bindings(checkpoint, "torch.distributed.checkpoint",
        ("load_state_dict", "save_state_dict", "load", "save", "FileSystemReader", "FileSystemWriter"),
        Fidelity.UNIMPLEMENTED, "Checkpoint I/O follows the existing explicit unsupported policy")
    register_api_bindings(checkpoint_sd, "torch.distributed.checkpoint.state_dict",
        ("StateDictOptions", "get_model_state_dict", "set_model_state_dict", "get_state_dict", "set_state_dict"),
        Fidelity.APPROXIMATE, "Local module state and registered FSDP load provider; "
        "full distributed checkpoint/options semantics are not implemented")
    register_api_bindings(rendezvous_mod, "torch.distributed.rendezvous", ("rendezvous",),
        Fidelity.APPROXIMATE, "Delegates supported URL, rank, world-size and timeout handling to native stores")
    register_api_bindings(c10d, "torch.distributed.distributed_c10d",
        ("is_xccl_available", "_get_default_group", "_get_default_store", "_get_default_timeout",
         "_resolve_process_group", "ProcessGroup", "ProcessGroupGloo", "ProcessGroupNCCL", "Work"),
        Fidelity.APPROXIMATE, "Exposes installation-owned WORLD/store and native group/work identities")
    register_api_bindings(c10d, "torch.distributed.distributed_c10d",
        ("_register_process_group", "_unregister_process_group"), Fidelity.UNIMPLEMENTED,
        "Named registration hooks are import-compatible no-ops")
    register_api_bindings(rpc, "torch.distributed.rpc", ("is_available", "init_rpc", "shutdown"),
        Fidelity.UNIMPLEMENTED, "RPC transport is unavailable; initialization/shutdown are placeholders")
    register_api_bindings(symmetric_memory, "torch.distributed._symmetric_memory",
        ("enable_symm_mem_for_group", "is_symm_mem_enabled_for_group", "rendezvous", "empty"),
        Fidelity.UNIMPLEMENTED, "Symmetric memory allocation and rendezvous are unavailable")
    register_api_bindings(sharded_tensor, "torch.distributed._shard.sharded_tensor",
        ("ShardedTensor", "init_from_local_shards", "empty"), Fidelity.UNIMPLEMENTED,
        "Placeholder type and local tensor/first-shard returns do not implement sharded storage")
    register_api_bindings(dist, "torch.distributed", ("DeviceMesh", "init_device_mesh"),
        Fidelity.UNIMPLEMENTED if dist.DeviceMesh is _DeviceMesh else Fidelity.APPROXIMATE,
        "Uses the installed mesh owner; the fallback only describes a singleton mesh")


def install(ctx):
    _install_distributed(ctx.jittor_module, ctx.registry)
