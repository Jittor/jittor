"""FSDP2/DTensor compatibility for the jittor torch shim.

World-size 1 keeps PyTorch FSDP2's original-parameter semantics: parameters stay
full-size and communication APIs are no-ops.  When Jittor's NCCL launcher has
initialized a multi-rank process group, this module enables a minimal true FSDP2
path for dense parameters: parameters are stored as rank-local shards, gathered
before forward, then resharded.  A small helper synchronizes sharded gradients for
tests/benchmarks while the broader torch optimizer integration is filled in.
"""
import contextlib
import enum
import os
import sys
import types

import numpy as np
import jittor as jt
from jittor import nn


def _ensure_module(name, parent=None, attr=None):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if parent is not None and attr:
        setattr(parent, attr, mod)
    return mod


def _prod(xs):
    out = 1
    for x in xs:
        try:
            out *= int(x)
        except Exception:
            pass
    return out


def _world_size():
    try:
        return int(getattr(jt, "world_size", 1))
    except Exception:
        return 1


def _rank():
    try:
        return int(getattr(jt, "rank", 0))
    except Exception:
        return 0


def _in_true_distributed():
    return _world_size() > 1 and (
        os.environ.get("JT_NCCL_WORLD_SIZE") is not None
        or os.environ.get("OMPI_COMM_WORLD_SIZE") is not None
        or getattr(jt, "in_mpi", False)
    )


def _nccl_ops():
    try:
        return getattr(jt.compile_extern, "nccl_ops", None)
    except Exception:
        return None


def _flatten_var(v):
    return v.reshape((-1,))


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _pad_flat(flat, padded_numel):
    n = int(flat.numel()) if callable(getattr(flat, "numel", None)) else int(np.prod(flat.shape))
    if n == int(padded_numel):
        return flat
    pad = jt.zeros((int(padded_numel) - n,), dtype=flat.dtype)
    return jt.concat([flat, pad], dim=0)


def _slice_flat(flat, start, length):
    start = int(start)
    length = int(length)
    return flat[start:start + length]


def _all_gather_shards(local_shard):
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_all_gather", None)):
        return ops.nccl_all_gather(local_shard)
    if callable(getattr(local_shard, "mpi_all_gather", None)):
        return local_shard.mpi_all_gather()
    raise RuntimeError("Jittor NCCL all_gather is not available; launch with jittor.distributed.launch and use_nccl=1")


def _reduce_scatter_padded(full_grad):
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_reduce_scatter", None)):
        return ops.nccl_reduce_scatter(full_grad)
    # Correct fallback for environments with all_reduce but without native
    # reduce_scatter.  It communicates more than needed, but preserves semantics.
    reduced = full_grad.mpi_all_reduce("sum")
    shard = int(reduced.shape[0]) // max(_world_size(), 1)
    return _slice_flat(reduced, _rank() * shard, shard)


def _param_numel(v):
    return int(np.prod(tuple(int(x) for x in v.shape)))


def _named_parameters_with_owner(module, recurse=True):
    out = []
    seen = set()

    def child_items(mod):
        try:
            items = mod.named_children()
            if items is not None:
                return list(items)
        except Exception:
            pass
        try:
            modules = getattr(mod, "_modules", None)
            if callable(modules):
                modules = modules()
            if isinstance(modules, dict):
                return list(modules.items())
        except Exception:
            pass
        return []

    def visit(mod, prefix=""):
        dc = getattr(mod, "__dict__", {})
        try:
            from jittor import nn as _nn
            if isinstance(mod, _nn.ParameterList):
                dc = mod.params
        except Exception:
            pass
        bufnames = getattr(mod, "__dict__", {}).get("_buffer_names", ())
        for name, value in list(dc.items()):
            if isinstance(name, str) and name.startswith("_"):
                continue
            if isinstance(value, jt.Var):
                if id(value) in seen:
                    continue
                if getattr(value, "is_buffer", False) or not getattr(value, "persistent", True) or name in bufnames:
                    continue
                seen.add(id(value))
                pname = f"{prefix}.{name}" if prefix else str(name)
                out.append((pname, mod, name, value))
        if recurse:
            for name, value in child_items(mod):
                if isinstance(value, nn.Module):
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    visit(value, child_prefix)

    visit(module)
    return out


class DeviceMesh:
    def __init__(self, device_type=None, mesh=None, *, mesh_dim_names=None,
                 _init_backend=True, **kwargs):
        self.device_type = device_type or ("cuda" if getattr(jt, "has_cuda", 0) else "cpu")
        if not isinstance(self.device_type, str):
            self.device_type = getattr(self.device_type, "type", "cpu")
        self.mesh = mesh if mesh is not None else (0,)
        if isinstance(self.mesh, int):
            self.shape = (int(self.mesh),)
        else:
            try:
                self.shape = tuple(int(x) for x in self.mesh)
            except Exception:
                self.shape = tuple(getattr(self.mesh, "shape", (1,)))
        if not self.shape:
            self.shape = (1,)
        self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names is not None else None
        self.ndim = len(self.shape)

    def __repr__(self):
        return "DeviceMesh(device_type=%r, mesh=%r, mesh_dim_names=%r)" % (
            self.device_type, self.mesh, self.mesh_dim_names)

    def __getitem__(self, key):
        return self

    def size(self, dim=None, *, mesh_dim=None):
        if mesh_dim is not None:
            dim = mesh_dim
        if dim is None:
            return _prod(self.shape)
        if isinstance(dim, str) and self.mesh_dim_names and dim in self.mesh_dim_names:
            dim = self.mesh_dim_names.index(dim)
        try:
            return int(self.shape[int(dim)])
        except Exception:
            return 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_rank(self, *args, **kwargs):
        return _rank() if self.size() > 1 else 0

    def get_local_rank(self, *args, **kwargs):
        try:
            return int(os.environ.get("JT_NCCL_LOCAL_RANK",
                                      os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            return 0

    def get_group(self, *args, **kwargs):
        return None

    def get_all_groups(self):
        return [self.get_group()]

    def get_coordinate(self):
        return tuple(0 for _ in range(self.ndim))

    def _flatten(self, mesh_dim_name=None):
        return self

    def _unflatten(self, mesh_dim_names=None):
        if mesh_dim_names is not None:
            self.mesh_dim_names = tuple(mesh_dim_names)
        return self

    @staticmethod
    def _concatenate(meshes, mesh_dim_name=None):
        meshes = list(meshes)
        return meshes[0] if meshes else DeviceMesh("cpu", (1,))

    @classmethod
    def from_group(cls, group, device_type=None, mesh=None, mesh_dim_names=None, **kwargs):
        return cls(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names, **kwargs)


def init_device_mesh(device_type=None, mesh_shape=None, *, mesh_dim_names=None, **kwargs):
    return DeviceMesh(device_type=device_type, mesh=mesh_shape or (1,),
                      mesh_dim_names=mesh_dim_names, **kwargs)


class Placement:
    def is_shard(self):
        return isinstance(self, Shard)

    def is_replicate(self):
        return isinstance(self, Replicate)

    def is_partial(self):
        return isinstance(self, Partial)

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == getattr(other, "__dict__", {})

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))


class Replicate(Placement):
    def __repr__(self):
        return "Replicate()"


class Shard(Placement):
    def __init__(self, dim=0):
        self.dim = int(dim)

    def __repr__(self):
        return "Shard(dim=%s)" % self.dim


class Partial(Placement):
    def __init__(self, reduce_op="sum"):
        self.reduce_op = reduce_op

    def __repr__(self):
        return "Partial(reduce_op=%r)" % self.reduce_op


def _mark_dtensor(tensor, device_mesh=None, placements=None):
    mesh = device_mesh or DeviceMesh("cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
    pls = tuple(placements or (Replicate(),))
    try:
        object.__setattr__(tensor, "_dtensor_device_mesh", mesh)
        object.__setattr__(tensor, "_dtensor_placements", pls)
        object.__setattr__(tensor, "_local_tensor", tensor)
        object.__setattr__(tensor, "device_mesh", mesh)
        object.__setattr__(tensor, "placements", pls)
        object.__setattr__(tensor, "_spec", types.SimpleNamespace(mesh=mesh, placements=pls))
        if not callable(getattr(tensor, "to_local", None)):
            object.__setattr__(tensor, "to_local", types.MethodType(lambda self, *a, **k: self, tensor))
        if not callable(getattr(tensor, "full_tensor", None)):
            object.__setattr__(tensor, "full_tensor", types.MethodType(lambda self, *a, **k: self, tensor))
        if not callable(getattr(tensor, "redistribute", None)):
            def _redistribute(self, device_mesh=None, placements=None, **kwargs):
                return _mark_dtensor(self, device_mesh or getattr(self, "_dtensor_device_mesh", None),
                                     placements or getattr(self, "_dtensor_placements", None))
            object.__setattr__(tensor, "redistribute", types.MethodType(_redistribute, tensor))
    except Exception:
        pass
    return tensor


class _DTensorMeta(type):
    def __instancecheck__(cls, obj):
        return hasattr(obj, "_dtensor_placements") or type.__instancecheck__(cls, obj)


class DTensor(metaclass=_DTensorMeta):
    def __init__(self, local_tensor, device_mesh=None, placements=None, **kwargs):
        self._local_tensor = local_tensor
        self.device_mesh = device_mesh or DeviceMesh("cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
        self.placements = tuple(placements or (Replicate(),))
        self._spec = types.SimpleNamespace(mesh=self.device_mesh, placements=self.placements)

    @staticmethod
    def from_local(local_tensor, device_mesh=None, placements=None, run_check=False,
                   shape=None, stride=None, grad_placements=None, **kwargs):
        return _mark_dtensor(local_tensor, device_mesh, placements)

    def to_local(self, *args, **kwargs):
        return self._local_tensor

    def full_tensor(self, *args, **kwargs):
        return self._local_tensor

    def redistribute(self, device_mesh=None, placements=None, **kwargs):
        self.device_mesh = device_mesh or self.device_mesh
        self.placements = tuple(placements or self.placements)
        self._spec = types.SimpleNamespace(mesh=self.device_mesh, placements=self.placements)
        return self

    def __getattr__(self, name):
        return getattr(self._local_tensor, name)

    def __array__(self, dtype=None):
        arr = self._local_tensor.numpy()
        return arr.astype(dtype) if dtype is not None else arr


def distribute_tensor(tensor, device_mesh=None, placements=None, src_data_rank=0, **kwargs):
    return _mark_dtensor(tensor, device_mesh, placements)


def distribute_module(module, device_mesh=None, partition_fn=None, input_fn=None,
                      output_fn=None, **kwargs):
    if callable(partition_fn):
        partition_fn("", module, device_mesh)
    object.__setattr__(module, "_distribute_module_applied", True)
    object.__setattr__(module, "_dtensor_device_mesh", device_mesh)
    if callable(input_fn):
        object.__setattr__(module, "_dtensor_input_fn", input_fn)
    if callable(output_fn):
        object.__setattr__(module, "_dtensor_output_fn", output_fn)
    return module


def is_dtensor(obj):
    return isinstance(obj, DTensor) or hasattr(obj, "_dtensor_placements")


def _shape_from_args(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(int(x) for x in args[0])
    return tuple(int(x) for x in args)


def _np_dtype(dtype=None):
    if dtype is None:
        return np.float32
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    if name in ("float", "float32"):
        return np.float32
    if name in ("double", "float64"):
        return np.float64
    if name in ("half", "float16"):
        return np.float16
    if name in ("bfloat16",):
        return np.float32
    if name in ("long", "int64"):
        return np.int64
    if name in ("int", "int32"):
        return np.int32
    if name in ("bool", "bool_"):
        return np.bool_
    return np.float32


def _dtensor_from_array(array, device_mesh=None, placements=None, dtype=None):
    tensor = jt.array(array)
    if dtype is not None:
        try:
            tensor = tensor.astype(dtype)
        except Exception:
            try:
                tensor = tensor.astype(str(dtype).split(".")[-1])
            except Exception:
                pass
    return _mark_dtensor(tensor, device_mesh, placements)


def empty(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.empty(_shape_from_args(size), dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def ones(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.ones(_shape_from_args(size), dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def zeros(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.zeros(_shape_from_args(size), dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def full(size, fill_value, *, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.full(_shape_from_args((size,)), fill_value,
                                       dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def rand(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.random.rand(*_shape_from_args(size)).astype(_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def randn(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.random.randn(*_shape_from_args(size)).astype(_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def linspace(start, end, steps, *, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(np.linspace(start, end, int(steps), dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


def logspace(start, end, steps, *, base=10.0, device_mesh=None, placements=None,
             dtype=None, **kwargs):
    return _dtensor_from_array(np.logspace(start, end, int(steps), base=base,
                                           dtype=_np_dtype(dtype)),
                               device_mesh, placements, dtype)


class StateDictType(enum.Enum):
    FULL_STATE_DICT = "full"
    LOCAL_STATE_DICT = "local"
    SHARDED_STATE_DICT = "sharded"


class ShardingStrategy(enum.Enum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    _HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"


class BackwardPrefetch(enum.Enum):
    BACKWARD_PRE = "backward_pre"
    BACKWARD_POST = "backward_post"


class CPUOffload:
    def __init__(self, offload_params=False):
        self.offload_params = bool(offload_params)


class _Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StateDictConfig(_Config):
    pass


class OptimStateDictConfig(_Config):
    pass


class FullStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu), rank0_only=bool(rank0_only))


class LocalStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedStateDictConfig(LocalStateDictConfig):
    pass


class FullOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=False, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu), rank0_only=bool(rank0_only))


class LocalOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedOptimStateDictConfig(LocalOptimStateDictConfig):
    pass


class StateDictSettings:
    def __init__(self, state_dict_type=StateDictType.FULL_STATE_DICT,
                 state_dict_config=None, optim_state_dict_config=None):
        self.state_dict_type = state_dict_type
        self.state_dict_config = state_dict_config
        self.optim_state_dict_config = optim_state_dict_config


class OptimStateKeyType(enum.Enum):
    PARAM_NAME = "param_name"
    PARAM_ID = "param_id"


class FlatParameter:
    def __new__(cls, data=None, requires_grad=True, *args, **kwargs):
        maker = getattr(jt, "_torch_make_parameter", None)
        if data is not None and callable(maker):
            return maker(data, requires_grad=requires_grad)
        return object.__new__(cls)

    def __init__(self, data=None, requires_grad=True, *args, **kwargs):
        self.data = data
        self.requires_grad = requires_grad


class MixedPrecisionPolicy:
    def __init__(self, param_dtype=None, reduce_dtype=None, output_dtype=None,
                 cast_forward_inputs=True, **kwargs):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.output_dtype = output_dtype
        self.cast_forward_inputs = cast_forward_inputs
        for k, v in kwargs.items():
            setattr(self, k, v)


class MixedPrecision(MixedPrecisionPolicy):
    pass


class OffloadPolicy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CPUOffloadPolicy(OffloadPolicy):
    def __init__(self, pin_memory=True, **kwargs):
        super().__init__(pin_memory=pin_memory, **kwargs)


class NoOffloadPolicy(OffloadPolicy):
    pass


class DataParallelMeshDims:
    def __init__(self, shard=None, replicate=None):
        self.shard = shard
        self.replicate = replicate
        self.shard_names = tuple(() if shard is None else
                                 (shard if isinstance(shard, (tuple, list)) else (shard,)))
        self.replicate_names = tuple(() if replicate is None else
                                     (replicate if isinstance(replicate, (tuple, list))
                                      else (replicate,)))


class UnshardHandle:
    def __init__(self, module=None):
        self.module = module

    def wait(self):
        return None


def _iter_modules(module, recurse=True):
    if recurse and hasattr(module, "modules"):
        try:
            return list(module.modules())
        except Exception:
            pass
    return [module]


def _iter_fsdp_modules(module, recurse=True):
    return [m for m in _iter_modules(module, recurse)
            if getattr(m, "_is_fsdp_module", False)]


def _apply_fsdp_attr(module, name, value, recurse=True):
    targets = _iter_fsdp_modules(module, recurse) or [module]
    for m in targets:
        st = getattr(m, "_fsdp_state", None)
        if st is None:
            st = types.SimpleNamespace()
            object.__setattr__(m, "_fsdp_state", st)
        setattr(st, name, value)
    return module


def _init_true_fsdp_state(module, state):
    if getattr(state, "true_fsdp_initialized", False):
        return state
    if not _in_true_distributed():
        state.true_fsdp_initialized = False
        return state
    ws = _world_size()
    rank = _rank()
    entries = []
    for name, owner, attr, param in _named_parameters_with_owner(module, recurse=True):
        numel = _param_numel(param)
        shard_numel = _ceil_div(numel, ws)
        padded_numel = shard_numel * ws
        flat_full = _pad_flat(_flatten_var(param), padded_numel)
        local = _slice_flat(flat_full, rank * shard_numel, shard_numel)
        local.sync()
        entries.append(types.SimpleNamespace(
            name=name,
            owner=owner,
            attr=attr,
            shape=tuple(int(x) for x in param.shape),
            dtype=param.dtype,
            numel=numel,
            padded_numel=padded_numel,
            shard_numel=shard_numel,
            shard=local,
            full_param=None,
        ))
        object.__setattr__(owner, attr, local)
    state.true_fsdp_initialized = True
    state.true_fsdp_rank = rank
    state.true_fsdp_world_size = ws
    state.true_fsdp_params = entries
    state.true_fsdp_unsharded = False
    return state


def _unshard_module_params(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if getattr(state, "true_fsdp_unsharded", False):
        return module
    for entry in state.true_fsdp_params:
        gathered = _all_gather_shards(entry.shard)
        full_flat = gathered if entry.padded_numel == entry.numel else _slice_flat(gathered, 0, entry.numel)
        full = full_flat.reshape(entry.shape)
        entry.full_param = full
        object.__setattr__(entry.owner, entry.attr, full)
    state.true_fsdp_unsharded = True
    return module


def _reshard_module_params(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if not getattr(state, "true_fsdp_unsharded", False):
        return module
    for entry in state.true_fsdp_params:
        object.__setattr__(entry.owner, entry.attr, entry.shard)
        # Keep the full Var from the just-finished forward alive for
        # sync_sharded_grads(loss): Jittor's autograd needs the exact Var object
        # that participated in the forward graph.
    state.true_fsdp_unsharded = False
    return module


def _execute_with_true_fsdp(module, orig_execute, *args, **kwargs):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return orig_execute(*args, **kwargs)
    _unshard_module_params(module)
    try:
        out = orig_execute(*args, **kwargs)
    finally:
        if getattr(state, "reshard_after_forward", True):
            _reshard_module_params(module)
    return out


def _install_true_fsdp_execute(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if getattr(module, "_fsdp_orig_execute", None) is not None:
        return module
    orig_execute = getattr(module, "execute", None)
    if not callable(orig_execute):
        return module
    object.__setattr__(module, "_fsdp_orig_execute", orig_execute)

    def _wrapped_execute(self, *args, **kwargs):
        return _execute_with_true_fsdp(self, self._fsdp_orig_execute, *args, **kwargs)

    object.__setattr__(module, "execute", types.MethodType(_wrapped_execute, module))
    return module


def sync_sharded_grads(module, loss=None, *, divide_by_world_size=True):
    """Return rank-local sharded gradients for a true-FSDP-managed module.

    The helper computes gradients against gathered full parameters, then
    reduce-scatters each flattened gradient so optimizers can update local shards.
    It is intentionally explicit: Jittor's generic optimizer integration does not
    yet know about FSDP2 sharded parameters.
    """
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        params = list(module.parameters())
        return jt.grad(loss, params) if loss is not None else []
    if loss is None:
        raise ValueError("sync_sharded_grads() requires a loss for true FSDP2")
    has_forward_params = all(getattr(entry, "full_param", None) is not None
                             for entry in state.true_fsdp_params)
    if not getattr(state, "true_fsdp_unsharded", False) and not has_forward_params:
        _unshard_module_params(module)
    full_params = [entry.full_param for entry in state.true_fsdp_params]
    full_grads = jt.grad(loss, full_params)
    sharded = []
    for entry, grad in zip(state.true_fsdp_params, full_grads):
        flat = _pad_flat(_flatten_var(grad), entry.padded_numel)
        shard = _reduce_scatter_padded(flat)
        if divide_by_world_size:
            shard = shard / max(int(state.true_fsdp_world_size), 1)
        shard = shard.stop_grad()
        sharded.append(shard)
    state.true_fsdp_last_grads = sharded
    return sharded


def sharded_sgd_step(module, loss, lr=1e-4, *, divide_by_world_size=True):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        params = list(module.parameters())
        grads = jt.grad(loss, params)
        for p, g in zip(params, grads):
            p.assign(p - g * lr)
        return grads
    grads = sync_sharded_grads(module, loss, divide_by_world_size=divide_by_world_size)
    for entry, grad in zip(state.true_fsdp_params, grads):
        entry.shard.assign((entry.shard - grad * lr).stop_grad())
        object.__setattr__(entry.owner, entry.attr, entry.shard)
        entry.full_param = None
    state.true_fsdp_unsharded = False
    return grads


def local_sharded_state_dict(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module.state_dict()
    return {entry.name: entry.shard for entry in state.true_fsdp_params}


class _FSDPModuleMeta(type):
    def __instancecheck__(cls, obj):
        return bool(getattr(obj, "_is_fsdp_module", False)) or type.__instancecheck__(cls, obj)


class FSDPModule(metaclass=_FSDPModuleMeta):
    def unshard(self, async_op=False):
        _unshard_module_params(self)
        return UnshardHandle(self) if async_op else None

    def reshard(self):
        _reshard_module_params(self)
        return None

    def reset_iter_state(self):
        return _apply_fsdp_attr(self, "iter_state_reset", True, False)

    def set_reshard_after_forward(self, value, recurse=True):
        return _apply_fsdp_attr(self, "reshard_after_forward", value, recurse)

    def set_requires_gradient_sync(self, value, recurse=True):
        return _apply_fsdp_attr(self, "requires_gradient_sync", bool(value), recurse)

    def set_requires_all_reduce(self, value, recurse=True):
        return _apply_fsdp_attr(self, "requires_all_reduce", bool(value), recurse)

    def set_all_reduce_hook(self, hook, *, stream=None):
        _apply_fsdp_attr(self, "all_reduce_hook", hook, False)
        return _apply_fsdp_attr(self, "all_reduce_hook_stream", stream, False)

    def set_allocate_memory_from_process_group_for_comm(self, enable):
        return _apply_fsdp_attr(self, "allocate_memory_from_process_group_for_comm",
                                bool(enable), False)

    def set_custom_all_gather(self, comm):
        return _apply_fsdp_attr(self, "custom_all_gather", comm, False)

    def set_custom_reduce_scatter(self, comm):
        return _apply_fsdp_attr(self, "custom_reduce_scatter", comm, False)

    def set_reduce_scatter_unused_params(self, value=True):
        return _apply_fsdp_attr(self, "reduce_scatter_unused_params", bool(value), False)

    def set_reduce_scatter_max_input_buffers(self, value):
        return _apply_fsdp_attr(self, "reduce_scatter_max_input_buffers", value, False)

    def set_separate_reduce_scatter_group(self, group):
        return _apply_fsdp_attr(self, "separate_reduce_scatter_group", group, False)

    def set_is_last_backward(self, value, recurse=True):
        return _apply_fsdp_attr(self, "is_last_backward", bool(value), recurse)

    def set_reshard_after_backward(self, value, recurse=True):
        return _apply_fsdp_attr(self, "reshard_after_backward", value, recurse)

    def set_unshard_in_backward(self, value, recurse=True):
        return _apply_fsdp_attr(self, "unshard_in_backward", bool(value), recurse)

    def set_modules_to_forward_prefetch(self, modules):
        return _apply_fsdp_attr(self, "modules_to_forward_prefetch", list(modules or ()), False)

    def set_modules_to_backward_prefetch(self, modules):
        return _apply_fsdp_attr(self, "modules_to_backward_prefetch", list(modules or ()), False)

    def set_gradient_divide_factor(self, factor, recurse=True):
        return _apply_fsdp_attr(self, "gradient_divide_factor", factor, recurse)

    def set_reduce_scatter_divide_factor(self, factor, recurse=True):
        return _apply_fsdp_attr(self, "reduce_scatter_divide_factor", factor, recurse)

    def set_force_sum_reduction_for_comms(self, value, recurse=True):
        return _apply_fsdp_attr(self, "force_sum_reduction_for_comms", bool(value), recurse)

    def set_symm_mem_for_comm(self, backend="NCCL", recurse=True):
        return _apply_fsdp_attr(self, "symm_mem_for_comm", backend, recurse)

    def set_post_optim_event(self, event, recurse=True):
        return _apply_fsdp_attr(self, "post_optim_event", event, recurse)

    def _get_fsdp_state(self):
        return getattr(self, "_fsdp_state", None)

    def sync_sharded_grads(self, loss, *, divide_by_world_size=True):
        return sync_sharded_grads(self, loss, divide_by_world_size=divide_by_world_size)

    def sharded_sgd_step(self, loss, lr=1e-4, *, divide_by_world_size=True):
        return sharded_sgd_step(self, loss, lr=lr, divide_by_world_size=divide_by_world_size)

    def local_sharded_state_dict(self):
        return local_sharded_state_dict(self)


_FSDP_METHODS = (
    "unshard", "reshard", "reset_iter_state", "set_reshard_after_forward",
    "set_requires_gradient_sync", "set_requires_all_reduce", "set_is_last_backward",
    "set_all_reduce_hook", "set_allocate_memory_from_process_group_for_comm",
    "set_custom_all_gather", "set_custom_reduce_scatter",
    "set_reduce_scatter_unused_params", "set_reduce_scatter_max_input_buffers",
    "set_separate_reduce_scatter_group",
    "set_reshard_after_backward", "set_unshard_in_backward",
    "set_modules_to_forward_prefetch", "set_modules_to_backward_prefetch",
    "set_gradient_divide_factor", "set_reduce_scatter_divide_factor",
    "set_force_sum_reduction_for_comms", "set_symm_mem_for_comm",
    "set_post_optim_event", "_get_fsdp_state",
    "sync_sharded_grads", "sharded_sgd_step", "local_sharded_state_dict",
)


def _inject_fsdp_methods(module):
    for name in _FSDP_METHODS:
        if not callable(getattr(module, name, None)):
            object.__setattr__(module, name, types.MethodType(getattr(FSDPModule, name), module))
    if isinstance(module, FSDPModule):
        return module
    cache = getattr(FSDPModule, "_jittor_class_cache", None)
    if cache is None:
        cache = FSDPModule._jittor_class_cache = {}
    cls = type(module)
    fsdp_cls = cache.get(cls)
    if fsdp_cls is None:
        try:
            fsdp_cls = type(cls.__name__ + "FSDP2Compat", (cls, FSDPModule),
                            {"__module__": cls.__module__})
            cache[cls] = fsdp_cls
        except Exception:
            fsdp_cls = None
    if fsdp_cls is not None:
        try:
            module.__class__ = fsdp_cls
        except Exception:
            pass
    return module


def fully_shard(module, *, mesh=None, reshard_after_forward=True,
                shard_placement_fn=None, mp_policy=None, offload_policy=None,
                ignored_params=None, dp_mesh_dims=None, **kwargs):
    if isinstance(module, (list, tuple)):
        for m in module:
            fully_shard(m, mesh=mesh, reshard_after_forward=reshard_after_forward,
                        shard_placement_fn=shard_placement_fn, mp_policy=mp_policy,
                        offload_policy=offload_policy, ignored_params=ignored_params,
                        dp_mesh_dims=dp_mesh_dims, **kwargs)
        return module
    if module is None or not hasattr(module, "parameters"):
        raise TypeError("fully_shard() expects a torch.nn.Module-compatible object")
    st = getattr(module, "_fsdp_state", None)
    if st is None:
        st = types.SimpleNamespace()
        object.__setattr__(module, "_fsdp_state", st)
    st.mesh = mesh or getattr(st, "mesh", None) or DeviceMesh("cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
    st.reshard_after_forward = reshard_after_forward
    st.shard_placement_fn = shard_placement_fn
    st.mp_policy = mp_policy if mp_policy is not None else MixedPrecisionPolicy()
    st.offload_policy = offload_policy if offload_policy is not None else NoOffloadPolicy()
    st.ignored_params = tuple(ignored_params or ())
    st.dp_mesh_dims = dp_mesh_dims
    st.kwargs = dict(kwargs)
    object.__setattr__(module, "_is_fsdp_module", True)
    object.__setattr__(module, "_is_fsdp_managed_module", True)
    object.__setattr__(module, "_fsdp_use_orig_params", True)
    _inject_fsdp_methods(module)
    _init_true_fsdp_state(module, st)
    _install_true_fsdp_execute(module)
    return module


def register_fsdp_forward_method(module, method_name):
    methods = getattr(module, "_fsdp_forward_methods", None)
    if methods is None:
        methods = set()
        object.__setattr__(module, "_fsdp_forward_methods", methods)
    methods.add(method_name)
    return module


@contextlib.contextmanager
def share_comm_ctx(modules=None):
    if modules is not None:
        for module in modules:
            _apply_fsdp_attr(module, "share_comm_ctx", True, False)
    yield


class FullyShardedDataParallel(nn.Module, FSDPModule):
    def __init__(self, module, *args, **kwargs):
        nn.Module.__init__(self)
        self.module = fully_shard(module, **kwargs)
        object.__setattr__(self, "_is_fsdp_module", True)
        object.__setattr__(self, "_fsdp_state", getattr(self.module, "_fsdp_state", None))

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    forward = execute

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.module.load_state_dict(state_dict, *args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self.module.buffers(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.module.named_buffers(*args, **kwargs)

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module, state_dict_type, state_dict_config=None,
                        optim_state_dict_config=None):
        old = getattr(module, "_fsdp_state_dict_type", None)
        object.__setattr__(module, "_fsdp_state_dict_type",
                           (state_dict_type, state_dict_config, optim_state_dict_config))
        try:
            yield
        finally:
            if old is None:
                try:
                    delattr(module, "_fsdp_state_dict_type")
                except Exception:
                    pass
            else:
                object.__setattr__(module, "_fsdp_state_dict_type", old)

    @staticmethod
    def set_state_dict_type(module, state_dict_type, state_dict_config=None,
                            optim_state_dict_config=None):
        object.__setattr__(module, "_fsdp_state_dict_type",
                           (state_dict_type, state_dict_config, optim_state_dict_config))
        return module

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(module, *args, **kwargs):
        yield module

    @staticmethod
    def optim_state_dict(module, optim, *args, **kwargs):
        return optim.state_dict() if hasattr(optim, "state_dict") else {}

    full_optim_state_dict = optim_state_dict

    @staticmethod
    def optim_state_dict_to_load(module, optim, optim_state_dict, *args, **kwargs):
        return optim_state_dict

    @staticmethod
    def shard_full_optim_state_dict(full_optim_state_dict, module, *args, **kwargs):
        return full_optim_state_dict

    @staticmethod
    def scatter_full_optim_state_dict(full_optim_state_dict, module, *args, **kwargs):
        return full_optim_state_dict or {}

    @staticmethod
    def rekey_optim_state_dict(optim_state_dict, optim_state_key_type, module, *args, **kwargs):
        return optim_state_dict


class ShardedGradScaler:
    def __init__(self, *args, **kwargs):
        self._enabled = kwargs.get("enabled", True)

    def scale(self, loss):
        return loss

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, *args, **kwargs):
        return None

    def unscale_(self, optimizer):
        return None

    def state_dict(self):
        return {"enabled": self._enabled}

    def load_state_dict(self, state_dict):
        self._enabled = state_dict.get("enabled", self._enabled)


class FSDPMeshInfo:
    def __init__(self, mesh=None, shard_mesh_dim=None, replicate_mesh_dim=None,
                 shard_mesh_size=1, replicate_mesh_size=1, **kwargs):
        self.mesh = mesh
        self.shard_mesh_dim = shard_mesh_dim
        self.replicate_mesh_dim = replicate_mesh_dim
        self.shard_mesh_size = shard_mesh_size
        self.replicate_mesh_size = replicate_mesh_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class ShardPlacementResult:
    def __init__(self, shard_dim=None, placements=None, **kwargs):
        self.shard_dim = shard_dim
        self.placements = tuple(placements or ())
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get_mesh_info(mesh=None, dp_mesh_dims=None, **kwargs):
    return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=getattr(dp_mesh_dims, "shard", None),
                        replicate_mesh_dim=getattr(dp_mesh_dims, "replicate", None))


class FSDPState:
    pass


TrainingState = enum.Enum("TrainingState", {"IDLE": "idle", "FORWARD": "forward", "BACKWARD": "backward"})
FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"


class DTensorSpec:
    def __init__(self, mesh=None, placements=None, tensor_meta=None, **kwargs):
        self.mesh = mesh
        self.device_mesh = mesh
        self.placements = tuple(placements or ())
        self.tensor_meta = tensor_meta
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get_module_fsdp_state(module):
    return getattr(module, "_fsdp_state", None)


def _get_module_fsdp_state_if_fully_sharded_module(module):
    return getattr(module, "_fsdp_state", None) if getattr(module, "_is_fsdp_module", False) else None


def _is_fsdp_managed_module(module):
    return bool(getattr(module, "_is_fsdp_managed_module", False))


def _lazy_init(state, module=None):
    return state


def _get_post_forward_mesh_info(*args, **kwargs):
    return _get_mesh_info(*args, **kwargs)


def compute_local_shape_and_global_offset(global_shape, mesh, placements):
    shape = tuple(int(x) for x in global_shape)
    offset = tuple(0 for _ in shape)
    return shape, offset


class ParallelStyle:
    def __init__(self, *args, **kwargs):
        self.args = args
        for k, v in kwargs.items():
            setattr(self, k, v)


def parallelize_module(module, device_mesh=None, parallelize_plan=None,
                       src_data_rank=0, **kwargs):
    object.__setattr__(module, "_parallelize_module_applied", True)
    object.__setattr__(module, "_parallelize_plan", parallelize_plan)
    return module


@contextlib.contextmanager
def loss_parallel(*args, **kwargs):
    yield


def _checkpoint_wrapper(module=None, *args, **kwargs):
    return (lambda m: m) if module is None else module


def _apply_activation_checkpointing(model, *args, **kwargs):
    return model


def _checkpoint(module, *args, **kwargs):
    return module(*args, **kwargs)


class ModuleWrapPolicy:
    def __init__(self, module_classes):
        self.module_classes = tuple(module_classes)

    def __call__(self, module, recurse, nonwrapped_numel):
        return isinstance(module, self.module_classes)


class CustomPolicy:
    def __init__(self, lambda_fn):
        self.lambda_fn = lambda_fn

    def __call__(self, module, recurse, nonwrapped_numel):
        return bool(self.lambda_fn(module, recurse, nonwrapped_numel))


def _install_wrap_helpers(fsdp_wrap_mod):
    @contextlib.contextmanager
    def enable_wrap(*args, **kwargs):
        yield

    def wrap(module, *args, **kwargs):
        return fully_shard(module, **{
            k: v for k, v in kwargs.items()
            if k in ("mesh", "reshard_after_forward", "mp_policy", "offload_policy")
        })

    fsdp_wrap_mod.enable_wrap = enable_wrap
    fsdp_wrap_mod.wrap = wrap
    fsdp_wrap_mod.always_wrap_policy = lambda *a, **k: True
    fsdp_wrap_mod.size_based_auto_wrap_policy = (
        lambda module, recurse, nonwrapped_numel, min_num_params=1e8, *a, **k:
        bool(nonwrapped_numel >= min_num_params))
    fsdp_wrap_mod.transformer_auto_wrap_policy = lambda *a, **k: False
    fsdp_wrap_mod.lambda_auto_wrap_policy = (
        lambda module, recurse, nonwrapped_numel, lambda_fn=None, *a, **k:
        bool(lambda_fn(module) if callable(lambda_fn) else False))
    fsdp_wrap_mod.ModuleWrapPolicy = ModuleWrapPolicy
    fsdp_wrap_mod.CustomPolicy = CustomPolicy
    fsdp_wrap_mod._or_policy = (
        lambda module, recurse, nonwrapped_numel, policies=None, *a, **k:
        any(policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
            for policy in (policies or ()) if callable(policy)))


def install(dist, torch_module=None):
    tensor_mod = _ensure_module("torch.distributed.tensor", dist, "tensor")
    tensor_legacy_mod = _ensure_module("torch.distributed._tensor", dist, "_tensor")
    tensor_api_mod = _ensure_module("torch.distributed.tensor._api")
    tensor_placement_mod = _ensure_module("torch.distributed.tensor.placement_types")
    tensor_spec_mod = _ensure_module("torch.distributed.tensor._dtensor_spec")
    tensor_utils_mod = _ensure_module("torch.distributed.tensor._utils")
    tensor_parallel_mod = _ensure_module("torch.distributed.tensor.parallel")
    tensor_parallel_api_mod = _ensure_module("torch.distributed.tensor.parallel.api")
    tensor_parallel_style_mod = _ensure_module("torch.distributed.tensor.parallel.style")
    tensor_parallel_loss_mod = _ensure_module("torch.distributed.tensor.parallel.loss")
    device_mesh_mod = _ensure_module("torch.distributed.device_mesh", dist, "device_mesh")
    tensor_legacy_device_mesh_mod = _ensure_module("torch.distributed._tensor.device_mesh")
    fsdp_mod = _ensure_module("torch.distributed.fsdp", dist, "fsdp")
    fsdp_api_mod = _ensure_module("torch.distributed.fsdp.api")
    fsdp_full_mod = _ensure_module("torch.distributed.fsdp.fully_sharded_data_parallel")
    fsdp_wrap_mod = _ensure_module("torch.distributed.fsdp.wrap")
    fsdp_traversal_mod = _ensure_module("torch.distributed.fsdp._traversal_utils")
    fsdp_runtime_mod = _ensure_module("torch.distributed.fsdp._runtime_utils")
    fsdp_top_common_mod = _ensure_module("torch.distributed.fsdp._common_utils")
    fsdp_state_mod = _ensure_module("torch.distributed.fsdp._fsdp_state")
    fsdp_scaler_mod = _ensure_module("torch.distributed.fsdp.sharded_grad_scaler")
    fsdp_fully_pkg = _ensure_module("torch.distributed.fsdp._fully_shard")
    fsdp_fully_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fully_shard")
    fsdp_fully_api_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_api")
    fsdp_common_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_common")
    fsdp_init_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_init")
    fsdp_fully_state_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_state")
    fsdp_param_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_param")
    fsdp_collectives_mod = _ensure_module("torch.distributed.fsdp._fully_shard._fsdp_collectives")
    comp_mod = _ensure_module("torch.distributed._composable", dist, "_composable")
    comp_fsdp_mod = _ensure_module("torch.distributed._composable.fsdp", comp_mod, "fsdp")
    comp_fsdp_fully_mod = _ensure_module("torch.distributed._composable.fsdp.fully_shard")
    comp_fsdp_api_mod = _ensure_module("torch.distributed._composable.fsdp._fsdp_api")
    functional_collectives_mod = _ensure_module("torch.distributed._functional_collectives")
    algorithms_mod = _ensure_module("torch.distributed.algorithms", dist, "algorithms")
    checkpoint_mod = _ensure_module("torch.distributed.algorithms._checkpoint",
                                    algorithms_mod, "_checkpoint")
    checkpoint_wrapper_mod = _ensure_module(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        checkpoint_mod, "checkpoint_wrapper")

    dist.__path__ = getattr(dist, "__path__", [])
    for pkg in (tensor_mod, tensor_legacy_mod, tensor_parallel_mod, fsdp_mod,
                fsdp_fully_pkg, comp_mod, comp_fsdp_mod, algorithms_mod,
                checkpoint_mod):
        pkg.__path__ = getattr(pkg, "__path__", [])

    fsdp_mod.api = fsdp_api_mod
    fsdp_mod.fully_sharded_data_parallel = fsdp_full_mod
    fsdp_mod.wrap = fsdp_wrap_mod
    fsdp_mod._traversal_utils = fsdp_traversal_mod
    fsdp_mod._runtime_utils = fsdp_runtime_mod
    fsdp_mod._common_utils = fsdp_top_common_mod
    fsdp_mod._fsdp_state = fsdp_state_mod
    fsdp_mod.sharded_grad_scaler = fsdp_scaler_mod
    fsdp_mod._fully_shard = fsdp_fully_pkg
    fsdp_fully_pkg._fully_shard = fsdp_fully_mod
    fsdp_fully_pkg._fsdp_api = fsdp_fully_api_mod
    fsdp_fully_pkg._fsdp_common = fsdp_common_mod
    fsdp_fully_pkg._fsdp_init = fsdp_init_mod
    fsdp_fully_pkg._fsdp_state = fsdp_fully_state_mod
    fsdp_fully_pkg._fsdp_param = fsdp_param_mod
    fsdp_fully_pkg._fsdp_collectives = fsdp_collectives_mod
    comp_fsdp_mod.fully_shard = comp_fsdp_fully_mod
    comp_fsdp_mod._fsdp_api = comp_fsdp_api_mod
    dist._functional_collectives = functional_collectives_mod

    exports = dict(
        DeviceMesh=DeviceMesh, init_device_mesh=init_device_mesh,
        DTensor=DTensor, Placement=Placement, Replicate=Replicate, Shard=Shard,
        Partial=Partial, distribute_tensor=distribute_tensor,
        distribute_module=distribute_module, is_dtensor=is_dtensor,
        StateDictType=StateDictType, ShardingStrategy=ShardingStrategy,
        BackwardPrefetch=BackwardPrefetch, CPUOffload=CPUOffload,
        StateDictConfig=StateDictConfig, OptimStateDictConfig=OptimStateDictConfig,
        MixedPrecision=MixedPrecision, MixedPrecisionPolicy=MixedPrecisionPolicy,
        OffloadPolicy=OffloadPolicy, CPUOffloadPolicy=CPUOffloadPolicy,
        NoOffloadPolicy=NoOffloadPolicy, DataParallelMeshDims=DataParallelMeshDims,
        FullStateDictConfig=FullStateDictConfig,
        LocalStateDictConfig=LocalStateDictConfig,
        ShardedStateDictConfig=ShardedStateDictConfig,
        FullOptimStateDictConfig=FullOptimStateDictConfig,
        LocalOptimStateDictConfig=LocalOptimStateDictConfig,
        ShardedOptimStateDictConfig=ShardedOptimStateDictConfig,
        StateDictSettings=StateDictSettings, OptimStateKeyType=OptimStateKeyType,
        FlatParameter=FlatParameter,
        UnshardHandle=UnshardHandle, FSDPModule=FSDPModule,
        FullyShardedDataParallel=FullyShardedDataParallel,
        ShardedGradScaler=ShardedGradScaler, fully_shard=fully_shard,
        register_fsdp_forward_method=register_fsdp_forward_method,
        share_comm_ctx=share_comm_ctx,
        sync_sharded_grads=sync_sharded_grads,
        sharded_sgd_step=sharded_sgd_step,
        local_sharded_state_dict=local_sharded_state_dict,
    )
    for mod in (fsdp_mod, fsdp_api_mod, fsdp_full_mod, fsdp_fully_pkg,
                fsdp_fully_mod, fsdp_fully_api_mod, comp_fsdp_mod,
                comp_fsdp_fully_mod, comp_fsdp_api_mod):
        for k, v in exports.items():
            setattr(mod, k, v)
    fsdp_mod.FSDP = FullyShardedDataParallel
    fsdp_full_mod.FSDP = FullyShardedDataParallel
    fsdp_scaler_mod.ShardedGradScaler = ShardedGradScaler
    fsdp_traversal_mod._get_fsdp_states = (
        lambda module: [getattr(m, "_fsdp_state") for m in _iter_fsdp_modules(module, True)
                        if hasattr(m, "_fsdp_state")])
    fsdp_traversal_mod._get_fsdp_handles = lambda module: []
    for mod in (fsdp_runtime_mod, fsdp_top_common_mod, fsdp_state_mod,
                fsdp_common_mod, fsdp_fully_state_mod):
        mod.FSDPState = FSDPState
        mod.TrainingState = TrainingState
        mod.FSDP_WRAPPED_MODULE = FSDP_WRAPPED_MODULE
        mod._lazy_init = _lazy_init
        mod._get_module_fsdp_state = _get_module_fsdp_state
        mod._get_fsdp_state = _get_module_fsdp_state
        mod._get_module_fsdp_state_if_fully_sharded_module = (
            _get_module_fsdp_state_if_fully_sharded_module)
        mod._is_fsdp_managed_module = _is_fsdp_managed_module
    fsdp_param_mod.FlatParameter = FlatParameter
    fsdp_collectives_mod.all_gather = lambda tensor, *a, **k: (
        _all_gather_shards(tensor) if _in_true_distributed() else tensor)
    fsdp_collectives_mod.reduce_scatter = lambda tensor, *a, **k: (
        _reduce_scatter_padded(tensor) if _in_true_distributed() else tensor)
    _install_wrap_helpers(fsdp_wrap_mod)

    tensor_factories = {
        "empty": empty,
        "ones": ones,
        "zeros": zeros,
        "full": full,
        "rand": rand,
        "randn": randn,
        "linspace": linspace,
        "logspace": logspace,
    }
    for mod in (tensor_mod, tensor_legacy_mod, tensor_api_mod, tensor_placement_mod,
                tensor_utils_mod):
        for k in ("DTensor", "Placement", "Replicate", "Shard", "Partial",
                  "DeviceMesh", "init_device_mesh", "distribute_tensor",
                  "distribute_module", "is_dtensor"):
            setattr(mod, k, exports[k])
        for k, v in tensor_factories.items():
            setattr(mod, k, v)
    tensor_spec_mod.DTensorSpec = DTensorSpec
    tensor_utils_mod.compute_local_shape_and_global_offset = compute_local_shape_and_global_offset
    tensor_mod.placement_types = tensor_placement_mod
    tensor_mod._api = tensor_api_mod
    tensor_mod._dtensor_spec = tensor_spec_mod
    tensor_mod._utils = tensor_utils_mod
    tensor_mod.parallel = tensor_parallel_mod
    tensor_mod.DeviceMesh = DeviceMesh
    tensor_legacy_mod.device_mesh = tensor_legacy_device_mesh_mod
    for mod in (device_mesh_mod, tensor_legacy_device_mesh_mod):
        mod.DeviceMesh = DeviceMesh
        mod.init_device_mesh = init_device_mesh
    parallel_classes = {}
    for name in ("ColwiseParallel", "RowwiseParallel", "SequenceParallel",
                 "PrepareModuleInput", "PrepareModuleOutput",
                 "PrepareModuleInputOutput"):
        parallel_classes[name] = type(name, (ParallelStyle,), {})
    for mod in (tensor_parallel_mod, tensor_parallel_style_mod):
        mod.ParallelStyle = ParallelStyle
        for name, cls in parallel_classes.items():
            setattr(mod, name, cls)
    for mod in (tensor_parallel_mod, tensor_parallel_api_mod):
        mod.parallelize_module = parallelize_module
    for mod in (tensor_parallel_mod, tensor_parallel_loss_mod):
        mod.loss_parallel = loss_parallel
    tensor_parallel_mod.api = tensor_parallel_api_mod
    tensor_parallel_mod.style = tensor_parallel_style_mod
    tensor_parallel_mod.loss = tensor_parallel_loss_mod

    device_mesh_mod.DeviceMesh = DeviceMesh
    device_mesh_mod.init_device_mesh = init_device_mesh
    dist.DeviceMesh = DeviceMesh
    dist.init_device_mesh = init_device_mesh
    dist.is_available = lambda *a, **k: True
    class AsyncCollectiveTensor:
        def __init__(self, tensor=None):
            self.tensor = tensor
        def wait(self):
            return self.tensor
        def __getattr__(self, name):
            return getattr(self.tensor, name)
    functional_collectives_mod.AsyncCollectiveTensor = AsyncCollectiveTensor

    checkpoint_wrapper_mod.checkpoint_wrapper = _checkpoint_wrapper
    checkpoint_wrapper_mod.apply_activation_checkpointing = _apply_activation_checkpointing
    checkpoint_wrapper_mod.offload_wrapper = lambda module, *a, **k: module
    checkpoint_wrapper_mod._CHECKPOINT_PREFIX = "_checkpoint_wrapped_module."
    checkpoint_wrapper_mod.CheckpointImpl = enum.Enum(
        "CheckpointImpl", {"NO_REENTRANT": "no_reentrant", "REENTRANT": "reentrant"})
    checkpoint_wrapper_mod.checkpoint = _checkpoint
    fsdp_common_mod.FSDPMeshInfo = FSDPMeshInfo
    fsdp_common_mod.ShardPlacementResult = ShardPlacementResult
    fsdp_init_mod._get_mesh_info = _get_mesh_info
    fsdp_init_mod._get_post_forward_mesh_info = _get_post_forward_mesh_info
    if torch_module is not None:
        try:
            torch_module["distributed"] = dist
        except Exception:
            setattr(torch_module, "distributed", dist)
    return dist


_install_fsdp2_distributed = install
