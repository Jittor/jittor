"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt

from ..context import registry_for


def _install_fsdp2_distributed(dist, torch_module=None, registry=None):
    """Install the single-process FSDP2/DTensor compatibility surface."""
    from jittor.compat.fsdp2 import installer as _fsdp2_installer
    return _fsdp2_installer.install_with_registry(
        dist, torch_module, registry=registry
    )

def _install_distributed(g, registry=None):
    """Install single-process torch.distributed stubs.

    Transformers 5 imports torch.distributed at module import time for tensor
    parallel helpers even when no distributed execution is requested. The jittor
    torch shim runs TRELLIS.2 as a single process, so report distributed support
    as unavailable while keeping the imported symbols present.
    """
    _modules = registry_for(g, registry).module_map
    import types as _types

    dist = _modules.get("torch.distributed")
    if dist is None:
        dist = _types.ModuleType("torch.distributed")
        _modules["torch.distributed"] = dist
    dist.is_available = lambda *a, **k: True
    dist.is_initialized = lambda *a, **k: False
    dist.get_rank = lambda *a, **k: 0
    dist.get_world_size = lambda *a, **k: 1
    dist.init_process_group = lambda *a, **k: None
    dist.destroy_process_group = lambda *a, **k: None
    dist.barrier = lambda *a, **k: None
    dist.all_reduce = lambda *a, **k: None
    dist.all_gather = lambda *a, **k: None
    dist.broadcast = lambda *a, **k: None
    def _all_gather_object(object_list, obj, *a, **k):
        if object_list:
            object_list[0] = obj
        return None
    def _broadcast_object_list(object_list, src=0, *a, **k):
        return object_list
    def _all_gather_into_tensor(output_tensor, input_tensor, *a, **k):
        try:
            output_tensor.assign(input_tensor.reshape(output_tensor.shape))
        except Exception:
            try:
                output_tensor.assign(input_tensor)
            except Exception:
                pass
        return None
    dist.all_gather_object = _all_gather_object
    dist.broadcast_object_list = _broadcast_object_list
    dist.all_gather_into_tensor = _all_gather_into_tensor
    dist.gather_object = lambda obj, object_gather_list=None, dst=0, *a, **k: _all_gather_object(object_gather_list or [], obj)
    dist.new_group = lambda *a, **k: dist.group.WORLD
    dist.new_subgroups_by_enumeration = lambda *a, **k: ([dist.group.WORLD], dist.group.WORLD)
    dist.get_global_rank = lambda group=None, group_rank=0: int(group_rank)
    dist.is_torchelastic_launched = lambda *a, **k: False
    class _ReduceOp:
        SUM = 0
        MEAN = 1
        AVG = 1
        MAX = 2
        MIN = 3
        PRODUCT = 4
    _ReduceOp.RedOpType = _ReduceOp
    dist.ReduceOp = getattr(dist, "ReduceOp", _ReduceOp)
    if not hasattr(dist.ReduceOp, "RedOpType"):
        dist.ReduceOp.RedOpType = dist.ReduceOp
    dist.GroupMember = getattr(dist, "GroupMember", type("GroupMember", (), {"WORLD": None}))
    dist.group = getattr(dist, "group", type("group", (), {"WORLD": None}))

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
    except Exception:
        const_mod.default_pg_timeout = getattr(const_mod, "default_pg_timeout", None)
    dist.constants = const_mod
    join_mod = _modules.get("torch.distributed.algorithms.join")
    if join_mod is None:
        join_mod = _types.ModuleType("torch.distributed.algorithms.join")
        _modules["torch.distributed.algorithms.join"] = join_mod
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
    join_mod.Join = Join
    join_mod.Joinable = Joinable
    join_mod.JoinHook = JoinHook
    dist.algorithms.join = join_mod

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
    dist.device_mesh.DeviceMesh = getattr(dist.device_mesh, "DeviceMesh", _DeviceMesh)
    dist.device_mesh.init_device_mesh = getattr(dist.device_mesh, "init_device_mesh", _init_device_mesh)
    dist.DeviceMesh = getattr(dist, "DeviceMesh", _DeviceMesh)
    dist.init_device_mesh = getattr(dist, "init_device_mesh", _init_device_mesh)
    dist.ProcessGroup = getattr(dist, "ProcessGroup", type("ProcessGroup", (), {
        "__init__": lambda self, *a, **k: None,
        "size": lambda self, *a, **k: 1,
        "rank": lambda self, *a, **k: 0,
    }))

    c10d = _modules.get("torch.distributed.distributed_c10d")
    if c10d is None:
        c10d = _types.ModuleType("torch.distributed.distributed_c10d")
        _modules["torch.distributed.distributed_c10d"] = c10d
    for name in dir(dist):
        if not name.startswith("__"):
            setattr(c10d, name, getattr(dist, name))
    for name in ("is_xccl_available", "is_nccl_available", "is_gloo_available",
                 "is_mpi_available", "is_ucc_available"):
        setattr(c10d, name, lambda *a, **k: False)
    c10d.ProcessGroup = dist.ProcessGroup
    c10d._get_default_group = lambda *a, **k: None
    c10d._get_default_store = lambda *a, **k: None
    c10d.Work = getattr(c10d, "Work", type("Work", (), {}))
    c10d.default_pg_timeout = getattr(c10d, "default_pg_timeout", None)
    dist.distributed_c10d = c10d

    rpc = _modules.get("torch.distributed.rpc")
    if rpc is None:
        rpc = _types.ModuleType("torch.distributed.rpc")
        _modules["torch.distributed.rpc"] = rpc
    rpc.is_available = lambda *a, **k: False
    rpc.init_rpc = lambda *a, **k: None
    rpc.shutdown = lambda *a, **k: None
    dist.rpc = rpc

    optim = _modules.get("torch.distributed.optim")
    if optim is None:
        optim = _types.ModuleType("torch.distributed.optim")
        _modules["torch.distributed.optim"] = optim
    dist.optim = optim

    dist.nn.all_reduce = lambda input, *a, **k: input
    _modules["torch.distributed.nn"] = dist.nn

    futures = _modules.get("torch.futures")
    if futures is None:
        futures = _types.ModuleType("torch.futures")
        _modules["torch.futures"] = futures
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
    futures.Future = Future
    g.futures = futures

    checkpoint = dist.checkpoint
    checkpoint.__path__ = getattr(checkpoint, "__path__", [])
    class FileSystemReader:
        def __init__(self, path, *a, **k):
            self.path = path
    class FileSystemWriter:
        def __init__(self, path, *a, **k):
            self.path = path
    checkpoint.FileSystemReader = FileSystemReader
    checkpoint.FileSystemWriter = FileSystemWriter
    checkpoint.load_state_dict = lambda state_dict, *a, **k: state_dict
    checkpoint.save_state_dict = lambda state_dict, *a, **k: state_dict
    checkpoint.load = lambda state_dict=None, *a, **k: state_dict
    checkpoint.save = lambda state_dict=None, *a, **k: state_dict
    checkpoint_sd = _types.ModuleType("torch.distributed.checkpoint.state_dict")
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
        if hasattr(model, "load_state_dict"):
            return model.load_state_dict(state_dict, strict=getattr(options, "strict", True))
        return None
    checkpoint_sd.StateDictOptions = StateDictOptions
    checkpoint_sd.get_model_state_dict = _get_model_state_dict
    checkpoint_sd.set_model_state_dict = _set_model_state_dict
    checkpoint_sd.get_state_dict = lambda model, optimizers=None, *a, **k: (
        _get_model_state_dict(model, *a, **k),
        optimizers.state_dict() if hasattr(optimizers, "state_dict") else {},
    )
    checkpoint_sd.set_state_dict = lambda model, optimizers=None, model_state_dict=None, optim_state_dict=None, *a, **k: (
        _set_model_state_dict(model, model_state_dict or {}, *a, **k)
    )
    checkpoint_fs = _types.ModuleType("torch.distributed.checkpoint.filesystem")
    checkpoint_fs.FileSystemReader = FileSystemReader
    checkpoint_fs.FileSystemWriter = FileSystemWriter
    checkpoint_fs.SerializationFormat = type("SerializationFormat", (), {
        "TORCH_SAVE": "torch_save",
        "SAFETENSORS": "safetensors",
    })
    checkpoint_fs._write_item = lambda *a, **k: None
    _modules["torch.distributed.checkpoint"] = checkpoint
    _modules["torch.distributed.checkpoint.state_dict"] = checkpoint_sd
    _modules["torch.distributed.checkpoint.filesystem"] = checkpoint_fs
    checkpoint.state_dict = checkpoint_sd
    checkpoint.filesystem = checkpoint_fs

    shard = dist._shard
    shard.__path__ = getattr(shard, "__path__", [])
    sharded_tensor = _types.ModuleType("torch.distributed._shard.sharded_tensor")
    class ShardedTensor:
        pass
    sharded_tensor.ShardedTensor = ShardedTensor
    sharded_tensor.init_from_local_shards = lambda shards, *a, **k: shards[0] if shards else None
    sharded_tensor.empty = lambda *a, **k: jt.empty(*a, **{kk: vv for kk, vv in k.items() if kk == "dtype"})
    shard.sharded_tensor = sharded_tensor
    _modules["torch.distributed._shard"] = shard
    _modules["torch.distributed._shard.sharded_tensor"] = sharded_tensor
    for _sub in ("api", "metadata", "reshard", "shard"):
        _m = _types.ModuleType("torch.distributed._shard.sharded_tensor." + _sub)
        _m.ShardedTensor = ShardedTensor
        _modules[_m.__name__] = _m

    class TCPStore:
        _data = {}
        def __init__(self, *a, **k):
            pass
        def set(self, key, value):
            self._data[str(key)] = value
        def get(self, key):
            return self._data.get(str(key), b"")
        def add(self, key, num):
            v = int(self._data.get(str(key), 0)) + int(num)
            self._data[str(key)] = v
            return v
        def wait(self, keys, *a, **k):
            return None
        def delete_key(self, key):
            self._data.pop(str(key), None)
            return True
    dist.TCPStore = TCPStore

    if not hasattr(g, "_C"):
        class _Accel:
            type = "cuda" if getattr(jt.compiler, "has_cuda", 0) else "cpu"
        class _CNS:
            @staticmethod
            def _get_accelerator():
                return _Accel()
        g._C = _CNS()
    _install_fsdp2_distributed(dist, g, registry=registry)
    g.distributed = dist


def install(ctx):
    _install_distributed(ctx.jittor_module, ctx.registry)
