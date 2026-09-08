"""Lightweight FSDP, tensor-parallel, and checkpoint compatibility types."""

import contextlib
import enum


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
    return FSDPMeshInfo(
        mesh=mesh,
        shard_mesh_dim=getattr(dp_mesh_dims, "shard", None),
        replicate_mesh_dim=getattr(dp_mesh_dims, "replicate", None),
    )


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


_EXPORTS = (
    "FSDPMeshInfo",
    "ShardPlacementResult",
    "_get_mesh_info",
    "FSDPState",
    "TrainingState",
    "FSDP_WRAPPED_MODULE",
    "DTensorSpec",
    "_get_module_fsdp_state",
    "_get_module_fsdp_state_if_fully_sharded_module",
    "_is_fsdp_managed_module",
    "_lazy_init",
    "_get_post_forward_mesh_info",
    "compute_local_shape_and_global_offset",
    "ParallelStyle",
    "parallelize_module",
    "loss_parallel",
    "_checkpoint_wrapper",
    "_apply_activation_checkpointing",
    "_checkpoint",
    "ModuleWrapPolicy",
    "CustomPolicy",
)
