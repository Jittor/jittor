"""Torch FSDP2 and DTensor compatibility for Jittor."""

import contextlib
import enum
import os
import sys
import types

import numpy as np

import jittor as jt
from jittor import nn

from . import common as _common
from . import compat_types as _compat_types
from . import config as _config
from . import dtensor as _dtensor
from . import shard as _shard
from . import grad_sync as _grad_sync
from . import optimizer as _optimizer
from . import api as _api
from . import installer as _installer


_IMPLEMENTATION_MODULES = (
    _common,
    _dtensor,
    _config,
    _shard,
    _grad_sync,
    _optimizer,
    _api,
    _compat_types,
    _installer,
)
_export_owners = {}
for _module in _IMPLEMENTATION_MODULES:
    for _name in _module._EXPORTS:
        if _name in _export_owners:
            raise RuntimeError(
                "duplicate FSDP2 export %r from %s and %s"
                % (_name, _export_owners[_name].__name__, _module.__name__)
            )
        globals()[_name] = getattr(_module, _name)
        _export_owners[_name] = _module


__all__ = (
    "BackwardPrefetch", "CPUOffload", "CPUOffloadPolicy", "CustomPolicy",
    "DTensor", "DTensorSpec", "DataParallelMeshDims", "DeviceMesh",
    "FSDPMeshInfo", "FSDPModule", "FSDPState", "FSDP_WRAPPED_MODULE",
    "FlatParameter", "FullOptimStateDictConfig", "FullStateDictConfig",
    "FullyShardedDataParallel", "LocalOptimStateDictConfig",
    "LocalStateDictConfig", "MixedPrecision", "MixedPrecisionPolicy",
    "ModuleWrapPolicy", "NoOffloadPolicy", "OffloadPolicy",
    "OptimStateDictConfig", "OptimStateKeyType", "ParallelStyle", "Partial",
    "Placement", "Replicate", "Shard", "ShardPlacementResult",
    "ShardedGradScaler", "ShardedOptimStateDictConfig",
    "ShardedStateDictConfig", "ShardingStrategy", "StateDictConfig",
    "StateDictSettings", "StateDictType", "TrainingState", "UnshardHandle",
    "clear_fsdp_optimizer_grads", "collect_fsdp_full_params_for_backward",
    "compute_local_shape_and_global_offset", "contextlib", "distribute_module",
    "distribute_tensor", "empty", "enum", "fill_fsdp_optimizer_grads_from_grad_map",
    "full", "fully_shard", "init_device_mesh", "install", "is_dtensor",
    "is_fsdp_managed_param", "jt", "linspace", "local_sharded_state_dict",
    "logspace", "loss_parallel", "nn", "np", "ones",
    "optimizer_has_fsdp_params", "optimizer_has_non_fsdp_params",
    "optimizer_step", "os", "parallelize_module", "rand", "randn",
    "refresh_optimizer_fsdp_params", "refresh_visible_full_grads",
    "register_fsdp_forward_method", "sharded_sgd_step", "share_comm_ctx",
    "sync_sharded_grads", "sys", "types", "zeros",
)


_current_module = sys.modules[__name__]
_legacy_name = "jittor.torch_fsdp2_compat"
_legacy_module = sys.modules.get(_legacy_name)
if _legacy_module is not None and _legacy_module is not _current_module:
    raise RuntimeError("legacy FSDP2 module name is already bound to another object")
sys.modules[_legacy_name] = _current_module
setattr(sys.modules["jittor"], "torch_fsdp2_compat", _current_module)

_SUBMODULE_NAMES = (
    "api", "common", "compat_types", "config", "dtensor", "grad_sync",
    "installer", "optimizer", "shard",
)
for _submodule_name in _SUBMODULE_NAMES:
    globals().pop(_submodule_name, None)


def __getattr__(name):
    if name in _SUBMODULE_NAMES:
        return sys.modules[__name__ + "." + name]
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


del _legacy_module, _module, _name, _submodule_name
