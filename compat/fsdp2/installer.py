"""Install the FSDP2/DTensor compatibility modules into the torch shim."""

import sys
import types

from . import api, common, compat_types, config, dtensor, grad_sync, optimizer, shard, public_helpers


_INSTALL_MARKER = "_jittor_fsdp2_install_complete"
_MODULE_GRAPH_ATTR = "_jittor_fsdp2_module_graph"


def _ensure_module(registry, name, parent=None, attr=None):
    return registry.ensure(name)


def _install_wrap_helpers(fsdp_wrap_mod):
    for name in public_helpers.WRAP_HELPERS:
        setattr(fsdp_wrap_mod, name, getattr(public_helpers, name))
    fsdp_wrap_mod.ModuleWrapPolicy = compat_types.ModuleWrapPolicy
    fsdp_wrap_mod.CustomPolicy = compat_types.CustomPolicy


class _ModuleGraph:
    """Per-install binding ledger; it does not own any public API implementation."""
    def __init__(self, registry):
        self.registry = registry
        self.entries = []

    def __call__(self, name, parent=None, attr=None):
        installed = _ensure_module(self.registry, name, parent, attr)
        self.entries.append((name, installed))
        return installed


def _registry_for(torch_module, registry=None):
    from jittor.compat.torch.context import registry_for

    root_module = getattr(registry, "root_module", None)
    if root_module is None:
        root_module = (
            torch_module
            if isinstance(torch_module, types.ModuleType)
            else sys.modules.get("torch") or sys.modules.get("jittor")
        )
    if root_module is None:
        raise RuntimeError("torch root module is not installed")
    return registry_for(root_module, registry)


def install(dist, torch_module=None):
    return install_with_registry(dist, torch_module, registry=None)


def install_with_registry(dist, torch_module=None, registry=None):
    registry = _registry_for(torch_module, registry)
    registry.publish("torch.distributed", dist)

    if getattr(dist, _INSTALL_MARKER, False):
        installed_graph = getattr(dist, _MODULE_GRAPH_ATTR, None)
        if installed_graph is not None:
            for name, installed_module in installed_graph:
                registry.publish(name, installed_module)
            # Replaying the child module graph binds this package attribute to
            # the importable submodule again. The package API itself owns the
            # callable, while the submodule remains available in sys.modules.
            composable_fsdp = next(
                (installed_module for name, installed_module in installed_graph
                 if name == "torch.distributed._composable.fsdp"),
                None,
            )
            if composable_fsdp is not None:
                composable_fsdp.fully_shard = api.fully_shard
            return dist

    module = _ModuleGraph(registry)
    module_graph = module.entries

    tensor_mod = module("torch.distributed.tensor", dist, "tensor")
    tensor_legacy_mod = module(
        "torch.distributed._tensor", dist, "_tensor")
    tensor_api_mod = module("torch.distributed.tensor._api")
    tensor_placement_mod = module(
        "torch.distributed.tensor.placement_types")
    tensor_spec_mod = module(
        "torch.distributed.tensor._dtensor_spec")
    tensor_utils_mod = module("torch.distributed.tensor._utils")
    tensor_parallel_mod = module("torch.distributed.tensor.parallel")
    tensor_parallel_api_mod = module(
        "torch.distributed.tensor.parallel.api")
    tensor_parallel_style_mod = module(
        "torch.distributed.tensor.parallel.style")
    tensor_parallel_loss_mod = module(
        "torch.distributed.tensor.parallel.loss")
    device_mesh_mod = module(
        "torch.distributed.device_mesh", dist, "device_mesh")
    tensor_legacy_device_mesh_mod = module(
        "torch.distributed._tensor.device_mesh")
    # Where the mesh lives now that the namespace has lost its underscore. Both
    # spellings answer, because code written against either torch is in use.
    tensor_device_mesh_mod = module(
        "torch.distributed.tensor.device_mesh", tensor_mod, "device_mesh")
    fsdp_mod = module("torch.distributed.fsdp", dist, "fsdp")
    fsdp_api_mod = module("torch.distributed.fsdp.api")
    fsdp_full_mod = module(
        "torch.distributed.fsdp.fully_sharded_data_parallel")
    fsdp_wrap_mod = module("torch.distributed.fsdp.wrap")
    fsdp_traversal_mod = module(
        "torch.distributed.fsdp._traversal_utils")
    fsdp_runtime_mod = module(
        "torch.distributed.fsdp._runtime_utils")
    fsdp_top_common_mod = module(
        "torch.distributed.fsdp._common_utils")
    fsdp_state_mod = module("torch.distributed.fsdp._fsdp_state")
    fsdp_scaler_mod = module(
        "torch.distributed.fsdp.sharded_grad_scaler")
    fsdp_fully_pkg = module("torch.distributed.fsdp._fully_shard")
    fsdp_fully_mod = module(
        "torch.distributed.fsdp._fully_shard._fully_shard")
    fsdp_fully_api_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_api")
    fsdp_common_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_common")
    fsdp_init_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_init")
    fsdp_fully_state_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_state")
    fsdp_param_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_param")
    fsdp_collectives_mod = module(
        "torch.distributed.fsdp._fully_shard._fsdp_collectives")
    comp_mod = module(
        "torch.distributed._composable", dist, "_composable")
    comp_fsdp_mod = module(
        "torch.distributed._composable.fsdp", comp_mod, "fsdp")
    comp_fsdp_fully_mod = module(
        "torch.distributed._composable.fsdp.fully_shard")
    comp_fsdp_api_mod = module(
        "torch.distributed._composable.fsdp._fsdp_api")
    functional_collectives_mod = module(
        "torch.distributed._functional_collectives")
    algorithms_mod = module(
        "torch.distributed.algorithms", dist, "algorithms")
    checkpoint_mod = module(
        "torch.distributed.algorithms._checkpoint", algorithms_mod, "_checkpoint")
    checkpoint_wrapper_mod = module(
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

    exports = {}
    for owner, names in (
        (dtensor, (
            "DeviceMesh", "init_device_mesh", "DTensor", "Placement",
            "Replicate", "Shard", "Partial", "distribute_tensor",
            "distribute_module", "is_dtensor",
        )),
        (config, (
            "StateDictType", "ShardingStrategy", "BackwardPrefetch",
            "CPUOffload", "StateDictConfig", "OptimStateDictConfig",
            "MixedPrecision", "MixedPrecisionPolicy", "OffloadPolicy",
            "CPUOffloadPolicy", "NoOffloadPolicy", "DataParallelMeshDims",
            "FullStateDictConfig", "LocalStateDictConfig",
            "ShardedStateDictConfig", "FullOptimStateDictConfig",
            "LocalOptimStateDictConfig", "ShardedOptimStateDictConfig",
            "StateDictSettings", "OptimStateKeyType", "FlatParameter",
            "UnshardHandle",
        )),
        (api, (
            "FSDPModule", "FullyShardedDataParallel", "ShardedGradScaler",
            "fully_shard", "register_fsdp_forward_method", "share_comm_ctx",
        )),
        (grad_sync, ("sync_sharded_grads",)),
        (optimizer, ("sharded_sgd_step", "local_sharded_state_dict")),
    ):
        exports.update((name, getattr(owner, name)) for name in names)
    for mod in (fsdp_mod, fsdp_api_mod, fsdp_full_mod, fsdp_fully_pkg,
                fsdp_fully_mod, fsdp_fully_api_mod, comp_fsdp_mod,
                comp_fsdp_fully_mod, comp_fsdp_api_mod):
        for k, v in exports.items():
            setattr(mod, k, v)
    fsdp_mod.FSDP = exports["FullyShardedDataParallel"]
    fsdp_full_mod.FSDP = exports["FullyShardedDataParallel"]
    fsdp_scaler_mod.ShardedGradScaler = exports["ShardedGradScaler"]
    fsdp_traversal_mod._get_fsdp_states = public_helpers._get_fsdp_states
    fsdp_traversal_mod._get_fsdp_handles = public_helpers._get_fsdp_handles
    for mod in (fsdp_runtime_mod, fsdp_top_common_mod, fsdp_state_mod,
                fsdp_common_mod, fsdp_fully_state_mod):
        mod.FSDPState = compat_types.FSDPState
        mod.TrainingState = compat_types.TrainingState
        mod.FSDP_WRAPPED_MODULE = compat_types.FSDP_WRAPPED_MODULE
        mod._lazy_init = compat_types._lazy_init
        mod._get_module_fsdp_state = compat_types._get_module_fsdp_state
        mod._get_fsdp_state = compat_types._get_module_fsdp_state
        mod._get_module_fsdp_state_if_fully_sharded_module = (
            compat_types._get_module_fsdp_state_if_fully_sharded_module)
        mod._is_fsdp_managed_module = compat_types._is_fsdp_managed_module
    fsdp_param_mod.FlatParameter = exports["FlatParameter"]
    fsdp_collectives_mod.all_gather = public_helpers.all_gather
    fsdp_collectives_mod.reduce_scatter = public_helpers.reduce_scatter
    _install_wrap_helpers(fsdp_wrap_mod)

    tensor_factories = {
        name: getattr(dtensor, name)
        for name in (
            "empty", "ones", "zeros", "full", "rand", "randn", "linspace",
            "logspace",
        )
    }
    for mod in (tensor_mod, tensor_legacy_mod, tensor_api_mod, tensor_placement_mod,
                tensor_utils_mod):
        for k in ("DTensor", "Placement", "Replicate", "Shard", "Partial",
                  "DeviceMesh", "init_device_mesh", "distribute_tensor",
                  "distribute_module", "is_dtensor"):
            setattr(mod, k, exports[k])
        for k, v in tensor_factories.items():
            setattr(mod, k, v)
    tensor_spec_mod.DTensorSpec = compat_types.DTensorSpec
    tensor_utils_mod.compute_local_shape_and_global_offset = (
        compat_types.compute_local_shape_and_global_offset)
    tensor_mod.placement_types = tensor_placement_mod
    tensor_mod._api = tensor_api_mod
    tensor_mod._dtensor_spec = tensor_spec_mod
    tensor_mod._utils = tensor_utils_mod
    tensor_mod.parallel = tensor_parallel_mod
    tensor_mod.DeviceMesh = exports["DeviceMesh"]
    tensor_legacy_mod.device_mesh = tensor_legacy_device_mesh_mod
    for mod in (device_mesh_mod, tensor_legacy_device_mesh_mod,
                tensor_device_mesh_mod):
        mod.DeviceMesh = exports["DeviceMesh"]
        mod.init_device_mesh = exports["init_device_mesh"]
    parallel_classes = {
        name: getattr(public_helpers, name) for name in public_helpers.PARALLEL_STYLES
    }
    for mod in (tensor_parallel_mod, tensor_parallel_style_mod):
        mod.ParallelStyle = compat_types.ParallelStyle
        for name, cls in parallel_classes.items():
            setattr(mod, name, cls)
    for mod in (tensor_parallel_mod, tensor_parallel_api_mod):
        mod.parallelize_module = compat_types.parallelize_module
    for mod in (tensor_parallel_mod, tensor_parallel_loss_mod):
        mod.loss_parallel = compat_types.loss_parallel
    tensor_parallel_mod.api = tensor_parallel_api_mod
    tensor_parallel_mod.style = tensor_parallel_style_mod
    tensor_parallel_mod.loss = tensor_parallel_loss_mod

    device_mesh_mod.DeviceMesh = exports["DeviceMesh"]
    device_mesh_mod.init_device_mesh = exports["init_device_mesh"]
    dist.DeviceMesh = exports["DeviceMesh"]
    dist.init_device_mesh = exports["init_device_mesh"]
    dist.is_available = public_helpers.is_available
    functional_collectives_mod.AsyncCollectiveTensor = public_helpers.AsyncCollectiveTensor

    checkpoint_wrapper_mod.checkpoint_wrapper = compat_types._checkpoint_wrapper
    checkpoint_wrapper_mod.apply_activation_checkpointing = (
        compat_types._apply_activation_checkpointing)
    checkpoint_wrapper_mod.offload_wrapper = public_helpers.offload_wrapper
    checkpoint_wrapper_mod._CHECKPOINT_PREFIX = "_checkpoint_wrapped_module."
    checkpoint_wrapper_mod.CheckpointImpl = public_helpers.CheckpointImpl
    checkpoint_wrapper_mod.checkpoint = compat_types._checkpoint
    fsdp_common_mod.FSDPMeshInfo = compat_types.FSDPMeshInfo
    fsdp_common_mod.ShardPlacementResult = compat_types.ShardPlacementResult
    fsdp_init_mod._get_mesh_info = compat_types._get_mesh_info
    fsdp_init_mod._get_post_forward_mesh_info = compat_types._get_post_forward_mesh_info
    public_helpers.register_helper_fidelity()
    setattr(dist, _INSTALL_MARKER, True)
    setattr(dist, _MODULE_GRAPH_ATTR, tuple(module_graph))
    registry.publish("torch.distributed", dist)
    return dist


_install_fsdp2_distributed = install


_EXPORTS = (
    "_ensure_module", "_install_wrap_helpers", "install",
    "_install_fsdp2_distributed",
)
