"""Install the FSDP2/DTensor compatibility modules into the torch shim."""

from .runtime import facade, preserve_facade_origins


def _ensure_module(name, parent=None, attr=None):
    mod = facade.sys.modules.get(name)
    if mod is None:
        mod = facade.types.ModuleType(name)
        facade.sys.modules[name] = mod
    if parent is not None and attr:
        setattr(parent, attr, mod)
    return mod


def _install_wrap_helpers(fsdp_wrap_mod):
    @facade.contextlib.contextmanager
    def enable_wrap(*args, **kwargs):
        yield

    def wrap(module, *args, **kwargs):
        return facade.fully_shard(module, **{
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
    fsdp_wrap_mod.ModuleWrapPolicy = facade.ModuleWrapPolicy
    fsdp_wrap_mod.CustomPolicy = facade.CustomPolicy
    fsdp_wrap_mod._or_policy = (
        lambda module, recurse, nonwrapped_numel, policies=None, *a, **k:
        any(policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)
            for policy in (policies or ()) if callable(policy)))
    preserve_facade_origins((
        fsdp_wrap_mod.enable_wrap,
        fsdp_wrap_mod.wrap,
        fsdp_wrap_mod.always_wrap_policy,
        fsdp_wrap_mod.size_based_auto_wrap_policy,
        fsdp_wrap_mod.transformer_auto_wrap_policy,
        fsdp_wrap_mod.lambda_auto_wrap_policy,
        fsdp_wrap_mod._or_policy,
    ))


def install(dist, torch_module=None):
    tensor_mod = facade._ensure_module("torch.distributed.tensor", dist, "tensor")
    tensor_legacy_mod = facade._ensure_module(
        "torch.distributed._tensor", dist, "_tensor")
    tensor_api_mod = facade._ensure_module("torch.distributed.tensor._api")
    tensor_placement_mod = facade._ensure_module(
        "torch.distributed.tensor.placement_types")
    tensor_spec_mod = facade._ensure_module(
        "torch.distributed.tensor._dtensor_spec")
    tensor_utils_mod = facade._ensure_module("torch.distributed.tensor._utils")
    tensor_parallel_mod = facade._ensure_module("torch.distributed.tensor.parallel")
    tensor_parallel_api_mod = facade._ensure_module(
        "torch.distributed.tensor.parallel.api")
    tensor_parallel_style_mod = facade._ensure_module(
        "torch.distributed.tensor.parallel.style")
    tensor_parallel_loss_mod = facade._ensure_module(
        "torch.distributed.tensor.parallel.loss")
    device_mesh_mod = facade._ensure_module(
        "torch.distributed.device_mesh", dist, "device_mesh")
    tensor_legacy_device_mesh_mod = facade._ensure_module(
        "torch.distributed._tensor.device_mesh")
    fsdp_mod = facade._ensure_module("torch.distributed.fsdp", dist, "fsdp")
    fsdp_api_mod = facade._ensure_module("torch.distributed.fsdp.api")
    fsdp_full_mod = facade._ensure_module(
        "torch.distributed.fsdp.fully_sharded_data_parallel")
    fsdp_wrap_mod = facade._ensure_module("torch.distributed.fsdp.wrap")
    fsdp_traversal_mod = facade._ensure_module(
        "torch.distributed.fsdp._traversal_utils")
    fsdp_runtime_mod = facade._ensure_module(
        "torch.distributed.fsdp._runtime_utils")
    fsdp_top_common_mod = facade._ensure_module(
        "torch.distributed.fsdp._common_utils")
    fsdp_state_mod = facade._ensure_module("torch.distributed.fsdp._fsdp_state")
    fsdp_scaler_mod = facade._ensure_module(
        "torch.distributed.fsdp.sharded_grad_scaler")
    fsdp_fully_pkg = facade._ensure_module("torch.distributed.fsdp._fully_shard")
    fsdp_fully_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fully_shard")
    fsdp_fully_api_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_api")
    fsdp_common_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_common")
    fsdp_init_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_init")
    fsdp_fully_state_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_state")
    fsdp_param_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_param")
    fsdp_collectives_mod = facade._ensure_module(
        "torch.distributed.fsdp._fully_shard._fsdp_collectives")
    comp_mod = facade._ensure_module(
        "torch.distributed._composable", dist, "_composable")
    comp_fsdp_mod = facade._ensure_module(
        "torch.distributed._composable.fsdp", comp_mod, "fsdp")
    comp_fsdp_fully_mod = facade._ensure_module(
        "torch.distributed._composable.fsdp.fully_shard")
    comp_fsdp_api_mod = facade._ensure_module(
        "torch.distributed._composable.fsdp._fsdp_api")
    functional_collectives_mod = facade._ensure_module(
        "torch.distributed._functional_collectives")
    algorithms_mod = facade._ensure_module(
        "torch.distributed.algorithms", dist, "algorithms")
    checkpoint_mod = facade._ensure_module(
        "torch.distributed.algorithms._checkpoint", algorithms_mod, "_checkpoint")
    checkpoint_wrapper_mod = facade._ensure_module(
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

    export_names = (
        "DeviceMesh", "init_device_mesh", "DTensor", "Placement", "Replicate",
        "Shard", "Partial", "distribute_tensor", "distribute_module",
        "is_dtensor", "StateDictType", "ShardingStrategy", "BackwardPrefetch",
        "CPUOffload", "StateDictConfig", "OptimStateDictConfig",
        "MixedPrecision", "MixedPrecisionPolicy", "OffloadPolicy",
        "CPUOffloadPolicy", "NoOffloadPolicy", "DataParallelMeshDims",
        "FullStateDictConfig", "LocalStateDictConfig", "ShardedStateDictConfig",
        "FullOptimStateDictConfig", "LocalOptimStateDictConfig",
        "ShardedOptimStateDictConfig", "StateDictSettings", "OptimStateKeyType",
        "FlatParameter", "UnshardHandle", "FSDPModule",
        "FullyShardedDataParallel", "ShardedGradScaler", "fully_shard",
        "register_fsdp_forward_method", "share_comm_ctx", "sync_sharded_grads",
        "sharded_sgd_step", "local_sharded_state_dict",
    )
    exports = {name: getattr(facade, name) for name in export_names}
    for mod in (fsdp_mod, fsdp_api_mod, fsdp_full_mod, fsdp_fully_pkg,
                fsdp_fully_mod, fsdp_fully_api_mod, comp_fsdp_mod,
                comp_fsdp_fully_mod, comp_fsdp_api_mod):
        for k, v in exports.items():
            setattr(mod, k, v)
    fsdp_mod.FSDP = exports["FullyShardedDataParallel"]
    fsdp_full_mod.FSDP = exports["FullyShardedDataParallel"]
    fsdp_scaler_mod.ShardedGradScaler = exports["ShardedGradScaler"]
    fsdp_traversal_mod._get_fsdp_states = (
        lambda module: [
            getattr(m, "_fsdp_state")
            for m in facade._iter_fsdp_modules(module, True)
            if hasattr(m, "_fsdp_state")
        ])
    fsdp_traversal_mod._get_fsdp_handles = lambda module: []
    for mod in (fsdp_runtime_mod, fsdp_top_common_mod, fsdp_state_mod,
                fsdp_common_mod, fsdp_fully_state_mod):
        mod.FSDPState = facade.FSDPState
        mod.TrainingState = facade.TrainingState
        mod.FSDP_WRAPPED_MODULE = facade.FSDP_WRAPPED_MODULE
        mod._lazy_init = facade._lazy_init
        mod._get_module_fsdp_state = facade._get_module_fsdp_state
        mod._get_fsdp_state = facade._get_module_fsdp_state
        mod._get_module_fsdp_state_if_fully_sharded_module = (
            facade._get_module_fsdp_state_if_fully_sharded_module)
        mod._is_fsdp_managed_module = facade._is_fsdp_managed_module
    fsdp_param_mod.FlatParameter = exports["FlatParameter"]
    fsdp_collectives_mod.all_gather = lambda tensor, *a, **k: (
        facade._all_gather_shards(tensor)
        if facade._in_true_distributed() else tensor)
    fsdp_collectives_mod.reduce_scatter = lambda tensor, *a, **k: (
        facade._reduce_scatter_padded(tensor)
        if facade._in_true_distributed() else tensor)
    facade._install_wrap_helpers(fsdp_wrap_mod)

    tensor_factories = {
        name: getattr(facade, name)
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
    tensor_spec_mod.DTensorSpec = facade.DTensorSpec
    tensor_utils_mod.compute_local_shape_and_global_offset = (
        facade.compute_local_shape_and_global_offset)
    tensor_mod.placement_types = tensor_placement_mod
    tensor_mod._api = tensor_api_mod
    tensor_mod._dtensor_spec = tensor_spec_mod
    tensor_mod._utils = tensor_utils_mod
    tensor_mod.parallel = tensor_parallel_mod
    tensor_mod.DeviceMesh = exports["DeviceMesh"]
    tensor_legacy_mod.device_mesh = tensor_legacy_device_mesh_mod
    for mod in (device_mesh_mod, tensor_legacy_device_mesh_mod):
        mod.DeviceMesh = exports["DeviceMesh"]
        mod.init_device_mesh = exports["init_device_mesh"]
    parallel_classes = {}
    for name in ("ColwiseParallel", "RowwiseParallel", "SequenceParallel",
                 "PrepareModuleInput", "PrepareModuleOutput",
                 "PrepareModuleInputOutput"):
        parallel_classes[name] = type(name, (facade.ParallelStyle,), {})
    for mod in (tensor_parallel_mod, tensor_parallel_style_mod):
        mod.ParallelStyle = facade.ParallelStyle
        for name, cls in parallel_classes.items():
            setattr(mod, name, cls)
    for mod in (tensor_parallel_mod, tensor_parallel_api_mod):
        mod.parallelize_module = facade.parallelize_module
    for mod in (tensor_parallel_mod, tensor_parallel_loss_mod):
        mod.loss_parallel = facade.loss_parallel
    tensor_parallel_mod.api = tensor_parallel_api_mod
    tensor_parallel_mod.style = tensor_parallel_style_mod
    tensor_parallel_mod.loss = tensor_parallel_loss_mod

    device_mesh_mod.DeviceMesh = exports["DeviceMesh"]
    device_mesh_mod.init_device_mesh = exports["init_device_mesh"]
    dist.DeviceMesh = exports["DeviceMesh"]
    dist.init_device_mesh = exports["init_device_mesh"]
    dist.is_available = lambda *a, **k: True
    class AsyncCollectiveTensor:
        def __init__(self, tensor=None):
            self.tensor = tensor
        def wait(self):
            return self.tensor
        def __getattr__(self, name):
            return getattr(self.tensor, name)
    functional_collectives_mod.AsyncCollectiveTensor = AsyncCollectiveTensor

    checkpoint_wrapper_mod.checkpoint_wrapper = facade._checkpoint_wrapper
    checkpoint_wrapper_mod.apply_activation_checkpointing = (
        facade._apply_activation_checkpointing)
    checkpoint_wrapper_mod.offload_wrapper = lambda module, *a, **k: module
    checkpoint_wrapper_mod._CHECKPOINT_PREFIX = "_checkpoint_wrapped_module."
    checkpoint_wrapper_mod.CheckpointImpl = facade.enum.Enum(
        "CheckpointImpl", {"NO_REENTRANT": "no_reentrant", "REENTRANT": "reentrant"})
    checkpoint_wrapper_mod.checkpoint = facade._checkpoint
    preserve_facade_origins((
        *parallel_classes.values(),
        fsdp_traversal_mod._get_fsdp_states,
        fsdp_traversal_mod._get_fsdp_handles,
        fsdp_collectives_mod.all_gather,
        fsdp_collectives_mod.reduce_scatter,
        fsdp_wrap_mod.enable_wrap,
        fsdp_wrap_mod.wrap,
        fsdp_wrap_mod.always_wrap_policy,
        fsdp_wrap_mod.size_based_auto_wrap_policy,
        fsdp_wrap_mod.transformer_auto_wrap_policy,
        fsdp_wrap_mod.lambda_auto_wrap_policy,
        fsdp_wrap_mod._or_policy,
        dist.is_available,
        AsyncCollectiveTensor,
        checkpoint_wrapper_mod.offload_wrapper,
        checkpoint_wrapper_mod.CheckpointImpl,
    ))
    fsdp_common_mod.FSDPMeshInfo = facade.FSDPMeshInfo
    fsdp_common_mod.ShardPlacementResult = facade.ShardPlacementResult
    fsdp_init_mod._get_mesh_info = facade._get_mesh_info
    fsdp_init_mod._get_post_forward_mesh_info = facade._get_post_forward_mesh_info
    if torch_module is not None:
        try:
            torch_module["distributed"] = dist
        except Exception:
            setattr(torch_module, "distributed", dist)
    return dist


_install_fsdp2_distributed = install


FACADE_EXPORTS = (
    "_ensure_module", "_install_wrap_helpers", "install",
    "_install_fsdp2_distributed",
)
preserve_facade_origins(
    tuple(globals()[name] for name in FACADE_EXPORTS if callable(globals()[name]))
)
