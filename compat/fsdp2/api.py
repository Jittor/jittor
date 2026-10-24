"""FSDP module API implementations exposed by the canonical package."""

import contextlib
import types

import jittor as jt
from jittor import nn
from jittor._core.dtypes import dtype_name

from . import common, config, dtensor, grad_sync, optimizer, shard
from ..diagnostics import EXPECTED, swallowed
from ..torch.types import device as _compat_device


class FSDPModule:
    """Public physical mixin used by ``fully_shard`` and its real subclasses.

    The native compatibility API retains its import/pickle owner. The Torch
    installation constructs its own physical public mixin from these methods;
    a private state flag must not turn arbitrary objects into either class.
    """

    def unshard(self, async_op=False):
        shard._unshard_module_params(self)
        return config.UnshardHandle(self) if async_op else None

    def reshard(self):
        shard._reshard_module_params(self)
        return None

    def reset_iter_state(self):
        return shard._apply_fsdp_attr(self, "iter_state_reset", True, False)

    def set_reshard_after_forward(self, value, recurse=True):
        return shard._apply_fsdp_attr(self, "reshard_after_forward", value, recurse)

    def set_requires_gradient_sync(self, value, recurse=True):
        shard._apply_fsdp_attr(self, "requires_all_reduce", bool(value), recurse)
        return shard._apply_fsdp_attr(self, "requires_gradient_sync", bool(value), recurse)

    def set_requires_all_reduce(self, value, recurse=True):
        if not value:
            for module in shard._iter_fsdp_modules(self, recurse):
                state = getattr(module, "_fsdp_state", None)
                replicate = getattr(state, "replicate_group", None)
                if replicate is not None and replicate.size() > 1:
                    raise NotImplementedError(
                        "FSDP2 set_requires_all_reduce(False) with a replicate "
                        "mesh requires partial-gradient accumulation; silently "
                        "omitting all-reduce would update replica weights differently")
        return shard._apply_fsdp_attr(self, "requires_all_reduce", bool(value), recurse)

    def set_all_reduce_hook(self, hook, *, stream=None):
        shard._apply_fsdp_attr(self, "all_reduce_hook", hook, False)
        return shard._apply_fsdp_attr(self, "all_reduce_hook_stream", stream, False)

    def set_allocate_memory_from_process_group_for_comm(self, enable):
        return shard._apply_fsdp_attr(
            self, "allocate_memory_from_process_group_for_comm", bool(enable), False)

    def set_custom_all_gather(self, comm):
        return shard._apply_fsdp_attr(self, "custom_all_gather", comm, False)

    def set_custom_reduce_scatter(self, comm):
        return shard._apply_fsdp_attr(self, "custom_reduce_scatter", comm, False)

    def set_reduce_scatter_unused_params(self, value=True):
        return shard._apply_fsdp_attr(
            self, "reduce_scatter_unused_params", bool(value), False)

    def set_reduce_scatter_max_input_buffers(self, value):
        return shard._apply_fsdp_attr(
            self, "reduce_scatter_max_input_buffers", value, False)

    def set_separate_reduce_scatter_group(self, group):
        return shard._apply_fsdp_attr(
            self, "separate_reduce_scatter_group", group, False)

    def set_is_last_backward(self, value, recurse=True):
        return shard._apply_fsdp_attr(self, "is_last_backward", bool(value), recurse)

    def set_reshard_after_backward(self, value, recurse=True):
        return shard._apply_fsdp_attr(self, "reshard_after_backward", value, recurse)

    def set_unshard_in_backward(self, value, recurse=True):
        return shard._apply_fsdp_attr(self, "unshard_in_backward", bool(value), recurse)

    def set_modules_to_forward_prefetch(self, modules):
        return shard._apply_fsdp_attr(
            self, "modules_to_forward_prefetch", list(modules or ()), False)

    def set_modules_to_backward_prefetch(self, modules):
        return shard._apply_fsdp_attr(
            self, "modules_to_backward_prefetch", list(modules or ()), False)

    def set_gradient_divide_factor(self, factor, recurse=True):
        return shard._apply_fsdp_attr(self, "gradient_divide_factor", factor, recurse)

    def set_reduce_scatter_divide_factor(self, factor, recurse=True):
        return shard._apply_fsdp_attr(
            self, "reduce_scatter_divide_factor", factor, recurse)

    def set_force_sum_reduction_for_comms(self, value, recurse=True):
        return shard._apply_fsdp_attr(
            self, "force_sum_reduction_for_comms", bool(value), recurse)

    def set_symm_mem_for_comm(self, backend="NCCL", recurse=True):
        return shard._apply_fsdp_attr(self, "symm_mem_for_comm", backend, recurse)

    def set_post_optim_event(self, event, recurse=True):
        return shard._apply_fsdp_attr(self, "post_optim_event", event, recurse)

    def _get_fsdp_state(self):
        return getattr(self, "_fsdp_state", None)

    def sync_sharded_grads(self, loss, *, divide_by_world_size=True):
        return grad_sync.sync_sharded_grads(
            self, loss, divide_by_world_size=divide_by_world_size)

    def sharded_sgd_step(self, loss, lr=1e-4, *, divide_by_world_size=True):
        return optimizer.sharded_sgd_step(
            self, loss, lr=lr, divide_by_world_size=divide_by_world_size)

    def local_sharded_state_dict(self):
        return optimizer.local_sharded_state_dict(self)


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
    # State flags are not Python inheritance. Publish the real public mixin and
    # fail explicitly if a class cannot be composed or reassigned.
    owner = getattr(type(module), "_nn_frontend_owner", None)
    marker = FSDPModule
    if owner is not None:
        marker = owner.fsdp_module_type
    if marker in type(module).__mro__:
        return module
    cache = getattr(marker, "_jittor_class_cache", None)
    if cache is None:
        cache = marker._jittor_class_cache = {}
    cls = type(module)
    fsdp_cls = cache.get(cls)
    if fsdp_cls is None:
        fsdp_cls = type("FSDP" + cls.__name__, (marker, cls),
                        {"__module__": "torch.distributed.fsdp._fully_shard._fully_shard"})
        cache[cls] = fsdp_cls
    module.__class__ = fsdp_cls
    return module


def _reject_unsupported_mesh(mesh, dp_mesh_dims=None):
    """Validate the actual sharding/replication axes before mutating a module."""
    if mesh is None:
        return
    if not isinstance(mesh, dtensor.DeviceMesh):
        raise TypeError("fully_shard mesh must be a DeviceMesh")
    if mesh.get_coordinate() is None:
        raise RuntimeError("fully_shard called on a rank outside its mesh")
    if mesh.ndim > 2 and dp_mesh_dims is None:
        raise ValueError("fully_shard needs explicit dp_mesh_dims for more than two axes")
    if dp_mesh_dims is not None:
        shard_names = dp_mesh_dims.shard_names
        replicate_names = dp_mesh_dims.replicate_names
        names = tuple(mesh.mesh_dim_names or ())
        selected = shard_names + replicate_names
        if not shard_names or len(set(selected)) != len(selected) or set(selected) != set(names):
            raise ValueError("dp_mesh_dims must partition mesh names into shard and replicate axes")


def _mesh_groups(mesh, dp_mesh_dims):
    if dp_mesh_dims is not None:
        shard_mesh = mesh[dp_mesh_dims.shard_names]._flatten()
        replicate_mesh = (mesh[dp_mesh_dims.replicate_names]._flatten()
                          if dp_mesh_dims.replicate_names else None)
        return shard_mesh.get_group(), (replicate_mesh.get_group()
                                       if replicate_mesh is not None else None)
    if mesh.ndim == 1:
        return mesh.get_group(), None
    return mesh.get_group(1), mesh.get_group(0)

def fully_shard(module, *, mesh=None, reshard_after_forward=True,
                shard_placement_fn=None, mp_policy=None, offload_policy=None,
                ignored_params=None, dp_mesh_dims=None, **kwargs):
    if isinstance(module, (list, tuple)):
        for m in module:
            fully_shard(
                m, mesh=mesh, reshard_after_forward=reshard_after_forward,
                shard_placement_fn=shard_placement_fn, mp_policy=mp_policy,
                offload_policy=offload_policy, ignored_params=ignored_params,
                dp_mesh_dims=dp_mesh_dims, **kwargs)
        return module
    if module is None or not hasattr(module, "parameters"):
        raise TypeError("fully_shard() expects a torch.nn.Module-compatible object")
    _reject_unsupported_mesh(mesh, dp_mesh_dims)
    st = getattr(module, "_fsdp_state", None)
    if st is not None and getattr(st, "true_fsdp_initialized", False) and mesh is not None:
        previous = st.mesh
        if previous.shape != mesh.shape or previous.mesh.tolist() != mesh.mesh.tolist():
            raise ValueError("fully_shard cannot change an initialized parameter mesh")
    if st is None:
        st = common.StateRecord()
        object.__setattr__(module, "_fsdp_state", st)
    st.mesh = mesh or getattr(st, "mesh", None) or dtensor.init_device_mesh(
        "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (common._world_size(),))
    shard_group, replicate_group = _mesh_groups(st.mesh, dp_mesh_dims)
    if getattr(st, "true_fsdp_initialized", False):
        previous_shard = st.shard_group
        previous_replicate = st.replicate_group
        if (previous_shard.ranks != shard_group.ranks or
                getattr(previous_replicate, "ranks", ()) != getattr(replicate_group, "ranks", ())):
            raise ValueError("fully_shard cannot change initialized sharding axes")
    st.shard_group, st.replicate_group = shard_group, replicate_group
    st.reshard_after_forward = reshard_after_forward
    st.shard_placement_fn = shard_placement_fn
    st.mp_policy = (
        mp_policy if mp_policy is not None else config.MixedPrecisionPolicy())
    st.offload_policy = (
        offload_policy if offload_policy is not None else config.NoOffloadPolicy())
    st.ignored_params = tuple(ignored_params or ())
    st.dp_mesh_dims = dp_mesh_dims
    st.kwargs = dict(kwargs)
    if common._in_true_distributed() and not getattr(st, "true_fsdp_initialized", False):
        target_type = st.mesh.device_type
        cuda_index = int(jt.current_device()) if target_type == "cuda" else None
        target_device = ("cuda:%d" % cuda_index
                         if target_type == "cuda" else target_type)
        ignored_ids = {id(param) for param in st.ignored_params}
        visited_modules = set()
        visited_vars = set()
        pending = [module]

        def move_managed(var):
            if str(var.device) == target_device:
                return
            if target_type == "cuda":
                moved = var.cuda(cuda_index)
            elif target_type == "cpu":
                moved = var.cpu()
            else:
                moved = var.to(target_type)
            optimizer._assign_preserve_trainability(var, moved)

        while pending:
            current = pending.pop()
            if id(current) in visited_modules:
                continue
            visited_modules.add(id(current))
            if current is not module and getattr(current, "_is_fsdp_module", False):
                continue
            for _, param in current.named_parameters(recurse=False):
                identity = id(param)
                if (identity in visited_vars or identity in ignored_ids or
                        shard.is_fsdp_managed_param(param)):
                    continue
                visited_vars.add(identity)
                move_managed(param)
            for _, buffer in current.named_buffers(recurse=False):
                identity = id(buffer)
                if identity in visited_vars:
                    continue
                visited_vars.add(identity)
                move_managed(buffer)
            pending.extend(child for _, child in current.named_children())
    object.__setattr__(module, "_is_fsdp_module", True)
    object.__setattr__(module, "_is_fsdp_managed_module", True)
    object.__setattr__(module, "_fsdp_use_orig_params", True)
    _inject_fsdp_methods(module)
    shard._init_true_fsdp_state(module, st)
    shard._install_true_fsdp_execute(module)
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
            shard._apply_fsdp_attr(module, "share_comm_ctx", True, False)
    yield


def _fsdp1_check_precision(module, precision):
    """Only accept policies that need no parameter/reduction/buffer cast.

    FSDP2 currently records its policy but does not implement mixed-precision
    storage or reduction. Mapping a spelling must not pretend otherwise.
    """
    if precision is None:
        return
    if not isinstance(precision, config.MixedPrecisionPolicy):
        raise TypeError("FSDP mixed_precision must be a MixedPrecision policy")
    if getattr(precision, "keep_low_precision_grads", False):
        raise NotImplementedError("FSDP1 low-precision gradients are not implemented")
    if getattr(precision, "output_dtype", None) is not None:
        raise NotImplementedError("FSDP1 output_dtype is not supported")
    for values, fields in ((module.parameters(), ("param_dtype", "reduce_dtype")),
                           (module.buffers(), ("buffer_dtype",))):
        for value in values:
            actual = dtype_name(value.dtype)
            if not (actual.startswith("float") or actual == "bfloat16"):
                continue
            for field in fields:
                requested = getattr(precision, field, None)
                if requested is not None and dtype_name(requested) != actual:
                    raise NotImplementedError(
                        "FSDP1 %s=%s requires an unimplemented mixed-precision cast (storage=%s)"
                        % (field, requested, actual))


def _fsdp1_prepare_module(module, args, options):
    """Validate before mutation; implement the bounded, materialized world-1 API.

    A no-options call retains the historical world-1 lightweight wrapper. Explicit
    FSDP1 options require original parameters; flattened-parameter FSDP1,
    auto-wrap, meta materialization, CPU offload and multi-rank FSDP1 are not
    implemented. Multi-rank users must use the separate fully_shard API.
    """
    if args:
        raise TypeError("FSDP1 options must be passed by keyword")
    options = dict(options)
    supported = {
        "process_group", "sharding_strategy", "cpu_offload", "auto_wrap_policy",
        "backward_prefetch", "mixed_precision", "ignored_modules", "param_init_fn",
        "device_id", "sync_module_states", "forward_prefetch", "limit_all_gathers",
        "use_orig_params", "ignored_states", "device_mesh",
    }
    unknown = set(options) - supported
    if unknown:
        raise TypeError("Unsupported FSDP1 options: %s" % ", ".join(sorted(unknown)))
    if common._world_size() != 1:
        raise NotImplementedError("FSDP1 compatibility supports world_size=1 only; use fully_shard for multi-rank")
    if options and options.get("use_orig_params", False) is not True:
        raise NotImplementedError("FSDP1 options require use_orig_params=True; flattened FSDP1 is not implemented")
    group = options.get("process_group")
    mesh = options.get("device_mesh")
    if group is not None and mesh is not None:
        raise ValueError("FSDP1 process_group and device_mesh are mutually exclusive")
    if group is not None:
        raise NotImplementedError("FSDP1 explicit process_group is not implemented; pass a device_mesh")
    _reject_unsupported_mesh(mesh)
    if mesh is not None and any(int(size) != 1 for size in mesh.shape):
        raise NotImplementedError("FSDP1 requires a singleton device_mesh")
    if options.get("auto_wrap_policy") is not None:
        raise NotImplementedError("FSDP1 auto_wrap_policy is not implemented; explicitly disable auto-wrap")
    offload = options.get("cpu_offload")
    if offload is not None:
        if not isinstance(offload, config.CPUOffload):
            raise TypeError("cpu_offload must be CPUOffload")
        if offload.offload_params:
            raise NotImplementedError("FSDP1 CPU parameter offload is not implemented")
    for name in ("ignored_modules", "ignored_states"):
        if options.get(name) is not None and tuple(options[name]):
            raise NotImplementedError("FSDP1 %s is not implemented" % name)
    for name in ("sync_module_states", "forward_prefetch", "limit_all_gathers"):
        if name in options and not isinstance(options[name], bool):
            raise TypeError("FSDP1 %s must be bool" % name)
    if options.get("forward_prefetch", False):
        raise NotImplementedError("FSDP1 forward prefetch is not implemented")
    backward = options.get("backward_prefetch")
    if backward is not None and not isinstance(backward, config.BackwardPrefetch):
        raise TypeError("FSDP1 backward_prefetch must be BackwardPrefetch or None")
    strategy = options.get("sharding_strategy", config.ShardingStrategy.FULL_SHARD)
    if not isinstance(strategy, config.ShardingStrategy):
        raise TypeError("FSDP1 sharding_strategy must be ShardingStrategy")
    if strategy in (config.ShardingStrategy.HYBRID_SHARD, config.ShardingStrategy._HYBRID_SHARD_ZERO2):
        raise NotImplementedError("FSDP1 hybrid sharding is not implemented")
    init = options.get("param_init_fn")
    if init is not None and not callable(init):
        raise TypeError("FSDP1 param_init_fn must be callable")
    parameters = tuple(module.parameters())
    buffers = tuple(module.buffers())
    for value in parameters + buffers:
        if getattr(value, "is_meta", False) or getattr(getattr(value, "device", None), "type", None) == "meta":
            raise NotImplementedError("FSDP1 meta materialization/param_init_fn is not implemented")
    precision = options.get("mixed_precision")
    _fsdp1_check_precision(module, precision)
    requested = options.get("device_id")
    target = _compat_device(requested) if requested is not None else None
    if target is not None and target.type not in ("cpu", "cuda", "npu"):
        raise NotImplementedError("FSDP1 supports CPU/CUDA/NPU placement only")
    if mesh is not None and target is not None and mesh.device_type != target.type:
        raise ValueError("FSDP1 device_id and device_mesh device types differ")
    if target is None and mesh is not None:
        target = _compat_device(mesh.device_type)
    if target is not None:
        # Module.to owns in-place Parameter identity and buffer traversal. Do
        # not reimplement tensor migration or change the ordinary factory mode.
        with jt.runtime.scope(use_cuda=jt.runtime.use_cuda):
            module.to(device=target)
        if tuple(id(value) for value in module.parameters()) != tuple(id(value) for value in parameters):
            raise RuntimeError("FSDP1 device migration replaced original Parameter objects")
    if mesh is None and target is not None:
        mesh = dtensor.init_device_mesh(target.type, (1,))
    return {"mesh": mesh, "mp_policy": precision,
            "reshard_after_forward": strategy == config.ShardingStrategy.FULL_SHARD}


def _fsdp1_check_forward_inputs(value, expected):
    # Input casting is also an executable contract: do not silently pretend a
    # stored policy has cast a half input into the full-precision module.
    # Resolve lazily: Torch installation imports this FSDP domain before its
    # public package has finished initialization. Jagged tensors deliberately
    # are not Var/Tensor subclasses; inspect their dtype without densifying.
    from ..torch import nested_tensor_type
    if isinstance(value, (jt.Var, nested_tensor_type())):
        actual = dtype_name(value.dtype)
        if (actual.startswith("float") or actual == "bfloat16") and actual != expected:
            raise NotImplementedError("FSDP1 mixed-precision forward input casting is not implemented")
    elif isinstance(value, dict):
        for child in value.values():
            _fsdp1_check_forward_inputs(child, expected)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _fsdp1_check_forward_inputs(child, expected)


class FullyShardedDataParallel(nn.Module, FSDPModule):
    def __init__(self, module, *args, **kwargs):
        nn.Module.__init__(self)
        mapped = _fsdp1_prepare_module(module, args, kwargs)
        self.module = fully_shard(module, **mapped)
        object.__setattr__(self, "_is_fsdp_module", True)
        object.__setattr__(self, "_fsdp_state", getattr(self.module, "_fsdp_state", None))
        object.__setattr__(self, "_fsdp1_precision", kwargs.get("mixed_precision"))
        self._fsdp_state.fsdp1_execution = "unsharded_world_size_1"
        self._fsdp_state.fsdp1_sync_module_states = bool(kwargs.get("sync_module_states", False))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Direct lookup avoids recursively forwarding before __init__ has
            # installed the wrapped module. Wrapper attributes take priority.
            module = object.__getattribute__(self, "module")
            return getattr(module, name)

    def execute(self, *args, **kwargs):
        precision = self._fsdp1_precision
        expected = getattr(precision, "param_dtype", None)
        if expected is not None and (getattr(precision, "cast_forward_inputs", False)
                                     or getattr(precision, "cast_root_forward_inputs", True)):
            _fsdp1_check_forward_inputs((args, kwargs), dtype_name(expected))
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

    def clip_grad_norm_(self, max_norm, norm_type=2.0):
        # This wrapper currently owns unsharded world-1 parameters only.
        # Reuse the public gradient collector and on-device clipping math;
        # a local norm is not a substitute for multi-rank FSDP reduction.
        if common._world_size() != 1:
            raise NotImplementedError("FSDP1 gradient clipping supports world_size=1 only")
        from ..torch.tensor_state import compatibility_owner
        torch = compatibility_owner(jt)
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm, norm_type)

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
                except (AttributeError, TypeError) as exc:
                    swallowed("fsdp2/api.py state_dict_type: delattr(module, '_fsdp_state_dict_type')", exc)
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
        state = getattr(module, "_fsdp_state", None)
        was_unsharded = bool(getattr(state, "true_fsdp_unsharded", False)) if state is not None else False
        if state is not None and getattr(state, "true_fsdp_initialized", False):
            shard._unshard_module_params(module)
        try:
            yield module
        finally:
            if state is not None and getattr(state, "true_fsdp_initialized", False) and not was_unsharded:
                shard._reshard_module_params(module)

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


_EXPORTS = (
    "FSDPModule", "_FSDP_METHODS",
    "_inject_fsdp_methods", "fully_shard", "register_fsdp_forward_method",
    "share_comm_ctx", "FullyShardedDataParallel", "ShardedGradScaler",
)
