"""FSDP module API implementations exposed by the canonical package."""

import abc
import contextlib
import types

import numpy as np

import jittor as jt
from jittor import nn

from . import common, config, dtensor, grad_sync, optimizer, shard
from ..diagnostics import EXPECTED, swallowed


class _FSDPModuleMeta(abc.ABCMeta):
    def __instancecheck__(cls, obj):
        marker = globals().get("FSDPModule")
        if cls is marker and bool(getattr(obj, "_is_fsdp_module", False)):
            return True
        return super().__instancecheck__(obj)


class FSDPModule(metaclass=_FSDPModuleMeta):
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
    for name in _FSDP_METHODS:
        if not callable(getattr(module, name, None)):
            object.__setattr__(
                module,
                name,
                types.MethodType(getattr(FSDPModule, name), module),
            )
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
        except EXPECTED as exc:
            swallowed("fsdp2/api.py _inject_fsdp_methods: fsdp_cls = type(cls.__name__ + 'FSDP2Compat', (cls, FSD...", exc)
            fsdp_cls = None
    if fsdp_cls is not None:
        try:
            module.__class__ = fsdp_cls
        except (AttributeError, TypeError) as exc:
            swallowed("fsdp2/api.py _inject_fsdp_methods: module.__class__ = fsdp_cls", exc)
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


# FSDP1 arguments that this wrapper accepts but cannot change the FSDP2 plan.
# use_orig_params is unconditional in FSDP2; prefetch/limit_all_gathers are
# scheduling knobs of the FSDP1 collectives, which this implementation does
# not have; device_id selects the device fully_shard already moves to.
_LEGACY_FSDP_NOOP = (
    "use_orig_params", "forward_prefetch", "limit_all_gathers",
    "backward_prefetch", "device_id", "_use_expandable_segments",
    "_device_handle",
)


def _legacy_strategy(value):
    if value is None:
        return config.ShardingStrategy.FULL_SHARD
    if isinstance(value, config.ShardingStrategy):
        return value
    try:
        return config.ShardingStrategy(value)
    except (ValueError, TypeError) as error:
        raise ValueError("unknown FSDP sharding_strategy %r" % (value,)) from error


def _legacy_mesh(device_mesh, process_group):
    if device_mesh is not None and process_group is not None:
        raise ValueError(
            "FullyShardedDataParallel accepts either device_mesh= or "
            "process_group=, not both")
    if device_mesh is not None:
        if not isinstance(device_mesh, dtensor.DeviceMesh):
            raise TypeError("FullyShardedDataParallel device_mesh must be a DeviceMesh")
        return device_mesh
    if process_group is None:
        return None
    return dtensor.DeviceMesh.from_group(
        process_group, "cuda" if getattr(jt, "has_cuda", 0) else "cpu")


def _legacy_ignored_params(module, ignored_states, ignored_modules):
    ordered = []
    seen = set()

    def add(parameter):
        if parameter is not None and id(parameter) not in seen:
            seen.add(id(parameter))
            ordered.append(parameter)

    for parameter in ignored_states or ():
        add(parameter)
    for ignored in ignored_modules or ():
        if ignored is None:
            continue
        for parameter in ignored.parameters(recurse=True):
            add(parameter)
    return tuple(ordered)


def _legacy_sync_module_states(module, mesh, enabled):
    """FSDP1's sync_module_states: rank 0's initial weights reach every rank."""
    if not enabled or not common._in_true_distributed():
        return
    from jittor.distributed import broadcast

    if mesh is None:
        group, ranks = None, list(range(common._world_size()))
    elif mesh.ndim == 1:
        group = mesh.get_group()
        ranks = [int(rank) for rank in mesh.mesh]
    else:
        flat = mesh._flatten()
        group = flat.get_group()
        ranks = [int(rank) for rank in flat.mesh]
    if len(ranks) <= 1:
        return
    source = ranks[0]
    for value in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        value.update(broadcast(value, src=source, group=group))


def _legacy_fsdp_plan(module, args, kwargs):
    """Translate FSDP1 constructor arguments into a fully_shard plan.

    The two names that change the plan are ``device_mesh`` (which axis to
    shard) and ``sharding_strategy`` (whether to shard at all).  Before this
    translation both landed in ``fully_shard(**kwargs)`` and were silently
    ignored: a NO_SHARD wrap with ``device_mesh=dp_mesh`` sharded the model
    over the *world* instead of not sharding it.
    """
    if len(args) > 1:
        raise TypeError(
            "FullyShardedDataParallel accepts at most one positional argument "
            "(process_group); use keyword arguments for the rest")
    process_group = kwargs.pop("process_group", args[0] if args else None)
    device_mesh = kwargs.pop("device_mesh", None)
    strategy = _legacy_strategy(kwargs.pop("sharding_strategy", None))
    cpu_offload = kwargs.pop("cpu_offload", None)
    auto_wrap_policy = kwargs.pop("auto_wrap_policy", None)
    mixed_precision = kwargs.pop("mixed_precision", None)
    ignored_states = kwargs.pop("ignored_states", None)
    ignored_modules = kwargs.pop("ignored_modules", None)
    param_init_fn = kwargs.pop("param_init_fn", None)
    sync_module_states = bool(kwargs.pop("sync_module_states", False))
    for name in _LEGACY_FSDP_NOOP:
        kwargs.pop(name, None)
    if kwargs:
        raise TypeError(
            "FullyShardedDataParallel got unexpected keyword argument(s): %s"
            % ", ".join(sorted(kwargs)))
    if auto_wrap_policy is not None:
        raise NotImplementedError(
            "FullyShardedDataParallel auto_wrap_policy= would create nested "
            "FSDP units; this compat wraps the module passed to it as one "
            "unit, so accepting the policy would silently change the plan")
    if param_init_fn is not None:
        raise NotImplementedError(
            "FullyShardedDataParallel param_init_fn= is not implemented")
    if cpu_offload is not None and getattr(cpu_offload, "offload_params", False):
        raise NotImplementedError(
            "FullyShardedDataParallel cpu_offload(offload_params=True) is not implemented")
    mp_policy = None
    if mixed_precision is not None:
        buffer_dtype = getattr(mixed_precision, "buffer_dtype", None)
        if buffer_dtype is not None:
            raise NotImplementedError(
                "FullyShardedDataParallel mixed_precision buffer_dtype= is not implemented")
        mp_policy = config.MixedPrecisionPolicy(
            param_dtype=getattr(mixed_precision, "param_dtype", None),
            reduce_dtype=getattr(mixed_precision, "reduce_dtype", None),
            cast_forward_inputs=getattr(mixed_precision, "cast_forward_inputs", True),
        )
    mesh = _legacy_mesh(device_mesh, process_group)
    _legacy_sync_module_states(module, mesh, sync_module_states)
    if mesh is None and strategy is config.ShardingStrategy.NO_SHARD:
        # FSDP1's default process group is the whole world, and NO_SHARD
        # replicates over it.  Without this the fallthrough would shard over
        # the world, which is exactly the bug this translation removes.
        mesh = dtensor.init_device_mesh(
            "cuda" if getattr(jt, "has_cuda", 0) else "cpu",
            (common._world_size(),), mesh_dim_names=("dp",))
    plan = {
        "mesh": mesh,
        "strategy": strategy,
        "mp_policy": mp_policy,
        "ignored_params": _legacy_ignored_params(
            module, ignored_states, ignored_modules),
        "reshard_after_forward": True,
        "dp_mesh_dims": None,
    }
    if strategy is config.ShardingStrategy.NO_SHARD:
        # NO_SHARD means every rank keeps the full parameters and only
        # gradients are synchronized over the data-parallel mesh.  Sharding
        # over a one-rank shard group expresses that without a second code
        # path: the shards ARE the full parameters, and the replicate group
        # performs the DDP-style mean the FSDP1 strategy promises.
        plan["replicate_mesh"] = mesh
    elif strategy is config.ShardingStrategy.SHARD_GRAD_OP:
        # FSDP1 keeps parameters unsharded after the forward pass.
        plan["reshard_after_forward"] = False
    elif strategy in (config.ShardingStrategy.HYBRID_SHARD,
                      config.ShardingStrategy._HYBRID_SHARD_ZERO2):
        names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
        if mesh is None or mesh.ndim != 2 or set(names) != {"shard", "replicate"}:
            raise NotImplementedError(
                "FullyShardedDataParallel HYBRID_SHARD needs a 2-D device_mesh "
                "with mesh_dim_names=('shard', 'replicate') so the sharding "
                "and replication axes are unambiguous")
        plan["dp_mesh_dims"] = config.DataParallelMeshDims(
            shard="shard", replicate="replicate")
        if strategy is config.ShardingStrategy._HYBRID_SHARD_ZERO2:
            plan["reshard_after_forward"] = False
    return plan


class FullyShardedDataParallel(nn.Module, FSDPModule):
    def __init__(self, module, *args, **kwargs):
        nn.Module.__init__(self)
        plan = _legacy_fsdp_plan(module, args, kwargs)
        replicate_mesh = plan.pop("replicate_mesh", None)
        strategy = plan.pop("strategy")
        mesh = plan.pop("mesh")
        if replicate_mesh is not None:
            # One-rank shard group + the data-parallel mesh as replication.
            # The singleton mesh is local to this rank, so it needs no
            # communicator (a size-1 group short-circuits every collective).
            mesh = dtensor.DeviceMesh(
                replicate_mesh.device_type,
                np.asarray([common._rank()], dtype=np.int64),
                _init_backend=False)
        wrapped = fully_shard(module, mesh=mesh, **plan)
        state = getattr(wrapped, "_fsdp_state", None)
        if state is not None:
            state.legacy_sharding_strategy = strategy
            if replicate_mesh is not None:
                state.replicate_group = (
                    replicate_mesh.get_group() if replicate_mesh.size() > 1 else None)
                state.mesh = replicate_mesh
        self.module = wrapped
        object.__setattr__(self, "_is_fsdp_module", True)
        object.__setattr__(self, "_fsdp_state", getattr(self.module, "_fsdp_state", None))

    def __getattr__(self, name):
        # FSDP1 forwards unknown attributes to the wrapped module; the example
        # reads model.config and model.loss_function after wrapping.
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is None:
                raise
            return getattr(module, name)

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
    "_FSDPModuleMeta", "FSDPModule", "_FSDP_METHODS",
    "_inject_fsdp_methods", "fully_shard", "register_fsdp_forward_method",
    "share_comm_ctx", "FullyShardedDataParallel", "ShardedGradScaler",
)
