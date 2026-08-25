"""FSDP module API implementations exposed by the canonical package."""

import abc
import contextlib
import types

import jittor as jt
from jittor import nn

from . import config, dtensor, grad_sync, optimizer, shard


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
        return shard._apply_fsdp_attr(self, "requires_gradient_sync", bool(value), recurse)

    def set_requires_all_reduce(self, value, recurse=True):
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
            fully_shard(
                m, mesh=mesh, reshard_after_forward=reshard_after_forward,
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
    st.mesh = mesh or getattr(st, "mesh", None) or dtensor.DeviceMesh(
        "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
    st.reshard_after_forward = reshard_after_forward
    st.shard_placement_fn = shard_placement_fn
    st.mp_policy = (
        mp_policy if mp_policy is not None else config.MixedPrecisionPolicy())
    st.offload_policy = (
        offload_policy if offload_policy is not None else config.NoOffloadPolicy())
    st.ignored_params = tuple(ignored_params or ())
    st.dp_mesh_dims = dp_mesh_dims
    st.kwargs = dict(kwargs)
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
