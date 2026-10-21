"""Read-only legacy hook names backed by the canonical runtime dispatch table."""

import sys
from types import MappingProxyType, ModuleType

from jittor._runtime.dispatch import _is_registered, dispatch_context, registered_kernel, try_dispatch


_OPERATIONS = MappingProxyType({
    "batch_norm_cuda": "nn.batch_norm.training",
    "batch_norm_eval_cuda": "nn.batch_norm.eval",
    "group_norm_cuda": "nn.group_norm",
    "rms_norm_cuda": "nn.rms_norm.inference",
    "rms_norm_training_cuda": "nn.rms_norm.training",
    "acl_grouped_add_rms_norm": "nn.grouped_add_rms_norm",
    "acl_grouped_bfloat16_rms_norm": "nn.grouped_bfloat16_rms_norm",
    "acl_grouped_dual_bfloat16_rms_norm": "nn.grouped_dual_bfloat16_rms_norm",
    "acl_expand_rotary_cache": "nn.expand_rotary_cache",
    "acl_grouped_qk_rms_norm_rotary": "nn.grouped_qk_rms_norm_rotary",
    "acl_constant_pad": "nn.constant_pad",
    "acl_embedding": "nn.embedding",
    "acl_silu_and_mul": "nn.silu_and_mul",
    "acl_scaled_dot_product_attention": "nn.scaled_dot_product_attention",
})


class _HookCall:
    __slots__ = ("_operation",)

    def __init__(self, operation):
        object.__setattr__(self, "_operation", operation)

    def __setattr__(self, name, value):
        raise AttributeError("backend hook references are read-only")

    def __call__(self, *args, **kwargs):
        return try_dispatch(self._operation, *args, **kwargs)

    def __getattr__(self, name):
        implementation = registered_kernel(self._operation, dispatch_context().backend)
        if implementation is None:
            raise AttributeError(name)
        return getattr(implementation, name)


_CALLS = MappingProxyType({name: _HookCall(operation)
                          for name, operation in _OPERATIONS.items()})


def __getattr__(name):
    operation = _OPERATIONS.get(name)
    if operation is None:
        raise AttributeError(name)
    # Not registered on any backend: nothing the context could select. Asked
    # on every call of the functions that consult a hook -- `pad` reads
    # `acl_constant_pad` each time -- and building the dispatch context and
    # taking the registry lock to learn "none" was most of what they cost.
    if not _is_registered(operation):
        return None
    if registered_kernel(operation, dispatch_context().backend) is None:
        return None
    return _CALLS[name]


def __dir__():
    return sorted(set(globals()) | set(_OPERATIONS))


class _HookModule(ModuleType):
    def __setattr__(self, name, value):
        if name in _OPERATIONS:
            raise AttributeError("backend hooks are read-only; use register_kernel")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _OPERATIONS:
            raise AttributeError("backend hooks are read-only; use register_kernel")
        super().__delattr__(name)


sys.modules[__name__].__class__ = _HookModule
