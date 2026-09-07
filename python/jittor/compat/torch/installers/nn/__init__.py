"""Torch NN compatibility, composed in the original installation order."""

from jittor import nn
from .module_methods import (
    _ORIG_MODULE_EXECUTE,
    _ORIG_MODULE_DISPATCH_CALL,
    _ORIG_MODULE_NAMED_PARAMETERS,
    _ORIG_MODULE_NAMED_BUFFERS,
    _ORIG_MODULE_NAMED_MODULES,
    _ORIG_MODULE_LOAD_STATE_DICT,
    _ORIG_MODULE_PARAMETERS,
    _dispatch_cache,
    _pipeline_state,
    _leaves_published,
    _IncompatibleKeys,
    _MODULE_FLOAT_DTYPES,
    _pipelining_from_environment,
    _execute,
    _forward_alias,
    _prefer_forward,
    _acl_bfloat16_rms_norm,
    _standard_rms_norm,
    set_execution_pipelining,
    get_execution_pipelining,
    _maybe_pipeline,
    _dispatch_module_call,
    _call,
    _named_parameters,
    _named_buffers,
    _named_modules,
    _find_state_target,
    _state_source_to_var,
    _preserve_target_dtypes_for_load,
    _state_dict_key_diff,
    _load_state_dict,
    _ParamList,
    _register_leaf_params,
    _parameters,
    _set_is_train,
    _train,
    _eval,
    _module_cast_var_if_needed,
    _module_cast_float_dtype,
    _module_replace_vars,
    _module_to_conversion,
    _module_to,
    _module_to_empty,
    _module_cuda,
    _module_npu,
    _module_cpu,
    _module_float,
    _module_double,
    _module_half,
    _zero_grad,
    _buffers,
    _get_submodule,
    _get_parameter,
    _get_buffer,
    _register_parameter,
    _module_type,
    _nonpersist_set,
    _install_module_methods,
)
from .extras import _install_nn_extras
from .functional import _install_functional
from .attention import install_attention
from .parity import install_parity


def install(ctx):
    _install_functional(ctx)
    install_attention(ctx)
    _install_nn_extras(nn, ctx.registry)
    ctx.registry.module_map["torch.nn"] = nn
    if hasattr(nn, "functional"):
        ctx.registry.module_map["torch.nn.functional"] = nn.functional
