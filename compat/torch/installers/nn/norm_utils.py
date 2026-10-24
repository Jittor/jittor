from ...grad import _clip_grads_with_norm_device, _get_total_norm_device
from ...context import get_install_context
from ...frontend import tensor_frontend

import jittor as _jt

def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False,
                    foreach=None):
    del foreach
    if _jt.core._is_tensor_carrier(tensors):
        tensors = [tensors]
    tensors = list(tensors)
    # Enable the canonical per-group shard reduction only for real tagged
    # gradients. Ordinary/DDP gradients are already replicated; unconditional
    # True would enter the reducer's WORLD path and count them more than once.
    shard_reduce = any(_jt.core._is_tensor_carrier(tensor)
                       and getattr(tensor, "_fsdp_norm_group", None) is not None
                       for tensor in tensors)
    tensor_type = get_install_context(_jt).state["Var"]
    with tensor_frontend(tensor_type, like=tensors[0] if tensors else None,
                         default_placement=not tensors):
        return _get_total_norm_device(
            tensors, norm_type, error_if_nonfinite, shard_reduce=shard_reduce)

def _clip_grads_with_norm_(parameters, max_norm, total_norm,
                           foreach=None):
    del foreach
    if _jt.core._is_tensor_carrier(parameters):
        parameters = [parameters]
    params = list(parameters)
    grads = []
    for parameter in params:
        # Gradients belong to parameters, not the most recently constructed
        # optimizer. Independent modules may have different optimizers.
        grad = getattr(parameter, "grad", None)
        if grad is not None:
            grads.append(grad)
    _clip_grads_with_norm_device(grads, max_norm, total_norm)
