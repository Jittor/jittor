"""One FusedSgdOp for a whole parameter list, however the backend runs it."""

import jittor as jt

from jittor._core.dtypes import dtype_name as _jittor_dtype_name


def _with_storage(tensor):
    """Give the tensor real, contiguous storage if it has none.

    A parameter can start life as a broadcast: `nn.LayerNorm`'s weight is
    `jt.ones(shape)`, which carries no storage of its own, and the in-place
    kernels reject that. The optimizer assigns every output of this op back
    onto the tensor it came from, so materialising here is invisible to the
    caller and happens once -- the assignment leaves real storage behind.
    """
    return tensor if tensor._storage_is_contiguous() else tensor.contiguous()


def fused_sgd_acl(parameters, velocities, gradients, lr, momentum,
                  weight_decay, dampening, nesterov):
    count = len(parameters)
    if count == 0 or any(len(values) != count for values in (velocities, gradients)):
        raise ValueError("fused SGD TensorLists must have one or more entries")
    for parameter, velocity, gradient in zip(parameters, velocities, gradients):
        tensors = (parameter, velocity, gradient)
        if any(list(tensor.shape) != list(parameter.shape) for tensor in tensors):
            raise ValueError("fused SGD tensors must have identical shapes")
        if any(_jittor_dtype_name(tensor.dtype) != _jittor_dtype_name(parameter.dtype)
               for tensor in tensors):
            raise TypeError("fused SGD tensors must have identical dtypes")
        if _jittor_dtype_name(parameter.dtype) != "float32":
            raise TypeError("fused SGD on ACL requires float32 tensors")
    if not jt.flags.use_acl:
        raise RuntimeError("fused SGD ACL op requires the ACL backend")
    result = jt.fused_sgd(
        [_with_storage(value) for value in parameters],
        [_with_storage(value) for value in velocities],
        [_with_storage(value) for value in gradients],
        float(lr), float(momentum), float(weight_decay), float(dampening),
        bool(nesterov), False,
    )
    return result[:count], result[count:]


__all__ = ["fused_sgd_acl"]
