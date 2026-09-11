"""One aclnnFusedSgd launch for a whole parameter list."""

import jittor as jt

from jittor._core.dtypes import dtype_name as _jittor_dtype_name


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
        if _jittor_dtype_name(parameter.dtype) not in ("bfloat16", "float16", "float32"):
            raise TypeError("fused SGD requires bfloat16, float16, or float32")
    if not jt.flags.use_acl:
        raise RuntimeError("fused SGD ACL op requires the ACL backend")
    result = jt.fused_sgd(
        list(parameters), list(velocities), list(gradients),
        float(lr), float(momentum), float(weight_decay), float(dampening),
        bool(nesterov), False,
    )
    return result[:count], result[count:]


__all__ = ["fused_sgd_acl"]
