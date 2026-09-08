"""ACL fused AdamW TensorList update."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt


def fused_adamw_acl(parameters, moments, variances, gradients, step, lr,
                    beta1, beta2, weight_decay, eps):
    count = len(parameters)
    if count == 0 or any(
            len(values) != count
            for values in (moments, variances, gradients)):
        raise ValueError("fused AdamW TensorLists must have one or more entries")
    for parameter, moment, variance, gradient in zip(
            parameters, moments, variances, gradients):
        tensors = (parameter, moment, variance, gradient)
        if any(list(tensor.shape) != list(parameter.shape) for tensor in tensors):
            raise ValueError("fused AdamW tensors must have identical shapes")
        if any(_jittor_dtype_name(tensor.dtype) != _jittor_dtype_name(parameter.dtype) for tensor in tensors):
            raise TypeError("fused AdamW tensors must have identical dtypes")
        if _jittor_dtype_name(parameter.dtype) not in ("bfloat16", "float16", "float32"):
            raise TypeError("fused AdamW requires bfloat16, float16, or float32")
    if step.numel() != 1 or _jittor_dtype_name(step.dtype) not in ("float32", "int64"):
        raise TypeError("fused AdamW step must be one float32 or int64 value")

    if not jt.flags.use_acl:
        raise RuntimeError("fused AdamW ACL op requires the ACL backend")
    result = jt.fused_adamw(
        list(parameters), list(moments), list(variances), list(gradients),
        step, float(lr), float(beta1), float(beta2),
        float(weight_decay), float(eps),
    )
    return result[:count], result[count:2 * count], result[2 * count:]


__all__ = ["fused_adamw_acl"]
