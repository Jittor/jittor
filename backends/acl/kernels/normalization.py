"""ACL normalization and grouped serving implementations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import math
from numbers import Real
import jittor as jt
from .ops.norms_op import (
    BatchNormACL,
    GroupNormACL,
    RmsNormACL,
    LayerNormACL,
    GroupedAddRmsNormACL,
    GroupedBFloat16RmsNormACL,
    GroupedDualBFloat16RmsNormACL,
)
from .ops.rope_op import ExpandRotaryCacheACL, GroupedQKRmsNormRotaryACL
from .tensor import embedding_acl


def layer_norm_acl(x, normalized_shape, weight, bias, eps, *, allow_bfloat16=False):
    if not isinstance(weight, jt.Var) or not isinstance(bias, jt.Var):
        return None
    return LayerNormACL(normalized_shape, eps, True)(x, weight, bias)


def _unit_weight(weight):
    unit = weight.__dict__.get("_torch_acl_rms_norm_unit_weight")
    if unit is None or tuple(unit.shape) != tuple(weight.shape):
        unit = jt.ones(weight.shape, dtype="bfloat16")
        unit.stop_grad()
        weight.__dict__["_torch_acl_rms_norm_unit_weight"] = unit
    return unit


def _batch_norm_eval_cuda_acl(x, weight, bias, running_mean, running_var, eps):
    # aclnnBatchNorm forwards x/weight/bias's own dtype (BatchNormACL passes
    # [x.dtype] * 3 through, unconditionally); float16 was verified correct
    # against a CPU reference, both with the running stats also float16 (a
    # plain module.to(dtype=float16) casts everything, including buffers --
    # ResNet-50 inference in this project's own fp16 benchmark does exactly
    # that) and with them left float32 (the mixed-precision convention some
    # training setups use instead). Restricting to float32-only forced every
    # float16 inference call through the generic decomposition instead of
    # this fused kernel.
    values = (x, weight, bias, running_mean, running_var)
    norm_dtypes = ("float32", "float16")
    if (
        all((isinstance(value, jt.Var) for value in values))
        and (_jittor_dtype_name(x.dtype) in norm_dtypes)
        and (_jittor_dtype_name(weight.dtype) == _jittor_dtype_name(x.dtype))
        and (_jittor_dtype_name(bias.dtype) == _jittor_dtype_name(x.dtype))
        and (_jittor_dtype_name(running_mean.dtype) in ("float32", _jittor_dtype_name(x.dtype)))
        and (_jittor_dtype_name(running_var.dtype) in ("float32", _jittor_dtype_name(x.dtype)))
        and isinstance(eps, Real)
    ):
        shape = tuple((int(size) for size in x.shape))
        epsilon = float(eps)
        if (
            len(shape) == 4
            and all((size > 0 for size in shape))
            and all((tuple(value.shape) == (shape[1],) for value in values[1:]))
            and math.isfinite(epsilon)
            and (epsilon > 0.0)
        ):
            return BatchNormACL(epsilon)(x, weight, bias, running_mean, running_var)
    return None


def _group_norm_cuda_acl(x, num_groups, weight, bias, eps):
    # aclnnGroupNorm takes the input dtype as-is (GroupNormOpRunner forwards
    # whatever dtype the Vars already have); float16 was verified correct
    # against a CPU reference (max abs diff ~0.007, well inside fp16 noise).
    # Restricting to float32 only forced every float16 call -- e.g. SD1.5's
    # VAE decoder, which runs its norms in fp16 -- through the generic
    # Sub/ReduceMean/Sqrt/RealDiv/Mul/Add decomposition instead of this one
    # fused kernel (>45% of that workload's device time in an aclprof
    # capture).
    norm_dtypes = ("float32", "float16")
    if (
        isinstance(x, jt.Var)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
        and isinstance(eps, Real)
        and (_jittor_dtype_name(x.dtype) in norm_dtypes)
        and (_jittor_dtype_name(weight.dtype) == _jittor_dtype_name(x.dtype))
        and (_jittor_dtype_name(bias.dtype) == _jittor_dtype_name(x.dtype))
    ):
        shape = tuple((int(size) for size in x.shape))
        groups = int(num_groups)
        epsilon = float(eps)
        spatial = 1
        for size in shape[2:]:
            spatial *= size
        # aclnnGroupNorm has a severe (~45x, measured 9ms vs 0.2ms on a
        # 910B3) performance cliff for a per-sample spatial extent (product
        # of every dim past the channel axis) from 2 to 7 elements inclusive
        # -- bisected directly against this exact kernel at fixed channel
        # and group counts: 1x1 and 3x3 are fast, 1x2/1x4/1x5/1x7/2x2/2x3 are
        # all ~45x slower, 2x4 and everything larger is fast again. Looks
        # like a hardcoded tile size of 8 with a catastrophic remainder
        # path, not anything this wrapper controls -- ddpm_unet_train hits
        # exactly this (its bottleneck block is 2x2), so route that narrow
        # band to the decomposed fallback instead of eating the cliff.
        pathological_spatial = 2 <= spatial <= 7
        if (
            len(shape) >= 2
            and all((size > 0 for size in shape))
            and (groups > 0)
            and (shape[1] % groups == 0)
            and (tuple(weight.shape) == (shape[1],))
            and (tuple(bias.shape) == (shape[1],))
            and math.isfinite(epsilon)
            and (epsilon > 0.0)
            and (not pathological_spatial)
        ):
            return GroupNormACL(groups, epsilon)(x, weight, bias)
    return None


def _rms_norm_cuda_acl(x, gamma, epsilon=1e-06):
    if isinstance(x, jt.Var) and isinstance(gamma, jt.Var) and isinstance(epsilon, Real):
        x_shape = tuple((int(size) for size in x.shape))
        gamma_shape = tuple((int(size) for size in gamma.shape))
        x_dtype = _jittor_dtype_name(x.dtype)
        gamma_dtype = _jittor_dtype_name(gamma.dtype)
        epsilon_value = float(epsilon)
        supported_gamma_dtypes = {
            "float32": ("float32",),
            "float16": ("float16", "float32"),
            "bfloat16": ("bfloat16", "float32"),
        }
        if (
            x_shape
            and all((size > 0 for size in x_shape))
            and (gamma_shape == (x_shape[-1],))
            and (gamma_dtype in supported_gamma_dtypes.get(x_dtype, ()))
            and (getattr(jt.flags, "no_grad", 0) or _jittor_dtype_name(x_dtype) in ("float32", "bfloat16"))
            and math.isfinite(epsilon_value)
            and (epsilon_value > 0)
        ):
            return RmsNormACL()(x, gamma, epsilon_value)
    return None


def _grouped_add_rms_norm_acl(x, residual, weight, eps):
    if not (
        getattr(jt.flags, "no_grad", 0)
        and isinstance(x, jt.Var)
        and isinstance(residual, jt.Var)
        and isinstance(weight, jt.Var)
        and isinstance(eps, Real)
    ):
        return None
    shape = tuple((int(size) for size in x.shape))
    epsilon = float(eps)
    if (
        not shape
        or any((size <= 0 for size in shape))
        or tuple(residual.shape) != shape
        or (tuple(weight.shape) != (shape[-1],))
        or (len({_jittor_dtype_name(x.dtype), _jittor_dtype_name(residual.dtype), _jittor_dtype_name(weight.dtype)}) != 1)
        or (_jittor_dtype_name(x.dtype) not in ("float16", "bfloat16", "float32"))
        or (not math.isfinite(epsilon))
        or (epsilon <= 0.0)
    ):
        return None
    return GroupedAddRmsNormACL()(x, residual, weight, epsilon)


def _grouped_bfloat16_rms_norm_acl(x, unit_weight, weight, eps):
    values = (x, unit_weight, weight)
    if not (
        getattr(jt.flags, "no_grad", 0)
        and all((isinstance(value, jt.Var) for value in values))
        and all((_jittor_dtype_name(value.dtype) == "bfloat16" for value in values))
        and isinstance(eps, Real)
    ):
        return None
    shape = tuple((int(size) for size in x.shape))
    epsilon = float(eps)
    if (
        not shape
        or any((size <= 0 for size in shape))
        or tuple(unit_weight.shape) != (shape[-1],)
        or (tuple(weight.shape) != (shape[-1],))
        or (not math.isfinite(epsilon))
        or (epsilon <= 0.0)
    ):
        return None
    return GroupedBFloat16RmsNormACL()(x, unit_weight, weight, epsilon)


def _grouped_dual_bfloat16_rms_norm_acl(first, second, first_weight, second_weight, eps):
    values = (first, second, first_weight, second_weight)
    if not (
        getattr(jt.flags, "no_grad", 0)
        and all((isinstance(value, jt.Var) for value in values))
        and all((_jittor_dtype_name(value.dtype) == "bfloat16" for value in values))
        and isinstance(eps, Real)
    ):
        return None
    first_shape = tuple((int(size) for size in first.shape))
    second_shape = tuple((int(size) for size in second.shape))
    epsilon = float(eps)
    if (
        not first_shape
        or not second_shape
        or any((size <= 0 for size in first_shape + second_shape))
        or (first_shape[-1] != second_shape[-1])
        or (tuple(first_weight.shape) != (first_shape[-1],))
        or (tuple(second_weight.shape) != (second_shape[-1],))
        or (not math.isfinite(epsilon))
        or (epsilon <= 0.0)
    ):
        return None

    return GroupedDualBFloat16RmsNormACL()(
        first,
        second,
        _unit_weight(first_weight),
        _unit_weight(second_weight),
        first_weight,
        second_weight,
        epsilon,
    )


def _expand_rotary_cache_acl(cache, rotary_dim):
    if (
        getattr(jt.flags, "no_grad", 0)
        and isinstance(cache, jt.Var)
        and (cache.ndim == 2)
        and (int(cache.shape[-1]) == int(rotary_dim))
        and (int(rotary_dim) > 0)
        and (int(rotary_dim) % 2 == 0)
        and (_jittor_dtype_name(cache.dtype) in ("float16", "bfloat16", "float32"))
    ):
        return ExpandRotaryCacheACL()(cache)
    return None


def _grouped_qk_rms_norm_rotary_acl(
    positions,
    query,
    key,
    query_weight,
    key_weight,
    cos_sin_cache,
    head_size,
    rotary_dim,
    is_neox,
    eps,
):
    values = (query, key, query_weight, key_weight, cos_sin_cache)
    if not (
        getattr(jt.flags, "no_grad", 0)
        and all((isinstance(value, jt.Var) for value in values))
        and isinstance(positions, jt.Var)
        and all((_jittor_dtype_name(value.dtype) == "bfloat16" for value in values))
        and (_jittor_dtype_name(positions.dtype) in ("int32", "int64"))
        and (int(positions.numel()) == 1)
        and is_neox
        and (int(head_size) == int(rotary_dim))
        and (int(head_size) > 0)
        and (int(head_size) % 64 == 0)
        and (cos_sin_cache.ndim == 2)
        and (int(cos_sin_cache.shape[-1]) == int(rotary_dim))
        and isinstance(eps, Real)
    ):
        return None
    query_shape = tuple((int(size) for size in query.shape))
    key_shape = tuple((int(size) for size in key.shape))
    head_size = int(head_size)
    epsilon = float(eps)
    if (
        len(query_shape) != 2
        or len(key_shape) != 2
        or query_shape[0] != 1
        or (key_shape[0] != 1)
        or (query_shape[-1] % head_size != 0)
        or (key_shape[-1] % head_size != 0)
        or (tuple(query_weight.shape) != (head_size,))
        or (tuple(key_weight.shape) != (head_size,))
        or (not math.isfinite(epsilon))
        or (epsilon <= 0.0)
    ):
        return None

    selected = embedding_acl(positions.reshape((1,)), cos_sin_cache)
    cos, sin = ExpandRotaryCacheACL()(selected)
    cos = cos.reshape((1, 1, 1, head_size))
    sin = sin.reshape((1, 1, 1, head_size))
    query_4d = query.reshape((1, query_shape[-1] // head_size, 1, head_size))
    key_4d = key.reshape((1, key_shape[-1] // head_size, 1, head_size))
    query_out, key_out = GroupedQKRmsNormRotaryACL()(
        query_4d,
        key_4d,
        _unit_weight(query_weight),
        _unit_weight(key_weight),
        query_weight,
        key_weight,
        cos,
        sin,
        epsilon,
    )
    return (query_out.reshape(query_shape), key_out.reshape(key_shape))
