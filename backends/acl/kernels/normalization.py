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
from ._soc import soc_name, supports_bfloat16
from .ops.vector_norm_op import VectorNormACL


def vector_norm_supported(x, ord=2, dim=None, keepdim=False):
    # The SDK avoids half intermediate square overflow/underflow. Keep the
    # measured last-axis scope; production timing did not establish a speedup,
    # and the multi-axis GRN candidate slowed the full model.
    from numbers import Integral
    if isinstance(dim, (tuple, list)):
        if len(dim) != 1:
            return False
        dim = dim[0]
    return (bool(jt.flags.no_grad) and soc_name().startswith("Ascend310P")
            and isinstance(x, jt.Var) and x.ndim > 0
            and all(int(size) > 0 for size in x.shape)
            and ord == 2 and isinstance(keepdim, bool)
            and isinstance(dim, Integral) and not isinstance(dim, bool)
            and dim in (-1, x.ndim - 1)
            and _jittor_dtype_name(x.dtype) in ("float16", "float32"))


def vector_norm_acl(x, ord=2, dim=None, keepdim=False):
    return VectorNormACL()(x.contiguous(), keepdim)


_NORM_F16_F32_BF16 = frozenset(("float32", "float16", "bfloat16"))


def _fast_dtype_name(dtype):
    if dtype.__class__ is str:
        return dtype if dtype in _NORM_F16_F32_BF16 else _jittor_dtype_name(dtype)
    name = getattr(dtype, "name", None)
    if name is not None and name in _NORM_F16_F32_BF16:
        return name
    s = str(dtype)
    return s if s in _NORM_F16_F32_BF16 else _jittor_dtype_name(dtype)


def layer_norm_acl(x, normalized_shape, weight, bias, eps, *, allow_bfloat16=False):
    if not isinstance(weight, jt.Var) or not isinstance(bias, jt.Var):
        return None
    mapped = getattr(getattr(jt.core, "ops", None), "mapped_layer_norm", None)
    if mapped is not None and isinstance(x, jt.Var) and isinstance(eps, Real):
        dt = _fast_dtype_name(x.dtype)
        if (
            dt in _NORM_F16_F32_BF16
            and (dt != "bfloat16" or supports_bfloat16())
            and _fast_dtype_name(weight.dtype) == dt
            and _fast_dtype_name(bias.dtype) == dt
        ):
            epsilon = float(eps)
            nd = 1 if isinstance(normalized_shape, int) else len(normalized_shape)
            if nd >= 1 and math.isfinite(epsilon) and epsilon > 0.0:
                return mapped(x, weight, bias, nd, epsilon)[0]
    return LayerNormACL(normalized_shape, eps, True)(x, weight, bias)


def _unit_weight(weight):
    unit = weight.__dict__.get("_torch_acl_rms_norm_unit_weight")
    if unit is None or tuple(unit.shape) != tuple(weight.shape):
        unit = jt.ones(weight.shape, dtype="bfloat16")
        unit.stop_grad()
        weight.__dict__["_torch_acl_rms_norm_unit_weight"] = unit
    return unit


def _batch_norm_eval_cuda_acl(x, weight, bias, running_mean, running_var, eps):
    if not (
        isinstance(x, jt.Var)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
        and isinstance(running_mean, jt.Var)
        and isinstance(running_var, jt.Var)
        and isinstance(eps, Real)
    ):
        return None
    x_dt = _fast_dtype_name(x.dtype)
    if (
        x_dt not in _NORM_F16_F32_BF16
        or (x_dt == "bfloat16" and not supports_bfloat16())
        or _fast_dtype_name(weight.dtype) != x_dt
        or _fast_dtype_name(bias.dtype) != x_dt
    ):
        return None
    rm_dt = _fast_dtype_name(running_mean.dtype)
    rv_dt = _fast_dtype_name(running_var.dtype)
    if (rm_dt != x_dt and rm_dt != "float32") or (rv_dt != x_dt and rv_dt != "float32"):
        return None
    x_shape = x.shape
    if len(x_shape) != 4:
        return None
    n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    if n <= 0 or c <= 0 or h <= 0 or w <= 0:
        return None
    if (
        len(weight.shape) != 1 or weight.shape[0] != c
        or len(bias.shape) != 1 or bias.shape[0] != c
        or len(running_mean.shape) != 1 or running_mean.shape[0] != c
        or len(running_var.shape) != 1 or running_var.shape[0] != c
    ):
        return None
    epsilon = float(eps)
    if not (math.isfinite(epsilon) and epsilon > 0.0):
        return None
    mapped = getattr(getattr(jt.core, "ops", None), "mapped_batch_norm", None)
    if mapped is not None:
        return mapped(x, weight, bias, running_mean, running_var, 0.1, epsilon, False, False)[0]
    return BatchNormACL(epsilon)(x, weight, bias, running_mean, running_var)


def _batch_norm_train_statistics_acl(x, weight, bias, eps):
    """Fused ACL training BatchNorm returning ``(y, batch_mean, batch_var)``."""
    norm_dtypes = ("float32", "float16", "bfloat16") if supports_bfloat16() else ("float32", "float16")
    if not (
        isinstance(x, jt.Var)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
        and isinstance(eps, Real)
    ):
        return None
    x_dtype = _jittor_dtype_name(x.dtype)
    if (
        x_dtype not in norm_dtypes
        or _jittor_dtype_name(weight.dtype) != x_dtype
        or _jittor_dtype_name(bias.dtype) != x_dtype
    ):
        return None
    shape = tuple(int(size) for size in x.shape)
    epsilon = float(eps)
    if (
        len(shape) != 4
        or any(size <= 0 for size in shape)
        or tuple(weight.shape) != (shape[1],)
        or tuple(bias.shape) != (shape[1],)
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        return None
    # BatchNorm training requires >1 spatial*batch elements per channel so
    # variance is well-defined; let the portable path handle 1-element edge cases.
    if shape[0] * shape[2] * shape[3] <= 1:
        return None
    dev = int(jt.core.dispatch_context([x])[1])
    channels = shape[1]
    dummy_mean = jt.zeros((channels,), dtype=x_dtype).to_device(dev).stop_grad()
    dummy_var = jt.ones((channels,), dtype=x_dtype).to_device(dev).stop_grad()
    y, save_mean, save_invstd = BatchNormACL(epsilon, momentum=0.0, is_train=True).call_with_stats(
        x, weight, bias, dummy_mean, dummy_var
    )
    with jt.no_grad():
        mean_out = save_mean.stop_grad()
        inv_f = save_invstd.stop_grad().float32()
        var_f = jt.maximum((1.0 / (inv_f * inv_f)) - epsilon, 0.0)
        var_out = var_f if x_dtype == "float32" else var_f.cast(x_dtype)
    return y, mean_out, var_out


def _batch_norm_train_running_acl(x, running_mean, running_var, weight, bias, momentum, eps):
    """Fused ACL training BatchNorm updating running stats inside one kernel.

    Returns ``(y, new_running_mean, new_running_var)`` in pure SSA form so
    Jittor's lazy graph always tracks the running-buffer dependency while
    avoiding dummy allocations, invstd->var inversion, and Python EMA kernels.
    """
    if not (
        isinstance(x, jt.Var)
        and isinstance(running_mean, jt.Var)
        and isinstance(running_var, jt.Var)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
        and isinstance(momentum, Real)
        and isinstance(eps, Real)
    ):
        return None
    x_dtype = _fast_dtype_name(x.dtype)
    if (
        x_dtype not in _NORM_F16_F32_BF16
        or (x_dtype == "bfloat16" and not supports_bfloat16())
        or _fast_dtype_name(weight.dtype) != x_dtype
        or _fast_dtype_name(bias.dtype) != x_dtype
    ):
        return None
    rm_dt = _fast_dtype_name(running_mean.dtype)
    rv_dt = _fast_dtype_name(running_var.dtype)
    if (rm_dt != x_dtype and rm_dt != "float32") or (rv_dt != x_dtype and rv_dt != "float32"):
        return None
    x_shape = x.shape
    if len(x_shape) != 4:
        return None
    n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    if n <= 0 or c <= 0 or h <= 0 or w <= 0 or (n * h * w <= 1):
        return None
    if (
        len(weight.shape) != 1 or weight.shape[0] != c
        or len(bias.shape) != 1 or bias.shape[0] != c
        or len(running_mean.shape) != 1 or running_mean.shape[0] != c
        or len(running_var.shape) != 1 or running_var.shape[0] != c
    ):
        return None
    epsilon = float(eps)
    mom = float(momentum)
    if not (math.isfinite(epsilon) and epsilon > 0.0 and math.isfinite(mom) and 0.0 <= mom <= 1.0):
        return None
    mapped = getattr(getattr(jt.core, "ops", None), "mapped_batch_norm", None)
    if mapped is not None:
        res = mapped(x, weight, bias, running_mean, running_var, mom, epsilon, True, True)
        return res[0], res[1], res[2]
    y, new_rm, new_rv = BatchNormACL(epsilon, momentum=mom, is_train=True).call_with_running_stats(
        x, weight, bias, running_mean, running_var
    )
    return y, new_rm.stop_grad(), new_rv.stop_grad()


def _group_norm_cuda_acl(x, num_groups, weight, bias, eps):
    # aclnnGroupNorm takes the input dtype as-is (GroupNormOpRunner forwards
    # whatever dtype the Vars already have); float16 was verified correct
    # against a CPU reference (max abs diff ~0.007, well inside fp16 noise).
    # Restricting to float32 only forced every float16 call -- e.g. SD1.5's
    # VAE decoder, which runs its norms in fp16 -- through the generic
    # Sub/ReduceMean/Sqrt/RealDiv/Mul/Add decomposition instead of this one
    # fused kernel (>45% of that workload's device time in an aclprof
    # capture).
    norm_dtypes = ("float32", "float16", "bfloat16") if supports_bfloat16() else ("float32", "float16")
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


def _rms_norm_acl_supported(x, gamma, epsilon, x_dtype):
    """The existing subset, checked on a fallback or native rejection only."""
    x_shape = tuple((int(size) for size in x.shape))
    gamma_shape = tuple((int(size) for size in gamma.shape))
    gamma_dtype = _jittor_dtype_name(gamma.dtype)
    supported_gamma_dtypes = {
        "float32": ("float32",),
        "float16": ("float16", "float32"),
        "bfloat16": ("bfloat16", "float32"),
    }
    return (
        x_shape
        and all((size > 0 for size in x_shape))
        and (gamma_shape == (x_shape[-1],))
        and (gamma_dtype in supported_gamma_dtypes.get(x_dtype, ()))
        and (getattr(jt.flags, "no_grad", 0) or x_dtype in ("float32", "bfloat16"))
        and math.isfinite(epsilon)
        and (epsilon > 0)
    )


def _rms_norm_cuda_acl(x, gamma, epsilon=1e-06):
    if isinstance(x, jt.Var) and isinstance(gamma, jt.Var) and isinstance(epsilon, Real):
        x_dtype = _jittor_dtype_name(x.dtype)
        epsilon_value = float(epsilon)
        if (x_dtype == "float16" and x.ndim and x.shape[-1] < 16
                and soc_name().startswith("Ascend310P")):
            # The small-width FP16 kernel can return incorrect rows on 310P.
            # Both mapped and fresh-descriptor CodeOp paths reproduce it;
            # leave these shapes with device primitives.
            return None
        if (
            jt.flags.no_grad
            and (x_dtype == "float16" or (x_dtype == "bfloat16" and supports_bfloat16()))
            and _fast_dtype_name(gamma.dtype) == x_dtype
        ):
            mapped = getattr(getattr(jt.core, "ops", None), "mapped_rms_norm", None)
            if mapped is not None:
                try:
                    return mapped(x, gamma, epsilon_value, x_dtype == "bfloat16")
                except RuntimeError:
                    # Inspect unsupported metadata only on failure. Valid inputs
                    # must expose native/backend failures rather than fall back.
                    if not _rms_norm_acl_supported(x, gamma, epsilon_value, x_dtype):
                        return None
                    raise
        if _rms_norm_acl_supported(x, gamma, epsilon_value, x_dtype):
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
