"""ACL neural-network kernels without Module or facade replacement."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from numbers import Integral

import jittor as jt
from .ops._code import ACL_FLOAT_DTYPES
from .ops.conv_op import ConvACL, ConvTranspose2dACL
from .ops.bmm_op import BmmACL
from .ops.cross_entropy_loss_op import CrossEntropyLossACL
from .ops.gelu_op import GeluACL
from .ops.grid_sample_op import GridSampler2DACL
from .ops.matmul_op import MatmulACL
from .ops.upsample_op import UpsampleNearest2dACL, UpsampleBicubic2dACL, UpsampleBilinear2dACL
from .ops.relu_op import ReLUACL, LeakyReLUACL
from .ops.silu_op import SiLUACL, SwishACL, SwiGluACL, SoftplusACL
from .ops.softmax_op import SoftmaxACL
from .ops.pool_op import PoolACL, AdaptiveAvgPool2dACL
from .ops.quant_op import DynamicBlockQuantACL, DynamicMxQuantACL, QuantMatmulACL
from .ops.rope_op import RotaryPositionEmbeddingACL
from ._soc import soc_name, supports_bfloat16, supports_fp8, allow_reduced_precision


#: Dtype sets the decline guards below test membership in. A Var's dtype is a
#: native NanoString that prints its canonical name, so the raw spelling
#: answers first and only an unusual one reaches `dtype_name`; the guards ran
#: two or three `dtype_name` calls each, on every activation, product and
#: norm. Spelled out here rather than imported from `ops._code`: the ACL
#: structure fixture replaces everything these modules import from `ops.` with
#: a recorder, and a decline must not depend on one.
_ACL_FLOATS = frozenset(ACL_FLOAT_DTYPES)
_F32_BF16 = frozenset(("float32", "bfloat16"))
_F16_F32_BF16 = frozenset(("float16", "float32", "bfloat16"))
_INT32_INT64 = frozenset(("int32", "int64"))


def _fast_dtype_name(dtype):
    if dtype.__class__ is str:
        return dtype if dtype in _F16_F32_BF16 else _jittor_dtype_name(dtype)
    name = getattr(dtype, "name", None)
    if name is not None and name in _F16_F32_BF16:
        return name
    s = str(dtype)
    return s if s in _F16_F32_BF16 else _jittor_dtype_name(dtype)


def _dtype_in(value, names):
    """Whether ``value``'s dtype is one of ``names``, a set of canonical names."""
    dtype = value.dtype
    return str(dtype) in names or _jittor_dtype_name(dtype) in names


def conv_acl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, _depthwise_fast_path=True
):
    mapped = getattr(getattr(jt.core, "ops", None), "mapped_conv2d", None)
    if (
        mapped is not None
        and isinstance(x, jt.Var)
        and isinstance(weight, jt.Var)
        and x.ndim == 4
        and weight.ndim == 4
        and (bias is None or (isinstance(bias, jt.Var) and bias.ndim == 1))
    ):
        dt = _fast_dtype_name(x.dtype)
        if (
            dt in _F16_F32_BF16
            and (dt != "bfloat16" or supports_bfloat16())
            and _fast_dtype_name(weight.dtype) == dt
            and (bias is None or _fast_dtype_name(bias.dtype) == dt)
        ):
            if isinstance(stride, int):
                sh = sw = stride
            else:
                sh, sw = int(stride[0]), int(stride[1])
            if isinstance(padding, int):
                ph = pw = padding
            else:
                ph, pw = int(padding[0]), int(padding[1])
            if isinstance(dilation, int):
                dh = dw = dilation
            else:
                dh, dw = int(dilation[0]), int(dilation[1])
            g = int(groups)
            if sh > 0 and sw > 0 and dh > 0 and dw > 0 and ph >= 0 and pw >= 0 and g > 0:
                inputs = [x, weight] if bias is None else [x, weight, bias]
                return mapped(
                    inputs, sh, sw, ph, pw, dh, dw, g,
                    1 if allow_reduced_precision() else 0,
                )
    return ConvACL()(x, weight, bias, stride, padding, dilation, groups)


def conv3d_acl(x, weight, stride=1, padding=0, dilation=1, groups=1):
    # nn.functional.conv3d owns its separately rounded bias addition, and its
    # selected kernel receives no bias argument. Both dimensions use the same
    # aclnnConvolution runner with rank-sized spatial attributes.
    if (not jt.flags.no_grad
            and (not x.is_stop_grad() or not weight.is_stop_grad())
            and soc_name().startswith("Ascend310P")):
        # On 310P the forward is supported, but this SDK's 3-D
        # aclnnConvolutionBackward rejects the architecture with error 161002.
        # The 2-D backward and a separately added bias are unaffected.
        raise NotImplementedError(
            "ACL Conv3d gradients are unavailable on Ascend310P via "
            "aclnnConvolutionBackward; this path supports inference only "
            "(use no_grad or stop gradients on both input and weight)")
    return ConvACL()(x, weight, None, stride, padding, dilation, groups)


def conv_transpose2d_supported(x, weight, bias, stride, padding, output_padding,
                               dilation, groups):
    # Validated on 310P only. A 910B keeps its existing path until its SDK
    # transpose forward/precision contract has independent device evidence.
    if (not soc_name().startswith("Ascend310P") or not jt.flags.no_grad
            or x.ndim != 4 or weight.ndim != 4
            or not _dtype_in(x, ("float16", "float32"))
            or str(weight.dtype) != str(x.dtype)
            or not x._storage_is_contiguous() or not weight._storage_is_contiguous()
            or any(int(size) <= 0 for size in tuple(x.shape) + tuple(weight.shape))
            or not isinstance(groups, Integral) or isinstance(groups, bool)
            or groups <= 0 or x.shape[1] != weight.shape[0] or x.shape[1] % groups):
        return False
    if _dtype_in(x, ("float32",)) and not allow_reduced_precision():
        return False
    for values in (stride, padding, output_padding, dilation):
        if (not isinstance(values, (tuple, list)) or len(values) != 2
                or any(not isinstance(value, Integral) or isinstance(value, bool) for value in values)):
            return False
    for size, kernel, step, pad, out_pad, dil in zip(
            x.shape[2:], weight.shape[2:], stride, padding, output_padding, dilation):
        if (step <= 0 or dil <= 0 or pad < 0 or out_pad < 0
                or out_pad >= max(step, dil)
                or (size - 1) * step - 2 * pad + dil * (kernel - 1) + out_pad + 1 <= 0):
            return False
    return bias is None or (
        bias.ndim == 1 and bias.shape[0] == weight.shape[1] * groups
        and str(bias.dtype) == str(x.dtype) and bias._storage_is_contiguous())


def conv_transpose2d_acl(x, weight, bias, stride, padding, output_padding,
                         dilation, groups):
    return ConvTranspose2dACL()(x, weight, bias, stride, padding,
                                output_padding, dilation, groups)


def bmm_acl(x1, x2, trans_a=False, trans_b=False):
    if trans_a and not trans_b:
        return BmmACL(False, True)(x1, x2)
    if trans_a:
        x1 = _transpose_last_two(x1)
    return BmmACL(trans_b)(x1, x2)


def grid_sample_supported(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    return (soc_name().startswith("Ascend310P") and bool(jt.flags.no_grad)
            and mode == "bilinear" and padding_mode == "zeros" and align_corners is False
            and input.ndim == 4 and grid.ndim == 4
            and input.shape[0] == grid.shape[0] and grid.shape[-1] == 2
            and _dtype_in(input, ("float16", "float32"))
            and str(input.dtype) == str(grid.dtype)
            and all(int(size) > 0 for size in tuple(input.shape) + tuple(grid.shape)))


def grid_sample_acl(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    return GridSampler2DACL()(input, grid)


def matmul_acl(x1, x2, trans_a=False, trans_b=False):
    if trans_a and not trans_b and x1.ndim >= 2 and x2.ndim >= 2 and (x1.ndim == 2 or x2.ndim > 2):
        return MatmulACL(False, True)(x1, x2)
    if trans_a:
        x1 = _transpose_last_two(x1)
    if trans_b:
        return MatmulACL(True)(x1, x2)
    if x1.ndim == 1 and x2.ndim == 1:
        return MatmulACL()(x1.reshape((1, -1)), x2.reshape((-1, 1))).reshape((1,))
    if x1.ndim == 1:
        output_shape = x2.shape[:-2] + x2.shape[-1:]
        return MatmulACL()(x1.reshape((1, -1)), x2).reshape(output_shape)
    if x2.ndim == 1:
        output_shape = x1.shape[:-1]
        return MatmulACL()(x1, x2.reshape((-1, 1))).reshape(output_shape)
    return MatmulACL()(x1, x2)


def _transpose_last_two(x):
    #: The core transpose op is a single aclnnPermute on this backend
    #: (the `transpose` row of `acl_ops` in `backends/acl/src/acl_op_exec.cc`),
    #: so there is nothing left for a CodeOp override to add here.
    axes = list(range(x.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return x.transpose(axes)


def resize_acl(input, size, mode="nearest", align_corners=False, tf_mode=False):
    if (mode == "bilinear" and soc_name().startswith("Ascend310P")
            and input.ndim == 4 and _dtype_in(input, ("float16", "float32"))
            and getattr(jt.flags, "no_grad", 0) and not tf_mode
            and all(int(value) > 0 for value in input.shape)):
        if isinstance(size, int):
            size = (size, size)
        return UpsampleBilinear2dACL()(input, size, align_corners)
    if mode == "nearest-exact":
        if (input.ndim != 4 or not getattr(jt.flags, "no_grad", 0)
                or not _dtype_in(input, ("float16", "float32"))):
            return None
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or any(int(source) <= 0 or int(target) % int(source)
                                 for source, target in zip(input.shape[-2:], size)):
            return None
        # For integer upsampling, floor((i+0.5)/scale)==floor(i/scale),
        # so the existing nearest SDK runner implements nearest-exact too.
        mode = "nearest"
    if (mode == "bicubic" and input.ndim == 4
            and _dtype_in(input, ("float16", "float32"))
            and getattr(jt.flags, "no_grad", 0)
            and not align_corners and not tf_mode
            and all(int(value) > 0 for value in input.shape)):
        if isinstance(size, int):
            size = (size, size)
        return UpsampleBicubic2dACL()(input, size)
    if (
        mode == "nearest"
        and input.ndim == 4
        and _dtype_in(input, _F16_F32_BF16)
        and (not align_corners)
        and (not tf_mode)
        and all((int(value) > 0 for value in input.shape))
    ):
        if isinstance(size, int):
            size = (size, size)
        return UpsampleNearest2dACL()(input, size)
    return None


def relu(x, inplace=False):
    if isinstance(x, jt.Var):
        dt = _fast_dtype_name(x.dtype)
        if dt in _F16_F32_BF16 and (dt != "bfloat16" or supports_bfloat16()):
            mapped = getattr(getattr(jt.core, "ops", None), "mapped_unary", None)
            if mapped is not None:
                return mapped(x, 8, 0.0, 0.0)
    return ReLUACL()(x)


def leaky_relu(x, scale=0.01, inplace=False):
    return LeakyReLUACL()(x, scale)


def gelu_acl(x, approximate="none"):
    """CANN V2 preserves the requested exact or tanh algorithm and its gradient."""
    if approximate not in ("none", "tanh"):
        return None
    if isinstance(x, jt.Var):
        dt = _fast_dtype_name(x.dtype)
        if dt in _F16_F32_BF16:
            if dt == "bfloat16" and not supports_bfloat16():
                return None
            mapped = getattr(getattr(jt.core, "ops", None), "mapped_unary", None)
            if mapped is not None:
                return mapped(x, 7, float(approximate == "tanh"), 0.0)
            return GeluACL()(x, approximate)
    return None


def cross_entropy_loss_acl(output, target, weight=None, ignore_index=None,
                           reduction="mean"):
    """One aclnn launch for the loss and one for its gradient.

    Everything this declines falls back to the portable definition: a class
    weight vector (CANN applies it inside the reduction this function keeps in
    Python), a non-float32 logit tensor, and any rank the portable code does
    not itself flatten to ``[N, C]``.
    """
    if weight is not None or reduction not in ("none", "mean", "sum"):
        return None
    if not isinstance(output, jt.Var) or not isinstance(target, jt.Var):
        return None
    if str(output.dtype) != "float32" and _jittor_dtype_name(output.dtype) != "float32":
        return None
    if not _dtype_in(target, _INT32_INT64):
        return None
    target_shape = target.shape
    if output.ndim == 4:
        classes = int(output.shape[1])
        output = output.transpose((0, 2, 3, 1)).reshape((-1, classes))
    elif output.ndim != 2:
        return None
    classes = int(output.shape[1])
    if classes <= 0 or int(output.shape[0]) <= 0:
        return None
    target = target.reshape((-1,))
    if int(target.shape[0]) != int(output.shape[0]):
        return None

    # jittor gives an out-of-range label -- and one equal to ignore_index --
    # weight zero and keeps it out of the mean's denominator. aclnnCrossEntropy
    # Loss instead gathers with the raw label and faults the AI Core outside
    # [0, C), ignoreIndex included, so the label reaching it is masked to 0 and
    # the weighting is applied to the per-sample loss it returns.
    valid = jt.logical_and(target >= 0, target < classes)
    if ignore_index is not None:
        valid = jt.logical_and(valid, target != ignore_index)
    # jittor's frontend accepts either label width, but aclnn's
    # CrossEntropyLoss takes DT_INT64 only: CANN 9.0.0 rejects a DT_INT32
    # target with AclNN_Parameter_Error. The mask is applied in the target's
    # own width and the result is widened before it reaches the aclnn op.
    in_range = (target * valid.cast(target.dtype)).cast("int64")
    target_weight = valid.float32()
    loss = CrossEntropyLossACL()(output, in_range) * target_weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / target_weight.sum()
    return loss.reshape(target_shape)


def _silu_acl(x, inplace=False):
    if isinstance(x, jt.Var) and _dtype_in(x, _F16_F32_BF16):
        mapped = getattr(getattr(jt.core, "ops", None), "mapped_unary", None)
        if mapped is not None:
            return mapped(x, 1, 0.0, 0.0)
        if _jittor_dtype_name(x.dtype) == "bfloat16":
            return SwishACL()(x)
        return SiLUACL()(x)
    return None


def _silu_and_mul_acl(x):
    if (
        getattr(jt.flags, "no_grad", 0)
        and isinstance(x, jt.Var)
        and _dtype_in(x, _F16_F32_BF16)
        and (x.ndim > 0)
        and (int(x.shape[-1]) > 0)
        and (int(x.shape[-1]) % 2 == 0)
    ):
        return SwiGluACL()(x, -1)
    return None


def softplus_acl(x, beta=1.0, threshold=20.0):
    if (
        isinstance(x, jt.Var)
        and _dtype_in(x, _F16_F32_BF16)
        and isinstance(beta, (int, float))
        and isinstance(threshold, (int, float))
        and float(beta) > 0.0
    ):
        mapped = getattr(getattr(jt.core, "ops", None), "mapped_unary", None)
        if mapped is not None:
            return mapped(x, 2, float(beta), float(threshold))
        return SoftplusACL()(x, float(beta), float(threshold))
    return None


def softmax_supported(x, log=False, zero_all_neg_inf=False, dim=None):
    return not zero_all_neg_inf


def softmax_acl(x, log=False, zero_all_neg_inf=False, dim=None):
    if zero_all_neg_inf:
        return None
    if dim is None:
        dim = 0 if x.ndim in (0, 1, 3) else 1
    if log:
        x = x.float32()
        m = jt.max(x, dim, keepdims=True)
        shifted = x - m
        return shifted - jt.log(jt.exp(shifted).sum(dim, keepdims=True))
    if (
        isinstance(x, jt.Var)
        and isinstance(dim, Integral)
        and not isinstance(dim, bool)
        and x.ndim >= 1
        and _dtype_in(x, _F16_F32_BF16)
    ):
        mapped = getattr(getattr(jt.core, "ops", None), "mapped_unary", None)
        if mapped is not None:
            return mapped(x, 6, float(int(dim)), 0.0)
    return SoftmaxACL()(x, dim)


def _pool_pair(value):
    if isinstance(value, Integral):
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(part, Integral) for part in value):
            return tuple(int(part) for part in value)
    return None


def pool_acl(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=None,
    return_indices=None,
    ceil_mode=False,
    count_include_pad=True,
    op="maximum",
):
    if (
        op not in ("maximum", "mean")
        or not isinstance(input, jt.Var)
        or not _dtype_in(input, _ACL_FLOATS)
        or input.ndim != 4
        or any(int(size) <= 0 for size in input.shape)
        or (return_indices and op != "maximum")
    ):
        return None
    kernel = _pool_pair(kernel_size)
    strides = kernel if stride is None else _pool_pair(stride)
    pads = _pool_pair(padding)
    dilations = (1, 1) if dilation is None else _pool_pair(dilation)
    if any(value is None for value in (kernel, strides, pads, dilations)):
        return None
    if (
        dilations != (1, 1)
        or any(size <= 0 for size in kernel + strides)
        or any(pad < 0 or 2 * pad > size for pad, size in zip(pads, kernel))
    ):
        return None
    result = PoolACL(kernel, strides, pads, 1, return_indices, ceil_mode, count_include_pad, op)(
        input
    )
    if return_indices:
        value, indices = result
        return value, indices.int64()
    return result


def _rotary_supported(x, freq_cos, freq_sin):
    try:
        x_shape = tuple(int(size) for size in x.shape)
        cos_shape = tuple(int(size) for size in freq_cos.shape)
        sin_shape = tuple(int(size) for size in freq_sin.shape)
    except (TypeError, ValueError):
        return False
    if len(x_shape) != 4 or len(cos_shape) != 4 or cos_shape != sin_shape:
        return False
    if any(size <= 0 for size in x_shape + cos_shape):
        return False
    if x_shape[-1] % 64 or x_shape[-1] > 1024:
        return False
    if any(scale not in (1, size) for scale, size in zip(cos_shape, x_shape)):
        return False
    return _jittor_dtype_name(x.dtype) == _jittor_dtype_name(freq_cos.dtype) == _jittor_dtype_name(freq_sin.dtype) and _jittor_dtype_name(x.dtype) in (
        "float16",
        "float32",
        "bfloat16",
    )


def rope_acl(xq, xk, freqs_cis=None, freq_sin=None, freq_cos=None):
    if freqs_cis is not None:
        freq_cos, freq_sin = freqs_cis[..., 0], freqs_cis[..., 1]
    if freq_cos is None or freq_sin is None:
        return None
    if not getattr(jt.flags, "no_grad", 0) and (
        _jittor_dtype_name(xq.dtype) not in ("float32", "bfloat16") or _jittor_dtype_name(xk.dtype) not in ("float32", "bfloat16")
    ):
        return None
    if not (
        _rotary_supported(xq, freq_cos, freq_sin) and _rotary_supported(xk, freq_cos, freq_sin)
    ):
        return None
    return (
        RotaryPositionEmbeddingACL()(xq, freq_cos, freq_sin),
        RotaryPositionEmbeddingACL()(xk, freq_cos, freq_sin),
    )


def adaptive_avg_pool2d_acl(x, output_size):
    return AdaptiveAvgPool2dACL()(x, output_size)


_FP8_COMPUTE_DTYPES = frozenset(("float8_e4m3fn", "float8_e5m2"))
_FP8_QUANT_IN_DTYPES = frozenset(("bfloat16", "float16"))
_FP8_OUT_DTYPES = frozenset(("bfloat16", "float16", "float32"))


def dynamic_block_quant_supported(x, row_block_size=1, col_block_size=128,
                                  dst_dtype="float8_e4m3fn", min_scale=0.0):
    if not supports_fp8() or x.ndim < 2:
        return False
    if not _dtype_in(x, _FP8_QUANT_IN_DTYPES):
        return False
    if _jittor_dtype_name(dst_dtype) not in _FP8_COMPUTE_DTYPES:
        return False
    row_blk = int(row_block_size)
    col_blk = int(col_block_size)
    if row_blk <= 0 or col_blk <= 0:
        return False
    m_dim = 1
    for size in x.shape[:-1]:
        m_dim *= int(size)
    k_dim = int(x.shape[-1])
    return m_dim > 0 and k_dim > 0 and (m_dim % row_blk == 0) and (k_dim % col_blk == 0)


def dynamic_block_quant_acl(x, row_block_size=1, col_block_size=128,
                            dst_dtype="float8_e4m3fn", min_scale=0.0):
    return DynamicBlockQuantACL()(
        x,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
        dst_dtype=dst_dtype,
        min_scale=min_scale,
    )


def dynamic_mx_quant_supported(x, axis=-1, block_size=32,
                               dst_dtype="float8_e4m3fn", scale_alg=0):
    if not supports_fp8() or x.ndim < 2:
        return False
    if not _dtype_in(x, _FP8_QUANT_IN_DTYPES):
        return False
    if _jittor_dtype_name(dst_dtype) not in _FP8_COMPUTE_DTYPES:
        return False
    blk = int(block_size)
    if blk != 32:
        return False
    norm_axis = int(axis) if int(axis) >= 0 else int(axis) + x.ndim
    if not (0 <= norm_axis < x.ndim):
        return False
    dim = int(x.shape[norm_axis])
    return dim > 0 and dim % 64 == 0


def dynamic_mx_quant_acl(x, axis=-1, block_size=32,
                         dst_dtype="float8_e4m3fn", scale_alg=0):
    return DynamicMxQuantACL()(
        x,
        axis=axis,
        block_size=block_size,
        dst_dtype=dst_dtype,
        scale_alg=scale_alg,
    )


def quant_matmul_supported(x1, x2, x1_scale, x2_scale, *, bias=None,
                           out_dtype="bfloat16", trans_a=False, trans_b=False,
                           trans_x2_scale=False, group_size=0):
    if not supports_fp8() or x1.ndim < 2 or x2.ndim != 2:
        return False
    if not (_dtype_in(x1, _FP8_COMPUTE_DTYPES) and _dtype_in(x2, _FP8_COMPUTE_DTYPES)):
        return False
    if _jittor_dtype_name(out_dtype) not in _FP8_OUT_DTYPES:
        return False
    s1_dt = _jittor_dtype_name(x1_scale.dtype)
    s2_dt = _jittor_dtype_name(x2_scale.dtype)
    if not ((s1_dt == "float32" and s2_dt == "float32")
            or (s1_dt == "float8_e8m0fnu" and s2_dt == "float8_e8m0fnu")):
        return False
    return True


def quant_matmul_acl(x1, x2, x1_scale, x2_scale, *, bias=None,
                     out_dtype="bfloat16", trans_a=False, trans_b=False,
                     trans_x2_scale=False, group_size=0):
    return QuantMatmulACL()(
        x1,
        x2,
        x1_scale,
        x2_scale,
        bias=bias,
        out_dtype=out_dtype,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_x2_scale=trans_x2_scale,
        group_size=group_size,
    )


def fp8_linear_acl(
    x,
    weight_fp8,
    weight_scale,
    bias=None,
    *,
    mode="block128",
    block_size=128,
    out_dtype=None,
):
    if not (
        jt.flags.no_grad
        and supports_fp8()
        and isinstance(x, jt.Var)
        and isinstance(weight_fp8, jt.Var)
        and isinstance(weight_scale, jt.Var)
        and x.ndim >= 2
        and weight_fp8.ndim == 2
        and _dtype_in(x, _FP8_QUANT_IN_DTYPES)
        and _dtype_in(weight_fp8, _FP8_COMPUTE_DTYPES)
        and weight_fp8._storage_is_contiguous()
        and weight_scale._storage_is_contiguous()
    ):
        return None
    target_dtype = _jittor_dtype_name(x.dtype) if out_dtype is None else _jittor_dtype_name(out_dtype)
    if target_dtype not in _FP8_OUT_DTYPES:
        return None
    k_dim = x.shape[-1]
    n_dim = weight_fp8.shape[0]
    if k_dim != weight_fp8.shape[1] or k_dim <= 0 or n_dim <= 0:
        return None
    scale_dtype = _jittor_dtype_name(weight_scale.dtype)
    sshape = weight_scale.shape
    if mode == "block128":
        blk = int(block_size)
        if (
            blk <= 0
            or (k_dim % blk != 0)
            or (n_dim % blk != 0)
            or scale_dtype != "float32"
            or len(sshape) != 2
            or sshape[0] != n_dim // blk
            or sshape[1] != k_dim // blk
        ):
            return None
        mode_id = 0
    elif mode == "mxfp8":
        blk = 32
        if (
            (k_dim % 64 != 0)
            or scale_dtype != "float8_e8m0fnu"
            or len(sshape) != 3
            or sshape[0] != n_dim
            or sshape[1] != k_dim // 64
            or sshape[2] != 2
        ):
            return None
        mode_id = 1
    else:
        return None
    x_in = x if x._storage_is_contiguous() else x.contiguous()
    out = jt.core.ops.mapped_fp8_linear(
        x_in, weight_fp8, weight_scale, mode_id, blk, target_dtype
    )
    if bias is not None:
        bias_cast = bias if _jittor_dtype_name(bias.dtype) == target_dtype else bias.cast(target_dtype)
        out = out + bias_cast
    return out


def linear_bias_supported(x, weight, bias):
    if jt.flags.amp_reg:
        return False
    if not (isinstance(x, jt.Var) and isinstance(weight, jt.Var) and isinstance(bias, jt.Var)):
        return False
    if x.ndim < 2 or weight.ndim != 2 or bias.ndim != 1:
        return False
    dt = _jittor_dtype_name(x.dtype)
    if dt not in _F16_F32_BF16:
        return False
    if dt == "bfloat16" and not supports_bfloat16():
        return False
    if _jittor_dtype_name(weight.dtype) != dt or _jittor_dtype_name(bias.dtype) != dt:
        return False
    if int(weight.shape[0]) != int(bias.shape[0]) or int(x.shape[-1]) != int(weight.shape[1]):
        return False
    if any(int(s) <= 0 for s in x.shape) or int(weight.shape[0]) <= 0:
        return False
    return getattr(getattr(jt.core, "ops", None), "mapped_addmm", None) is not None


def linear_bias_acl(x, weight, bias):
    x_in = x if x._storage_is_contiguous() else x.contiguous()
    w_in = weight if weight._storage_is_contiguous() else weight.contiguous()
    b_in = bias if bias._storage_is_contiguous() else bias.contiguous()
    return jt.core.ops.mapped_addmm(x_in, w_in, b_in, allow_reduced_precision())


def grouped_mm_acl(mat_a, mat_b, offs=None):
    if offs is None:
        return None
    if not (isinstance(mat_a, jt.Var) and isinstance(mat_b, jt.Var) and isinstance(offs, jt.Var)):
        return None
    if mat_a.ndim != 2 or mat_b.ndim != 3 or offs.ndim != 1:
        return None
    dt = _jittor_dtype_name(mat_a.dtype)
    if dt not in _F16_F32_BF16 or _jittor_dtype_name(mat_b.dtype) != dt:
        return None
    if dt == "bfloat16" and not supports_bfloat16():
        return None
    if _jittor_dtype_name(offs.dtype) not in ("int32", "int64"):
        return None
    mapped = getattr(getattr(jt.core, "ops", None), "mapped_grouped_matmul", None)
    if mapped is None:
        return None
    a_in = mat_a if mat_a._storage_is_contiguous() else mat_a.contiguous()
    offs_in = offs if _jittor_dtype_name(offs.dtype) == "int64" else offs.cast("int64")
    if not offs_in._storage_is_contiguous():
        offs_in = offs_in.contiguous()
    return mapped(a_in, mat_b, offs_in)
