"""ACL neural-network kernels without Module or facade replacement."""

from numbers import Integral

import jittor as jt
from .ops._code import ACL_FLOAT_DTYPES
from .ops.conv_op import ConvACL
from .ops.bmm_op import BmmACL
from .ops.matmul_op import MatmulACL
from .ops.transpose_op import TransPoseACL
from .ops.upsample_op import UpsampleNearest2dACL
from .ops.relu_op import ReLUACL, LeakyReLUACL
from .ops.silu_op import SiLUACL, SwishACL, SwiGluACL
from .ops.softmax_op import SoftmaxACL
from .ops.pool_op import PoolACL
from .ops.rope_op import RotaryPositionEmbeddingACL


def conv_acl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, _depthwise_fast_path=True
):
    return ConvACL()(x, weight, bias, stride, padding, dilation, groups)


def bmm_acl(x1, x2, trans_a=False, trans_b=False):
    if trans_a:
        x1 = _transpose_last_two(x1)
    return BmmACL(trans_b)(x1, x2)


def matmul_acl(x1, x2, trans_a=False, trans_b=False):
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
    axes = list(range(x.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return TransPoseACL()(x, axes)


def resize_acl(input, size, mode="nearest", align_corners=False, tf_mode=False):
    if (
        mode == "nearest"
        and input.ndim == 4
        and (str(input.dtype) in ("float16", "float32", "bfloat16"))
        and (not align_corners)
        and (not tf_mode)
        and all((int(value) > 0 for value in input.shape))
    ):
        if isinstance(size, int):
            size = (size, size)
        return UpsampleNearest2dACL()(input, size)
    return None


def relu(x, inplace=False):
    return ReLUACL()(x)


def leaky_relu(x, scale=0.01, inplace=False):
    return LeakyReLUACL()(x, scale)


def _silu_acl(x, inplace=False):
    if isinstance(x, jt.Var) and str(x.dtype) in ("float32", "bfloat16"):
        if str(x.dtype) == "bfloat16":
            return SwishACL()(x)
        return SiLUACL()(x)
    return None


def _silu_and_mul_acl(x):
    if (
        getattr(jt.flags, "no_grad", 0)
        and isinstance(x, jt.Var)
        and (str(x.dtype) in ("float16", "bfloat16", "float32"))
        and (x.ndim > 0)
        and (int(x.shape[-1]) > 0)
        and (int(x.shape[-1]) % 2 == 0)
    ):
        return SwiGluACL()(x, -1)
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
        or str(input.dtype) not in ACL_FLOAT_DTYPES
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
    return str(x.dtype) == str(freq_cos.dtype) == str(freq_sin.dtype) and str(x.dtype) in (
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
        str(xq.dtype) not in ("float32", "bfloat16") or str(xk.dtype) not in ("float32", "bfloat16")
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
