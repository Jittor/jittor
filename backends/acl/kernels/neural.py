"""ACL neural-network kernels without Module or facade replacement."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from numbers import Integral

import jittor as jt
from .ops._code import ACL_FLOAT_DTYPES
from .ops.conv_op import ConvACL
from .ops.bmm_op import BmmACL
from .ops.cross_entropy_loss_op import CrossEntropyLossACL
from .ops.gelu_op import GeluACL
from .ops.matmul_op import MatmulACL
from .ops.upsample_op import UpsampleNearest2dACL
from .ops.relu_op import ReLUACL, LeakyReLUACL
from .ops.silu_op import SiLUACL, SwishACL, SwiGluACL
from .ops.softmax_op import SoftmaxACL
from .ops.pool_op import PoolACL
from .ops.rope_op import RotaryPositionEmbeddingACL


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


def _dtype_in(value, names):
    """Whether ``value``'s dtype is one of ``names``, a set of canonical names."""
    dtype = value.dtype
    return str(dtype) in names or _jittor_dtype_name(dtype) in names


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
    #: The core transpose op is a single aclnnPermute on this backend
    #: (the `transpose` row of `acl_ops` in `backends/acl/src/acl_op_exec.cc`),
    #: so there is nothing left for a CodeOp override to add here.
    axes = list(range(x.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return x.transpose(axes)


def resize_acl(input, size, mode="nearest", align_corners=False, tf_mode=False):
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
    return ReLUACL()(x)


def leaky_relu(x, scale=0.01, inplace=False):
    return LeakyReLUACL()(x, scale)


def gelu_acl(x, approximate="none"):
    """CANN's aclnnGelu covers the exact form only; tanh keeps the generic path."""
    if approximate != "none":
        return None
    if isinstance(x, jt.Var) and _dtype_in(x, _F16_F32_BF16):
        return GeluACL()(x)
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
    in_range = target * valid.cast(target.dtype)
    target_weight = valid.float32()
    loss = CrossEntropyLossACL()(output, in_range) * target_weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / target_weight.sum()
    return loss.reshape(target_shape)


def _silu_acl(x, inplace=False):
    if isinstance(x, jt.Var) and _dtype_in(x, _F32_BF16):
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
