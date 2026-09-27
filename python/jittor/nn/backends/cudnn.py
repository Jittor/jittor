"""cuDNN adapters shared by convolution functional implementations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import os

import jittor as jt
from jittor._runtime.dispatch import optional_kernel, register_kernel
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.core_api import _output_requires_grad
from jittor.nn.functional._amp import bias_for_compute_dtype

from jittor.backends.cuda.kernels.nn.channel_bias_cuda import _channel_bias_add_cuda


# Why cuDNN at all: jittor's default conv (reindex + broadcast + reduce) fuses
# the *forward* fine, but its *backward* materializes a dense
# ``[N,Cout,Cin,oh,ow,Kh,Kw]`` intermediate (~30 GB for a 256->256 3x3 conv on
# a 2x80x80 batch), OOMing two-stage detectors at batch>=2. cuDNN computes both
# directions in bounded memory.
#
# The backward is the C++ op's (``CudnnConvOp::grad``), not this layer's. It
# used to be written out again here in a ``jt.Function``, because autodiff
# through the raw op returned a wrongly shaped gradient -- the op's ``grad``
# compared against a layout name it never receives and handed the backward ops
# NHWC positions for an NCHW input. That was fixed in the op; the copy here
# stayed, and being a copy it kept winning, so the fix had no effect on
# anything that went through ``conv2d``. One definition now.

#: Where `_inference_filter` keeps a weight's OHWI copy: (weight version, copy).
_FILTER_CACHE = "_jittor_conv_filter"

#: Whether a half-precision convolution without a backward keeps an OHWI copy
#: of its filter (see `_inference_filter`). The copy costs one more copy of the
#: convolution weights -- 1.1 GB for the SD1.5 UNet -- for 13% of a sampling
#: step; set this False where memory is the tighter bound.
cache_half_filters = True


def _inference_filter(x, weight, groups):
    """The filter and its layout for a convolution with no backward.

    For half precision cuDNN runs NHWC kernels, and handed an OIHW filter it
    converts it on every call -- 51 of the 54 ms its layout conversions took in
    a 20-step SD1.5 sample, for weights that never change there. The OHWI copy
    is made once per weight version (the Var a parameter holds; loading or
    replacing it moves the holder to another) and kept on the weight itself,
    so it lives exactly as long as the parameter. A 1x1 filter is the same
    bytes in either layout and needs no copy. With a backward -- training, or
    an input that needs a gradient -- the weight changes every step, a copy
    would cost what the conversion does, and the filter stays as it is.
    """
    if (groups != 1 or _jittor_dtype_name(weight.dtype) not in ("float16", "bfloat16")
            or _output_requires_grad(x, weight)):
        return weight, "oihw"
    out_channels, in_channels, kh, kw = (int(size) for size in weight.shape)
    if kh == 1 and kw == 1:
        return weight.reshape((out_channels, 1, 1, in_channels)), "ohwi"
    if not cache_half_filters:
        return weight, "oihw"
    state = getattr(weight, "__dict__", None)
    if state is None:
        return weight, "oihw"
    version = weight.var_ptr
    cached = state.get(_FILTER_CACHE)
    if cached is None or cached[0] != version:
        cached = (version, weight.transpose(0, 2, 3, 1).clone().stop_grad())
        state[_FILTER_CACHE] = cached
    return cached[1], "ohwi"


def _supports_conv2d(x, weight, bias, stride, padding, dilation, groups,
                     *, _depthwise_fast_path=True):
    return x.dtype == weight.dtype and get_library_ops("cudnn") is not None


@optional_kernel("conv2d", "cuda", dtypes={"float16", "float32", "bfloat16"},
                 supports=_supports_conv2d, priority=10)
def _try_cudnn_conv2d(x, weight, bias, stride, padding, dilation, groups,
                      *, _depthwise_fast_path=True):
    ''' Return a cuDNN-backed conv2d result, or None if cuDNN isn't applicable
    so the caller falls back to the reindex path. '''
    sh, sw = stride   if isinstance(stride, tuple)   else (stride, stride)
    ph, pw = padding  if isinstance(padding, tuple)  else (padding, padding)
    dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    filter_, layout = _inference_filter(x, weight, groups)
    y = get_library_ops("cudnn").cudnn_conv(x, filter_, sh, sw, ph, pw, dh, dw, groups,
                                            "abcd", layout)
    if bias is not None:
        fast = _channel_bias_add_cuda(y, bias)
        y = fast if fast is not None else y + bias_for_compute_dtype(
            y, bias).broadcast(y.shape, [0, 2, 3])
    return y

# Same story for the transpose: the forward *is* the conv-backward-x op, and
# ``CudnnConvBackwardXOp::grad`` defines its backward. This layer only has to
# work out the output size cuDNN needs told.

def _supports_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    return get_library_ops("cudnn") is not None


@optional_kernel("conv_transpose2d", "cuda", dtypes={"float32"},
                 supports=_supports_conv_transpose2d)
def _try_cudnn_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    ''' cuDNN-backed conv_transpose2d, or None to fall back to the reindex path. '''
    sh, sw = stride         if isinstance(stride, tuple)         else (stride, stride)
    ph, pw = padding        if isinstance(padding, tuple)        else (padding, padding)
    oph, opw = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    dh, dw = dilation       if isinstance(dilation, tuple)       else (dilation, dilation)
    H, W = x.shape[2], x.shape[3]
    Kh, Kw = weight.shape[2], weight.shape[3]
    oH = (H - 1) * sh - 2 * ph + dh * (Kh - 1) + oph + 1
    oW = (W - 1) * sw - 2 * pw + dw * (Kw - 1) + opw + 1
    y = get_library_ops("cudnn").cudnn_conv_backward_x(
        weight, x, oH, oW, sh, sw, ph, pw, dh, dw, groups)
    if isinstance(bias, jt.Var):
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y

# cuDNN 3D convolution needs fp32 accumulation and tensor-op math enabled for
# fp16/bf16 descriptors on some CUDA/cuDNN combinations. The C++ op configures
# that path by default; keep a fallback switch for isolating driver regressions.
_CUDNN_3D_HALF_DTYPES = ("float16", "bfloat16")

def _cudnn_conv3d_fp16_safe(op, x, weight, *args):
    xd, wd = _jittor_dtype_name(x.dtype), _jittor_dtype_name(weight.dtype)
    half = xd if xd in _CUDNN_3D_HALF_DTYPES else (wd if wd in _CUDNN_3D_HALF_DTYPES else None)
    if half is None:
        return op(x, weight, *args)
    if os.environ.get("JITTOR_CUDNN3D_HALF_NATIVE", "1") != "0":
        return op(x, weight, *args)
    # Run in fp32 (cuDNN has a working fp32 3D-conv algo), then cast back.
    y = op(x.float32(), weight.float32(), *args)
    return y.cast(half)


def _supports_conv3d(x, weight, stride, padding, dilation, groups):
    return get_library_ops("cudnn") is not None


@optional_kernel("conv3d", "cuda", supports=_supports_conv3d)
def _try_cudnn_conv3d(x, weight, stride, padding, dilation, groups):
    return _cudnn_conv3d_fp16_safe(get_library_ops("cudnn").cudnn_conv3d,
                                  x, weight, *stride, *padding, *dilation, groups)


def _supports_conv_transpose3d(x, weight, output_shape, stride, padding, dilation, groups):
    return get_library_ops("cudnn") is not None


@optional_kernel("conv_transpose3d", "cuda", supports=_supports_conv_transpose3d)
def _try_cudnn_conv_transpose3d(x, weight, output_shape, stride, padding, dilation, groups):
    return _cudnn_conv3d_fp16_safe(get_library_ops("cudnn").cudnn_conv3d_backward_x,
                                  weight, x, *output_shape, *stride, *padding, *dilation, groups)


for _backend in ("rocm_legacy", "corex_legacy"):
    register_kernel("conv2d", _backend, _try_cudnn_conv2d.__wrapped__,
                    dtypes={"float16", "float32", "bfloat16"},
                    supports=_supports_conv2d, priority=10)
    register_kernel("conv_transpose2d", _backend, _try_cudnn_conv_transpose2d.__wrapped__,
                    dtypes={"float32"}, supports=_supports_conv_transpose2d)
    register_kernel("conv3d", _backend, _try_cudnn_conv3d.__wrapped__,
                    supports=_supports_conv3d)
    register_kernel("conv_transpose3d", _backend, _try_cudnn_conv_transpose3d.__wrapped__,
                    supports=_supports_conv_transpose3d)
del _backend
