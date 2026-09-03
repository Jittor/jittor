"""cuDNN adapters shared by convolution functional implementations."""

import os

import jittor as jt


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

def _try_cudnn_conv2d(x, weight, bias, stride, padding, dilation, groups):
    ''' Return a cuDNN-backed conv2d result, or None if cuDNN isn't applicable
    so the caller falls back to the reindex path. Low-precision tensors use
    cuDNN only for inference; their training backward stays on the established
    reindex implementation. '''
    if not (jt.flags.use_cuda and getattr(jt, "cudnn", None)):
        return None
    x_dtype, weight_dtype = str(x.dtype), str(weight.dtype)
    if x_dtype != weight_dtype:
        return None
    if x_dtype != "float32" and not (
            x_dtype in ("float16", "bfloat16")
            and getattr(jt.flags, "no_grad", 0)):
        return None
    sh, sw = stride   if isinstance(stride, tuple)   else (stride, stride)
    ph, pw = padding  if isinstance(padding, tuple)  else (padding, padding)
    dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    y = jt.cudnn.ops.cudnn_conv(x, weight, sh, sw, ph, pw, dh, dw, groups)
    if bias is not None:
        fast = jt.nn._channel_bias_add_cuda(y, bias)
        y = fast if fast is not None else y + bias.broadcast(y.shape, [0, 2, 3])
    return y

# Same story for the transpose: the forward *is* the conv-backward-x op, and
# ``CudnnConvBackwardXOp::grad`` defines its backward. This layer only has to
# work out the output size cuDNN needs told.

def _try_cudnn_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    ''' cuDNN-backed conv_transpose2d, or None to fall back to the reindex path. '''
    if not (jt.flags.use_cuda and getattr(jt, "cudnn", None)):
        return None
    if str(x.dtype) != "float32" or str(weight.dtype) != "float32":
        return None
    sh, sw = stride         if isinstance(stride, tuple)         else (stride, stride)
    ph, pw = padding        if isinstance(padding, tuple)        else (padding, padding)
    oph, opw = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    dh, dw = dilation       if isinstance(dilation, tuple)       else (dilation, dilation)
    H, W = x.shape[2], x.shape[3]
    Kh, Kw = weight.shape[2], weight.shape[3]
    oH = (H - 1) * sh - 2 * ph + dh * (Kh - 1) + oph + 1
    oW = (W - 1) * sw - 2 * pw + dw * (Kw - 1) + opw + 1
    y = jt.cudnn.ops.cudnn_conv_backward_x(
        weight, x, oH, oW, sh, sw, ph, pw, dh, dw, groups)
    if isinstance(bias, jt.Var):
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y

# cuDNN 3D convolution needs fp32 accumulation and tensor-op math enabled for
# fp16/bf16 descriptors on some CUDA/cuDNN combinations. The C++ op configures
# that path by default; keep a fallback switch for isolating driver regressions.
_CUDNN_3D_HALF_DTYPES = ("float16", "bfloat16")

def _cudnn_conv3d_fp16_safe(op, x, weight, *args):
    xd, wd = str(x.dtype), str(weight.dtype)
    half = xd if xd in jt.nn._CUDNN_3D_HALF_DTYPES else (wd if wd in jt.nn._CUDNN_3D_HALF_DTYPES else None)
    if half is None:
        return op(x, weight, *args)
    if os.environ.get("JITTOR_CUDNN3D_HALF_NATIVE", "1") != "0":
        return op(x, weight, *args)
    # Run in fp32 (cuDNN has a working fp32 3D-conv algo), then cast back.
    y = op(x.float32(), weight.float32(), *args)
    return y.cast(half)
