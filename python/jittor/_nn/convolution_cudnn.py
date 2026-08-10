"""cuDNN adapters shared by convolution functional implementations."""

import os

from .runtime import jt, preserve_facade_origins


class _CudnnConv2d(jt.Function):
    ''' Memory-efficient 2D convolution backed by cuDNN.

    jittor's default conv (reindex + broadcast + reduce) fuses the *forward*
    fine, but its *backward* materializes a dense ``[N,Cout,Cin,oh,ow,Kh,Kw]``
    intermediate (e.g. ~30 GB for a 256->256 3x3 conv on a 2x80x80 batch),
    OOMing two-stage detectors at batch>=2. cuDNN computes both directions in
    bounded memory. jittor's *autodiff* through the raw ``cudnn_conv`` op is
    broken in this build (wrong grad shape, grad.cc:232), so we supply the
    backward explicitly via ``cudnn_conv_backward_{x,w}``. Numerically matches
    the reindex path to ~1e-7 (cuDNN accumulation order) and aligns with real
    torch's cuDNN better than the reindex path does. '''
    def execute(self, x, w, sh, sw, ph, pw, dh, dw, groups):
        self.saved = (x, w, sh, sw, ph, pw, dh, dw, groups)
        return jt.cudnn.ops.cudnn_conv(x, w, sh, sw, ph, pw, dh, dw, groups)
    def grad(self, gy):
        x, w, sh, sw, ph, pw, dh, dw, groups = self.saved
        H, W = x.shape[2], x.shape[3]
        Kh, Kw = w.shape[2], w.shape[3]
        gx = jt.cudnn.ops.cudnn_conv_backward_x(w, gy, H, W, sh, sw, ph, pw, dh, dw, groups)
        gw = jt.cudnn.ops.cudnn_conv_backward_w(x, gy, Kh, Kw, sh, sw, ph, pw, dh, dw, groups)
        return gx, gw, None, None, None, None, None, None, None

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
    y = jt.nn._CudnnConv2d()(x, weight, sh, sw, ph, pw, dh, dw, groups)
    if bias is not None:
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y

class _CudnnConvT2d(jt.Function):
    ''' Memory-efficient 2D transpose-convolution backed by cuDNN. Same rationale
    as _CudnnConv2d: jittor's reindex transpose-conv backward materializes a dense
    intermediate. Transpose-conv forward IS the conv-backward-x op; its own
    backward maps to cudnn_conv (grad input) + cudnn_conv_backward_w (grad weight).
    Validated to match the reindex path exactly for the mask-head 14->28 deconv. '''
    def execute(self, x, w, sh, sw, ph, pw, oph, opw, dh, dw, groups):
        N, C, H, W = x.shape
        Kh, Kw = w.shape[2], w.shape[3]
        oH = (H - 1) * sh - 2 * ph + dh * (Kh - 1) + oph + 1
        oW = (W - 1) * sw - 2 * pw + dw * (Kw - 1) + opw + 1
        self.saved = (x, w, sh, sw, ph, pw, dh, dw, groups)
        return jt.cudnn.ops.cudnn_conv_backward_x(w, x, oH, oW, sh, sw, ph, pw, dh, dw, groups)
    def grad(self, gy):
        x, w, sh, sw, ph, pw, dh, dw, groups = self.saved
        Kh, Kw = w.shape[2], w.shape[3]
        gx = jt.cudnn.ops.cudnn_conv(gy, w, sh, sw, ph, pw, dh, dw, groups)
        gw = jt.cudnn.ops.cudnn_conv_backward_w(gy, x, Kh, Kw, sh, sw, ph, pw, dh, dw, groups)
        return gx, gw, None, None, None, None, None, None, None, None, None

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
    y = jt.nn._CudnnConvT2d()(x, weight, sh, sw, ph, pw, oph, opw, dh, dw, groups)
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


_FACADE_SYMBOLS = (
    _CudnnConv2d, _try_cudnn_conv2d,
    _CudnnConvT2d, _try_cudnn_conv_transpose2d,
    _cudnn_conv3d_fp16_safe,
)
preserve_facade_origins(_FACADE_SYMBOLS)
