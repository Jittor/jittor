"""cuDNN adapters shared by convolution functional implementations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import os

import jittor as jt
from jittor._runtime.dispatch import native_rule, optional_kernel, register_kernel
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.core_api import _output_requires_grad
from jittor.nn.functional._amp import bias_for_compute_dtype
from jittor.nn.functional._layout import channels_last_source, channels_last_view, offer_channels_last



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

#: Where `_inference_filter` keeps, for the weight version it moved, the dense
#: OHWI tensor the weight is now a view of: (weight version, OHWI tensor). The
#: same storage, not a copy.
_FILTER_OHWI = "_jittor_conv_filter"

#: Whether a half-precision convolution without a backward moves its weight
#: into OHWI storage (see `_inference_filter`). The values do not change and
#: nothing is duplicated; only the weight's strides do.
channels_last_filters = True


def _is_ohwi_storage(weight):
    """Whether an OIHW-shaped weight is a view of dense OHWI storage."""
    if weight._storage_is_contiguous() or weight._storage_offset():
        return False
    o, i, kh, kw = (int(size) for size in weight.shape)
    return tuple(weight._storage_strides()) == (kh * kw * i, 1, kw * i, i)


def _inference_filter(x, weight, groups):
    """The filter and its layout for a convolution with no backward.

    For half precision cuDNN runs NHWC kernels, and handed an OIHW filter it
    converts it on every call -- 51 of the 54 ms of layout conversions in a
    20-step SD1.5 sample, for weights that never change there. So the weight
    itself moves into OHWI storage: its values, shape and every reader's
    answer stay what they were (it is read through OIHW strides), and cuDNN
    is handed the same bytes as OHWI. A copy kept beside the weight did the
    same at the price of the convolution weights twice over, 1.1 GB for the
    UNet.

    Only a materialized weight moves -- a parameter, not a filter computed in
    the call -- and never while a step is being traced, where rebinding a
    parameter reads as a state change. A 1x1 filter is the same bytes in
    either layout and is only relabelled. With a backward nothing moves; a
    weight moved earlier still reads right, through its strides.
    """
    if (groups != 1 or _jittor_dtype_name(weight.dtype) not in ("float16", "bfloat16")
            or _output_requires_grad(x, weight)):
        return weight, "oihw"
    out_channels, in_channels, kh, kw = (int(size) for size in weight.shape)
    if kh == 1 and kw == 1 and weight._storage_is_contiguous():
        return weight.reshape((out_channels, 1, 1, in_channels)), "ohwi"
    state = getattr(weight, "__dict__", None)
    if state is None:
        return weight, "oihw"
    moved = state.get(_FILTER_OHWI)
    if moved is not None and moved[0] == weight.var_ptr:
        return moved[1], "ohwi"
    from jittor._runtime import step_capture
    if (not channels_last_filters or not weight.is_finished
            or step_capture.tracing() or jt.flags.keep_graph):
        return weight, "oihw"
    dense = weight.transpose(0, 2, 3, 1).clone()
    with jt.flag_scope(transpose_storage_view=1):
        view = dense.transpose(0, 3, 1, 2)
    # Made under `no_grad`, the view is born stopped; a parameter that takes a
    # gradient has to go on taking one.
    view = view.stop_grad() if weight.is_stop_grad() else view.start_grad()
    # Settled now, so the weight answers for its strides; a view moves no data.
    jt.sync([view, dense], False, False)
    weight.update(view)
    state[_FILTER_OHWI] = (weight.var_ptr, dense.stop_grad())
    return dense, "ohwi"


#: Whether a half-precision convolution that records no gradient hands out its
#: result in channels-last storage, as an NCHW view of NHWC memory. Tensor-core
#: kernels compute in NHWC, so with NCHW activations cuDNN converts every input
#: on the way in and every output on the way out, through a workspace -- 147 MB
#: for one ResNet-50 layer at batch 64. The elementwise ops in between keep a
#: view's layout (`propagate_storage_layout`), so the next convolution receives
#: NHWC memory again and reads it as it is. Anything that needs dense NCHW gets
#: it by the ordinary contiguous copy, so values never depend on this.
channels_last_activations = True

_HALF = ("float16", "bfloat16")

#: Whether a convolution that records a gradient runs channels-last where
#: that pays, float32 included. cuDNN's tensor-core kernels compute in NHWC for
#: training too: handed NCHW, every forward, data-gradient and filter-gradient
#: call converts its operands on the way in and its result on the way out --
#: 7.4 ms of a 78 ms ResNet-50 training step, whose convolutions ran in 41.3 ms
#: NCHW against 37.1 NHWC. It only pays while what reads the result keeps the
#: layout: a batch norm does (its NHWC kernels), a group norm, an
#: interpolation or a reshape would convert it back, and a DDPM UNet trained
#: 13% slower with every convolution channels-last. So a convolution whose
#: input is NCHW hands out NCHW and offers the channels-last result
#: (`offer_channels_last`) for a reader that can use it; one whose input is
#: already channels-last stays channels-last. Gradients follow the same
#: layout back (`TransposeOp::grad` keeps a view a view). Grouped convolutions
#: stay NCHW: their channels-last kernels are the slow ones.
channels_last_training = True


def _training_channels_last(x, weight, bias, groups):
    """Whether a call that records a gradient may run channels-last."""
    dtype = _jittor_dtype_name(x.dtype)
    return (channels_last_training and groups == 1
            and (dtype in _HALF or dtype == "float32")
            and not jt.flags.no_grad and _output_requires_grad(x, weight, bias))


@native_rule("same_float:cudnn_conv")
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
    cudnn = get_library_ops("cudnn")

    def channels_last(source):
        y = cudnn.cudnn_conv(
            x if source is None else source, filter_, sh, sw, ph, pw, dh, dw, groups,
            "abcd" if source is None else "acdb", layout, "acdb")
        if bias is not None:
            y = y + bias_for_compute_dtype(y, bias)
        return channels_last_view(y)

    training = _training_channels_last(x, weight, bias, groups)
    if training or (channels_last_activations and _jittor_dtype_name(x.dtype) in _HALF
                    and (jt.flags.no_grad or not _output_requires_grad(x, weight, bias))):
        source = channels_last_source(x)
        if source is None:
            # A dense copy still to be made of a channels-last activation --
            # Diffusers takes `.contiguous()` before every shortcut convolution
            # -- read where it lies instead: the copy, a transpose of the whole
            # activation (823 us for an SD1.5 VAE one), is then never made.
            if x._is_pending_contiguous():
                source = channels_last_source(x._input(0))
        if source is not None or not training:
            return channels_last(source)
    y = cudnn.cudnn_conv(x, filter_, sh, sw, ph, pw, dh, dw, groups, "abcd", layout)
    if bias is not None:
        y = y + bias_for_compute_dtype(y, bias).broadcast(y.shape, [0, 2, 3])
    if training:
        offer_channels_last(y, lambda: channels_last(None))
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
