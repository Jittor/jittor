"""oneDNN adapters for the CPU rows of the convolution kernel table.

Why this file exists at all: the CPU used to reach oneDNN's convolution
without a kernel row, by ``ConvTuner`` recognising the fused
``reindex * broadcast -> reduce`` subgraph and relaying it to ``mkl_conv``.
That relay is dead -- ``BroadcastToOp`` became a storage descriptor
(``share_with`` plus stride 0, ``OpType::other``), so it no longer enters a
fused op and the tuner's ``is_op(broadcast_to())`` pattern never matches.
Nothing failed loudly; the convolution simply ran as the generic fused kernel.
Measured on this machine, ``8x64x56x56`` against a ``64x64x3x3`` filter:

  jittor nn.Conv2d   0.1079 s    17.1 GFLOP/s
  torch conv2d (CPU) 0.0031 s   602.0 GFLOP/s      35x

and the generic path's *backward* is worse than its forward: it materializes a
dense ``[N,Cout,Cin,oh,ow,Kh,Kw]`` intermediate, which is the same reason the
cuDNN row exists (see ``backends/cudnn.py``).

Registering the row makes the CPU reach oneDNN the way CUDA reaches cuDNN,
without depending on a pattern match over the fused graph.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from jittor._runtime.dispatch import optional_kernel
from jittor._runtime.backend_libraries import get_library_ops


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _onednn_conv_ops():
    """The oneDNN operator module, loaded on demand, or None."""
    ops = get_library_ops("mkl", load=True)
    if ops is None or not hasattr(ops, "mkl_conv"):
        return None
    return ops


def _supports_conv2d(x, weight, bias, stride, padding, dilation, groups,
                     *, _depthwise_fast_path=True):
    """oneDNN's convolution: float32, NCHW/oihw, and both operands dense.

    float32 only is the library's own declaration, not a conservatism of this
    file -- ``mkl_capabilities.cc`` registers the capability with
    ``f32_only`` and ``MklConvOp`` asserts on anything else, and an assert
    aborts rather than declines. The layout strings are the ones every caller
    in ``conv2d`` passes; oneDNN takes others, but nothing here produces them.

    Density is required because the JIT body hands ``mem_ptr`` to oneDNN with
    the shape alone, so a strided view would be read as though it were dense.
    """
    if _jittor_dtype_name(x.dtype) != "float32":
        return False
    if x.dtype != weight.dtype:
        return False
    if bias is not None and bias.dtype != x.dtype:
        return False
    if not (x._storage_is_contiguous() and weight._storage_is_contiguous()):
        return False
    return _onednn_conv_ops() is not None


@optional_kernel("conv2d", "cpu", dtypes={"float32"}, supports=_supports_conv2d)
def _try_onednn_conv2d(x, weight, bias, stride, padding, dilation, groups,
                       *, _depthwise_fast_path=True):
    """A oneDNN-backed conv2d result, or None so the caller falls back."""
    ops = _onednn_conv_ops()
    if ops is None:
        return None
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    y = ops.mkl_conv(x, weight, sh, sw, ph, pw, dh, dw, groups,
                     "abcd", "oihw", "abcd")
    if bias is not None:
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y


# Same story one row down: conv_transpose's *forward* is the conv-backward-x
# op, and the CPU had no row for it either. Measured, `8x64x28x28` through a
# `64x64x3x3` filter at stride 2: 2.4364 s against torch's 0.0013 s, 1874x --
# worse than the forward convolution because the generic path's
# `broadcast * broadcast -> reindex_reduce` is a scatter, not a gather.


def _supports_conv_transpose2d(x, weight, bias, stride, padding, output_padding,
                               dilation, groups):
    if _jittor_dtype_name(x.dtype) != "float32":
        return False
    if x.dtype != weight.dtype:
        return False
    if bias is not None and bias.dtype != x.dtype:
        return False
    if groups != 1:
        return False
    if not (x._storage_is_contiguous() and weight._storage_is_contiguous()):
        return False
    ops = get_library_ops("mkl", load=True)
    return ops is not None and hasattr(ops, "mkl_conv_backward_x")


@optional_kernel("conv_transpose2d", "cpu", dtypes={"float32"},
                 supports=_supports_conv_transpose2d)
def _try_onednn_conv_transpose2d(x, weight, bias, stride, padding,
                                 output_padding, dilation, groups):
    """A oneDNN-backed conv_transpose2d result, or None to fall back."""
    ops = get_library_ops("mkl", load=True)
    if ops is None or not hasattr(ops, "mkl_conv_backward_x"):
        return None
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    dh, dw = _pair(dilation)
    H, W = x.shape[2], x.shape[3]
    kh, kw = weight.shape[2], weight.shape[3]
    # The size the *forward* convolution would have consumed. output_padding is
    # smaller than the stride, so this stays the size the op's own consistency
    # check derives back.
    oh = (H - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    ow = (W - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    y = ops.mkl_conv_backward_x(weight, x, oh, ow, sh, sw, ph, pw, dh, dw,
                                groups, "abcd", "oihw", "abcd")
    if bias is not None:
        y = y + bias.broadcast(y.shape, [0, 2, 3])
    return y
