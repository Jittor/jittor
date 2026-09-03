"""One-, two-, and three-dimensional convolution functionals."""

import jittor as jt
from jittor.misc import _pair, _triple

from ..backends import cudnn as _cudnn


def _check_conv2d_output_size(x, oh, ow, kernel_size, stride, padding, dilation):
    """Reject a geometry whose output has no elements, with the numbers in it."""
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f"Conv2d output size is non-positive (oh={oh}, ow={ow}): input "
            f"{tuple(x.shape)} is too small for kernel {tuple(kernel_size)}, "
            f"stride {tuple(stride)}, padding {tuple(padding)}, "
            f"dilation {tuple(dilation)}.")


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
           *, _depthwise_fast_path=True):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

    This is the only 2-D convolution in the package: ``nn.Conv2d.execute`` and
    ``nn.Conv._conv_forward`` both call it, so a layer computes the same thing
    however it is invoked.  It used to be transcribed twice, and the two copies
    had drifted: the module set ``{"G": G, "C": C}`` compile options on both operands
    while this one set ``{"G"}`` on one, the module validated the input rank,
    the channel count and the output size while this one validated nothing, and
    the module took a CUDA depthwise fast path that this one did not.

    ``_depthwise_fast_path`` is private: it exists so the depthwise CUDA kernel
    can be cross-checked against the generic grouped path
    (``tests/nn/test_depthwise_conv.py``).  Nothing else should pass it.

    :param x: the input image
    :type x: jt.Var

    :param weight: the convolution kernel
    :type weight: jt.Var

    :param bias: the bias after convolution
    :type bias: jt,Var, optional

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    Example:

    >>> x = jt.randn(4, 24, 100, 100)
    >>> w = jt.randn(32, 24, 3, 3)
    >>> y = nn.conv2d(x, w)
    '''
    padding = _pair(padding)
    stride = _pair(stride)
    dilation = _pair(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    # Clear, torch-grade errors for the two most common Conv2d misuses, instead
    # of an empty `AssertionError:` (channel mismatch) or a cryptic "not enough
    # values to unpack" (wrong ndim). These used to live only in the module.
    if x.ndim != 4:
        raise ValueError(
            f"Conv2d expected a 4-D input (N, C, H, W), but got a {x.ndim}-D input "
            f"of shape {tuple(x.shape)}.")
    in_channels = weight.shape[1] * groups
    if x.shape[1] != in_channels:
        raise ValueError(
            f"Conv2d expected input with {in_channels} channels (in_channels), "
            f"but got {x.shape[1]} channels; input shape {tuple(x.shape)}.")
    Kh, Kw = weight.shape[-2:]
    out_height = (x.shape[2]+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
    out_width = (x.shape[3]+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
    # Before any fast path: cuDNN and the depthwise kernel do not check this,
    # so validating inside the reindex branches only would leave CUDA silent.
    _check_conv2d_output_size(x, out_height, out_width, (Kh, Kw), stride,
                              padding, dilation)
    # Depthwise CUDA kernel; on CPU DepthwiseConv itself calls back into the
    # grouped path below, so it is only worth taking when it is really CUDA.
    if (_depthwise_fast_path
            and groups == out_channels == x.shape[1]
            and jt.flags.use_cuda and jt.compiler.is_cuda):
        y = jt.nn.DepthwiseConv(stride, padding, dilation)(x, weight)
        if bias is not None:
            y = y + bias.broadcast(y.shape, [0, 2, 3])
        return y
    # cuDNN path (memory-efficient fwd+bwd); falls back to reindex below on
    # CPU / no-cuDNN / non-float32. See nn/backends/cudnn.py.
    _y = _cudnn._try_cudnn_conv2d(
        x, weight, bias, stride, padding, dilation, groups)
    if _y is not None:
        return _y
    if groups == 1:
        N,C,H,W = x.shape
        oh, ow = out_height, out_width
        with jt.flag_scope(amp_reg=jt.flags.amp_reg | jt.amp_flags.keep_reduce
                      | jt.amp_flags.reduce16_no_fp32_acc):
            xx = x.reindex([N,out_channels,C,oh,ow,Kh,Kw], [
                    'i0', # Nid
                    'i2', # Cid
                    f'i3*{stride[0]}-{padding[0]}+i5*{dilation[0]}', # Hid+Khid
                    f'i4*{stride[1]}-{padding[1]}+i6*{dilation[1]}', # Wid+KWid
                ])
            ww = weight.broadcast(xx.shape, [0,3,4])
            yy = xx*ww
            y = yy.sum([2,5,6]) # Kc, Kh, Kw
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
    else:
        N,C,H,W = x.shape
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        oh, ow = out_height, out_width
        xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
            ])
        # w: [oc, CpG, Kh, Kw]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
        # Both operands, and both keys: the module set {"G": G, "C": C} on xx
        # and ww, this one set {"G": G} on xx alone, so the same layer produced
        # two different fused-op keys depending on how it was called.
        ww.compile_options = xx.compile_options = {"G":G,"C":C}
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y


def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ''' Applies a 3D convolution over an input signal composed of several input planes.

    :param x: the input volume
    :type x: jt.Var

    :param weight: the convolution kernel
    :type weight: jt.Var

    :param bias: the bias after convolution
    :type bias: jt,Var, optional

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    Example:

    >>> x = jt.randn(4, 24, 50, 50, 50)
    >>> w = jt.randn(32, 24, 3, 3, 3)
    >>> y = nn.conv2d(x, w)
    '''
    padding = _triple(padding)
    stride = _triple(stride)
    dilation = _triple(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if jt.flags.use_cuda and jt.cudnn:
        y = _cudnn._cudnn_conv3d_fp16_safe(
            jt.cudnn.ops.cudnn_conv3d, x, weight,
            *stride, *padding, *dilation, groups)
    elif groups == 1:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,out_channels,C,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i4*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
                f'i5*{stride[2]}-{padding[2]}+i8*{dilation[2]}', # Did+KDid
            ])
        ww = weight.broadcast(xx.shape, [0,3,4,5])
        yy = xx*ww
        y = yy.sum([2,6,7,8]) # Kc, Kh, Kw,Kd
    else:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,G,oc//G,CpG,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i7*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i8*{dilation[1]}', # Wid+KWid
                f'i6*{stride[2]}-{padding[2]}+i9*{dilation[2]}', # Did+KDid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kd, Kh, Kw]; the broadcast shape must match xx's axis
        # order [N,G,oc//G,CpG,od,oh,ow,Kd,Kh,Kw], so i7/i8/i9 are Kd/Kh/Kw.
        ww = weight.reindex([N, G, oc//G, CpG, od, oh, ow, Kd, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i7',
                'i8',
                'i9'
            ])
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, od, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5',
                'i6'
            ])

    if bias is not None:
        b = bias.broadcast(y.shape, [0,2,3,4])
        y = y + b
    return y


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ''' Applies a 1D convolution over an input signal composed of several input
    planes. Torch-compatible functional interface.

    :param input: the input var of shape ``(N, C_in, L)``
    :type input: jt.Var

    :param weight: the convolution kernel of shape ``(C_out, C_in//groups, kW)``
    :type weight: jt.Var

    :param bias: the optional bias of shape ``(C_out,)``. Default: None
    :type bias: jt.Var, optional

    :param stride: stride of the convolution. Default: 1
    :param padding: zero-padding added to both sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param groups: number of blocked connections. Default: 1

    Example:
        >>> x = jt.randn(4, 8, 100)
        >>> w = jt.randn(16, 8, 3)
        >>> y = nn.conv1d(x, w, stride=2, padding=1)
    '''
    if input.dim() != 3:
        raise RuntimeError(f'Expected 3D input to conv1d, but got input of size: {input.shape}')
    if weight.dim() != 3:
        raise RuntimeError(f'Expected 3D weight to conv1d, but got weight of size: {weight.shape}')
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    padding = padding[0] if isinstance(padding, (tuple, list)) else padding
    dilation = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
    # reuse the 2D conv by adding a singleton width dimension
    x = input.unsqueeze(-1)
    w = weight.unsqueeze(-1)
    y = jt.nn.conv2d(x, w, bias, (stride, 1), (padding, 0), (dilation, 1), groups)
    return y.squeeze(-1)
