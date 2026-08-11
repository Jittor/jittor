"""One-, two-, and three-dimensional convolution functionals."""

import jittor as jt


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

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
    padding = jt.nn._pair(padding)
    stride = jt.nn._pair(stride)
    dilation = jt.nn._pair(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    # cuDNN path (memory-efficient fwd+bwd); falls back to reindex below on
    # CPU / no-cuDNN / non-float32. See _CudnnConv2d.
    _y = jt.nn._try_cudnn_conv2d(x, weight, bias, stride, padding, dilation, groups)
    if _y is not None:
        return _y
    if groups == 1:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
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
        Kh, Kw = weight.shape[-2:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kh, Kw]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
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
    padding = jt.nn._triple(padding)
    stride = jt.nn._triple(stride)
    dilation = jt.nn._triple(dilation)
    out_channels = weight.shape[0]
    if groups <= 0:
        raise ValueError("groups must be a positive integer")
    if jt.flags.use_cuda and jt.cudnn:
        y = jt.nn._cudnn_conv3d_fp16_safe(jt.cudnn.ops.cudnn_conv3d, x, weight, *stride, *padding, *dilation, groups)
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
        # w: [oc, CpG, Kh, Kw, Kd]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, od, Kh, Kw, Kd], [
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
