"""Transposed convolution functional implementations."""

import jittor as jt


def conv_transpose(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if groups == 1:
        x = input
        if x.dim() != 4:
            raise RuntimeError(f'Expected 4D input to conv_transpose, but got input of size: {x.shape}')
        N,C,H,W = x.shape
        i,o,h,w = weight.shape
        assert C==i
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise RuntimeError("non-positive stride is not supported")
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert output_padding[0] < max(stride[0], dilation[0]) and \
            output_padding[1] < max(stride[1], dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        h_out = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        w_out = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, o, h_out, w_out)
        # cuDNN path (memory-efficient fwd+bwd); falls back to reindex below.
        _y = jt.nn._try_cudnn_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, 1)
        if _y is not None:
            return _y
        shape = (N, i, o, H, W, h, w)
        xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
        ww = weight.broadcast(shape, (0, 3, 4)) # N,H,W
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # N
            'i2', # o
            f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
            f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
        ])
        if isinstance(bias, jt.Var):
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        else:
            assert not bias, "Bias should be none or jittor var"
        return y
    else:
        if input.dim() != 4:
            raise RuntimeError(f'Expected 4D input to conv_transpose, but got input of size: {input.shape}')
        N,C,H,W = input.shape
        i,o,h,w = weight.shape
        G = groups
        oc = o * G
        CpG = C // G # channels per group
        assert C % G == 0
        assert C==i, (C, i)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        if stride[0] <= 0 or stride[1] <= 0:
            raise RuntimeError("non-positive stride is not supported")
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert output_padding[0] < max(stride[0], dilation[0]) and \
            output_padding[1] < max(stride[1], dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        oh = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        ow = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, oc, oh, ow)
        shape = [N,G,oc//G,CpG,oh,ow,h,w]
        xx = input.reindex(shape, [
            'i0',
            f'i1*{oc//G}+i2',
            'i4',
            'i5'
        ])
        ww = weight.reindex(shape, [
            f'i1*{oc//G}+i2',
            'i3',
            'i6',
            'i7'
        ])
        ww.compile_options = xx.compile_options = {"G":G,"C":C}
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # Nid
            f'i1*{CpG}+i3', # Gid
            f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
            f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
        ])
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input
    if x.dim() != 5:
        raise RuntimeError(f'Expected 5D input to conv_transpose3d, but got input of size: {x.shape}')
    N,C,D,H,W = x.shape
    i,o,d,h,w = weight.shape
    assert C==i
    assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
    if stride[0] <= 0 or stride[1] <= 0 or stride[2] <= 0:
        raise RuntimeError("non-positive stride is not supported")
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    # added
    padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
    output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
    assert output_padding[0] < max(stride[0], dilation[0]) and \
        output_padding[1] < max(stride[1], dilation[1]) and \
        output_padding[2] < max(stride[2], dilation[2]), \
        "output padding must be smaller than max(stride, dilation)"

    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation

    d_out = (D-1) * stride_d + output_padding[0] - 2*padding_d + 1 + (d-1)*dilation_d
    h_out = (H-1) * stride_h + output_padding[1] - 2*padding_h + 1 + (h-1)*dilation_h
    w_out = (W-1) * stride_w + output_padding[2] - 2*padding_w + 1 + (w-1)*dilation_w
    out_shape = (N, o, d_out, h_out, w_out)
    if jt.flags.use_cuda and jt.cudnn:
        # fp16/bf16 3D transposed-conv hits the same missing-cuDNN-algo wall as
        # the forward conv3d; reuse the fp32-fallback wrapper.
        return jt.nn._cudnn_conv3d_fp16_safe(jt.cudnn.ops.cudnn_conv3d_backward_x, weight, x, *out_shape[2:], *stride, *padding, *dilation, groups)
    shape = (N, i, o, D, H, W, d, h, w)
    xx = x.broadcast(shape, (2, 6, 7, 8)) # i,h,w
    ww = weight.broadcast(shape, (0, 3, 4, 5)) # N,H,W
    y = (ww*xx).reindex_reduce("add", out_shape, [
        'i0', # N
        'i2', # o
        f'i3*{stride_d}-{padding_d}+i6*{dilation_d}', # Did+Kdid
        f'i4*{stride_h}-{padding_h}+i7*{dilation_h}', # Hid+Khid
        f'i5*{stride_w}-{padding_w}+i8*{dilation_w}', # Wid+KWid
    ])
    if isinstance(bias, jt.Var):
        b = bias.broadcast(y.shape, [0,2,3,4])
        y = y + b
    else:
        assert not bias, "Bias should be none or jittor var"
    return y


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    ''' Applies a 1D transposed convolution operator over an input signal.
    Torch-compatible functional interface.

    :param input: the input var of shape ``(N, C_in, L)``
    :type input: jt.Var

    :param weight: the kernel of shape ``(C_in, C_out//groups, kW)``
    :type weight: jt.Var

    :param bias: the optional bias of shape ``(C_out,)``. Default: None
    :type bias: jt.Var, optional

    :param stride: stride of the convolution. Default: 1
    :param padding: ``dilation * (kW - 1) - padding`` zero-padding. Default: 0
    :param output_padding: additional size added to the output. Default: 0
    :param groups: number of blocked connections. Default: 1
    :param dilation: spacing between kernel elements. Default: 1

    Example:
        >>> x = jt.randn(4, 8, 50)
        >>> w = jt.randn(8, 16, 3)
        >>> y = nn.conv_transpose1d(x, w, stride=2)
    '''
    if input.dim() != 3:
        raise RuntimeError(f'Expected 3D input to conv_transpose1d, but got input of size: {input.shape}')
    if weight.dim() != 3:
        raise RuntimeError(f'Expected 3D weight to conv_transpose1d, but got weight of size: {weight.shape}')
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    padding = padding[0] if isinstance(padding, (tuple, list)) else padding
    output_padding = output_padding[0] if isinstance(output_padding, (tuple, list)) else output_padding
    dilation = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
    x = input.unsqueeze(-1)
    w = weight.unsqueeze(-1)
    y = jt.nn.conv_transpose(x, w, bias, (stride, 1), (padding, 0), (output_padding, 0), groups, (dilation, 1))
    return y.squeeze(-1)
