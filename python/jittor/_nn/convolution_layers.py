"""One- and two-dimensional convolution layer implementations."""

import math

from .runtime import jt, preserve_facade_origins


class Conv(jt.Module):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv2d(24, 32, 3)
    >>> conv = nn.Conv2d(24, 32, (3,3))
    >>> conv = nn.Conv2d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv2d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        # padding_mode/device/dtype accepted for torch.nn.Conv2d compatibility.
        # jittor pads with zeros; non-'zeros' padding_mode is not yet implemented
        # (warn rather than silently differ).
        self.padding_mode = padding_mode
        if padding_mode not in ('zeros',):
            jt.LOG.w(f"Conv: padding_mode={padding_mode!r} not implemented, using 'zeros'")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be greater than zero, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be greater than zero, got {out_channels}")
        if groups <= 0:
            raise ValueError(f"groups must must be greater than zero, got {groups}")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        if isinstance(kernel_size, tuple):
            for size in kernel_size:
                if size <= 0:
                    raise ValueError(f"kernel_size must be greater than zero, got {kernel_size}")
        else:
            if kernel_size <= 0:
                raise ValueError(f"kernel_size must be greater than zero, got {kernel_size}")
        if isinstance(stride, tuple):
            for size in stride:
                if size <= 0:
                    raise ValueError(f"stride must be greater than zero, got {stride}")
        else:
            if stride <= 0:
                raise ValueError(f"stride must be greater than zero, got {stride}")
        if isinstance(padding, (tuple, list)):
            for size in padding:
                if size < 0:
                    raise ValueError(f"padding must be nonnegative, got {padding}")
        else:
            if padding < 0:
                raise ValueError(f"padding must be nonnegative, got {padding}")
        if isinstance(dilation, (tuple, list)):
            for size in dilation:
                if size <= 0:
                    raise ValueError(f"dilation must be greater than zero, got {dilation}")
        else:
            if dilation <= 0:
                raise ValueError(f"dilation must be greater than zero, got {dilation}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # torch accepts int OR sequence (list/tuple); _pair normalizes int->2-tuple
        # and passes sequences through, so a *list* kernel_size no longer falls into
        # the scalar branch (which produced nested ([k,k],[k,k]) and crashed init).
        self.kernel_size = jt.nn._pair(kernel_size)
        self.stride = jt.nn._pair(stride)
        self.padding = jt.nn._pair(padding)
        self.dilation = jt.nn._pair(dilation)
        self.groups = groups
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
        if self.is_depthwise_conv and jt.flags.use_cuda and jt.compiler.is_cuda:
            self.depthwise_conv = jt.nn.DepthwiseConv(stride, padding, dilation)
        Kh, Kw = self.kernel_size

        self.weight = jt.nn.init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = jt.nn.init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        # Clear, torch-grade errors for the two most common Conv2d misuses, instead of an
        # empty `AssertionError:` (channel mismatch) or a cryptic "not enough values to
        # unpack" (wrong ndim). Covers all paths (depthwise / groups==1 / grouped) at once.
        if x.ndim != 4:
            raise ValueError(
                f"Conv2d expected a 4-D input (N, C, H, W), but got a {x.ndim}-D input "
                f"of shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv2d expected input with {self.in_channels} channels (in_channels), "
                f"but got {x.shape[1]} channels; input shape {tuple(x.shape)}.")
        if hasattr(self, 'depthwise_conv'):
            y = self.depthwise_conv(x, self.weight)
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        # cuDNN path (memory-efficient fwd+bwd); falls back to reindex below on
        # CPU / no-cuDNN / non-float32. See _CudnnConv2d.
        _y = jt.nn._try_cudnn_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if _y is not None:
            return _y
        if self.groups == 1:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            if oh<=0 or ow<=0:
                raise ValueError(
                    f"Conv2d output size is non-positive (oh={oh}, ow={ow}): input "
                    f"{tuple(x.shape)} is too small for kernel {tuple(self.kernel_size)}, "
                    f"stride {tuple(self.stride)}, padding {tuple(self.padding)}, "
                    f"dilation {tuple(self.dilation)}.")
            with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
                xx = x.reindex([N,self.out_channels,C,oh,ow,Kh,Kw], [
                    'i0', # Nid
                    'i2', # Cid
                    f'i3*{self.stride[0]}-{self.padding[0]}+i5*{self.dilation[0]}', # Hid+Khid
                    f'i4*{self.stride[1]}-{self.padding[1]}+i6*{self.dilation[1]}', # Wid+KWid
                ])
                ww = self.weight.broadcast(xx.shape, [0,3,4])
                yy = xx*ww
                y = yy.sum([2,5,6]) # Kc, Kh, Kw
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            if oh<=0 or ow<=0:
                raise ValueError(
                    f"Conv2d output size is non-positive (oh={oh}, ow={ow}): input "
                    f"{tuple(x.shape)} is too small for kernel {tuple(self.kernel_size)}, "
                    f"stride {tuple(self.stride)}, padding {tuple(self.padding)}, "
                    f"dilation {tuple(self.dilation)}.")
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            # w: [oc, CpG, Kh, Kw]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            yy = xx*ww
            y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y

    def _conv_forward(self, input, weight, bias=None):
        # torch nn.Conv2d API: apply the conv with an externally supplied weight
        # (and bias). Used by mmdet's NormedConv2d (seesaw loss / normed heads),
        # which normalizes the weight then calls self._conv_forward(x, weight_, bias).
        return jt.nn.conv2d(input, weight, bias, self.stride, self.padding,
                            self.dilation, self.groups)


class Conv1d(jt.Module):
    ''' Applies a 1D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv1d(24, 32, 3)
    >>> conv = nn.Conv1d(24, 32, (3,3))
    >>> conv = nn.Conv1d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv1d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        assert in_channels > 0, 'in_channels must be positive'
        assert out_channels > 0, 'out_channels must be positive'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, 1)
        self.stride = (stride, 1)
        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.groups = groups
        self.bias = bias
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        # using list to escape module dfs
        self._conv = [jt.nn.Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)]
        self.weight = self._conv[0].weight.squeeze(-1)
        self.bias = self._conv[0].bias

    def execute(self, x):
        if x.dim() != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        N,C,D = x.shape
        assert C==self.in_channels
        self._conv[0].weight = self.weight.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self._conv[0](x)
        y = x.squeeze(-1)
        return y


_FACADE_SYMBOLS = (Conv, Conv1d)
preserve_facade_origins(_FACADE_SYMBOLS)
