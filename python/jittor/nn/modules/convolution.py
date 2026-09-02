"""One- and two-dimensional convolution layer implementations."""

import math

import jittor as jt


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
        # Descriptive only. The depthwise CUDA kernel is selected per call by
        # jt.nn.conv2d, not decided here: deciding it in __init__ meant a layer
        # built before `jt.flags.use_cuda = 1` never took the fast path.
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
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
        return self._conv_forward(x, self.weight, self.bias)

    def _conv_forward(self, input, weight, bias=None):
        # torch nn.Conv2d API: apply the conv with an externally supplied weight
        # (and bias). Used by mmdet's NormedConv2d (seesaw loss / normed heads),
        # which normalizes the weight then calls self._conv_forward(x, weight_, bias).
        #
        # execute() goes through here too, so this module holds parameters and
        # nothing else: there is one 2-D convolution and it lives in
        # jt.nn.conv2d. The two used to be independent transcriptions that had
        # already drifted apart in compile options, validation and the CUDA
        # depthwise path -- and _conv_forward called the functional one, so the
        # same layer computed different things depending on the entry point.
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
