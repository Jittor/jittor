"""Three-dimensional convolution layer implementation."""

import math

import jittor as jt


class Conv3d(jt.Module):
    ''' Applies a 3D convolution over an input signal composed of several input planes.

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

    >>> conv = nn.Conv3d(24, 32, 3)
    >>> conv = nn.Conv3d(24, 32, (3,3))
    >>> conv = nn.Conv3d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv3d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 50, 50, 50)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # torch accepts int OR any sequence (list/tuple) for these; the old
        # `isinstance(x, tuple)` test sent a *list* kernel_size (e.g. Qwen2.5-VL's
        # Conv3d patch_embed uses [t,p,p]) into the scalar branch -> nested
        # ([k,k,k],...) -> weight-shape build crashes. _triple normalizes int->3-tuple
        # and passes sequences through (matching torch's _triple).
        self.kernel_size = jt.nn._triple(kernel_size)
        self.stride = jt.nn._triple(stride)
        self.padding = jt.nn._triple(padding)
        self.dilation = jt.nn._triple(dilation)
        self.groups = groups
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw, Kd = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = jt.nn.init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw, Kd], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = jt.nn.init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return jt.nn.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
