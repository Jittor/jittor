"""One- and two-dimensional convolution layer implementations."""

import math

import jittor as jt
from jittor.misc import _pair
from jittor.nn.functional.convolution import same_padding_pairs

#: torch's ``padding_mode`` values. ``'zeros'`` is the kernels' own padding;
#: the others pad the input explicitly and convolve with none, as torch does.
_PADDING_MODES = ('zeros', 'reflect', 'replicate', 'circular')


def _check_padding_options(padding, padding_mode, stride):
    """Validate torch's ``padding_mode`` and string ``padding`` at construction."""
    if padding_mode not in _PADDING_MODES:
        raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'"
                         .format(_PADDING_MODES, padding_mode))
    if isinstance(padding, str):
        if padding not in ('same', 'valid'):
            raise ValueError("Invalid padding string {!r}, should be one of "
                             "{{'valid', 'same'}}".format(padding))
        if padding == 'same' and any(int(value) != 1 for value in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")


def _reversed_padding_repeated_twice(padding, kernel_size, dilation):
    """``F.pad`` widths (last dimension first) for an explicit ``padding_mode``."""
    if padding == 'same':
        pairs = same_padding_pairs(kernel_size, dilation)
    elif padding == 'valid':
        pairs = [(0, 0)] * len(kernel_size)
    else:
        pairs = [(int(value), int(value)) for value in padding]
    widths = []
    for before, after in reversed(pairs):
        widths.extend((before, after))
    return tuple(widths)


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

    :param padding: Padding added to all four sides of the input, or torch's
        ``'valid'`` / ``'same'`` (stride 1 only). Default: 0
    :type padding: int, tuple or str, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    :param padding_mode: ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    :type padding_mode: str, optional

    Example:

    >>> conv = nn.Conv2d(24, 32, 3)
    >>> conv = nn.Conv2d(24, 32, (3,3))
    >>> conv = nn.Conv2d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv2d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100, 100)
    >>> output = conv(input)
    '''
    #: Class default, so a subclass that skips ``__init__`` still convolves.
    padding_mode = 'zeros'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        # device/dtype accepted for torch.nn.Conv2d compatibility.
        _check_padding_options(padding, padding_mode, _pair(stride))
        self.padding_mode = padding_mode
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
        if isinstance(padding, str):
            pass                      # validated above
        elif isinstance(padding, (tuple, list)):
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
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        # A string stays a string, as on torch's layers; nn.conv2d resolves it.
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        # torch's name and layout for the widths an explicit padding_mode uses.
        self._reversed_padding_repeated_twice = _reversed_padding_repeated_twice(
            self.padding, self.kernel_size, self.dilation)
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
        if self.padding_mode != 'zeros':
            input = jt.nn.pad(input, self._reversed_padding_repeated_twice,
                              mode=self.padding_mode)
            return jt.nn.conv2d(input, weight, bias, self.stride, 0,
                                self.dilation, self.groups)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        assert in_channels > 0, 'in_channels must be positive'
        assert out_channels > 0, 'out_channels must be positive'
        _check_padding_options(padding, padding_mode, _pair(stride))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, 1)
        self.stride = (stride, 1)
        # The inner height-wise Conv resolves a string itself: the width axis
        # has kernel 1, so 'same' adds nothing there.
        self.padding = padding if isinstance(padding, str) else (padding, 0)
        self.dilation = (dilation, 1)
        self.padding_mode = padding_mode
        self.groups = groups
        self.bias = bias
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        # using list to escape module dfs
        self._conv = [jt.nn.Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, padding_mode=padding_mode)]
        self.weight = self._conv[0].weight.squeeze(-1)
        self.bias = self._conv[0].bias

    def execute(self, x):
        if x.dim() != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        N,C,D = x.shape
        assert C==self.in_channels
        self._conv[0].weight = self.weight.unsqueeze(-1)
        # The bias needs the same re-sync as the weight, and for the same
        # reason: the inner Conv is held in a list to escape module traversal,
        # so whatever replaces this module's parameters -- `load_state_dict`,
        # `.to()`, the offload manager -- updates `self.bias` and leaves the
        # inner one behind. Without this line the convolution runs with the
        # *initialisation* value, which is `uniform(-1/sqrt(fan_in),
        # 1/sqrt(fan_in))`: uncorrelated with the checkpoint's bias, identical
        # in distribution from run to run, and different in value every run
        # because it is drawn from the RNG. That is what made the audio VAE's
        # output move between processes while its weights checked out.
        self._conv[0].bias = self.bias
        x = x.unsqueeze(-1)
        x = self._conv[0](x)
        y = x.squeeze(-1)
        return y
