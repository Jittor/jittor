"""Pooling modules: parameter ownership and calls to stateless functions."""

from jittor._core.module import Module
from ..functional.pooling.average import adaptive_avg_pool2d, avg_pool2d, avg_pool3d
from ..functional.pooling.core_2d import _pool2d, _pool2d_parameters
from ..functional.pooling.core_3d import _pool3d, _pool3d_parameters
from ..functional.pooling.adaptive import _adaptive_max_pool2d, _adaptive_max_pool2d_parameters
from ..functional.pooling.adaptive import _adaptive_avg_pool3d, _adaptive_avg_pool3d_parameters
from ..functional.pooling.adaptive import _adaptive_max_pool3d, _adaptive_max_pool3d_parameters
from ..functional.pooling.pooling_1d import _adaptive_avg_pool1d
from ..functional.pooling.pooling_1d import _max_pool1d, _max_pool1d_parameters
from ..functional.pooling.pooling_1d import _avg_pool1d, _avg_pool1d_parameters
from ..functional.pooling.unpool import _max_unpool2d, _max_unpool2d_parameters
from ..functional.pooling.unpool import _max_unpool3d, _max_unpool3d_parameters
from ..functional.pooling.entrypoints import _no_dilation


class Pool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op='maximum'):
        parameters = _pool2d_parameters(kernel_size, stride, padding, dilation, return_indices, ceil_mode, count_include_pad, op)
        self.ceil_mode = parameters["ceil_mode"]
        self.count_include_pad = parameters["count_include_pad"]
        self.kernel_size = parameters["kernel_size"]
        self.op = parameters["op"]
        self.padding = parameters["padding"]
        self.return_indices = parameters["return_indices"]
        self.stride = parameters["stride"]

    def execute(self, x):
        return _pool2d(x,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            kernel_size=self.kernel_size,
            op=self.op,
            padding=self.padding,
            return_indices=self.return_indices,
            stride=self.stride,
        )


class Pool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op='maximum'):
        parameters = _pool3d_parameters(kernel_size, stride, padding, dilation, return_indices, ceil_mode, count_include_pad, op)
        self.ceil_mode = parameters["ceil_mode"]
        self.count_include_pad = parameters["count_include_pad"]
        self.kernel_size = parameters["kernel_size"]
        self.op = parameters["op"]
        self.padding = parameters["padding"]
        self.return_indices = parameters["return_indices"]
        self.stride = parameters["stride"]

    def execute(self, x):
        return _pool3d(x,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            kernel_size=self.kernel_size,
            op=self.op,
            padding=self.padding,
            return_indices=self.return_indices,
            stride=self.stride,
        )


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size, return_indices=False):
        parameters = _adaptive_max_pool2d_parameters(output_size, return_indices)
        self.output_size = parameters["output_size"]
        self.return_indices = parameters["return_indices"]

    def execute(self, x):
        return _adaptive_max_pool2d(x,
            output_size=self.output_size,
            return_indices=self.return_indices,
        )


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        parameters = _adaptive_avg_pool3d_parameters(output_size)
        self.output_size = parameters["output_size"]

    def execute(self, x):
        return _adaptive_avg_pool3d(x,
            output_size=self.output_size,
        )


class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size, return_indices=False):
        parameters = _adaptive_max_pool3d_parameters(output_size, return_indices)
        self.output_size = parameters["output_size"]
        self.return_indices = parameters["return_indices"]

    def execute(self, x):
        return _adaptive_max_pool3d(x,
            output_size=self.output_size,
            return_indices=self.return_indices,
        )


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        return _adaptive_avg_pool1d(x,
            output_size=self.output_size,
        )


class MaxPool1d(Module):
    '''1D max pooling, (N,C,L) -> (N,C,Lout). torch-compatible.

    Implemented with reindex+reduce rather than the 2D Pool because Pool rejects a
    size-1 spatial dim. Padding positions map out-of-bounds -> -inf so they never win
    a max (matches torch, which pads with -inf for max pooling).'''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=None, ceil_mode=False):
        parameters = _max_pool1d_parameters(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.ceil_mode = parameters["ceil_mode"]
        self.kernel_size = parameters["kernel_size"]
        self.padding = parameters["padding"]
        self.return_indices = parameters["return_indices"]
        self.stride = parameters["stride"]

    def execute(self, x):
        return _max_pool1d(x,
            ceil_mode=self.ceil_mode,
            kernel_size=self.kernel_size,
            padding=self.padding,
            return_indices=self.return_indices,
            stride=self.stride,
        )


class AvgPool1d(Module):
    '''1D average pooling, (N,C,L) -> (N,C,Lout). torch-compatible.

    count_include_pad=True (torch default) divides every window by kernel_size, so
    padded (out-of-bounds) positions contribute 0 to the sum but still count in the
    denominator; =False divides by the number of real (non-pad) elements.'''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        parameters = _avg_pool1d_parameters(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.ceil_mode = parameters["ceil_mode"]
        self.count_include_pad = parameters["count_include_pad"]
        self.kernel_size = parameters["kernel_size"]
        self.padding = parameters["padding"]
        self.stride = parameters["stride"]

    def execute(self, x):
        return _avg_pool1d(x,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )


class MaxUnpool2d(Module):
    ''' MaxUnpool2d is the invert version of MaxPool2d with indices.
    It takes the output index of MaxPool2d as input.
    The element will be zero if it is not the max pooled value.

    Example::

    >>> import jittor as jt
    >>> from jittor import nn

    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = jt.array([[[[ 1.,  2,  3,  4,0],
                            [ 5,  6,  7,  8,0],
                            [ 9, 10, 11, 12,0],
                            [13, 14, 15, 16,0],
                            [0,  0,  0,  0, 0]]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.shape)
    jt.array([[[[   0.,  0.,   0.,   0.,   0.],
                [   0.,  6.,   0.,   8.,   0.],
                [   0.,  0.,   0.,   0.,   0.],
                [   0., 14.,   0.,  16.,   0.],
                [   0.,  0.,   0.,   0.,   0.]]]])
    '''
    def __init__(self, kernel_size, stride=None):
        parameters = _max_unpool2d_parameters(kernel_size, stride)
        self.kernel_size = parameters["kernel_size"]
        self.stride = parameters["stride"]

    def execute(self, x, id, output_size=None):
        return _max_unpool2d(x, id, output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )


class MaxUnpool3d(Module):
    ''' MaxUnpool3d is the invert version of MaxPool3d with indices.
    It takes the output index of MaxPool3d as input.
    The element will be zero if it is not the max pooled value.
    '''
    def __init__(self, kernel_size, stride=None):
        parameters = _max_unpool3d_parameters(kernel_size, stride)
        self.kernel_size = parameters["kernel_size"]
        self.stride = parameters["stride"]

    def execute(self, x, id, output_size=None):
        return _max_unpool3d(x, id, output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        if _no_dilation(dilation): dilation = None
        self._layer = Pool(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")

    def execute(self, x):
        return self._layer(x)


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        if _no_dilation(dilation): dilation = None
        self._layer = Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")

    def execute(self, x):
        return self._layer(x)


class AvgPool2d(Module):
    '''2D average pooling, torch-compatible (N,C,H,W) -> (N,C,Hout,Wout).

    ``count_include_pad`` selects the averaging denominator exactly as PyTorch
    documents it: when ``True`` (default) padded zeros are counted, when ``False``
    only real input elements are.  ``ceil_mode`` overshoot beyond the input is
    never counted as padding.  ``jittor.pool.AvgPool2d`` forwards here, so the two
    spellings are the same numbers.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        return avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AvgPool3d(Module):
    '''3D average pooling, torch-compatible (N,C,D,H,W) -> (N,C,Do,Ho,Wo).

    Same semantics as :class:`AvgPool2d` one rank up, and literally the same
    implementation (:func:`jittor.nn.functional.avg_pool3d`).  Before this the
    2-D and 3-D members of ``jt.nn`` followed two different averaging rules.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        return avg_pool3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AdaptiveAvgPool2d(Module):
    '''2D adaptive average pooling, torch-compatible (N,C,H,W) -> (N,C,Oh,Ow).

    Uses torch's variable-width overlapping bins
    ``hstart=floor(i*H/Oh)``, ``hend=ceil((i+1)*H/Oh)`` (and likewise for W) and
    divides by the real bin size, so it matches PyTorch even when the output size
    does not divide the input size (the common diffusers / classifier-head case).
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        return adaptive_avg_pool2d(x, self.output_size)
