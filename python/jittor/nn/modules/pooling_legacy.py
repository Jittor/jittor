"""Legacy pooling classes kept for historical imports and pickle globals."""

from jittor._core.module import Module
from .pooling import AvgPool2d as _AvgPool2d, AvgPool3d as _AvgPool3d
from ..functional.pooling.adaptive import _legacy_adaptive_avg_pool2d


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        return _legacy_adaptive_avg_pool2d(x,
            output_size=self.output_size,
        )


class AvgPool2d(Module):
    '''Deprecated spelling of :class:`jittor.nn.AvgPool2d`; forwards to it.

    This class used to wrap ``Pool(op="mean")``, which ignored
    ``count_include_pad`` outside its ceil_mode kernel and used jittor's
    uncorrected ceil_mode output size.  ``_AvgPool2d`` and
    ``jt.pool.AvgPool2d`` therefore returned different numbers -- and different
    shapes -- for the same arguments.  Now there is one implementation.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = _AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def execute(self, x):
        return self.layer(x)


class AvgPool3d(Module):
    '''Deprecated spelling of :class:`jittor.nn.AvgPool3d`; forwards to it.'''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = _AvgPool3d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def execute(self, x):
        return self.layer(x)
