"""Legacy pooling functional entry points and thin module wrappers."""

import jittor as jt


def argmax_pool(x, size, stride, padding=0):
    if stride <= 0:
        raise RuntimeError("stride must be > 0, but got {}".format(stride))
    return jt.pool.pool(x, size, "maximum", padding, stride)


def pool(x, kernel_size, op, padding=0, stride=None):
    return jt.pool.Pool(kernel_size, stride, padding, op=op)(x)


def pool3d(x, kernel_size, op, padding=0, stride=None):
    return jt.pool.Pool3d(kernel_size, stride, padding, op=op)(x)


class AvgPool2d(jt.Module):
    '''Deprecated spelling of :class:`jittor.nn.AvgPool2d`; forwards to it.

    This class used to wrap ``Pool(op="mean")``, which ignored
    ``count_include_pad`` outside its ceil_mode kernel and used jittor's
    uncorrected ceil_mode output size.  ``jt.nn.AvgPool2d`` and
    ``jt.pool.AvgPool2d`` therefore returned different numbers -- and different
    shapes -- for the same arguments.  Now there is one implementation.
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = jt.nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def execute(self, x):
        return self.layer(x)


class AvgPool3d(jt.Module):
    '''Deprecated spelling of :class:`jittor.nn.AvgPool3d`; forwards to it.'''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = jt.nn.AvgPool3d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def execute(self, x):
        return self.layer(x)


def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    '''Deprecated spelling of :func:`jittor.nn.avg_pool2d`; forwards to it.'''
    return jt.nn.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)


def _no_dilation(dilation):
    # torch's default dilation=1 (or (1,1)) means *no* dilation, which jittor's
    # Pool expresses as None. Normalize int/tuple/list all-ones -> None.
    if dilation is None or dilation == 1: return True
    if isinstance(dilation, (tuple, list)): return all(d == 1 for d in dilation)
    return False


class MaxPool2d(jt.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        if jt.pool._no_dilation(dilation): dilation = None
        self._layer = jt.pool.Pool(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")

    def execute(self, x):
        return self._layer(x)


class MaxPool3d(jt.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        if jt.pool._no_dilation(dilation): dilation = None
        self._layer = jt.pool.Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")

    def execute(self, x):
        return self._layer(x)


def max_pool2d(x=None, kernel_size=None, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, input=None):
    if x is None: x = input          # torch uses the keyword `input` (mmdet DropBlock)
    return jt.pool.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)


def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
    return jt.pool.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)


_PUBLIC_SYMBOLS = (
    argmax_pool, pool, pool3d, AvgPool2d, AvgPool3d, avg_pool2d, _no_dilation,
    MaxPool2d, MaxPool3d, max_pool2d, max_pool3d,
)
