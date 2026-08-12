"""Torch-compatible average-pooling modules for :mod:`jittor.nn`."""

import jittor as jt

from ..functional.pooling import (
    adaptive_avg_pool2d as adaptive_avg_pool2d,
    avg_pool2d as avg_pool2d,
)


class AvgPool2d(jt.Module):
    '''2D average pooling, torch-compatible (N,C,H,W) -> (N,C,Hout,Wout).

    Unlike ``jittor.pool.AvgPool2d`` this honours ``count_include_pad`` exactly as
    PyTorch documents it: when ``True`` (default) padded zeros are counted in the
    averaging denominator; when ``False`` only real input elements are.  ``ceil_mode``
    overshoot beyond the input is never counted as padding (matches torch).
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        return jt.nn.avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


class AdaptiveAvgPool2d(jt.Module):
    '''2D adaptive average pooling, torch-compatible (N,C,H,W) -> (N,C,Oh,Ow).

    Uses torch's variable-width overlapping bins
    ``hstart=floor(i*H/Oh)``, ``hend=ceil((i+1)*H/Oh)`` (and likewise for W) and
    divides by the real bin size, so it matches PyTorch even when the output size
    does not divide the input size (the common diffusers / classifier-head case).
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        return jt.nn.adaptive_avg_pool2d(x, self.output_size)
