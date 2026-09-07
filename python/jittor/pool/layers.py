"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import MaxPool2d, MaxPool3d
from jittor.nn.modules.pooling_legacy import AvgPool2d, AvgPool3d
from jittor.nn.functional.pooling import argmax_pool, avg_pool2d, max_pool2d, max_pool3d, pool, pool3d
from jittor.nn.functional.pooling.entrypoints import _no_dilation

_PUBLIC_SYMBOLS = (
    argmax_pool, pool, pool3d, AvgPool2d, AvgPool3d, avg_pool2d, _no_dilation,
    MaxPool2d, MaxPool3d, max_pool2d, max_pool3d,
)
