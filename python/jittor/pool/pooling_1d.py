"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import AdaptiveAvgPool1d, AvgPool1d, MaxPool1d

_PUBLIC_SYMBOLS = (AdaptiveAvgPool1d, MaxPool1d, AvgPool1d)
