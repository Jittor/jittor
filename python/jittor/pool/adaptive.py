"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import AdaptiveAvgPool3d, AdaptiveMaxPool2d, AdaptiveMaxPool3d
from jittor.nn.modules.pooling_legacy import AdaptiveAvgPool2d

_PUBLIC_SYMBOLS = (
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
)
__all__ = (
    "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "AdaptiveAvgPool3d",
    "AdaptiveMaxPool3d",
)
