"""Stateless pooling API; implementations live in rank and operation owners."""

import sys as _sys
from ._state import PoolingStateView as _PoolingStateView
from .average import adaptive_avg_pool2d, avg_pool2d, avg_pool3d
from .entrypoints import argmax_pool, max_pool2d, max_pool3d, pool, pool2d, pool3d

__all__ = [
    "adaptive_avg_pool2d", "avg_pool2d", "avg_pool3d",
    "argmax_pool", "max_pool2d", "max_pool3d", "pool", "pool2d", "pool3d",
]

_sys.modules[__name__].__class__ = _PoolingStateView
