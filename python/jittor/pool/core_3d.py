"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import Pool3d
from jittor.nn.functional.pooling.core_3d import (
    _triple as _triple,
    _pool_output_size as _pool_output_size,
)

_PUBLIC_SYMBOLS = (_triple, Pool3d)
__all__ = ("_triple", "Pool3d")
