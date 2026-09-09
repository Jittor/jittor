"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import Pool

_PUBLIC_SYMBOLS = (Pool,)
__all__ = ("Pool",)
