"""Deprecated compatibility imports; implementation lives in jittor.nn."""

from jittor.nn.modules.pooling import MaxUnpool2d, MaxUnpool3d

_PUBLIC_SYMBOLS = (MaxUnpool2d, MaxUnpool3d)
