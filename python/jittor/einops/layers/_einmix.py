"""Deprecated same-object alias of :mod:`jittor.contrib.einops.layers._einmix`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.einops.layers._einmix")
_sys.modules[__name__] = _canonical
