"""Deprecated same-object alias of :mod:`jittor.contrib.einops`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.einops")
_sys.modules[__name__] = _canonical
