"""Deprecated same-object alias of :mod:`jittor.contrib.math_util.igamma`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.math_util.igamma")
_sys.modules[__name__] = _canonical
