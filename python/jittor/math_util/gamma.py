"""Deprecated same-object alias of :mod:`jittor.contrib.math_util.gamma`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.math_util.gamma")
_sys.modules[__name__] = _canonical
