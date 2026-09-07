"""Deprecated same-object alias of :mod:`jittor.contrib.ccl.ccl_3d`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.ccl.ccl_3d")
_sys.modules[__name__] = _canonical
