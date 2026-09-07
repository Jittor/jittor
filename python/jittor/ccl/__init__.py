"""Deprecated same-object alias of :mod:`jittor.contrib.ccl`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.ccl")
_sys.modules[__name__] = _canonical
