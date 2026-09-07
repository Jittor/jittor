"""Deprecated same-object alias of :mod:`jittor.contrib.loss3d.emd`."""

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("jittor.contrib.loss3d.emd")
_sys.modules[__name__] = _canonical
