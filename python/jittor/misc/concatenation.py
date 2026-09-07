"""Deprecated import alias; use :mod:`jittor.ops.concatenation`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("jittor.ops.concatenation")
