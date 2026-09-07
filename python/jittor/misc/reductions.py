"""Deprecated import alias; use :mod:`jittor.ops.reductions`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("jittor.ops.reductions")
