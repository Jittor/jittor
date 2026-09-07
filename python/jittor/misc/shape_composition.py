"""Deprecated import alias; use :mod:`jittor.ops.shape_composition`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("jittor.ops.shape_composition")
