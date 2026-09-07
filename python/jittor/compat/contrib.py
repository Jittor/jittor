"""Deprecated import alias for :mod:`jittor.contrib`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("jittor.contrib")
