"""Deployed ``torch`` entry point backed by the canonical Jittor module.

This file is copied to ``site-packages/torch/__init__.py``.  Torch API
registration belongs to :mod:`jittor.compat.torch`; the deployed package only
selects the public module identity.
"""

import sys as _sys

import jittor as _jittor
from jittor.compat import torch as _torch_compat

_torch_compat.install(_jittor)
_sys.modules[__name__] = _jittor
