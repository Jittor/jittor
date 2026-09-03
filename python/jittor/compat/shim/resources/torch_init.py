"""Deployed ``torch`` entry point backed by the canonical Jittor module.

This file is copied to ``site-packages/torch/__init__.py``.  Torch API
registration belongs to :mod:`jittor.compat.torch`; the deployed package only
selects the public module identity.
"""

import os as _os
import sys as _sys

_sys.modules[__name__]._jittor_torch_shim_placeholder = True
_os.environ["JITTOR_TORCH_SHIM"] = "1"

import jittor as _jittor  # noqa: E402
from jittor.compat.shim import activate as _activate  # noqa: E402

_activate()
_sys.modules[__name__] = _jittor
