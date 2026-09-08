"""Deployed independent ``torch`` entry point using the native Jittor graph.

This file is copied to ``site-packages/torch/__init__.py``.  Torch API
registration belongs to :mod:`jittor.compat.torch`; the deployed package only
selects the public module identity.
"""

import os as _os
import sys as _sys

_requested_frontend = _os.environ.get("JITTOR_TORCH_INDEPENDENT")
if _requested_frontend is not None and _requested_frontend.strip().lower() not in ("1", "true", "yes", "on"):
    raise RuntimeError("JITTOR_TORCH_INDEPENDENT requests the removed legacy frontend; remove it")
_sys.modules[__name__]._jittor_torch_shim_placeholder = True
_os.environ["JITTOR_TORCH_SHIM"] = "1"

import jittor as _jittor  # noqa: E402
from jittor.compat.shim import activate as _activate  # noqa: E402

_activation = _activate()
_sys.modules[__name__] = _activation["torch"]
