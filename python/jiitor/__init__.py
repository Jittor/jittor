"""Compatibility alias for the common ``jiitor`` misspelling.

The real framework package is ``jittor``.  Some torch-drop-in experiments use
``import jiitor as torch``; keep that spelling working by returning the live
``jittor`` module object.
"""
import sys
import os

# ``import jiitor as torch`` must not import a real site-packages torch while
# Jittor is still initializing. Jittor's legacy CUDA runtime workaround probes
# torch unless this flag is disabled before importing jittor.
os.environ.setdefault("FIX_TORCH_ERROR", "0")
os.environ.setdefault("JITTOR_TORCH_STRICT_BOOTSTRAP", "1")

import jittor as _jittor

try:
    _jittor.flags.torch_shim = 1
except Exception as _e:
    raise RuntimeError("jiitor torch bootstrap failed") from _e

sys.modules[__name__] = _jittor
sys.modules["torch"] = _jittor
