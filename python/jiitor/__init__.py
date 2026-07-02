"""Compatibility alias for the common ``jiitor`` misspelling.

The real framework package is ``jittor``.  Some torch-drop-in experiments use
``import jiitor as torch``; keep that spelling working by returning the live
``jittor`` module object.
"""
import sys
import jittor as _jittor

sys.modules[__name__] = _jittor
sys.modules["torch"] = _jittor
