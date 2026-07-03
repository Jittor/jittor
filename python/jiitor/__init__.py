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

import jittor as _jittor

try:
    from jittor.torch_shim import bootstrap as _jt_torch_bootstrap
    _jt_torch_bootstrap.enable(
        project_root=os.environ.get("JITTOR_TORCH_RUNTIME_ROOT") or os.getcwd(),
        auto_scan_extensions=True,
        build_extensions=True,
        local_home=False,
        configure_cuda=False,
        verbose=False,
    )
except Exception as _e:
    try:
        from jittor.compiler import LOG as _LOG
        _LOG.w(f"jiitor torch bootstrap skipped: {_e}")
    except Exception:
        pass

sys.modules[__name__] = _jittor
sys.modules["torch"] = _jittor
