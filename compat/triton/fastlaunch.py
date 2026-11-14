"""Optional C launcher for the Triton bridge's kernel launches.

The Python launch path builds a ``ctypes`` parameter array (one cast per
argument) before every launch.  This module compiles a tiny extension once
into the Jittor cache and hands the packed slots to ``cuLaunchKernel`` in
one call.  ``JITTOR_TRITON_FASTLAUNCH=0`` or any build/import failure falls
back to the ctypes path, so the bridge keeps working without a compiler.
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig

_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fastlaunch.c")
_module = None
_tried = False


def _cache_dir():
    base = os.environ.get("JITTOR_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "jittor")
    return os.path.join(base, "cache", "fastlaunch")


def _build():
    """Compile the extension into the cache; returns its path or None."""
    try:
        with open(_SOURCE, "rb") as handle:
            digest = hashlib.sha1(handle.read()).hexdigest()[:12]
        # An extension module's file name has to match its PyInit_ symbol,
        # so each build gets its own directory named by content.
        out_dir = os.path.join(_cache_dir(), digest)
        target = os.path.join(out_dir, "_fastlaunch.so")
        if os.path.exists(target):
            return target
        os.makedirs(out_dir, exist_ok=True)
        include = sysconfig.get_paths().get("include")
        compiler = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
        compiler = compiler.split()[0]
        tmp = target + ".%d.tmp" % os.getpid()
        subprocess.run([compiler, "-O2", "-shared", "-fPIC",
                        "-I" + include, _SOURCE, "-o", tmp, "-ldl"],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        os.replace(tmp, target)
        return target
    except Exception:
        return None


def load():
    """The compiled module, or None when it is disabled or not buildable."""
    global _module, _tried
    if _tried:
        return _module
    _tried = True
    if os.environ.get("JITTOR_TRITON_FASTLAUNCH", "1") == "0":
        return None
    path = _build()
    if path is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("_fastlaunch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module = module
    except Exception:
        _module = None
    return _module
