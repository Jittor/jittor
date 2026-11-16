"""Compile the optional CUDA launcher with an ABI-specific cache identity.

An unavailable compiler or Python development headers leaves the ctypes path
usable with a diagnostic. Unexpected implementation errors remain visible.
"""

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import warnings

_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastlaunch.c")
_module = None
_tried = False


def _cache_dir():
    base = os.environ.get("JITTOR_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "jittor")
    return os.path.join(base, "cache", "fastlaunch")


def _build():
    """Compile for this Python ABI, preserving compiler wrappers and flags."""
    include = sysconfig.get_paths()["include"]
    compiler = shlex.split(os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc")
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    recipe = [compiler, include, suffix, sys.version, sysconfig.get_platform()]
    with open(_SOURCE, "rb") as handle:
        digest = hashlib.sha256(handle.read() + json.dumps(recipe).encode()).hexdigest()[:20]
    out_dir = os.path.join(_cache_dir(), digest)
    target = os.path.join(out_dir, "_fastlaunch" + suffix)
    if os.path.exists(target):
        return target
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="build-", suffix=".so", dir=out_dir)
    os.close(fd)
    try:
        subprocess.run(compiler + ["-O2", "-shared", "-fPIC", "-I" + include,
                                   _SOURCE, "-o", tmp, "-ldl"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=120)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def load():
    """Return the launcher, or diagnose a known optional-toolchain failure."""
    global _module, _tried
    # Honor a per-call opt-out even after the module has been loaded.
    if os.environ.get("JITTOR_TRITON_FASTLAUNCH", "1") == "0":
        return None
    if _tried:
        return _module
    try:
        path = _build()
        spec = importlib.util.spec_from_file_location("_fastlaunch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (OSError, ImportError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        warnings.warn("Triton C launcher unavailable; using ctypes: " + str(detail),
                      RuntimeWarning, stacklevel=2)
        _tried = True
        return None
    _module = module
    _tried = True
    return _module
