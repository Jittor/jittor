"""Lazy public API for the canonical Jittor Torch shim domain."""

from __future__ import absolute_import

import importlib


_PUBLIC = {
    "NativeExtension": ("jittor.compat.shim.discovery", "NativeExtension"),
    "scan_extension_dirs": ("jittor.compat.shim.discovery", "scan_extension_dirs"),
    "build_extension_dirs": ("jittor.compat.shim.build", "build_extension_dirs"),
    "activate": ("jittor.compat.shim.runtime", "activate"),
    "activation_status": ("jittor.compat.shim.runtime", "activation_status"),
    "enable": ("jittor.compat.shim.runtime", "activate"),
}

__all__ = sorted(_PUBLIC)


def __getattr__(name):
    target = _PUBLIC.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
