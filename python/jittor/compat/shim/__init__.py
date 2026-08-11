"""Canonical runtime and deployment helpers for the Jittor Torch shim."""

import importlib
import importlib.util
import sys

from .bootstrap import (
    NativeExtension,
    build_extension_dirs,
    enable,
    scan_extension_dirs,
)

__all__ = [
    "NativeExtension",
    "build_extension_dirs",
    "enable",
    "scan_extension_dirs",
]


_LEGACY_ALIAS_TARGETS = {
    "jittor.torch_shim.bootstrap": "jittor.compat.shim.bootstrap",
    "jittor.torch_shim.deploy": "jittor.compat.shim.deploy",
    "jittor.torch_shim.cpp_extension": "jittor.compat.shim.cpp_extension",
    "jittor.torch_shim.cpp_extension.torch_utils": (
        "jittor.compat.shim.cpp_extension.torch_utils"
    ),
    "jittor.torch_shim.torch_utils": "jittor.compat.shim.cpp_extension.torch_utils",
    "jittor.torch_shim.flashattn_jittor": (
        "jittor.compat.shim.backends.flash_attention"
    ),
    "jittor.torch_shim.flashattn": "jittor.compat.shim.backends.flash_attention",
    "jittor.torch_shim.flash_attention": (
        "jittor.compat.shim.backends.flash_attention"
    ),
    "jittor.torch_shim.readonly_extensions": (
        "jittor.compat.shim.extensions.readonly"
    ),
    "jittor.torch_shim.readonly": "jittor.compat.shim.extensions.readonly",
    "jittor.compat.shim.torch_utils": "jittor.compat.shim.cpp_extension.torch_utils",
    "jittor.compat.shim.flashattn_jittor": (
        "jittor.compat.shim.backends.flash_attention"
    ),
    "jittor.compat.shim.flashattn": "jittor.compat.shim.backends.flash_attention",
    "jittor.compat.shim.readonly_extensions": "jittor.compat.shim.extensions.readonly",
    "jittor.compat.shim.readonly": "jittor.compat.shim.extensions.readonly",
}


class _LegacyAliasLoader:
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        return None


class _LegacyAliasFinder:
    def find_spec(self, fullname, path=None, target=None):
        canonical = _LEGACY_ALIAS_TARGETS.get(fullname)
        if canonical is None:
            return None
        loader = _LegacyAliasLoader(canonical)
        is_package = canonical == "jittor.compat.shim.cpp_extension"
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


_legacy_alias_finder = _LegacyAliasFinder()


def install_legacy_aliases():
    """Publish historical paths lazily without executing canonical modules twice."""
    package = sys.modules[__name__]
    sys.modules["jittor.torch_shim"] = package
    if _legacy_alias_finder not in sys.meta_path:
        sys.meta_path.insert(0, _legacy_alias_finder)
    for alias, canonical in _LEGACY_ALIAS_TARGETS.items():
        module = sys.modules.get(canonical)
        if module is not None:
            sys.modules[alias] = module
    return dict(_LEGACY_ALIAS_TARGETS)


install_legacy_aliases()
