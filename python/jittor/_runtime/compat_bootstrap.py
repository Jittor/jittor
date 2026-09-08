"""Native-owned, stdlib-only entry points for optional compatibility domains."""

import importlib
import os
import sys
from types import SimpleNamespace

from .import_aliases import install_aliases, register_alias_provider


for _prefix in ("jittor.torch_compat", "jittor.torch_fsdp2_compat",
                "jittor.torch_shim", "jittor.triton_shim"):
    register_alias_provider(_prefix, "jittor.compat._aliases")
del _prefix


def is_truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _requested(environ):
    torch = sys.modules.get("torch")
    return bool(is_truthy(environ.get("JITTOR_TORCH_SHIM"))
                or environ.get("JITTOR_TORCH_PROJECT_ROOT")
                or environ.get("JITTOR_TORCH_RUNTIME_ROOT")
                or getattr(torch, "_jittor_torch_shim_placeholder", False))


def _compat_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name in ("jittor.compat", name):
            raise ModuleNotFoundError(
                "Torch compatibility was requested but is not installed; "
                "install the jittor-torch distribution", name=error.name
            ) from error
        raise


def prepare_import_environment(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    if not _requested(environ):
        return SimpleNamespace(active=False)
    preflight = _compat_module("jittor.compat.shim.preflight")
    return preflight.prepare_import_environment(argv=argv, environ=environ)


def compose(root_module, core_flags, strict=True, preflight=None):
    """Keep plain native startup independent of the optional distribution."""
    if getattr(preflight, "active", False) or _requested(os.environ):
        runtime = _compat_module("jittor.compat.runtime")
        return runtime.compose(root_module, core_flags, strict=strict, preflight=preflight)
    aliases = install_aliases(root_module)
    return SimpleNamespace(torch_reports=(), integrations={}, aliases=aliases)
