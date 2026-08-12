"""Delay independent PyTorch imports until pytest executes a test module."""

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys


def _real_torch_site():
    raw_site = os.environ.get("REAL_TORCH_SITE", "").strip()
    if not raw_site:
        return None
    site = Path(raw_site).expanduser().resolve()
    return site if site.is_dir() else None


def _loaded_torch_is_jittor_shim():
    module = sys.modules.get("torch")
    if module is None:
        return False
    return (
        getattr(module, "__name__", "torch") != "torch"
        or bool(getattr(module, "_jittor_torch_shim_placeholder", False))
        or hasattr(module, "_torch_compat_install_context")
    )


def _loaded_module_is_from_site(module_name, site):
    module = sys.modules.get(module_name)
    origin = getattr(module, "__file__", None)
    if not origin:
        return False
    try:
        return site in Path(origin).resolve().parents
    except OSError:
        return False


def _site_spec(module_name, site):
    return importlib.machinery.PathFinder.find_spec(module_name, [str(site)])


def modules_available(*module_names):
    """Return whether top-level optional dependencies are discoverable without importing."""
    top_level_names = {module_name.partition(".")[0] for module_name in module_names}
    for module_name in top_level_names:
        try:
            if module_name == "torch":
                if _loaded_torch_is_jittor_shim():
                    return False
                module = sys.modules.get("torch")
                if module is not None:
                    if getattr(module, "__name__", None) != "torch":
                        return False
                    continue
            site = _real_torch_site()
            if module_name in ("torch", "torchvision") and site is not None:
                if _loaded_module_is_from_site(module_name, site):
                    continue
                if _site_spec(module_name, site) is None:
                    return False
            elif module_name == "torchvision":
                module = sys.modules.get(module_name)
                if module is not None and "jittor" in str(
                    getattr(module, "__file__", "")
                ):
                    return False
                spec = importlib.util.find_spec(module_name)
                origin = str(getattr(spec, "origin", "")) if spec else ""
                if spec is None or "jittor" in origin:
                    return False
            elif importlib.util.find_spec(module_name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def import_torch_modules(*module_names):
    """Apply Jittor's runtime workaround and import optional Torch modules."""
    import jittor as jt

    owner = sys.modules.get("torch")
    if owner is None or _loaded_torch_is_jittor_shim():
        raise RuntimeError("independent Torch was not preloaded before Jittor")
    jt.dirty_fix_pytorch_runtime_error()
    site = _real_torch_site()
    site_text = str(site) if site is not None else None
    if site_text is not None:
        sys.path[:] = [path for path in sys.path if path != site_text]
        sys.path.insert(0, site_text)
    try:
        modules = tuple(importlib.import_module(name) for name in module_names)
    finally:
        if site_text is not None:
            sys.path[:] = [path for path in sys.path if path != site_text]
            sys.path.append(site_text)
    if sys.modules.get("torch") is not owner:
        raise RuntimeError("Torch namespace owner changed during oracle import")
    if site is not None:
        for name, module in zip(module_names, modules):
            if name.partition(".")[0] in ("torch", "torchvision"):
                origin = Path(getattr(module, "__file__", "")).resolve()
                if site not in origin.parents:
                    raise RuntimeError(
                        "{} did not come from REAL_TORCH_SITE: {}".format(name, origin)
                    )
    return modules
