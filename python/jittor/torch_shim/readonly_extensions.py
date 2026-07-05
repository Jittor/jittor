"""Selective boundary policies for known torch native extensions."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from typing import Dict, Iterable, Optional, Set, Tuple


_READONLY_BORROW_ATTR = "_jittor_torch_ext_readonly_borrow"
_FORCE_CPU_ATTR = "_jittor_torch_force_cpu"
_MISSING = object()

_DEFAULT_READONLY_FUNCTIONS: Dict[str, Tuple[str, ...]] = {
    "diff_gaussian_rasterization._C": (
        "rasterize_gaussians",
        "rasterize_gaussians_backward",
        "mark_visible",
        "fusedssim",
        "fusedssim_backward",
    ),
    "fused_ssim_cuda": (
        "fusedssim",
        "fusedssim_backward",
    ),
    "simple_knn._C": (
        "distCUDA2",
    ),
}

def _is_falsey(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def _is_var(obj) -> bool:
    try:
        import jittor as jt

        return isinstance(obj, jt.Var)
    except Exception:
        return False


def _iter_vars(obj, seen: Set[int]):
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if _is_var(obj):
        yield obj
        return
    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_vars(item, seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_vars(item, seen)


def _mark_readonly(args, kwargs):
    saved = []
    seen: Set[int] = set()
    for tensor in _iter_vars((args, kwargs), seen):
        try:
            if getattr(tensor, _FORCE_CPU_ATTR, False):
                continue
        except Exception:
            continue
        try:
            old_value = getattr(tensor, _READONLY_BORROW_ATTR)
        except AttributeError:
            old_value = _MISSING
        except Exception:
            continue
        try:
            setattr(tensor, _READONLY_BORROW_ATTR, True)
        except Exception:
            continue
        saved.append((tensor, old_value))
    return saved


def _restore(saved) -> None:
    for tensor, old_value in reversed(saved):
        try:
            if old_value is _MISSING:
                delattr(tensor, _READONLY_BORROW_ATTR)
            else:
                setattr(tensor, _READONLY_BORROW_ATTR, old_value)
        except Exception:
            pass


def _wrap_readonly_function(fn):
    if getattr(fn, "_jittor_readonly_borrow_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        saved = _mark_readonly(args, kwargs)
        try:
            return fn(*args, **kwargs)
        finally:
            _restore(saved)

    wrapped._jittor_readonly_borrow_wrapped = True
    return wrapped


def _patch_module(module, readonly_functions: Iterable[str]) -> None:
    for name in readonly_functions:
        try:
            fn = getattr(module, name)
        except AttributeError:
            continue
        if callable(fn):
            try:
                setattr(module, name, _wrap_readonly_function(fn))
            except Exception:
                pass


class _ExtensionPolicyLoader(importlib.abc.Loader):
    def __init__(self, loader, readonly_functions: Tuple[str, ...]):
        self.loader = loader
        self.readonly_functions = readonly_functions

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        _patch_module(module, self.readonly_functions)


class _ExtensionPolicyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, readonly_registry: Dict[str, Tuple[str, ...]]):
        self.readonly_registry = readonly_registry

    def find_spec(self, fullname, path=None, target=None):
        readonly_functions = self.readonly_registry.get(fullname, ())
        if not readonly_functions:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _ExtensionPolicyLoader):
            return spec
        spec.loader = _ExtensionPolicyLoader(spec.loader, readonly_functions)
        return spec


def install_readonly_extension_borrow(registry=None) -> None:
    """Install import-time wrappers for native extension boundary policies."""

    readonly_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_READONLY_BORROW")) else dict(
        registry or _DEFAULT_READONLY_FUNCTIONS
    )
    if not readonly_reg:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _ExtensionPolicyFinder):
            finder.readonly_registry.update(readonly_reg)
            break
    else:
        sys.meta_path.insert(0, _ExtensionPolicyFinder(readonly_reg))
    for name in readonly_reg:
        module = sys.modules.get(name)
        if module is not None:
            _patch_module(module, readonly_reg.get(name, ()))
