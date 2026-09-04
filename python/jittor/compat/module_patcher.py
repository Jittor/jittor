"""Shared import-time patch registration for optional compatibility packages.

Third-party adapters can expose a registrar through the
``jittor.module_patches`` entry-point group. A registrar receives
``register_module_patch`` and may register any number of exact module paths.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import sys
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from ._entry_points import entry_points as _entry_points
from .diagnostics import EXPECTED, swallowed
from .transaction import TransactionConflict


MODULE_PATCH_ENTRY_POINT = "jittor.module_patches"
_MISSING = object()
_REGISTRY: Dict[str, List[Callable[[ModuleType], object]]] = {}
_ENTRY_POINTS_LOADED = set()
_LOCK = threading.RLock()
_FINDER = None
_LAST_REPORT = None


@dataclass(frozen=True)
class PatchResult:
    """Outcome of one entry-point load or module patch invocation."""

    kind: str
    name: str
    callback: str
    status: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class PatchReport:
    """Immutable report returned after installing or applying patches."""

    results: Tuple[PatchResult, ...]
    finder_installed: bool

    @property
    def failures(self) -> Tuple[PatchResult, ...]:
        return tuple(item for item in self.results if item.status == "failed")

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class MethodPatch:
    """Handle used to restore an attribute only while our replacement owns it."""

    owner: object
    name: str
    replacement: object
    had_local_value: bool
    local_value: object


class MethodPatchConflict(RuntimeError):
    """Raised when a method no longer has the value a patch expected."""


def _callback_name(callback: object) -> str:
    return "%s.%s" % (
        getattr(callback, "__module__", type(callback).__module__),
        getattr(callback, "__qualname__", getattr(callback, "__name__", type(callback).__name__)),
    )


def register_module_patch(path: str, fn: Callable[[ModuleType], object]) -> Callable[[ModuleType], object]:
    """Register *fn* for the exact import path and return *fn*.

    Re-registering the same callable is a no-op, which makes adapter install
    functions safe to call repeatedly.
    """

    if not isinstance(path, str) or not path or path.startswith("."):
        raise ValueError("module patch path must be a non-empty absolute module name")
    if not callable(fn):
        raise TypeError("module patch must be callable")
    with _LOCK:
        callbacks = _REGISTRY.setdefault(path, [])
        if not any(item is fn for item in callbacks):
            callbacks.append(fn)
    return fn


def registered_module_patches() -> Mapping[str, Tuple[Callable[[ModuleType], object], ...]]:
    """Return a read-only snapshot of the current process registry."""

    with _LOCK:
        return {path: tuple(callbacks) for path, callbacks in _REGISTRY.items()}


def patch_method(owner: object, name: str, replacement: object, expected: object = _MISSING) -> MethodPatch:
    """Install *replacement* and return a conflict-aware restoration handle."""

    if not isinstance(name, str) or not name:
        raise ValueError("method name must be a non-empty string")
    current = getattr(owner, name, _MISSING)
    if expected is not _MISSING and current is not expected:
        raise MethodPatchConflict("%s changed before it could be patched" % name)
    namespace = getattr(owner, "__dict__", {})
    had_local = name in namespace
    local_value = namespace.get(name, _MISSING)
    setattr(owner, name, replacement)
    return MethodPatch(owner, name, replacement, had_local, local_value)


def restore_method(patch: MethodPatch) -> bool:
    """Restore a method when it is still owned by *patch*.

    Returning ``False`` means another participant replaced the method after us;
    its value is deliberately left untouched.
    """

    namespace = getattr(patch.owner, "__dict__", {})
    current = (
        namespace.get(patch.name, _MISSING)
        if patch.name in namespace
        else getattr(patch.owner, patch.name, _MISSING)
    )
    if current is not patch.replacement:
        return False
    if patch.had_local_value:
        setattr(patch.owner, patch.name, patch.local_value)
    else:
        delattr(patch.owner, patch.name)
    return True


def _register_entry_point_value(value: object) -> None:
    if isinstance(value, Mapping):
        for path, callback in value.items():
            register_module_patch(path, callback)
        return
    if not callable(value):
        raise TypeError("module patch entry point must load a registrar or mapping")
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        accepts_args = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        result = value(register_module_patch) if positional or accepts_args else value()
    else:
        result = value(register_module_patch)
    if result is not None and result is not value:
        _register_entry_point_value(result)


def _load_entry_point_patches() -> List[PatchResult]:
    results = []
    try:
        entry_points = _entry_points(MODULE_PATCH_ENTRY_POINT)
    except EXPECTED as exc:
        swallowed("module_patcher.py _load_entry_point_patches: entry_points = _entry_points(MODULE_PATCH_ENTRY_POINT)", exc)
        return [PatchResult("entry_point", MODULE_PATCH_ENTRY_POINT, "discovery", "failed", repr(exc))]
    for entry_point in entry_points:
        key = (getattr(entry_point, "name", ""), getattr(entry_point, "value", repr(entry_point)))
        if key in _ENTRY_POINTS_LOADED:
            results.append(PatchResult("entry_point", key[0], key[1], "already_loaded"))
            continue
        try:
            _register_entry_point_value(entry_point.load())
        except EXPECTED as exc:
            swallowed("module_patcher.py _load_entry_point_patches: _register_entry_point_value(entry_point.load())", exc)
            results.append(PatchResult("entry_point", key[0], key[1], "failed", repr(exc)))
            continue
        _ENTRY_POINTS_LOADED.add(key)
        results.append(PatchResult("entry_point", key[0], key[1], "loaded"))
    return results


def _apply_module_patches(module: ModuleType) -> List[PatchResult]:
    with _LOCK:
        callbacks = tuple(_REGISTRY.get(module.__name__, ()))
    results = []
    for callback in callbacks:
        callback_name = _callback_name(callback)
        try:
            changed = callback(module)
        except EXPECTED as exc:
            swallowed("module_patcher.py _apply_module_patches: changed = callback(module)", exc)
            results.append(PatchResult("module", module.__name__, callback_name, "failed", repr(exc)))
            continue
        status = "patched" if changed is not False else "unchanged"
        results.append(PatchResult("module", module.__name__, callback_name, status))
    return results


class _ModulePatchLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        global _LAST_REPORT
        results = tuple(_apply_module_patches(module))
        _LAST_REPORT = PatchReport(results, True)


class _ModulePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        with _LOCK:
            registered = fullname in _REGISTRY
        if not registered:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _ModulePatchLoader):
            spec.loader = _ModulePatchLoader(spec.loader)
        return spec


def install_module_patches(load_entry_points: bool = True, transaction=None) -> PatchReport:
    """Load adapter registrations, patch loaded modules, and install one finder."""

    global _FINDER, _LAST_REPORT
    with _LOCK:
        old_registry = {path: list(callbacks) for path, callbacks in _REGISTRY.items()}
        old_loaded = set(_ENTRY_POINTS_LOADED)
        old_finder = _FINDER
        results = _load_entry_point_patches() if load_entry_points else []
        for path in tuple(_REGISTRY):
            module = sys.modules.get(path)
            if isinstance(module, ModuleType):
                results.extend(_apply_module_patches(module))
        if _FINDER is None or _FINDER not in sys.meta_path:
            _FINDER = _ModulePatchFinder()
            sys.meta_path.insert(0, _FINDER)
            if transaction is not None:
                finder = _FINDER
                def restore_finder(f=finder):
                    if f not in sys.meta_path:
                        raise TransactionConflict("module patch finder replaced externally")
                    sys.meta_path.remove(f)
                transaction.record_undo(restore_finder)
        if transaction is not None:
            committed_registry = {path: tuple(callbacks)
                                   for path, callbacks in _REGISTRY.items()}
            committed_loaded = set(_ENTRY_POINTS_LOADED)
            def restore_registry():
                current = {path: tuple(callbacks)
                           for path, callbacks in _REGISTRY.items()}
                if current != committed_registry or set(_ENTRY_POINTS_LOADED) != committed_loaded:
                    raise TransactionConflict("module patch registry changed externally")
                _REGISTRY.clear()
                _REGISTRY.update({path: list(callbacks)
                                  for path, callbacks in old_registry.items()})
                _ENTRY_POINTS_LOADED.clear()
                _ENTRY_POINTS_LOADED.update(old_loaded)
                globals()['_FINDER'] = old_finder
            transaction.record_undo(restore_registry)
        report = PatchReport(tuple(results), True)
        _LAST_REPORT = report
        return report


def last_module_patch_report() -> Optional[PatchReport]:
    return _LAST_REPORT


def uninstall_module_patches() -> bool:
    """Remove this module's finder; registrations and patched values remain."""

    global _FINDER
    with _LOCK:
        if _FINDER is None or _FINDER not in sys.meta_path:
            return False
        sys.meta_path.remove(_FINDER)
        _FINDER = None
        return True


__all__ = [
    "MODULE_PATCH_ENTRY_POINT",
    "MethodPatch",
    "MethodPatchConflict",
    "PatchReport",
    "PatchResult",
    "install_module_patches",
    "last_module_patch_report",
    "patch_method",
    "register_module_patch",
    "registered_module_patches",
    "restore_method",
    "uninstall_module_patches",
]
