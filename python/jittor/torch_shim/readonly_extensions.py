"""Selective boundary policies for known torch native extensions."""

from __future__ import annotations

import functools
import os
import contextlib
from typing import Dict, Iterable, Optional, Set, Tuple

from jittor.compat.module_patcher import install_module_patches, register_module_patch


_READONLY_BORROW_ATTR = "_jittor_torch_ext_readonly_borrow"
_FORCE_CPU_ATTR = "_jittor_torch_force_cpu"
_MISSING = object()
_VAR_TYPE = None
_READONLY_REGISTRY: Dict[str, Tuple[str, ...]] = {}
_COPY_SCOPE_REGISTRY: Dict[str, Tuple[str, ...]] = {}
_SCRATCH_BORROW_REGISTRY: Dict[str, Tuple[str, ...]] = {}
_READONLY_ARG_REGISTRY: Dict[str, Dict[str, Tuple[int, ...]]] = {}

_DEFAULT_READONLY_FUNCTIONS: Dict[str, Tuple[str, ...]] = {}
_DEFAULT_READONLY_ARG_FUNCTIONS: Dict[str, Dict[str, Tuple[int, ...]]] = {}
_DEFAULT_SCRATCH_BORROW_FUNCTIONS: Dict[str, Tuple[str, ...]] = {}
_DEFAULT_COPY_SCOPE_FUNCTIONS: Dict[str, Tuple[str, ...]] = {}


def _is_falsey(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def _get_var_type():
    global _VAR_TYPE
    if _VAR_TYPE is None:
        import jittor as jt

        _VAR_TYPE = jt.Var
    return _VAR_TYPE


def _is_var(obj) -> bool:
    try:
        return isinstance(obj, _get_var_type())
    except Exception:
        return False


def _iter_vars(obj, seen: Set[int]):
    if _is_var(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        yield obj
        return
    if not isinstance(obj, (tuple, list, dict)):
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_vars(item, seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_vars(item, seen)


def _mark_readonly_tensor(tensor, saved) -> None:
    try:
        if getattr(tensor, _FORCE_CPU_ATTR, False):
            return
    except Exception:
        return
    try:
        old_value = getattr(tensor, _READONLY_BORROW_ATTR)
    except AttributeError:
        old_value = _MISSING
    except Exception:
        return
    try:
        setattr(tensor, _READONLY_BORROW_ATTR, True)
    except Exception:
        return
    saved.append((tensor, old_value))


def _mark_readonly(args, kwargs):
    saved = []
    seen: Set[int] = set()
    try:
        var_type = _get_var_type()
    except Exception:
        var_type = None

    for item in args:
        if var_type is not None and isinstance(item, var_type):
            oid = id(item)
            if oid not in seen:
                seen.add(oid)
                _mark_readonly_tensor(item, saved)
        elif isinstance(item, (tuple, list, dict)):
            for tensor in _iter_vars(item, seen):
                _mark_readonly_tensor(tensor, saved)

    if not kwargs:
        return saved

    for tensor in _iter_vars(kwargs, seen):
        _mark_readonly_tensor(tensor, saved)
    return saved


def _mark_readonly_arg_positions(args, positions: Tuple[int, ...]):
    saved = []
    seen: Set[int] = set()
    argc = len(args)
    for pos in positions:
        if pos < 0:
            pos += argc
        if pos < 0 or pos >= argc:
            continue
        for tensor in _iter_vars(args[pos], seen):
            _mark_readonly_tensor(tensor, saved)
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


@contextlib.contextmanager
def _borrow_scope():
    overrides = {
        "JITTOR_TORCH_EXT_UNSAFE_BORROW_INPUTS": "1",
        "JITTOR_TORCH_EXT_BORROW_INPUTS": "1",
    }
    old = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _readonly_borrow_mode() -> str:
    return str(os.environ.get("JITTOR_TORCH_EXT_READONLY_BORROW_MODE", "mark")).strip().lower()


def _wrap_readonly_function(fn):
    if getattr(fn, "_jittor_readonly_borrow_wrapped", False):
        return fn

    mode = _readonly_borrow_mode()
    if mode == "scope":
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with _borrow_scope():
                return fn(*args, **kwargs)
    else:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            saved = _mark_readonly(args, kwargs)
            try:
                return fn(*args, **kwargs)
            finally:
                _restore(saved)

    wrapped._jittor_readonly_borrow_wrapped = True
    return wrapped


def _wrap_scratch_borrow_function(fn):
    if getattr(fn, "_jittor_scratch_borrow_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        saved = _mark_readonly(args, kwargs)
        try:
            return fn(*args, **kwargs)
        finally:
            _restore(saved)

    wrapped._jittor_scratch_borrow_wrapped = True
    return wrapped


def _wrap_readonly_arg_function(fn, positions: Tuple[int, ...]):
    if getattr(fn, "_jittor_readonly_arg_borrow_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        saved = _mark_readonly_arg_positions(args, positions)
        try:
            return fn(*args, **kwargs)
        finally:
            _restore(saved)

    wrapped._jittor_readonly_arg_borrow_wrapped = True
    wrapped._jittor_readonly_arg_positions = positions
    return wrapped


@contextlib.contextmanager
def _copy_scope():
    overrides = {
        "JITTOR_TORCH_EXT_UNSAFE_BORROW_INPUTS": "0",
        "JITTOR_TORCH_EXT_UNSAFE_FAST_METADATA": "0",
        "JITTOR_TORCH_EXT_BORROW_INPUTS": "0",
        "JITTOR_TORCH_EXT_FAST_METADATA": "0",
    }
    old = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _wrap_copy_scope_function(fn):
    if getattr(fn, "_jittor_ext_copy_scope_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _copy_scope():
            return fn(*args, **kwargs)

    wrapped._jittor_ext_copy_scope_wrapped = True
    return wrapped


def _patch_module(module, readonly_functions: Iterable[str],
                  copy_scope_functions: Iterable[str],
                  scratch_borrow_functions: Iterable[str],
                  readonly_arg_functions: Dict[str, Tuple[int, ...]]) -> None:
    for name in copy_scope_functions:
        try:
            fn = getattr(module, name)
        except AttributeError:
            continue
        if callable(fn):
            try:
                setattr(module, name, _wrap_copy_scope_function(fn))
            except Exception:
                pass
    for name in scratch_borrow_functions:
        try:
            fn = getattr(module, name)
        except AttributeError:
            continue
        if callable(fn):
            try:
                setattr(module, name, _wrap_scratch_borrow_function(fn))
            except Exception:
                pass
    for name, positions in readonly_arg_functions.items():
        try:
            fn = getattr(module, name)
        except AttributeError:
            continue
        if callable(fn):
            try:
                setattr(module, name, _wrap_readonly_arg_function(fn, positions))
            except Exception:
                pass
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


def _patch_registered_module(module) -> bool:
    name = module.__name__
    _patch_module(
        module,
        _READONLY_REGISTRY.get(name, ()),
        _COPY_SCOPE_REGISTRY.get(name, ()),
        _SCRATCH_BORROW_REGISTRY.get(name, ()),
        _READONLY_ARG_REGISTRY.get(name, {}),
    )
    return True


def register_readonly_extension_borrow(registry=None, copy_scope_registry=None,
                                       scratch_borrow_registry=None,
                                       readonly_arg_registry=None,
                                       register_patch=register_module_patch) -> None:
    """Register explicit boundary policies without installing an import finder."""

    readonly_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_READONLY_BORROW")) else dict(
        registry or _DEFAULT_READONLY_FUNCTIONS
    )
    copy_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_COPY_SCOPE")) else dict(
        copy_scope_registry or _DEFAULT_COPY_SCOPE_FUNCTIONS
    )
    scratch_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_SCRATCH_BORROW")) else dict(
        scratch_borrow_registry or _DEFAULT_SCRATCH_BORROW_FUNCTIONS
    )
    readonly_arg_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_READONLY_ARG_BORROW")) else dict(
        readonly_arg_registry or _DEFAULT_READONLY_ARG_FUNCTIONS
    )
    if not readonly_reg and not copy_reg and not scratch_reg and not readonly_arg_reg:
        return
    _READONLY_REGISTRY.update(readonly_reg)
    _COPY_SCOPE_REGISTRY.update(copy_reg)
    _SCRATCH_BORROW_REGISTRY.update(scratch_reg)
    _READONLY_ARG_REGISTRY.update(readonly_arg_reg)
    paths = set(readonly_reg) | set(copy_reg) | set(scratch_reg) | set(readonly_arg_reg)
    for name in paths:
        register_patch(name, _patch_registered_module)


def install_readonly_extension_borrow(registry=None, copy_scope_registry=None,
                                      scratch_borrow_registry=None,
                                      readonly_arg_registry=None) -> None:
    """Register policies and install the shared import-time patch mechanism."""

    register_readonly_extension_borrow(
        registry=registry,
        copy_scope_registry=copy_scope_registry,
        scratch_borrow_registry=scratch_borrow_registry,
        readonly_arg_registry=readonly_arg_registry,
    )
    install_module_patches()
