"""Selective boundary policies for known torch native extensions."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
import contextlib
from typing import Dict, Iterable, Optional, Set, Tuple


_READONLY_BORROW_ATTR = "_jittor_torch_ext_readonly_borrow"
_FORCE_CPU_ATTR = "_jittor_torch_force_cpu"
_MISSING = object()
_VAR_TYPE = None

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

_DEFAULT_COPY_SCOPE_FUNCTIONS: Dict[str, Tuple[str, ...]] = {
    # o_voxel's GLB export keeps native mesh/rasterization objects alive across
    # several pybind calls. Keep TRELLIS inference on the low-overhead extension
    # path, but use the conservative tensor boundary for this export phase.
    "o_voxel.postprocess": (
        "to_glb",
    ),
    # nvdiffrast keeps CUDA raster/texture state around Python wrapper calls and
    # is commonly used after TRELLIS export for preview rendering. Use a stable
    # extension boundary for its public torch ops while keeping inference fast.
    "nvdiffrast.torch.ops": (
        "rasterize",
        "interpolate",
        "texture",
        "texture_construct_mip",
        "antialias",
        "antialias_construct_topology_hash",
    ),
}


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


def _patch_module(module, readonly_functions: Iterable[str], copy_scope_functions: Iterable[str]) -> None:
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
    def __init__(self, loader, readonly_functions: Tuple[str, ...], copy_scope_functions: Tuple[str, ...]):
        self.loader = loader
        self.readonly_functions = readonly_functions
        self.copy_scope_functions = copy_scope_functions

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        _patch_module(module, self.readonly_functions, self.copy_scope_functions)


class _ExtensionPolicyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, readonly_registry: Dict[str, Tuple[str, ...]],
                 copy_scope_registry: Dict[str, Tuple[str, ...]]):
        self.readonly_registry = readonly_registry
        self.copy_scope_registry = copy_scope_registry

    def find_spec(self, fullname, path=None, target=None):
        readonly_functions = self.readonly_registry.get(fullname, ())
        copy_scope_functions = self.copy_scope_registry.get(fullname, ())
        if not readonly_functions and not copy_scope_functions:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if isinstance(spec.loader, _ExtensionPolicyLoader):
            return spec
        spec.loader = _ExtensionPolicyLoader(spec.loader, readonly_functions, copy_scope_functions)
        return spec


def install_readonly_extension_borrow(registry=None, copy_scope_registry=None) -> None:
    """Install import-time wrappers for native extension boundary policies."""

    readonly_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_READONLY_BORROW")) else dict(
        registry or _DEFAULT_READONLY_FUNCTIONS
    )
    copy_reg = {} if _is_falsey(os.environ.get("JITTOR_TORCH_EXT_COPY_SCOPE")) else dict(
        copy_scope_registry or _DEFAULT_COPY_SCOPE_FUNCTIONS
    )
    if not readonly_reg and not copy_reg:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _ExtensionPolicyFinder):
            finder.readonly_registry.update(readonly_reg)
            finder.copy_scope_registry.update(copy_reg)
            break
    else:
        sys.meta_path.insert(0, _ExtensionPolicyFinder(readonly_reg, copy_reg))
    for name in set(readonly_reg) | set(copy_reg):
        module = sys.modules.get(name)
        if module is not None:
            _patch_module(module, readonly_reg.get(name, ()), copy_reg.get(name, ()))
