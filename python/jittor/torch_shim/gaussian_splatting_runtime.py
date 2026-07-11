"""Runtime patches for graphdeco Gaussian Splatting under the torch shim."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


_FALSEY = {"0", "false", "no", "off"}
_GAUSSIAN_MODEL_MODULE = "scene.gaussian_model"
_LPIPS_MODULE = "lpipsPyTorch"
_PATCH_MODULES = {_GAUSSIAN_MODEL_MODULE, _LPIPS_MODULE}


def _is_falsey(value) -> bool:
    return str(value or "").strip().lower() in _FALSEY


def _enabled() -> bool:
    return not _is_falsey(os.environ.get("JITTOR_GS_RUNTIME_PATCHES", "1"))


def _save_ply_enabled() -> bool:
    return not _is_falsey(os.environ.get("JITTOR_GS_BATCH_SAVE_PLY", "1"))


def _patch_gaussian_model_module(mod) -> bool:
    if not _save_ply_enabled():
        return False
    cls = getattr(mod, "GaussianModel", None)
    if cls is None or getattr(cls, "_jittor_gs_batch_save_ply", False):
        return False

    original = getattr(cls, "save_ply", None)
    if original is None:
        return False

    def save_ply(self, path):
        try:
            import jittor as jt
            import numpy as np
            from plyfile import PlyData, PlyElement
            from utils.system_utils import mkdir_p

            mkdir_p(os.path.dirname(path))
            tensors = [
                self._xyz.detach(),
                self._features_dc.detach().transpose(1, 2).reshape(
                    self._features_dc.shape[0], -1
                ),
                self._features_rest.detach().transpose(1, 2).reshape(
                    self._features_rest.shape[0], -1
                ),
                self._opacity.detach(),
                self._scaling.detach(),
                self._rotation.detach(),
            ]
            xyz, f_dc, f_rest, opacities, scale, rotation = jt.fetch_sync(tensors)
            dtype_full = [
                (attribute, "f4") for attribute in self.construct_list_of_attributes()
            ]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            names = elements.dtype.names
            col = 0
            for arr in (xyz, f_dc, f_rest, opacities, scale, rotation):
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if col == 3:
                    for _ in range(3):
                        elements[names[col]] = 0.0
                        col += 1
                for j in range(arr.shape[1]):
                    elements[names[col]] = arr[:, j]
                    col += 1
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(path)
        except Exception:
            return original(self, path)

    save_ply._jittor_gs_original = original
    cls.save_ply = save_ply
    cls._jittor_gs_batch_save_ply = True
    return True


def _patch_lpips_module(mod) -> bool:
    original = getattr(mod, "lpips", None)
    criterion_cls = getattr(mod, "LPIPS", None)
    if original is None or criterion_cls is None or getattr(
            original, "_jittor_gs_cached_criterion", False):
        return False

    criteria = {}

    def lpips(x, y, net_type="alex", version="0.1"):
        device = x.device
        key = (str(net_type), str(version), str(device))
        criterion = criteria.get(key)
        if criterion is None:
            criterion = criterion_cls(net_type, version).to(device)
            criteria[key] = criterion
        return criterion(x, y)

    lpips._jittor_gs_cached_criterion = True
    lpips._jittor_gs_original = original
    lpips._jittor_gs_criteria = criteria
    mod.lpips = lpips
    return True


def _patch_loaded_modules() -> bool:
    patched = False
    mod = sys.modules.get(_GAUSSIAN_MODEL_MODULE)
    if mod is not None:
        patched = _patch_gaussian_model_module(mod) or patched
    mod = sys.modules.get(_LPIPS_MODULE)
    if mod is not None:
        patched = _patch_lpips_module(mod) or patched
    return patched


def install() -> None:
    if not _enabled():
        return
    _patch_loaded_modules()
    for finder in sys.meta_path:
        if isinstance(finder, _GaussianSplattingRuntimeFinder):
            return
    sys.meta_path.insert(0, _GaussianSplattingRuntimeFinder())


class _GaussianSplattingRuntimeLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        if module.__name__ == _GAUSSIAN_MODEL_MODULE:
            _patch_gaussian_model_module(module)
        elif module.__name__ == _LPIPS_MODULE:
            _patch_lpips_module(module)


class _GaussianSplattingRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCH_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _GaussianSplattingRuntimeLoader):
            spec.loader = _GaussianSplattingRuntimeLoader(spec.loader)
        return spec
