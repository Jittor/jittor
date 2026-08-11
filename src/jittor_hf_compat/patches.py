"""Narrow Transformers version-drift patches."""

from __future__ import annotations

import sys


def _patch_transformers_legacy_tied_weights_keys(module) -> bool:
    """Accept the list-form tied-weight declaration used by older model code."""
    model_type = getattr(module, "PreTrainedModel", None)
    original = getattr(model_type, "get_expanded_tied_weights_keys", None)
    if original is None or getattr(
        original, "_jittor_hf_compat_tied_weights", False
    ):
        return False

    def input_embedding_weight_name(self):
        try:
            embedding = self.get_input_embeddings()
        except Exception:
            embedding = None
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            try:
                parameters = self.named_parameters(remove_duplicate=False)
            except TypeError:
                parameters = self.named_parameters()
            for name, parameter in parameters:
                if parameter is weight:
                    return name
        return "model.embed_tokens.weight"

    def get_expanded_tied_weights_keys(self, *args, **kwargs):
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, (list, tuple)):
            source = input_embedding_weight_name(self)
            self._tied_weights_keys = {
                target: source for target in tied if target != source
            }
        return original(self, *args, **kwargs)

    get_expanded_tied_weights_keys._jittor_hf_compat_tied_weights = True
    model_type.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys
    return True


def _refresh_transformers_lazy_torch_backend(lazy_module_type) -> None:
    for module in list(sys.modules.values()):
        if not isinstance(module, lazy_module_type):
            continue
        missing_map = getattr(module, "_object_missing_backend", None)
        if not isinstance(missing_map, dict):
            continue
        for name, missing in list(missing_map.items()):
            if "torch" not in missing:
                continue
            remaining = [backend for backend in missing if backend != "torch"]
            if remaining:
                missing_map[name] = remaining
            else:
                missing_map.pop(name, None)
            cached = getattr(module, "__dict__", {}).get(name)
            if getattr(cached, "_backends", None) is not None:
                try:
                    delattr(module, name)
                except Exception:
                    module.__dict__.pop(name, None)


def _patch_transformers_legacy_symbols(module) -> bool:
    """Restore removed import helpers used by trust-remote-code models."""
    changed = False
    torch_module = sys.modules.get("torch")
    if (
        getattr(torch_module, "__name__", "") == "jittor"
        and not getattr(module, "_jittor_hf_compat_torch_backend", False)
    ):

        def is_torch_available():
            return True

        def get_torch_version():
            return "2.4.0"

        for name in ("is_torch_available", "get_torch_version"):
            previous = getattr(module, name, None)
            clear = getattr(previous, "cache_clear", None)
            if callable(clear):
                clear()
        module.is_torch_available = is_torch_available
        module.get_torch_version = get_torch_version
        mapping = getattr(module, "BACKENDS_MAPPING", None)
        if isinstance(mapping, dict) and "torch" in mapping:
            mapping["torch"] = (
                is_torch_available,
                getattr(module, "PYTORCH_IMPORT_ERROR", ""),
            )
        utilities = sys.modules.get("transformers.utils")
        if utilities is not None:
            utilities.is_torch_available = is_torch_available
            utilities.get_torch_version = get_torch_version
        generic = sys.modules.get("transformers.utils.generic")
        if generic is not None:
            generic._is_torch_available = True
        lazy_module_type = getattr(module, "_LazyModule", None)
        if isinstance(lazy_module_type, type):
            _refresh_transformers_lazy_torch_backend(lazy_module_type)
        module._jittor_hf_compat_torch_backend = True
        changed = True

    if not hasattr(module, "is_torch_fx_available"):

        def is_torch_fx_available():
            try:
                __import__("torch.fx")
                return True
            except Exception:
                return False

        module.is_torch_fx_available = is_torch_fx_available
        utilities = sys.modules.get("transformers.utils")
        if utilities is not None and not hasattr(
            utilities, "is_torch_fx_available"
        ):
            utilities.is_torch_fx_available = is_torch_fx_available
        changed = True
    return changed


def _patch_transformers_default_rope_scaling(module) -> bool:
    """Restore ``None`` for MiniCPM's no-op modern RoPE placeholder."""
    config_type = getattr(module, "PretrainedConfig", None)
    original = getattr(config_type, "__init__", None)
    if original is None or getattr(
        original, "_jittor_hf_compat_default_rope", False
    ):
        return False

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        scaling = getattr(self, "rope_scaling", None)
        if not (
            isinstance(scaling, dict)
            and "type" not in scaling
            and scaling.get("rope_type") == "default"
        ):
            return
        architectures = getattr(self, "architectures", None) or ()
        model_type = getattr(self, "model_type", "") or ""
        if any("MiniCPM" in name for name in architectures) or model_type.startswith(
            "minicpm"
        ):
            self.rope_scaling = None

    init._jittor_hf_compat_default_rope = True
    config_type.__init__ = init
    return True


_MODULE_PATCHES = (
    (
        "transformers.utils.import_utils",
        _patch_transformers_legacy_symbols,
    ),
    (
        "transformers.modeling_utils",
        _patch_transformers_legacy_tied_weights_keys,
    ),
    (
        "transformers.configuration_utils",
        _patch_transformers_default_rope_scaling,
    ),
)


def register_patches(register) -> None:
    for path, callback in _MODULE_PATCHES:
        register(path, callback)


def install():
    from jittor.compat.module_patcher import (
        install_module_patches,
        register_module_patch,
    )

    register_patches(register_module_patch)
    return install_module_patches()


__all__ = ["install", "register_patches"]
