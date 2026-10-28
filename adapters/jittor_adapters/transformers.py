"""Align optional Transformers backend probes with the Jittor frontend."""
from functools import lru_cache, wraps
import importlib.util
import inspect
import sys

from ._common import require_version, required_patch, replace, replace_bound_aliases, UnsupportedAdapterVersion

SUPPORTED_VERSIONS = frozenset(("4.56.2", "5.5.3", "5.9.0", "5.13.0"))


@required_patch
def patch_import_utils(module):
    require_version("transformers", SUPPORTED_VERSIONS)
    package_probe = vars(module).get("_is_package_available")
    if (not callable(package_probe) or
            "return_version" not in inspect.signature(package_probe).parameters):
        raise UnsupportedAdapterVersion("Transformers package probe signature changed")

    torch_module = sys.modules.get("torch")
    torch_api_version = getattr(torch_module, "__torch_version__", None)
    if not isinstance(torch_api_version, str):
        raise UnsupportedAdapterVersion("Jittor Torch API version is unavailable")

    if not getattr(package_probe, "_jittor_transformers_torch_version", False):
        @wraps(package_probe)
        def guarded_package_probe(package, return_version=False):
            if package == "torch":
                return (True, torch_api_version) if return_version else True
            return package_probe(package, return_version=return_version)

        guarded_package_probe._jittor_transformers_torch_version = True
        guarded_package_probe._jittor_original_probe = package_probe
        replace(module, "_is_package_available", guarded_package_probe, package_probe)
        replace_bound_aliases(
            "transformers", "_is_package_available", package_probe, guarded_package_probe)
        for cache_name in ("is_torch_available", "get_torch_version"):
            cached = vars(module).get(cache_name)
            clear = getattr(cached, "cache_clear", None)
            if clear is not None:
                clear()

    original = vars(module).get("is_torch_npu_available")
    original_torchvision = vars(module).get("is_torchvision_available")
    original_torchvision_v2 = vars(module).get("is_torchvision_v2_available")
    backend_mapping = vars(module).get("BACKENDS_MAPPING")
    if not callable(original) or "check_device" not in inspect.signature(original).parameters:
        raise UnsupportedAdapterVersion("Transformers NPU probe signature changed")
    if not callable(original_torchvision) or inspect.signature(original_torchvision).parameters:
        raise UnsupportedAdapterVersion("Transformers torchvision probe signature changed")
    if not callable(original_torchvision_v2) or inspect.signature(original_torchvision_v2).parameters:
        raise UnsupportedAdapterVersion("Transformers torchvision v2 probe signature changed")
    torchvision_backend = backend_mapping.get("torchvision") if backend_mapping is not None else None
    if (not isinstance(torchvision_backend, tuple) or len(torchvision_backend) != 2
            or torchvision_backend[0] is not original_torchvision):
        raise UnsupportedAdapterVersion("Transformers torchvision backend mapping changed")
    if getattr(original, "_jittor_transformers_npu_guard", False):
        return False

    @lru_cache()
    def guarded(check_device=False):
        return False

    guarded._jittor_transformers_npu_guard = True
    guarded._jittor_original_probe = original
    replace(module, "is_torch_npu_available", guarded, original)
    replace_bound_aliases("transformers", "is_torch_npu_available", original, guarded)

    @lru_cache()
    def guarded_torchvision():
        if original_torchvision():
            return True
        return importlib.util.find_spec("torchvision") is not None

    @lru_cache()
    def guarded_torchvision_v2():
        if original_torchvision():
            return original_torchvision_v2()
        return guarded_torchvision()

    guarded_torchvision._jittor_original_probe = original_torchvision
    guarded_torchvision_v2._jittor_original_probe = original_torchvision_v2
    replace(module, "is_torchvision_available", guarded_torchvision, original_torchvision)
    replace_bound_aliases(
        "transformers", "is_torchvision_available", original_torchvision, guarded_torchvision)
    replace(module, "is_torchvision_v2_available", guarded_torchvision_v2, original_torchvision_v2)
    replace_bound_aliases(
        "transformers", "is_torchvision_v2_available", original_torchvision_v2, guarded_torchvision_v2)
    patched_backend_mapping = backend_mapping.copy()
    patched_backend_mapping["torchvision"] = (guarded_torchvision, torchvision_backend[1])
    replace(module, "BACKENDS_MAPPING", patched_backend_mapping, backend_mapping)
    return True


def register(register_module_patch):
    from .transformers_loading import register as register_materialized_loading
    from .transformers_qwen35 import register as register_qwen35

    register_materialized_loading(register_module_patch)
    register_qwen35(register_module_patch)
    register_module_patch("transformers.utils.import_utils", patch_import_utils)
