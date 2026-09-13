"""Align optional Transformers backend probes with the Jittor frontend."""
from functools import lru_cache
import importlib.util
import inspect

from ._common import require_version, required_patch, replace, replace_bound_aliases, UnsupportedAdapterVersion

SUPPORTED_VERSIONS = frozenset(("4.56.2", "5.5.3"))


@required_patch
def patch_import_utils(module):
    require_version("transformers", SUPPORTED_VERSIONS)
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
    register_module_patch("transformers.utils.import_utils", patch_import_utils)
