"""Keep native torch_npu discovery out of the independent Jittor frontend."""
from functools import lru_cache, wraps
import inspect

from ._common import require_version, required_patch, replace, replace_bound_aliases, UnsupportedAdapterVersion

SUPPORTED_VERSIONS = frozenset(("4.56.2", "5.5.3"))


@required_patch
def patch_import_utils(module):
    require_version("transformers", SUPPORTED_VERSIONS)
    original = vars(module).get("is_torch_npu_available")
    if not callable(original) or "check_device" not in inspect.signature(original).parameters:
        raise UnsupportedAdapterVersion("Transformers NPU probe signature changed")
    if getattr(original, "_jittor_transformers_npu_guard", False):
        return False

    @lru_cache()
    def guarded(check_device=False):
        return False

    guarded._jittor_transformers_npu_guard = True
    guarded._jittor_original_probe = original
    replace(module, "is_torch_npu_available", guarded, original)
    replace_bound_aliases("transformers", "is_torch_npu_available", original, guarded)
    return True


@required_patch
def patch_modeling_utils(module):
    """Do not turn Jittor's active backend into a Transformers device_map.

    Jittor factories follow the active runtime backend, so the torch facade's
    empty tensor and get_default_device() both report NPU even when the caller
    never selected a global default device. Transformers interprets that pair
    as an explicit torch device context and requires Accelerate. The adapter
    cannot distinguish that implicit placement from a real-device context, so
    normal from_pretrained loading ignores inferred real devices. Meta remains
    meaningful for deferred construction and is preserved.
    """
    require_version("transformers", SUPPORTED_VERSIONS)
    original = vars(module).get("get_torch_context_manager_or_global_device")
    if not callable(original) or inspect.signature(original).parameters:
        raise UnsupportedAdapterVersion(
            "Transformers global-device probe signature changed")
    if getattr(original, "_jittor_transformers_device_guard", False):
        return False

    @wraps(original)
    def guarded():
        inferred = original()
        if inferred is None or str(inferred).split(":", 1)[0] == "meta":
            return inferred
        return None

    guarded._jittor_transformers_device_guard = True
    guarded._jittor_original_probe = original
    replace(
        module,
        "get_torch_context_manager_or_global_device",
        guarded,
        original,
    )
    replace_bound_aliases(
        "transformers",
        "get_torch_context_manager_or_global_device",
        original,
        guarded,
    )
    return True


def register(register_module_patch):
    register_module_patch("transformers.utils.import_utils", patch_import_utils)
    register_module_patch("transformers.modeling_utils", patch_modeling_utils)
