"""Keep native torch_npu discovery out of the independent Jittor frontend."""
from functools import lru_cache
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


def register(register_module_patch):
    register_module_patch("transformers.utils.import_utils", patch_import_utils)
