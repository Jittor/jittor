"""Do not infer Ascend hardware from a portable Torch namespace.

verl 0.9.1 selects private attention helpers merely because torch.npu exists.
Jittor publishes device namespaces across backends, so retain the upstream
strict/public is_available() probe for this private dispatch as well. No Torch
attributes, tensor computations, imports or device placement are replaced.
Upstream: 1876b06d0a3e4e71e06230be10af14492ca8a75b.
Revalidate on upstream device.py or compat module-patcher/public NPU probe changes.
"""
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import sys
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)

SUPPORTED_VERSIONS = frozenset(("0.9.1",))
MODULE = "verl.utils.device"
FUNCTION = "is_torch_npu_available"
SOURCE_SHA256 = "0ce776f6e083e1919d4be28092b49b8a7f666aad1da88b4509aeb625c0868617"
FUNCTION_SHA256 = "e198a3436c4cf595411387a47349c2ad54be16c66f695a53997dbf4cb40c7a36"
POLICY = "verl_v091_portable_npu_namespace_v1"


@required_patch
def patch_device_probe(module):
    require_version("verl", SUPPORTED_VERSIONS)
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected verl device probe module")
    path = inspect.getsourcefile(module)
    if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl device probe module source SHA256 changed")
    current = vars(module).get(FUNCTION)
    already_patched = getattr(current, "_jittor_verl_device_probe", None) == POLICY
    original = getattr(current, "_jittor_verl_original", None) if already_patched else current
    if (not isinstance(original, types.FunctionType) or original.__closure__
            or original.__module__ != MODULE or original.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("verl device probe ownership changed")
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != FUNCTION_SHA256:
        raise UnsupportedAdapterVersion("verl device probe function source SHA256 changed")
    if already_patched:
        return False

    @wraps(original)
    def guarded(check_device=True):
        # The non-strict namespace-presence branch is not hardware discovery in
        # the portable frontend. Delegate to upstream's public strict probe.
        return original(check_device=check_device if check_device else True)

    guarded._jittor_verl_device_probe = POLICY
    guarded._jittor_verl_original = original
    replace(module, FUNCTION, guarded, original)
    replace_bound_aliases("verl", FUNCTION, original, guarded)
    return True


@required_patch
def patch_deferred_device_probe(module):
    root = sys.modules.get("verl")
    # The pinned root imports device.py through protocol before publishing its
    # version. Validate/install at root completion, never infer wheel metadata.
    if root is not None and "__version__" not in vars(root):
        return False
    return patch_device_probe(module)


@required_patch
def patch_root_device_probe(module):
    current = sys.modules.get(MODULE)
    if current is None:
        return False  # Register the later module hook; do not force an import.
    return patch_device_probe(current)


def register(register_module_patch):
    register_module_patch(MODULE, patch_deferred_device_probe)
    register_module_patch("verl", patch_root_device_probe)
