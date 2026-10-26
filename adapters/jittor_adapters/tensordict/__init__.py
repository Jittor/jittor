"""TensorDict 0.10's private leaf registry, outside the Torch type system.

Jittor's jagged frontend object must not take TensorDict's numpy/as_tensor
conversion path. Registration alone is insufficient in 0.10: two validators
leave ``is_tc`` unbound for a registered non-Tensor leaf. Only that exact leaf
type is handled here; all other inputs retain the original validator.

No framework attribute, isinstance result, payload, or gradient is rewritten.
Registering this adapter never imports TensorDict or creates a tensor.
"""
from functools import wraps
import inspect
import sys

from .._common import require_version, required_patch, replace, UnsupportedAdapterVersion
from .jagged_rebuild import patch_deferred_rebuild, patch_loaded_rebuild
from .consolidated_transport import patch_deferred_transport, patch_loaded_transport


SUPPORTED_VERSIONS = frozenset(("0.10.0",))
_VALIDATORS = ("_validate_value_generic", "_validate_value_devicefree")


def _validator(original, nested_type, base, check_device):
    @wraps(original)
    def validate(self, value, non_blocking=False, *, check_shape=True):
        if not isinstance(value, nested_type):
            return original(self, value, non_blocking=non_blocking, check_shape=check_shape)
        if check_shape and value.shape[:self.batch_dims] != self.batch_size:
            raise RuntimeError(
                "batch dimension mismatch, got self.batch_size=%s and value.shape=%s."
                % (self.batch_size, value.shape))
        if check_device and self.device is not None and value.device != self.device:
            recorder = base._device_recorder
            if recorder.marked and self.device.type != "cuda":
                recorder.record_transfer(self.device)
            value = value.to(self.device, non_blocking=non_blocking)
        return value

    validate._jittor_tensordict_nested = nested_type
    return validate


@required_patch
def patch_tensordict(module):
    require_version("tensordict", SUPPORTED_VERSIONS)
    from jittor.compat.torch import nested_tensor_type

    nested_type = nested_tensor_type()
    if not isinstance(nested_type, type):
        raise UnsupportedAdapterVersion("frontend nested_tensor_type() did not return a type")
    base = sys.modules.get("tensordict.base")
    owner = getattr(base, "TensorDictBase", None)
    if owner is None or not callable(getattr(base, "_register_tensor_class", None)):
        raise UnsupportedAdapterVersion("TensorDict leaf registration contract changed")
    originals = {}
    for name in _VALIDATORS:
        original = getattr(owner, name, None)
        if not callable(original) or set(inspect.signature(original).parameters) != {
                "self", "value", "non_blocking", "check_shape"}:
            raise UnsupportedAdapterVersion("TensorDict validator contract changed: " + name)
        originals[name] = original
    recorder = getattr(base, "_device_recorder", None)
    if recorder is None or not hasattr(recorder, "marked") or not callable(getattr(recorder, "record_transfer", None)):
        raise UnsupportedAdapterVersion("TensorDict device recorder contract changed")

    # _register_tensor_class rebinds a tuple without updating imported aliases.
    # Publish the same registration with the shared reversible mutation API,
    # including _td/tensorclass aliases, preserving each one's other classes.
    registries = []
    for path, target in tuple(sys.modules.items()):
        if path != "tensordict" and not path.startswith("tensordict."):
            continue
        if target is None or "_ACCEPTED_CLASSES" not in vars(target):
            continue
        accepted = vars(target)["_ACCEPTED_CLASSES"]
        if not isinstance(accepted, tuple) or not all(isinstance(item, type) for item in accepted):
            raise UnsupportedAdapterVersion("TensorDict accepted-class registry changed: " + path)
        registries.append((target, accepted))
    if not any(target is base for target, _ in registries):
        raise UnsupportedAdapterVersion("TensorDict base leaf registry is absent")

    changed = False
    for target, accepted in registries:
        if nested_type not in accepted:
            replace(target, "_ACCEPTED_CLASSES", accepted + (nested_type,), accepted)
            changed = True
    for name, original in originals.items():
        if getattr(original, "_jittor_tensordict_nested", None) is nested_type:
            continue
        validate = _validator(original, nested_type, base, name == "_validate_value_generic")
        replace(owner, name, validate, original)
        changed = True
    changed = patch_loaded_rebuild() or changed
    return patch_loaded_transport() or changed


def register(register_module_patch):
    register_module_patch("tensordict.utils", patch_deferred_rebuild)
    register_module_patch("tensordict._reductions", patch_deferred_transport)
    # Root finishes after _version and all validators/aliases are available.
    register_module_patch("tensordict", patch_tensordict)
