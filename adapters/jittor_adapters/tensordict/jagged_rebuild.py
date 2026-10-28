"""TensorDict 0.10's private packed-jagged reconstruction, via public APIs.

The upstream private constructor shares values storage but detaches autograd.
Keep that contract; do not create a Torch SymInt registry or skip consolidation.
Only the public Jittor nested leaf type is intercepted. Revalidate on the pinned
helper, nested public construction, or compat owned-patcher changes.
"""
from functools import wraps
import hashlib
import inspect
import sys
import types

from .._common import (require_version, required_patch,
                      replace_bound_aliases, UnsupportedAdapterVersion)

SUPPORTED_VERSIONS = frozenset(("0.10.0",))
MODULE = "tensordict.utils"
FUNCTION = "_rebuild_njt_from_njt"
FUNCTION_SHA256 = "7b07a0b409541b4ba2cfe233fd472bf7bf1cc7fb6a317c7e9cae027d55a9614a"
POLICY = "tensordict_v010_public_detached_jagged_v1"
LEAF_FUNCTION = "_is_leaf_nontensor"
LEAF_SHA256 = "bab60822dcbaba2c7e76f1b00df812b416e7b0dcbbf75cd728f6b21088dd15b3"


def _check_owned_handle(current, module, name, original):
    from jittor.compat.module_patcher import MethodPatch
    handle = getattr(current, "_jittor_tensordict_patch_handle", None)
    if (not isinstance(handle, MethodPatch) or handle.owner is not module
            or handle.name != name or handle.replacement is not current
            or not handle.had_local_value or handle.local_value is not original):
        raise UnsupportedAdapterVersion("TensorDict helper replacement ownership changed")


@required_patch
def patch_rebuild(module):
    require_version("tensordict", SUPPORTED_VERSIONS)
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected TensorDict jagged helper module")
    current = vars(module).get(FUNCTION)
    patched = getattr(current, "_jittor_tensordict_rebuild", None) == POLICY
    original = getattr(current, "_jittor_tensordict_original", None) if patched else current
    if (not isinstance(original, types.FunctionType) or original.__closure__
            or original.__module__ != MODULE or original.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("TensorDict jagged helper ownership changed")
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != FUNCTION_SHA256:
        raise UnsupportedAdapterVersion("TensorDict jagged helper source SHA256 changed")
    if patched:
        _check_owned_handle(current, module, FUNCTION, original)
        return False
    from jittor.compat.torch import nested_tensor_type, nested_tensor_rebuild
    nested_type = nested_tensor_type()
    if not isinstance(nested_type, type):
        raise UnsupportedAdapterVersion("frontend nested_tensor_type() did not return a type")

    @wraps(original)
    def rebuild(x, values, offsets, lengths):
        if not isinstance(x, nested_type):
            return original(x, values, offsets, lengths)
        has_lengths = x.lengths() is not None
        # Pinned 0.10 extract_kwargs keeps lengths=None; its new_thing lookup
        # therefore copies symbolic identity only in the lengths-present case.
        # The compatibility owner, not this leaf adapter, owns that binding.
        return nested_tensor_rebuild(
            x, values.detach(), offsets,
            lengths=lengths if has_lengths else None,
            preserve_ragged_identity=has_lengths and lengths is not None)

    rebuild._jittor_tensordict_rebuild = POLICY
    rebuild._jittor_tensordict_original = original
    from jittor.compat.module_patcher import patch_method
    rebuild._jittor_tensordict_patch_handle = patch_method(
        module, FUNCTION, rebuild, expected=original)
    replace_bound_aliases("tensordict", FUNCTION, original, rebuild)
    return True


@required_patch
def patch_deferred_rebuild(module):
    root = sys.modules.get("tensordict")
    if root is not None and "__version__" not in vars(root):
        return False
    return patch_rebuild(module)


def patch_loaded_rebuild():
    module = sys.modules.get(MODULE)
    if module is None:
        return False
    base = sys.modules.get("tensordict.base")
    if base is None:
        raise UnsupportedAdapterVersion("TensorDict base module is absent")
    leaf_changed = patch_leaf_predicate(base)
    return patch_rebuild(module) or leaf_changed


@required_patch
def patch_leaf_predicate(module):
    require_version("tensordict", SUPPORTED_VERSIONS)
    if module.__name__ != "tensordict.base":
        raise UnsupportedAdapterVersion("unexpected TensorDict leaf predicate module")
    current = vars(module).get(LEAF_FUNCTION)
    patched = getattr(current, "_jittor_tensordict_leaf", None) == POLICY
    original = getattr(current, "_jittor_tensordict_original", None) if patched else current
    if (not isinstance(original, types.FunctionType) or original.__closure__
            or original.__module__ != module.__name__ or original.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("TensorDict leaf predicate ownership changed")
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != LEAF_SHA256:
        raise UnsupportedAdapterVersion("TensorDict leaf predicate source SHA256 changed")
    if patched:
        _check_owned_handle(current, module, LEAF_FUNCTION, original)
        return False
    from jittor.compat.torch import nested_tensor_type
    nested_type = nested_tensor_type()
    if not isinstance(nested_type, type):
        raise UnsupportedAdapterVersion("frontend nested_tensor_type() did not return a type")

    @wraps(original)
    def is_leaf(cls):
        if isinstance(cls, type) and issubclass(cls, nested_type):
            return True
        return original(cls)

    is_leaf._jittor_tensordict_leaf = POLICY
    is_leaf._jittor_tensordict_original = original
    from jittor.compat.module_patcher import patch_method
    is_leaf._jittor_tensordict_patch_handle = patch_method(
        module, LEAF_FUNCTION, is_leaf, expected=original)
    replace_bound_aliases("tensordict", LEAF_FUNCTION, original, is_leaf)
    return True
