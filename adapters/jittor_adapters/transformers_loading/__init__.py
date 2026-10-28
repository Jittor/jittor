"""Opt-in, materialized CPU construction for the pinned HF 5.9 loading policy.

This is NOT a meta-device implementation. It consumes CPU storage for the full
model during construction, then leaves checkpoint loading, missing-key/buffer
initialization, and device dispatch to unmodified upstream code. No tensor or
model values are written here. Only inherited, unchanged HF initialization
policies are supported; a custom loader bypassing this selector is out of scope.
"""
import hashlib
import inspect
import os
from contextlib import contextmanager

from .._common import require_version, required_patch, UnsupportedAdapterVersion


SUPPORTED_VERSIONS = frozenset(("5.9.0",))
ENVIRONMENT_VARIABLE = "JITTOR_TRANSFORMERS_MATERIALIZED_INIT"
# Exact inspect.getsource() bytes, including decorators and class indentation.
SOURCE_SHA256 = {
    "get_init_context": "85673c5340632d9d83e0bfd513a728a7f2a7347dacba87d412202471a24cc0e2",
    "no_init_weights": "1efb0d0d1fa2d202924ca22dc92bad70566ebb1f1b10f93d422e9c4e983c92b1",
}


def _enabled():
    value = os.environ.get(ENVIRONMENT_VARIABLE)
    if value not in (None, "0", "1"):
        raise ValueError(ENVIRONMENT_VARIABLE + " must be unset, '0', or '1'")
    return value == "1"


def _verify_source(function, name):
    try:
        source = inspect.getsource(function)
    except (TypeError, OSError) as error:
        raise UnsupportedAdapterVersion("HF loading source is required: " + name) from error
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != SOURCE_SHA256[name]:
        raise UnsupportedAdapterVersion("HF loading source changed: " + name)


@required_patch
def patch_modeling_utils(module):
    if not _enabled():
        return False
    require_version("transformers", SUPPORTED_VERSIONS)
    owner = module.PreTrainedModel
    descriptor = vars(owner).get("get_init_context")
    if not isinstance(descriptor, classmethod):
        raise UnsupportedAdapterVersion("HF get_init_context must be an owned classmethod")
    original = descriptor.__func__
    if getattr(original, "_jittor_transformers_materialized_init", False):
        return False
    if (not inspect.isfunction(original) or original.__closure__
            or hasattr(original, "__wrapped__") or original.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("HF get_init_context ownership/wrapper changed")
    _verify_source(original, "get_init_context")
    expected_parameters = ("cls", "dtype", "is_quantized", "_is_ds_init_called", "allow_all_kernels")
    if tuple(inspect.signature(original).parameters) != expected_parameters:
        raise UnsupportedAdapterVersion("HF get_init_context signature changed")
    init = module.init
    no_init = init.no_init_weights
    wrapped = getattr(no_init, "__wrapped__", None)
    if (not inspect.isfunction(no_init) or not inspect.isgeneratorfunction(wrapped)
            or wrapped.__globals__ is not vars(init)
            or no_init.__code__ is not contextmanager(lambda: None).__code__
            or no_init.__closure__ is None or len(no_init.__closure__) != 1
            or no_init.__closure__[0].cell_contents is not wrapped):
        raise UnsupportedAdapterVersion("HF no_init_weights contextmanager wrapper changed")
    _verify_source(init.no_init_weights, "no_init_weights")
    if inspect.signature(init.no_init_weights).parameters:
        raise UnsupportedAdapterVersion("HF no_init_weights signature changed")
    # All factories must remain the ones validated when this selector was
    # installed. We preserve their returned contexts instead of reimplementing
    # dtype, weight-tying, hub patches, or safe-creation behavior.
    dependencies = (
        (module, "local_torch_dtype", module.local_torch_dtype),
        (init, "no_tie_weights", init.no_tie_weights),
        (module, "apply_patches", module.apply_patches),
        (init, "meta_device_safe_creation_ops", init.meta_device_safe_creation_ops),
        (init, "no_init_weights", init.no_init_weights),
    )
    if not all(callable(value) for _, _, value in dependencies):
        raise UnsupportedAdapterVersion("HF construction context factory is missing")

    def get_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
        if not _enabled():
            # A process can disable the opt-in after registration without
            # leaving its later loads silently on the materialized strategy.
            return original(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
        require_version("transformers", SUPPORTED_VERSIONS)
        if (is_quantized or _is_ds_init_called or allow_all_kernels
                or module.is_deepspeed_zero3_enabled()):
            raise UnsupportedAdapterVersion("materialized loading excludes quantization, DeepSpeed, and hub kernels")
        if cls.is_remote_code():
            raise UnsupportedAdapterVersion("materialized loading excludes remote/custom model code")
        if inspect.getattr_static(cls, "get_init_context") is not replacement:
            raise UnsupportedAdapterVersion("custom get_init_context overrides are unsupported")
        for target, name, expected in dependencies:
            if getattr(target, name, None) is not expected:
                raise UnsupportedAdapterVersion("HF construction context was replaced: " + name)
        contexts = original(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
        if (not isinstance(contexts, list) or len(contexts) != 5
                or not isinstance(contexts[3], module.torch.device)
                or contexts[3].type != "meta"
                or any(not callable(getattr(context, "__enter__", None))
                       or not callable(getattr(context, "__exit__", None)) for context in contexts)):
            raise UnsupportedAdapterVersion("HF construction context layout changed")
        # HF exits this list's ContextManagers block before loading weights and
        # before _finalize_model_loading initializes missing/nonpersistent data.
        # Its official no_init_weights therefore prevents premature initialized
        # flags, but is gone when its official missing-key initialization runs.
        return contexts[:3] + [init.no_init_weights(), module.torch.device("cpu")] + contexts[4:]

    get_init_context.__name__ = original.__name__
    get_init_context.__qualname__ = original.__qualname__
    get_init_context._jittor_transformers_materialized_init = True
    get_init_context._jittor_original = original
    replacement = classmethod(get_init_context)
    # Compare the raw descriptor immediately before publication. patch_method's
    # optional expected= uses bound getattr identity, unsuitable for classmethods.
    # The shared runtime hook records/restores the original raw descriptor.
    if vars(owner).get("get_init_context") is not descriptor:
        raise UnsupportedAdapterVersion("HF get_init_context changed during installation")
    from jittor.compat.module_patcher import patch_method

    patch_method(owner, "get_init_context", replacement)
    return True


def register(register_module_patch):
    """Only register a future module callback; never import Transformers here."""
    if _enabled():
        register_module_patch("transformers.modeling_utils", patch_modeling_utils)
