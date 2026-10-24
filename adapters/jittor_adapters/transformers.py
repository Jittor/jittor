"""Align optional Transformers backend probes with the Jittor frontend."""
from collections.abc import Mapping
from functools import lru_cache, wraps
import importlib.util
import inspect
import os
import sys
from types import SimpleNamespace

from ._common import require_version, required_patch, replace, replace_bound_aliases, UnsupportedAdapterVersion
from ._accelerate_peft import patch_accelerate_peft_fsdp_utils

SUPPORTED_VERSIONS = frozenset(("4.56.2", "5.5.3"))
ACCELERATE_SUPPORTED_VERSIONS = frozenset(("1.10.1",))


def _frontend_npu_available(check_device=False):
    """Use the frontend's public backend; never import a native Torch extension."""
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    if check_device and npu.device_count() < 1:
        return False
    return bool(npu.is_available())


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
        return _frontend_npu_available(check_device)

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


@required_patch
def patch_accelerate_imports(module):
    """Keep Accelerate's import-only native torch_npu branches inactive."""
    require_version("accelerate", ACCELERATE_SUPPORTED_VERSIONS)
    original = vars(module).get("is_npu_available")
    if not callable(original) or "check_device" not in inspect.signature(original).parameters:
        raise UnsupportedAdapterVersion("Accelerate NPU probe signature changed")
    if getattr(original, "_jittor_accelerate_npu_bootstrap", False):
        return False

    phase = SimpleNamespace(enabled=False)

    @lru_cache()
    def bootstrap(check_device=False):
        return _frontend_npu_available(check_device) if phase.enabled else False

    bootstrap._jittor_accelerate_npu_bootstrap = True
    bootstrap._jittor_accelerate_npu_phase = phase
    bootstrap._jittor_original_probe = original
    replace(module, "is_npu_available", bootstrap, original)
    replace_bound_aliases("accelerate", "is_npu_available", original, bootstrap)
    return True


@required_patch
def patch_accelerate(module):
    """Publish real NPU detection after all pinned import-only branches finish.

    Accelerate 1.10.1 imports torch_npu at module scope in state, accelerator
    and utils.modeling when its probe returns true. Those side-effect-only
    imports belong to native PyTorch. Once the root package is constructed,
    every runtime consumer must use the independent frontend's actual device.
    """
    require_version("accelerate", ACCELERATE_SUPPORTED_VERSIONS)
    imports = sys.modules.get("accelerate.utils.imports")
    original = vars(imports).get("is_npu_available") if imports is not None else None
    if not getattr(original, "_jittor_accelerate_npu_bootstrap", False):
        raise UnsupportedAdapterVersion("Accelerate NPU bootstrap contract changed")
    phase = original._jittor_accelerate_npu_phase
    if phase.enabled:
        return False
    for path in ("accelerate.state", "accelerate.accelerator", "accelerate.utils.modeling"):
        consumer = sys.modules.get(path)
        if consumer is None or vars(consumer).get("is_npu_available") is not original:
            raise UnsupportedAdapterVersion("Accelerate NPU consumer changed: " + path)

    # Keep the probe identity stable across both phases. Independent callback
    # hooks own disjoint slots, so cleanup cannot mistake the second phase for
    # an external replacement of the bootstrap callback's attribute.
    replace(phase, "enabled", True, False)
    original.cache_clear()
    return True


@required_patch
def patch_flash_attention_utils(module):
    """Keep native TorchNPU FlashAttention discovery private to its provider.

    Public NPU capability belongs to the frontend device. Transformers 4.56.2
    also uses that public probe to import native ``torch_npu`` FlashAttention,
    including an unconditional mask-alignment import during Trainer loading.
    The independent frontend does not supply that native extension. Retain
    upstream FA2/FA3 discovery and reject unsupported explicit FA requests.
    """
    version = require_version("transformers", SUPPORTED_VERSIONS)
    if version != "4.56.2":
        # Preserve the existing 5.5.3 adapter boundary. This private module's
        # NPU extension contract has only been audited on pinned 4.56.2.
        return False
    original = vars(module).get("is_torch_npu_available")
    alignment = vars(module).get("flash_attn_supports_top_left_mask")
    lazy_imports = vars(module).get("_lazy_imports")
    probes = [vars(module).get(name) for name in (
        "is_flash_attn_2_available", "is_flash_attn_3_available",
        "is_flash_attn_greater_or_equal_2_10")]
    if (not callable(original) or "check_device" not in inspect.signature(original).parameters
            or not callable(alignment) or inspect.signature(alignment).parameters
            or not callable(lazy_imports)
            or tuple(inspect.signature(lazy_imports).parameters) != ("implementation",)
            or not all(callable(probe) for probe in probes)):
        raise UnsupportedAdapterVersion("Transformers private FlashAttention provider contract changed")
    if getattr(original, "_jittor_native_flash_npu_guard", False):
        return False
    fa2, fa3, fa2_new_mask = probes

    def native_provider_available(check_device=False):
        return False

    def supports_top_left_mask():
        if fa3():
            return False
        if fa2():
            return not fa2_new_mask()
        return False

    native_provider_available._jittor_native_flash_npu_guard = True
    native_provider_available._jittor_original_probe = original
    supports_top_left_mask._jittor_original_probe = alignment
    # Deliberately do not replace public NPU aliases: this slot describes a
    # native FlashAttention provider, while Trainer/Accelerate use the device.
    replace(module, "is_torch_npu_available", native_provider_available, original)
    replace(module, "flash_attn_supports_top_left_mask", supports_top_left_mask, alignment)
    replace_bound_aliases(
        "transformers", "flash_attn_supports_top_left_mask", alignment, supports_top_left_mask)
    return True


_FSDP_POLICY_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "float32": "float32",
    "torch.float32": "float32",
}


def _resolve_fsdp_policy_dtype(value, torch_module):
    if not isinstance(value, str):
        return value
    name = value.strip().lower()
    if name in ("none", "no"):
        return None
    dtype_name = _FSDP_POLICY_DTYPES.get(name)
    if dtype_name is None:
        raise ValueError("unsupported FSDP mixed precision dtype: " + value)
    return getattr(torch_module, dtype_name)


def _apply_trainer_fsdp_policy(trainer, torch_module):
    """Apply an explicit TrainingArguments FSDP precision policy to Accelerate."""
    args = getattr(trainer, "args", None)
    fsdp_config = getattr(args, "fsdp_config", None)
    if not isinstance(fsdp_config, Mapping):
        return False
    configured_policy = fsdp_config.get("mixed_precision_policy")
    if configured_policy is None:
        return False

    accelerator = getattr(trainer, "accelerator", None)
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "fsdp_plugin", None)
    if plugin is None:
        raise ValueError("fsdp_config.mixed_precision_policy requires an active FSDP plugin")

    fsdp_version = int(getattr(plugin, "fsdp_version", 1))
    fsdp_api = getattr(getattr(torch_module, "distributed", None), "fsdp", None)
    if fsdp_version == 2:
        policy_type = getattr(fsdp_api, "MixedPrecisionPolicy", None)
        valid_fields = {"param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"}
    elif fsdp_version == 1:
        policy_type = getattr(fsdp_api, "MixedPrecision", None)
        valid_fields = {"param_dtype", "reduce_dtype", "buffer_dtype"}
    else:
        raise ValueError("unsupported FSDP version for mixed precision policy: " + str(fsdp_version))
    if not callable(policy_type):
        raise UnsupportedAdapterVersion("Torch FSDP mixed precision policy type is unavailable")

    if isinstance(configured_policy, policy_type):
        policy = configured_policy
    elif isinstance(configured_policy, Mapping):
        unknown = set(configured_policy) - valid_fields
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise ValueError("unsupported FSDP mixed precision policy fields: " + fields)
        if not configured_policy:
            raise ValueError("fsdp_config.mixed_precision_policy must not be empty")
        policy = policy_type(**{
            key: _resolve_fsdp_policy_dtype(value, torch_module)
            for key, value in configured_policy.items()
        })
    else:
        raise TypeError("fsdp_config.mixed_precision_policy must be a mapping or FSDP policy object")

    previous = getattr(plugin, "mixed_precision_policy", None)
    plugin.mixed_precision_policy = policy
    try:
        validate = getattr(plugin, "validate_mixed_precision_policy", None)
        if callable(validate):
            validate()
    except Exception:
        plugin.mixed_precision_policy = previous
        raise
    return True


@required_patch
def patch_trainer(module):
    """Bridge mixed precision config omitted by the pinned Trainer constructor."""
    version = require_version("transformers", SUPPORTED_VERSIONS)
    if version != "4.56.2":
        return False
    trainer_type = vars(module).get("Trainer")
    original = getattr(trainer_type, "__init__", None) if isinstance(trainer_type, type) else None
    if not callable(original):
        raise UnsupportedAdapterVersion("Transformers Trainer constructor is unavailable")
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError) as exc:
        raise UnsupportedAdapterVersion("Transformers Trainer constructor signature is unavailable") from exc
    if "args" not in parameters:
        raise UnsupportedAdapterVersion("Transformers Trainer constructor args contract changed")
    if getattr(original, "_jittor_fsdp_policy_bridge", False):
        return False
    torch_module = vars(module).get("torch")
    if torch_module is None:
        raise UnsupportedAdapterVersion("Transformers Trainer no longer exposes its Torch frontend")

    @wraps(original)
    def init(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _apply_trainer_fsdp_policy(self, torch_module)
        return result

    init._jittor_fsdp_policy_bridge = True
    init._jittor_original_init = original
    replace(trainer_type, "__init__", init, original)
    return True



@required_patch
def patch_accelerate_fsdp_utils(module):
    """Restore pinned FSDP2 sharded optimizer checkpoints with FQN keys.

    Accelerate saves FSDP2 with ``get_optimizer_state_dict``, but its loader
    passes a fresh optimizer's empty integer-keyed state to DCP. DCP loads only
    the requested keys, so Adam moments and update counters are omitted.
    """
    require_version("accelerate", ACCELERATE_SUPPORTED_VERSIONS)
    original = vars(module).get("load_fsdp_optimizer")
    prepare_options = vars(module).get("_prepare_sd_options")
    parameters = ("fsdp_plugin", "accelerator", "optimizer", "model", "input_dir",
                  "optimizer_index", "adapter_only")
    if (not callable(original) or tuple(inspect.signature(original).parameters) != parameters
            or not callable(prepare_options)
            or tuple(inspect.signature(prepare_options).parameters) != ("fsdp_plugin",)
            or vars(module).get("OPTIMIZER_NAME") != "optimizer"):
        raise UnsupportedAdapterVersion("Accelerate FSDP optimizer checkpoint contract changed")
    if getattr(original, "_jittor_fsdp_optimizer_restore", False):
        return False

    @wraps(original)
    def load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir,
                            optimizer_index=0, adapter_only=False):
        from torch.distributed.fsdp import StateDictType

        if (fsdp_plugin.fsdp_version != 2
                or fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT):
            return original(fsdp_plugin, accelerator, optimizer, model, input_dir,
                            optimizer_index=optimizer_index, adapter_only=adapter_only)

        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint.state_dict import (
            get_optimizer_state_dict, set_optimizer_state_dict,
        )

        accelerator.wait_for_everyone()
        options = prepare_options(fsdp_plugin)
        checkpoint_dir = (os.path.join(input_dir, "optimizer_%d" % optimizer_index)
                          if "optimizer" not in os.fspath(input_dir) else input_dir)
        state = {"optimizer": get_optimizer_state_dict(model, optimizer, options=options)}
        dist_cp.load(state, checkpoint_id=checkpoint_dir,
                     storage_reader=dist_cp.FileSystemReader(checkpoint_dir))
        set_optimizer_state_dict(model, optimizer, state["optimizer"], options=options)

    load_fsdp_optimizer._jittor_fsdp_optimizer_restore = True
    load_fsdp_optimizer._jittor_original_loader = original
    replace(module, "load_fsdp_optimizer", load_fsdp_optimizer, original)
    replace_bound_aliases("accelerate", "load_fsdp_optimizer", original, load_fsdp_optimizer)
    replace_bound_aliases("transformers", "load_fsdp_optimizer", original, load_fsdp_optimizer)
    return True


def register(register_module_patch):
    register_module_patch("transformers.utils.import_utils", patch_import_utils)
    register_module_patch("transformers.modeling_flash_attention_utils", patch_flash_attention_utils)
    register_module_patch("transformers.trainer", patch_trainer)
    register_module_patch("accelerate.utils.imports", patch_accelerate_imports)
    register_module_patch("accelerate.utils.fsdp_utils", patch_accelerate_fsdp_utils)
    register_module_patch("accelerate.utils.fsdp_utils", patch_accelerate_peft_fsdp_utils)
    register_module_patch("accelerate", patch_accelerate)
