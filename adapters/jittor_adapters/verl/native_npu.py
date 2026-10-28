"""Opt-in selection of original HF Qwen3/Qwen3.5 operators for ACL.

verl's private initializer globally replaces HF model mathematics with
torch_npu fused implementations. That vendor-specific choice is not a Torch
API. For the pinned dense Qwen3 scope, retain the original HF operators instead;
their tensor operations still execute through the actual Jittor frontend.
This module neither implements kernels nor imports/imitates torch_npu.

Host lifecycle contracts do not establish device/numerical/performance support.
Revalidate when the pinned sources or public namespace_owner/patch lifecycle
interfaces change. The worker must independently observe backend residency,
fallback counts and all numerical snapshots before claiming real-NPU support.
"""
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)


SUPPORTED_VERSIONS = frozenset(("0.9.1",))
TRANSFORMERS_VERSIONS = frozenset(("5.9.0",))
OPT_IN = "JITTOR_VERL_NATIVE_NPU_KERNELS"
POLICY = "retain_hf_qwen3_family_ops_for_jittor_acl_v2"
MODULE = "verl.workers.engine.fsdp.utils"
FUNCTION = "apply_npu_fsdp_patches"
SOURCE_SHA256 = "d4d836445d9160834a3bbc91cbf874f3db816c0b86e4fc63352d0e1b0c91a5b7"
FUNCTION_SHA256 = "d4b53f1c8e48086217e21d6bc55cab34f50714c1b4a6c67b9e85c7c638f7714d"
HF_MODULE = "transformers.models.qwen3.modeling_qwen3"
HF_SOURCE_SHA256 = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
HF_FUNCTIONS = {
    "Qwen3RMSNorm.forward": "790fa612e4fa57acfcbea31da45f92ba91f9016f7260ef539f61c5530ba06a5f",
    "Qwen3MLP.forward": "03df2d7685cf8446594448cbd33ef676a2b088779d99d9803abe3bed2ec7ad9b",
    "apply_rotary_pos_emb": "e2975ea7060de67704dfef2f64c821064e66fc292923e37f1d2689077e846b94",
}
QWEN35_HF_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
QWEN35_HF_SOURCE_SHA256 = "d67880b98f47d55a9d40679c901a740aa0618b97458824cab2abe9a1f7861be6"
QWEN35_HF_FUNCTIONS = {
    "Qwen3_5RMSNormGated.forward": "9e4a919ddd7afcb4301dd2bdaab79f193dd5c0732482e8ac1dae510c3bf2ded9",
    "Qwen3_5RMSNorm.forward": "350041feaa40063352594579e212741fdff004594990c36162458d778dae1a9e",
    "apply_rotary_pos_emb": "ab8feb64f19555713efc4c559ce306b35205b1197f8431784585999d9a5b6662",
}


def report_torch_api_version():
    """Ask the frontend owner to expose its Torch API level for verl gates."""
    if not enabled():
        return False
    frontend = sys.modules.get("torch")
    reporter = getattr(frontend, "compat_report_torch_api_version", None)
    if not callable(reporter):
        raise UnsupportedAdapterVersion("Jittor Torch frontend lacks compat_report_torch_api_version()")
    reported = reporter(True)
    api = getattr(frontend, "__torch_version__", None)
    if reported != api or not isinstance(api, str):
        raise UnsupportedAdapterVersion("Torch API version owner reported an inconsistent value")
    return True


def enabled():
    value = os.environ.get(OPT_IN)
    if value not in (None, "", "0", "1"):
        raise UnsupportedAdapterVersion(OPT_IN + " must be absent, 0 or 1")
    return value == "1"


def _require_runtime():
    if not enabled():
        raise UnsupportedAdapterVersion("native NPU policy requires explicit " + OPT_IN + "=1")
    require_version("verl", SUPPORTED_VERSIONS)
    require_version("transformers", TRANSFORMERS_VERSIONS)
    from jittor.compat.torch import namespace_owner

    frontend = sys.modules.get("torch")
    if frontend is None or namespace_owner(frontend) is None:
        raise UnsupportedAdapterVersion("native NPU policy requires the actual Jittor Torch namespace")
    npu = getattr(frontend, "npu", None)
    available = getattr(npu, "is_available", None)
    if not callable(available) or not available():
        raise UnsupportedAdapterVersion("native NPU policy requires an available NPU backend")
    if "torch_npu" in sys.modules or "verl.models.transformers.npu_patch" in sys.modules:
        raise UnsupportedAdapterVersion("native NPU policy refuses an already loaded vendor patch/runtime")


def _check_module_source(module, digest):
    if not isinstance(module, types.ModuleType):
        raise UnsupportedAdapterVersion("required upstream module is not loaded")
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("upstream module source is required") from error
    if hashlib.sha256(source).hexdigest() != digest:
        raise UnsupportedAdapterVersion("upstream module source SHA256 changed: " + module.__name__)


def _check_function(function, module, digest):
    if (not isinstance(function, types.FunctionType) or function.__closure__
            or function.__globals__ is not vars(module) or function.__module__ != module.__name__):
        raise UnsupportedAdapterVersion("upstream function ownership changed")
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("upstream function source is required") from error
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != digest:
        raise UnsupportedAdapterVersion("upstream function source SHA256 changed: " + function.__qualname__)


def _check_hf_operators(module_name=None, source_digest=None, functions=None):
    module_name = HF_MODULE if module_name is None else module_name
    source_digest = HF_SOURCE_SHA256 if source_digest is None else source_digest
    functions = HF_FUNCTIONS if functions is None else functions
    module = sys.modules.get(module_name)
    if module is None:
        # Ray trainer workers import verl's FSDP helpers before constructing the
        # HF model.  This opt-in policy owns the dependency and must not depend
        # on the driver's incidental import order.  Importing the pinned model
        # definition here loads code only; model weights are still constructed
        # later by verl's normal engine path.
        module = importlib.import_module(module_name)
    _check_module_source(module, source_digest)
    for path, digest in functions.items():
        function = module
        for name in path.split("."):
            function = getattr(function, name, None)
        _check_function(function, module, digest)


def _check_config(model_config):
    # These are third-party configuration types, never replacements for a
    # framework Tensor/type or a second model implementation.
    from verl.workers.config import HFModelConfig
    from transformers import Qwen3Config

    if type(model_config) is not HFModelConfig:
        raise UnsupportedAdapterVersion("native NPU policy requires exact HFModelConfig")
    hf = model_config.hf_config
    if type(hf) is Qwen3Config:
        family_config = hf
        expected_model_type = "qwen3"
        operator_spec = (HF_MODULE, HF_SOURCE_SHA256, HF_FUNCTIONS)
    else:
        if (type(hf).__module__ != "transformers.models.qwen3_5.configuration_qwen3_5"
                or type(hf).__name__ != "Qwen3_5Config"):
            raise UnsupportedAdapterVersion(
                "native NPU policy requires exact Qwen3Config or Qwen3_5Config/Qwen3_5TextConfig"
            )
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

        if type(hf) is not Qwen3_5Config or type(getattr(hf, "text_config", None)) is not Qwen3_5TextConfig:
            raise UnsupportedAdapterVersion(
                "native NPU policy requires exact Qwen3Config or Qwen3_5Config/Qwen3_5TextConfig"
            )
        family_config = hf.text_config
        expected_model_type = "qwen3_5"
        operator_spec = (QWEN35_HF_MODULE, QWEN35_HF_SOURCE_SHA256, QWEN35_HF_FUNCTIONS)
    if (model_config.model_type not in ("language_model", "value_model")
            or hf.model_type != expected_model_type
            or getattr(hf, "_attn_implementation", None) != "eager"
            or getattr(family_config, "_attn_implementation", None) != "eager"):
        raise UnsupportedAdapterVersion(
            "native NPU policy requires dense Qwen3/Qwen3.5 actor/critic eager initialization config"
        )
    for name in ("use_liger", "use_fused_kernels", "use_remove_padding", "trust_remote_code",
                 "enable_activation_offload", "enable_gradient_checkpointing"):
        if getattr(model_config, name, None) is not False:
            raise UnsupportedAdapterVersion("native NPU policy requires " + name + "=False")
    # verl 0.9.1 composes the Megatron LoRA option block into every model
    # config, including FSDP models with rank=0.  The presence of that block
    # therefore does not mean LoRA is active; only its activation-bearing
    # fields do.  Keep rejecting both FSDP and Megatron LoRA requests.
    lora = getattr(model_config, "lora", None)
    if lora is not None and not isinstance(lora, dict):
        raise UnsupportedAdapterVersion("native NPU policy requires a mapping LoRA config")
    megatron_lora_enabled = bool(lora) and (
        lora.get("rank", 0) not in (None, 0)
        or bool(lora.get("adapter_path"))
        or lora.get("merge", False) is not False
    )
    if (getattr(model_config, "lora_rank", None) != 0 or getattr(model_config, "lora_adapter_path", None)
            or megatron_lora_enabled or getattr(model_config, "external_lib", None)
            or getattr(hf, "quantization_config", None) is not None
            or getattr(family_config, "quantization_config", None) is not None
            or getattr(family_config, "num_experts", 0) not in (None, 0)
            or getattr(family_config, "num_local_experts", 0) not in (None, 0)):
        raise UnsupportedAdapterVersion("native NPU policy excludes LoRA, external models, quantization and MoE")
    tiled = getattr(model_config, "tiled_mlp", None)
    mtp = getattr(model_config, "mtp", None)
    if (not isinstance(tiled, dict) or tiled.get("enabled") is not False
            or any(getattr(mtp, name, None) is not False
                   for name in ("enable", "enable_train", "enable_rollout"))):
        raise UnsupportedAdapterVersion("native NPU policy excludes tiled MLP and MTP")
    return operator_spec


def _make_policy(original):
    def apply_native_npu_fsdp_policy(model_config=None):
        """Retain the pinned HF operators, after validating the explicit policy."""
        _require_runtime()
        operator_spec = _check_config(model_config)
        _check_hf_operators(*operator_spec)
        # Deliberately do not run the upstream global torch_npu monkeypatch.
        # There is no substituted forward, kernel, tensor or model state here.
        return None

    apply_native_npu_fsdp_policy.__qualname__ = "apply_native_npu_fsdp_policy"
    apply_native_npu_fsdp_policy._jittor_verl_native_npu_policy = POLICY
    apply_native_npu_fsdp_policy._jittor_verl_original = original
    return apply_native_npu_fsdp_policy


@required_patch
def patch_native_npu_initializer(module):
    _require_runtime()
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected verl NPU initializer module")
    _check_module_source(module, SOURCE_SHA256)
    current = vars(module).get(FUNCTION)
    if (getattr(current, "_jittor_verl_native_npu_policy", None) == POLICY
            and getattr(current, "__module__", None) == __name__):
        _check_function(current._jittor_verl_original, module, FUNCTION_SHA256)
        _check_hf_operators()
        return False
    _check_function(current, module, FUNCTION_SHA256)
    _check_hf_operators()
    replacement = _make_policy(current)
    replace(module, FUNCTION, replacement, current)
    replace_bound_aliases("verl", FUNCTION, current, replacement)
    return True


def register(register_module_patch):
    if enabled():
        register_module_patch(MODULE, patch_native_npu_initializer)
