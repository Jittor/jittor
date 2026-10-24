"""Read-only observation of the actual Qwen3 NPU initialization policy.

No imports of model/adapter modules, replacement functions, or tensor work.
The native oracle deliberately keeps verl's torch_npu fused implementations;
the candidate retains the original HF formulas under a source-pinned opt-in.
"""
import hashlib
import inspect
import os
from pathlib import Path
import sys

ENVIRONMENT_KEY = "JITTOR_VERL_NATIVE_NPU_KERNELS"
POLICY = "retain_hf_qwen3_ops_for_jittor_acl_v1"
HF_MODULE = "transformers.models.qwen3.modeling_qwen3"
NPU_MODULE = "verl.models.transformers.npu_patch"
HELPER_MODULE = "verl.workers.engine.fsdp.utils"
ADAPTER_MODULE = "jittor_adapters.verl.native_npu"
HELPER_HASH = "d4d836445d9160834a3bbc91cbf874f3db816c0b86e4fc63352d0e1b0c91a5b7"
FUNCTIONS = {
    "jittor": {
        "Qwen3RMSNorm.forward": ("Qwen3RMSNorm.forward", "790fa612e4fa57acfcbea31da45f92ba91f9016f7260ef539f61c5530ba06a5f"),
        "Qwen3MLP.forward": ("Qwen3MLP.forward", "03df2d7685cf8446594448cbd33ef676a2b088779d99d9803abe3bed2ec7ad9b"),
        "apply_rotary_pos_emb": ("apply_rotary_pos_emb", "e2975ea7060de67704dfef2f64c821064e66fc292923e37f1d2689077e846b94"),
    },
    "torch": {
        "Qwen3RMSNorm.forward": ("rms_norm_forward_npu", "94802065baf26be1adb68f5c932c57c243f797d97217a3b1ba19218a2f3a6034"),
        "Qwen3MLP.forward": ("silu_forward_npu", "68fd8986f64a1824d03f5fbcb85c2739a10581347fb3ef058b8ddec3efae3208"),
        "apply_rotary_pos_emb": ("apply_rotary_pos_emb_npu", "eee63fe35d837005bd51c7861ae22c3f5afac2219b64e61d3742d32b43975bfc"),
    },
}
MODULE_HASHES = {
    HF_MODULE: "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
    NPU_MODULE: "3b78eee4ef8ae9631884e3ed1a5edd9015f799053f9338c50de262668e9f2964",
}
SCOPE = "Qwen3 RMSNorm/MLP/RoPE bindings after worker initialization; not a kernel performance claim"


def expected_policy(framework, adapter_source_sha256=None):
    if framework not in FUNCTIONS:
        raise ValueError("unknown NPU provenance framework")
    candidate = framework == "jittor"
    owner = HF_MODULE if candidate else NPU_MODULE
    return {
        "scope": SCOPE, "requested_native_ops": candidate,
        "policy": POLICY if candidate else "upstream_torch_npu_fusions",
        "helper_module_sha256": HELPER_HASH,
        "helper_implementation": (ADAPTER_MODULE + ".apply_native_npu_fsdp_policy" if candidate
                                  else HELPER_MODULE + ".apply_npu_fsdp_patches"),
        "adapter_source_sha256": adapter_source_sha256 if candidate else None,
        "functions": {name: {"module": owner, "qualname": qualname, "source_sha256": digest,
                             "module_sha256": MODULE_HASHES[owner]}
                      for name, (qualname, digest) in FUNCTIONS[framework].items()},
    }


def validate_npu_policy(value, framework):
    digest = value.get("adapter_source_sha256") if isinstance(value, dict) else None
    if framework == "jittor" and (not isinstance(digest, str) or len(digest) != 64
                                  or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("missing native NPU adapter source provenance")
    if value != expected_policy(framework, digest):
        raise ValueError("missing or unsupported NPU initialization provenance")


def _module_hash(module):
    path = inspect.getsourcefile(module) if module is not None else None
    if not path:
        raise ValueError("NPU policy module source unavailable")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe_npu_policy(framework):
    if framework not in FUNCTIONS:
        raise ValueError("unknown NPU provenance framework")
    requested = os.environ.get(ENVIRONMENT_KEY, "0")
    if requested not in ("0", "1"):
        raise ValueError("invalid native NPU opt-in")
    helper_module = sys.modules.get(HELPER_MODULE)
    helper = getattr(helper_module, "apply_npu_fsdp_patches", None)
    model = sys.modules.get(HF_MODULE)
    if not callable(helper) or model is None:
        raise ValueError("NPU initialization functions were not loaded")
    active = getattr(helper, "_jittor_verl_native_npu_policy", None)
    if (active == POLICY) != (framework == "jittor"):
        raise ValueError("NPU initialization activation mismatch")
    adapter = sys.modules.get(ADAPTER_MODULE)
    observed = {
        "scope": SCOPE, "requested_native_ops": requested == "1",
        "policy": active if active else "upstream_torch_npu_fusions",
        "helper_module_sha256": _module_hash(helper_module),
        "helper_implementation": helper.__module__ + "." + helper.__qualname__,
        "adapter_source_sha256": _module_hash(adapter) if active else None,
        "functions": {},
    }
    for name in FUNCTIONS[framework]:
        function = model
        for part in name.split("."):
            function = getattr(function, part)
        if not inspect.isfunction(function) or hasattr(function, "__wrapped__"):
            raise ValueError("wrapped or non-function Qwen3 NPU binding: " + name)
        observed["functions"][name] = {
            "module": function.__module__, "qualname": function.__qualname__,
            "source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
            "module_sha256": _module_hash(sys.modules.get(function.__module__)),
        }
    validate_npu_policy(observed, framework)
    return observed
