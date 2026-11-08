"""Install Torch NPU primitives over the shared native device facade.

The shared CUDA/device installer already publishes ACL-backed discovery,
selection, memory, capability and RNG APIs. Keep those owners and add the
torch_npu primitive namespace without replacing working queries with stubs.
"""
import types

import jittor as jt

from ..fidelity import Fidelity, register_api_bindings


def _acl_build():
    """Whether this Jittor was built against ACL, not merely able to load it."""
    return getattr(jt.compiler, "has_acl", 0) and getattr(
        getattr(jt.compiler, "build_config", None), "backend", None) == "acl"


def _require_acl():
    # The shared device facade owns discovery, selection, memory and RNG.
    # Do not replace its implemented APIs with this installer's former stubs.
    from .cuda.api import _npu_is_available
    if not _npu_is_available():
        raise RuntimeError("The Jittor ACL/NPU backend has no available devices")


def npu_rms_norm(input, weight, epsilon=1e-6):
    """``torch_npu.npu_rms_norm``, as an ACL tensor graph.

    Returns the pair torch_npu returns, with None for the reciprocal-norm
    output the fused kernel also produces and callers rarely read.
    """
    _require_acl()
    mean_square = jt.mean(input * input, dim=-1, keepdims=True)
    return input * jt.rsqrt(mean_square + epsilon) * weight, None


def npu_rotary_mul(input, cos, sin):
    _require_acl()
    half = input.shape[-1] // 2
    rotated = jt.concat((-input[..., half:], input[..., :half]), dim=-1)
    return input * cos + rotated * sin


def npu_swiglu(input, dim=-1):
    _require_acl()
    gate, value = jt.chunk(input, 2, dim=dim)
    return (gate * jt.sigmoid(gate)) * value


def _unsupported(*args, **kw):
    raise NotImplementedError(
        "This NPU API is outside the Jittor ACL device facade")


#: These stream/property APIs still lack a native implementation. Memory,
#: capability and RNG queries retain the real shared-device implementations.
_REFUSED = (
    "get_device_properties", "current_stream", "default_stream", "set_stream",
    "Stream", "Event", "set_compile_mode", "ipc_collect",
)

# Device and RNG APIs already have one implementation in the shared facade.
# This installer adds only the torch_npu primitive adapters and strict refusals.
_SUPPORTED = ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu")


def install(ctx):
    target, registry = ctx.target_namespace, ctx.registry
    # Keep the NPU namespace created by the CUDA installer: it owns the
    # device-scoped RNG and random submodule APIs used by Transformers. This
    # installer adds primitive adapters without overriding those device APIs.
    npu = registry.get("torch.npu")
    if npu is None:
        npu = types.ModuleType("torch.npu")
    npu.__path__ = []
    for name in _SUPPORTED:
        setattr(npu, name, globals()[name])
    for name in _REFUSED:
        setattr(npu, name, _unsupported)
    registry.publish("torch.npu", npu, replace=True)
    target.npu = npu

    register_api_bindings(
        npu, "torch.npu",
        ("is_available", "device_count", "current_device", "set_device",
         "device", "is_bf16_supported"),
        Fidelity.EXACT,
        "Native ACL discovery and selection; the index accepts an int, a "
        "string, a torch.device or None, as torch's does.")
    register_api_bindings(
        npu, "torch.npu", ("synchronize", "empty_cache"), Fidelity.APPROXIMATE,
        "Submits Jittor's pending work and waits for every device it "
        "touched, which is stronger than torch's per-device wait.")
    register_api_bindings(
        npu, "torch.npu", ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu"),
        Fidelity.APPROXIMATE,
        "Executed as ACL tensor graphs rather than the fused CANN kernels; "
        "npu_rms_norm does not produce the auxiliary reciprocal-norm output.")
    register_api_bindings(
        npu, "torch.npu", _REFUSED, Fidelity.UNIMPLEMENTED,
        "Refused rather than answered: no synthetic device metadata, memory "
        "peaks or streams.")

    if not _acl_build():
        return
    # `import torch_npu` is what a Torch model does before calling any of the
    # three primitives above. Publishing a facade is only safe where no real
    # one is loaded -- otherwise the native extension and this would both
    # claim the name and the model would get whichever imported last.
    if registry.get("torch_npu") is not None:
        return
    facade = types.ModuleType("torch_npu")
    facade.__file__ = __file__
    facade.__jittor_acl_facade__ = True
    facade.__jittor_provider__ = "jittor.backends.acl"
    facade.npu = npu
    facade.npu_rms_norm = npu_rms_norm
    facade.npu_rotary_mul = npu_rotary_mul
    facade.npu_swiglu = npu_swiglu
    # Transformers imports this even for eager attention. A refusal is the
    # honest export; a silent unfused stand-in would be a different model.
    facade.npu_fusion_attention = _unsupported
    register_api_bindings(
        facade, "torch_npu",
        ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu"), Fidelity.APPROXIMATE,
        "The same ACL tensor graphs torch.npu publishes.")
    register_api_bindings(
        facade, "torch_npu", ("npu_fusion_attention",), Fidelity.UNIMPLEMENTED,
        "Fused NPU attention is not implemented by this device facade.")
    registry.publish("torch_npu", facade, bind_parent=False)
