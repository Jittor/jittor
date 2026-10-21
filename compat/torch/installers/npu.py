"""A Torch NPU device facade over the native ACL runtime.

``torch.npu`` is one of the device namespaces the CUDA installer creates, and
until now every one of them got the same placeholders: ``is_available()``
answered False and ``device_count()`` answered 0 on a machine with eight
910Bs. A downstream library that asks before choosing a device -- which is
what Transformers, ms-swift and DeepSpeed all do -- was told there was no NPU
and went to the host.

What is published here is discovery, selection and synchronization, plus the
three ``torch_npu`` primitives a Torch model reaches for directly. It is not
the torch_npu extension: streams, RNG state and the memory statistics have
their own contracts and are left refusing rather than answered with a zero.
"""
import contextlib
import types

import jittor as jt

from ..fidelity import Fidelity, register_api_bindings
from ..types import device as TorchDevice


def _acl_build():
    """Whether this Jittor was built against ACL, not merely able to load it."""
    return getattr(jt.compiler, "has_acl", 0) and getattr(
        getattr(jt.compiler, "build_config", None), "backend", None) == "acl"


def device_count():
    if not _acl_build():
        return 0
    # Ask the runtime, not the environment: a CANN or driver failure has to
    # surface, never be rounded down to "no devices".
    return int(jt.core.backend_device_count("acl"))


def is_available():
    return device_count() > 0


def _require_acl():
    if not is_available():
        raise RuntimeError("The Jittor ACL/NPU backend has no available devices")


def current_device():
    _require_acl()
    return int(jt.core.current_device())


def _index(value=None):
    _require_acl()
    if value is None:
        return current_device()
    if isinstance(value, bool):
        raise TypeError("NPU device index must not be bool")
    if isinstance(value, int):
        index = value
    else:
        parsed = TorchDevice(value)
        if parsed.type != "npu":
            raise ValueError("Expected an npu device, got {}".format(value))
        index = current_device() if parsed.index is None else parsed.index
    if index < 0 or index >= device_count():
        raise ValueError("NPU device index out of range: {}".format(index))
    return int(index)


def set_device(value):
    # The native setter owns the ACL context and the device-switch hooks.
    jt.core.set_device(_index(value))


@contextlib.contextmanager
def device(value=None):
    previous = current_device()
    selected = _index(value)
    if selected != previous:
        set_device(selected)
    try:
        yield
    finally:
        if selected != previous:
            set_device(previous)


def synchronize(value=None):
    _index(value)
    # Jittor's work is lazy, so a device wait alone would wait for nothing.
    # This submits what is pending and waits for every device it touched,
    # which is stronger than the per-device wait torch documents.
    jt.sync_all(True)


def empty_cache():
    _require_acl()
    jt.sync_all(True)
    jt.gc()


def is_bf16_supported():
    _require_acl()
    return True


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


#: Published as refusals rather than left to the shared placeholders, which
#: answer a question they cannot answer: a zero peak or an absent stream reads
#: as a measurement, and a caller cannot tell it from one.
_REFUSED = (
    "is_initialized", "get_device_name", "get_device_properties",
    "reset_max_memory_allocated", "reset_peak_memory_stats", "memory_stats",
    "memory_allocated", "max_memory_allocated", "memory_reserved",
    "max_memory_reserved", "mem_get_info",
    "current_stream", "default_stream", "set_stream", "Stream", "Event",
    "set_compile_mode", "ipc_collect",
)

_SUPPORTED = ("is_available", "device_count", "current_device", "set_device",
              "device", "synchronize", "empty_cache", "is_bf16_supported",
              "npu_rms_norm", "npu_rotary_mul", "npu_swiglu")


def install(ctx):
    target, registry = ctx.target_namespace, ctx.registry
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
