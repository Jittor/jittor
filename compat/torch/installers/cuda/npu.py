"""NPU device protocol backed by the selected native ACL runtime.

Only discovery, selection and synchronization are implemented here. Streams,
RNG state and memory APIs require their own contracts; they are not CUDA aliases.
"""

import jittor as jt

from ...types import device as TorchDevice
from ...fidelity import Fidelity, register_api_bindings


def device_count():
    """Count runtime-visible ACL devices, never CUDA devices or env entries."""
    if not bool(getattr(jt.compiler, "has_acl", False)):
        return 0
    return int(jt.get_device_count())


def is_available():
    return device_count() > 0


def _require_npu():
    count = device_count()
    if not count:
        raise RuntimeError("Jittor NPU device APIs require an available ACL backend")
    return count


def current_device():
    _require_npu()
    return int(jt.current_device())


def _device_index(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = TorchDevice(value)
    if isinstance(value, TorchDevice):
        if value.type != "npu":
            raise ValueError("Expected a npu device, but got: %s" % value)
        return current_device() if value.index is None else value.index
    if value is None:
        return current_device()
    raise TypeError("NPU device must be an int, string, torch.device or None")


def _checked_index(index):
    count = _require_npu()
    if index >= count:
        raise RuntimeError("Invalid NPU device ordinal %d; %d device(s) visible" % (index, count))
    return index


def set_device(device):
    index = _device_index(device)
    if index < 0:
        return None
    jt.set_device(_checked_index(index))


def synchronize(device=None):
    index = _device_index(device)
    _checked_index(current_device() if index < 0 else index)
    # Materialize pending Jittor graphs and wait for all touched devices. This
    # is stronger than a per-device wait and is explicitly APPROXIMATE below.
    return jt.sync_all(True)


def install(ctx):
    namespace = ctx.registry.module_map["torch.npu"]
    names = ("is_available", "device_count", "current_device", "set_device", "synchronize")
    implementations = (is_available, device_count, current_device, set_device, synchronize)
    for name, implementation in zip(names, implementations):
        setattr(namespace, name, implementation)
    register_api_bindings(
        namespace,
        "torch.npu",
        names,
        Fidelity.APPROXIMATE,
        "Native ACL discovery/selection for integer, string, device and None arguments; "
        "synchronize waits for all touched devices. No streams, RNG or memory support claimed.",
    )
