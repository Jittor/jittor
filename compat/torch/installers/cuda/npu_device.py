"""NPU device selection and metadata backed by the native ACL runtime.

Selecting the current accelerator does not change Torch's factory default.
This module does not claim support for the remaining NPU facade operations.
"""
import operator

import jittor as jt

from .npu_memory import _device_index


def device_count():
    if not getattr(jt.compiler, "has_acl", False):
        return 0
    count = int(jt.core.get_device_count())
    if count < 0:
        raise RuntimeError("ACL runtime returned a negative device count")
    return count


def is_available():
    return device_count() > 0


def current_device():
    # The shared parser reads native current_device, validates the ACL backend
    # and visible ordinal, and refuses an uninitialized (-1) device.
    return _device_index()


def set_device(device):
    # torch_npu 2.7.1 npu/utils.py only calls _npu_setDevice when the
    # resolved integer is >= 0. Preserve negative-integer no-op semantics
    # without initializing or querying a backend. Booleans remain invalid.
    if device is not None and not isinstance(device, (str, bool)) and not hasattr(device, "type"):
        device = operator.index(device)
        if device < 0:
            return
    # This is the native device owner, not a CUDA facade or a factory scope.
    jt.core.set_device(_device_index(device))


def get_device_name(device=None):
    index = _device_index(device)
    from jittor.backends.acl import get_device_name as query
    return query(index)


def install(npu):
    # Explicitly replace generic device placeholders. Keep the independently
    # installed name and memory providers untouched.
    for function in (is_available, device_count, current_device, set_device):
        setattr(npu, function.__name__, function)
