"""NPU memory metrics backed by ACL and native allocator events.

There is deliberately no CUDA-driver, process-total, or Python peak fallback.
Memory queries must not change Torch's ordinary factory default device.
Allocated/reserved metrics cover the standard SFRL/Temp allocator pools only,
not external ACL allocations or process-wide memory. ``mem_get_info`` returns
driver-wide free/total bytes. Custom nested caching pools are not supported by
this accounting contract.
"""
import operator
import re
import types

import jittor as jt


def _device_index(device=None):
    if not getattr(jt.compiler, "has_acl", False):
        raise RuntimeError("NPU device queries require the ACL backend")
    if device is None:
        index = jt.core.current_device()
    elif isinstance(device, str):
        match = re.fullmatch(r"npu(?::([0-9]+))?", device)
        if match is None:
            raise ValueError("Expected an NPU device, got %r" % device)
        index = jt.core.current_device() if match.group(1) is None else int(match.group(1))
    elif hasattr(device, "type"):
        if device.type != "npu":
            raise ValueError("Expected an NPU device, got %r" % device)
        index = device.index
        if index is None:
            index = jt.core.current_device()
    else:
        index = device
    if isinstance(index, bool):
        raise TypeError("NPU device index must be an integer, not bool")
    index = operator.index(index)
    if not 0 <= index < jt.core.get_device_count():
        raise ValueError("Invalid NPU device index: %s" % index)
    return index


def _stats(device):
    return jt.core.device_pool_memory_stats(_device_index(device))


def memory_allocated(device=None):
    return int(_stats(device)[0])


def memory_reserved(device=None):
    return int(_stats(device)[1])


def max_memory_allocated(device=None):
    return int(_stats(device)[2])


def max_memory_reserved(device=None):
    return int(_stats(device)[3])


def reset_peak_memory_stats(device=None):
    jt.core.reset_device_memory_peaks(_device_index(device))


def reset_max_memory_allocated(device=None):
    # Like Torch, the deprecated spelling resets both allocator peaks.
    reset_peak_memory_stats(device)


def mem_get_info(device=None):
    free, total = jt.core.device_memory_info(_device_index(device))
    return int(free), int(total)


def empty_cache():
    jt.core.gc_device_cache(_device_index())


def install(npu, modules):
    memory = types.ModuleType("torch.npu.memory")
    for function in (memory_allocated, memory_reserved, max_memory_allocated,
                     max_memory_reserved, reset_peak_memory_stats,
                     reset_max_memory_allocated, mem_get_info, empty_cache):
        setattr(npu, function.__name__, function)
        setattr(memory, function.__name__, function)
    npu.memory = memory
    modules["torch.npu.memory"] = memory
