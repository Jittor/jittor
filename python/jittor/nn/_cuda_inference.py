"""Shared contracts for private CUDA inference capabilities."""

import jittor as jt


def device_index(value):
    get_device = getattr(value, "get_device", None)
    if callable(get_device):
        try:
            device = int(get_device())
        except (TypeError, ValueError):
            device = -1
        if device >= 0:
            return device
    location = value.location()
    if location == "device":
        return 0
    if location == "cpu":
        return -1
    return 0 if jt.flags.use_cuda else -1
