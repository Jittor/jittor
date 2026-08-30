"""Shared contracts for private CUDA inference capabilities."""

import jittor as jt

_SOURCE_CACHE = {}


def cached_source(template, params):
    """``template % params``, memoised across calls.

    These kernels are specialised on shape-derived constants, so a decode step
    formats the same few sources over and over -- ~3us of string work against a
    ~1us kernel, repeated for every layer of every forward pass. The parameter
    sets are drawn from a handful of layer shapes, so the cache stays small.
    """
    key = (template, tuple(sorted(params.items())))
    source = _SOURCE_CACHE.get(key)
    if source is None:
        source = template % params
        _SOURCE_CACHE[key] = source
    return source


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
