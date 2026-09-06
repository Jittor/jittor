"""Shared contracts for private CUDA inference capabilities."""

from jittor._runtime.dispatch import dispatch_context

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


def on_acl():
    """Whether the native runtime currently selects the legacy ACL backend."""
    return dispatch_context().backend == "acl_legacy"


def device_index(value):
    """Resolve placement through the same context as registered kernels."""
    return dispatch_context(value).device_id
