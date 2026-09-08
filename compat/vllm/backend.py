from ..diagnostics import EXPECTED, swallowed
"""What this backend's substituted attention implementation actually does.

vLLM's FlashAttention backend declares the shape of the cache vLLM then
allocates for it, and it has spelled that shape both ways across versions:
``[num_blocks, 2, block_size, num_kv_heads, head_size]``, and the same axes
with the key/value one first. The reads and writes are this package's -- they
go to :func:`jittor.nn.paged_attention` and :func:`jittor.nn.reshape_and_cache`,
which take the block-major spelling -- so the answer is to declare that
spelling, not to teach two kernels to index a cache two ways.

The same reasoning covers the head sizes it accepts. vLLM's list is its
compiled kernel's; that kernel is not what runs here, and jittor's paged
attention takes any head size the fused decode path handles -- or falls back to
a portable one that has no limit at all. A model outside vLLM's list would
otherwise be refused by a check that no longer describes anything.

Overriding a backend's own declarations is what substituting its implementation
entitles us to do: they belong to whoever does the work.
"""

_DECLARED = "_jittor_cache_layout"

# Deliberately distinct sizes, so which axis carried which is unambiguous in
# the answer. The block size has to stay a multiple of 16; vLLM rejects the rest.
_PROBE = (3, 16, 5, 7)


def _block_major_shape(num_blocks, block_size, num_kv_heads, head_size, **_):
    return (num_blocks, 2, block_size, num_kv_heads, head_size)


def _contiguous_stride_order(*_args, **_kwargs):
    return (0, 1, 2, 3, 4)


def declare_cache_layout(module):
    """Point the FlashAttention backend's cache shape at the layout we read."""

    backend = getattr(module, "FlashAttentionBackend", None)
    if backend is None or getattr(backend, _DECLARED, False):
        return False
    try:
        probe = tuple(backend.get_kv_cache_shape(*_PROBE))
    except EXPECTED as exc:
        swallowed("vllm/backend.py declare_cache_layout: probe = tuple(backend.get_kv_cache_shape(*_PROBE))", exc)
        return False
    setattr(backend, _DECLARED, True)
    if probe[:2] != (2, _PROBE[0]):
        # Block-major already, or a spelling this does not recognise. Either
        # way there is nothing here to correct.
        return False
    backend.get_kv_cache_shape = staticmethod(_block_major_shape)
    backend.get_kv_cache_stride_order = staticmethod(_contiguous_stride_order)
    return True


_HEAD_SIZES = "_jittor_head_sizes"

#: The fused decode kernel's ceiling. Above it the portable path still runs, but
#: declaring more than the fast path covers would be promising the wrong thing.
_MAX_HEAD_SIZE = 256


def _supported_head_sizes():
    return list(range(1, _MAX_HEAD_SIZE + 1))


def _validate_head_size(head_size):
    if not 1 <= int(head_size) <= _MAX_HEAD_SIZE:
        raise ValueError(
            "Head size %s is not supported by this backend. Supported head "
            "sizes are 1 to %s." % (head_size, _MAX_HEAD_SIZE))


def declare_head_sizes(module):
    """Widen the head sizes the backend accepts to the ones we implement."""

    backend = getattr(module, "FlashAttentionBackend", None)
    if backend is None or getattr(backend, _HEAD_SIZES, False):
        return False
    if not hasattr(backend, "validate_head_size"):
        # Newer vLLM validates elsewhere; nothing here to widen.
        return False
    setattr(backend, _HEAD_SIZES, True)
    backend.validate_head_size = staticmethod(_validate_head_size)
    if hasattr(backend, "get_supported_head_sizes"):
        backend.get_supported_head_sizes = staticmethod(_supported_head_sizes)
    return True


def declare_backend(module):
    """Declare both, so a caller reading either gets this implementation's."""

    layout = declare_cache_layout(module)
    return declare_head_sizes(module) or layout


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.v1.attention.backends.flash_attn": declare_backend,
}
