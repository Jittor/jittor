"""The paged KV cache layout this backend reads.

vLLM's FlashAttention backend declares the shape of the cache vLLM then
allocates for it, and it has spelled that shape both ways across versions:
``[num_blocks, 2, block_size, num_kv_heads, head_size]``, and the same axes
with the key/value one first. The reads and writes are this package's -- they
go to :func:`jittor.nn.paged_attention` and :func:`jittor.nn.reshape_and_cache`,
which take the block-major spelling -- so the answer is to declare that
spelling, not to teach two kernels to index a cache two ways.

Overriding a backend's own declaration is what substituting its implementation
entitles us to do: the layout belongs to whoever reads the cache.
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
    except Exception:
        return False
    setattr(backend, _DECLARED, True)
    if probe[:2] != (2, _PROBE[0]):
        # Block-major already, or a spelling this does not recognise. Either
        # way there is nothing here to correct.
        return False
    backend.get_kv_cache_shape = staticmethod(_block_major_shape)
    backend.get_kv_cache_stride_order = staticmethod(_contiguous_stride_order)
    return True


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.v1.attention.backends.flash_attn": declare_cache_layout,
}
