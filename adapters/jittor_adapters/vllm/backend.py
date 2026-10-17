import os

from jittor.compat.diagnostics import EXPECTED, swallowed
from jittor.compat.transaction import set_attr
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
    set_attr(backend, _DECLARED, True)
    if probe[:2] != (2, _PROBE[0]):
        # Block-major already, or a spelling this does not recognise. Either
        # way there is nothing here to correct.
        return False
    set_attr(backend, "get_kv_cache_shape", staticmethod(_block_major_shape))
    set_attr(backend, "get_kv_cache_stride_order", staticmethod(_contiguous_stride_order))
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
    set_attr(backend, _HEAD_SIZES, True)
    set_attr(backend, "validate_head_size", staticmethod(_validate_head_size))
    if hasattr(backend, "get_supported_head_sizes"):
        set_attr(backend, "get_supported_head_sizes", staticmethod(_supported_head_sizes))
    return True


def declare_backend(module):
    """Declare both, so a caller reading either gets this implementation's."""

    layout = declare_cache_layout(module)
    sizes = declare_head_sizes(module)
    blocks = declare_acl_block_size(module)
    return layout or sizes or blocks


def declare_acl_block_size(module):
    """Use CANN's validated 128-token page layout on ACL only.

    The ACL paged-decode kernel consumes block-128 KV descriptors. CUDA keeps
    upstream's full block-size policy; unsupported explicit NPU sizes fail at
    validation instead of reaching a mismatched cache kernel.
    """
    backend = getattr(module, "FlashAttentionBackend", None)
    marker = "_jittor_acl_block_size"
    if backend is None or getattr(backend, marker, False):
        return False
    if not (hasattr(backend, "get_supported_kernel_block_sizes")
            and hasattr(backend, "get_preferred_block_size")):
        return False

    supported = backend.get_supported_kernel_block_sizes
    preferred = backend.get_preferred_block_size

    def acl_active():
        import jittor as jt
        return bool(getattr(jt.compiler, "has_acl", 0)
                    and getattr(jt.flags, "use_acl", 0))

    def supported_block_sizes():
        return [128] if acl_active() else supported()

    @classmethod
    def preferred_block_size(cls, default_block_size):
        return 128 if acl_active() else preferred(default_block_size)

    set_attr(backend, "get_supported_kernel_block_sizes",
             staticmethod(supported_block_sizes))
    set_attr(backend, "get_preferred_block_size", preferred_block_size)
    set_attr(backend, marker, True)
    return True


_DIRECT_ALL_REDUCE = "_jittor_direct_all_reduce"
_DIRECT_RESULT = "_jittor_direct_all_reduce_result"


def patch_cuda_all_reduce(module):
    """Use Jittor's out-of-place NCCL result directly.

    vLLM's PyTorch fallback clones the input before calling its in-place
    collective.  Jittor's NCCL op is already out-of-place, so that device copy
    is unnecessary.  For TP=2, mark the NCCL result itself as an output hint
    to bound Jittor's lazy graph.  This keeps the lifetime boundary that a bare
    return lacks without constructing a discarded clone holder on every
    collective.  Larger groups retain the holder publication path: TP=4
    batched decode regresses when that holder is removed because its graph
    scheduling/overlap differs from TP=2.  The larger-group holder starts
    empty: its old contents are overwritten by ``update``, so cloning the
    input first only launched a discarded device copy.  Set
    ``JITTOR_VLLM_EMPTY_ALL_REDUCE_HOLDER=0`` to restore that legacy clone.
    """

    communicator = getattr(module, "CudaCommunicator", None)
    if communicator is None or getattr(communicator, _DIRECT_ALL_REDUCE, False):
        return False
    original = communicator.all_reduce

    def all_reduce(self, input_):
        pynccl = getattr(self, "pynccl_comm", None)
        group = getattr(self, "device_group", None)
        native = getattr(group, "_all_reduce", None)
        pynccl_disabled = pynccl is None or getattr(pynccl, "disabled", False)
        if pynccl_disabled and callable(native):
            reduced = native(input_, "sum")
            direct_result = self.__dict__.get(_DIRECT_RESULT)
            if direct_result is None:
                size = getattr(group, "size", None)
                direct_result = callable(size) and int(size()) <= 2
                self.__dict__[_DIRECT_RESULT] = direct_result
            if direct_result:
                return reduced.out_hint()
            if os.environ.get(
                    "JITTOR_VLLM_EMPTY_ALL_REDUCE_HOLDER", "1") != "0":
                import jittor as jt
                output = jt.empty(input_.shape, dtype=input_.dtype)
            else:
                output = input_.clone()
            output.update(reduced)
            return output
        return original(self, input_)

    set_attr(communicator, "all_reduce", all_reduce)
    set_attr(communicator, _DIRECT_ALL_REDUCE, True)
    return True


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.v1.attention.backends.flash_attn": declare_backend,
    "vllm.distributed.device_communicators.cuda_communicator":
        patch_cuda_all_reduce,
}
