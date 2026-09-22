"""Flash-attention as a CUDA kernel for `nn.scaled_dot_product_attention`.

`python/jittor/nn/functional/attention.py` has always asked
`try_dispatch("nn.scaled_dot_product_attention", ...)`. Nothing answered on
CUDA: `backends/acl/kernels/install.py:53` registers that name for Ascend and
only for Ascend, so every native call fell through to the math lowering, which
materialises an N x N score matrix. Measured on this tree, fp16, interleaved
medians: `0.0081s` against `0.0030s` on the MiniMax-H3 video VAE's attention
shape (`2x32x1797x1797x64`), i.e. **2.72x**; `1.14x` at 1024 tokens and `1.23x`
at 2048. The gap grows with sequence length, which is what O(N^2) memory
against O(N) predicts.

This registers the bridge that already builds the official flash-attention
kernels (`jittor.compat.shim.backends.flash_attention`) under that name, so
`jt.nn.scaled_dot_product_attention` gets it too rather than only code that
goes through the torch shim.

**It cannot activate unless flash-attention is already installed.**
`load_backend_for` needs a discoverable source checkout -- in practice
`JITTOR_FLASH_ATTN_JITTOR_SRC`. Without one it returns no backend, this kernel
returns None, `try_dispatch` reports None and the caller takes the same math
path it always did, bit for bit. So the numerics change only for someone who
went and installed flash, which is the same bargain torch's own SDPA makes.

Every unsupported case returns None rather than raising, because returning None
is how a kernel declines: `try_dispatch` passes it straight back and the caller
falls through. The declines here are masks, dropout, non-4-D inputs, unequal
q/k/v head counts, and head dimensions flash has no template for.

The import is deferred to first use. The bridge pulls in the torch shim's
C++ extension builder, and core jittor must not pay for that at import time --
or need it at all when nobody asks for attention.
"""
import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import register_kernel

#: Head dimensions the official kernels are instantiated for.
_TEMPLATE_DIMS = (32, 64, 96, 128, 160, 192, 224, 256)

#: dtype names flash accepts, mapped to the bridge's spelling.
_DTYPES = {"float16": "fp16", "bfloat16": "bf16"}


def _bridge():
    """The flash-attention loader, or None when it cannot be imported."""
    from jittor.compat.shim.backends import flash_attention
    return flash_attention


def _template_dim(head_dim):
    for candidate in _TEMPLATE_DIMS:
        if head_dim <= candidate:
            return candidate
    return None


def _flash_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                        dropout_p=0.0, is_causal=False,
                                        scale=None):
    if attn_mask is not None or float(dropout_p or 0.0) != 0.0:
        return None
    q_shape, k_shape, v_shape = (tuple(query.shape), tuple(key.shape),
                                 tuple(value.shape))
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return None
    # Grouped-query attention would need the head-count broadcast the bridge's
    # caller does; declining is cheaper than getting it subtly wrong.
    if not (q_shape[1] == k_shape[1] == v_shape[1]):
        return None
    if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
        return None
    if k_shape[2] != v_shape[2] or q_shape[3] != k_shape[3] \
            or q_shape[3] != v_shape[3]:
        return None
    # flash aligns a causal mask to the bottom right when the two lengths
    # differ; `attention.py`'s lowering builds `triu(..., diagonal=1)`, which
    # is top-left. They agree only for a square score matrix, so anything else
    # would quietly return a different answer than the path it replaces.
    if is_causal and q_shape[2] != k_shape[2]:
        return None

    dtype = _DTYPES.get(_jittor_dtype_name(query.dtype))
    if dtype is None or _jittor_dtype_name(key.dtype) != _jittor_dtype_name(query.dtype) \
            or _jittor_dtype_name(value.dtype) != _jittor_dtype_name(query.dtype):
        return None
    head_dim = int(q_shape[3])
    template = _template_dim(head_dim)
    if template is None:
        return None

    bridge = _bridge()
    if not bridge.enabled():
        return None
    backend, capability_miss = bridge.load_backend_for(template, dtype)
    if backend is None or capability_miss is not None:
        # `required()` is the caller's request to hear about a flash that was
        # asked for and could not be built, rather than silently getting the
        # slow path and wondering later.
        if bridge.required():
            raise RuntimeError(
                "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but the native "
                "flash-attn backend is unavailable for head_dim=%d %s: %s"
                % (template, dtype,
                   capability_miss or bridge.last_error() or "unknown error"))
        return None
    call = getattr(backend, "flash_attn_func", None)
    if not callable(call):
        return None

    batch, heads, q_len, _ = (int(size) for size in q_shape)
    kv_len = int(k_shape[2])
    softmax_scale = (1.0 / (head_dim ** 0.5)) if scale is None else float(scale)
    # flash wants (batch, seq, heads, dim). `clone` materialises a row-major
    # tensor: the extension reads the buffer directly, and handing it a lazy
    # permute expression leaves it holding transient metadata.
    axes = (0, 2, 1, 3)
    q_dense = query.permute(*axes).reshape((batch, q_len, heads, head_dim)).clone()
    k_dense = key.permute(*axes).reshape((batch, kv_len, heads, head_dim)).clone()
    v_dense = value.permute(*axes).reshape((batch, kv_len, heads, head_dim)).clone()

    out = call(q_dense, k_dense, v_dense, 0.0, softmax_scale, bool(is_causal))
    if out is None:
        return None
    # `clone`, for the same reason the inputs are cloned: `out` is owned by the
    # extension, and handing the graph a lazy reshape/permute of that buffer
    # leaves a view onto memory the bridge may reclaim. Without this, a math
    # attention later in the same process died inside `matmul` with a
    # NanoVector whose packed shape was garbage.
    return out.reshape((batch, q_len, heads, head_dim)).permute(*axes).clone()


register_kernel("nn.scaled_dot_product_attention", "cuda",
                _flash_scaled_dot_product_attention,
                dtypes=("float16", "bfloat16"))
