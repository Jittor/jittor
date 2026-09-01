"""Public fused primitives for inference serving.

Serving stacks express a transformer layer through a handful of fused shapes
rather than the elementwise ops they decompose into: a gated activation, a
residual-and-normalise step, and rotary position embedding applied in place to
packed query/key. Each already exists here as an inference CUDA kernel; this
module is where they are named, so a serving stack can reach them without
reaching into private modules.

Every entry point takes an accelerator fused path when its preconditions hold --
inference mode, a supported backend and dtype -- and otherwise computes the same
result from ordinary ops, so the same call works on CPU and under autograd.
"""

import jittor as jt

from .rms_norm_cuda import _fused_add_rms_norm_cuda, _rms_norm_cuda
from .rope_cuda import _rotary_embedding_cuda
from .swiglu_cuda import _silu_and_mul_cuda

__all__ = ["silu_and_mul", "rms_norm", "fused_add_rms_norm", "rotary_embedding"]


def silu_and_mul(x):
    """Return ``silu(x[..., :d]) * x[..., d:]`` where ``d`` is half the width.

    The gate and the value arrive interleaved in one tensor because the two
    projections behind them are fused into a single matmul.
    """
    acl_backend = getattr(jt.nn, "_silu_and_mul_acl", None)
    if acl_backend is not None:
        fused = acl_backend(x)
        if fused is not None:
            return fused
    fused = _silu_and_mul_cuda(x)
    if fused is not None:
        return fused
    d = int(x.shape[-1]) // 2
    return jt.nn.silu(x[..., :d]) * x[..., d:]


def rms_norm(x, weight, eps=1e-6):
    """Root-mean-square normalise ``x`` over its last axis and scale."""
    backend = getattr(jt.nn, "_rms_norm_cuda", _rms_norm_cuda)
    backend_weight = weight
    if (
        getattr(jt.compiler, "has_acl", 0)
        and getattr(jt.flags, "use_acl", 0)
        and jt.flags.use_cuda
        and getattr(jt.flags, "no_grad", 0)
        and str(x.dtype) == "float32"
        and str(weight.dtype) in ("float16", "bfloat16")
    ):
        backend_weight = weight.__dict__.get("_serving_float32_weight")
        if backend_weight is None:
            backend_weight = weight.float32()
            weight.__dict__["_serving_float32_weight"] = backend_weight
    fused = backend(x, backend_weight, eps)
    if fused is not None:
        return fused
    dtype = x.dtype
    value = x.float32()
    scale = jt.rsqrt((value * value).mean(-1, keepdims=True) + eps)
    return (value * scale).cast(dtype) * weight


def fused_add_rms_norm(x, residual, weight, eps=1e-6):
    """Add the residual, then normalise -- returning ``(normalised, residual)``.

    The updated residual is the sum, which the next layer adds onto in turn;
    returning both is what lets the two steps share one pass over the data.
    """
    fused = _fused_add_rms_norm_cuda(x, residual, weight, eps)
    if fused is not None:
        return fused
    total = x + residual
    return rms_norm(total, weight, eps), total


def rotary_embedding(positions, query, key, cos_sin_cache, head_size=None,
                     is_neox=True, rotary_dim=None):
    """Apply rotary position embedding to packed ``query`` and ``key``.

    ``cos_sin_cache`` is indexed by position and holds the cosines in its first
    half and the sines in its second -- the form a serving stack builds once at
    start-up. ``query`` and ``key`` are packed as ``[..., heads * head_size]``;
    only the leading ``rotary_dim`` of each head is rotated, the rest passes
    through, which is what lets a partially-rotary model use the same call.

    Returns the rotated ``(query, key)``. ``key`` may be None.
    """
    if head_size is None:
        head_size = int(cos_sin_cache.shape[-1])
    head_size = int(head_size)
    if rotary_dim is None:
        rotary_dim = int(cos_sin_cache.shape[-1])
    rotary_dim = int(rotary_dim)
    acl_rotated = _rotary_embedding_acl(
        positions, query, key, cos_sin_cache,
        head_size, is_neox, rotary_dim)
    if acl_rotated is not None:
        return acl_rotated
    if key is not None:
        fused = _rotary_embedding_cuda(
            positions, query, key, cos_sin_cache,
            head_size=head_size, rotary_dim=rotary_dim, is_neox_style=is_neox)
        if fused is not None:
            return fused
    half = rotary_dim // 2
    flat = positions.reshape((-1,))
    cache = cos_sin_cache[flat]
    cos = cache[:, :half]
    sin = cache[:, half:rotary_dim]

    def rotate(packed):
        if packed is None:
            return None
        shape = tuple(int(size) for size in packed.shape)
        heads = shape[-1] // head_size
        view = packed.reshape((-1, heads, head_size))
        span = view[:, :, :rotary_dim]
        tail = view[:, :, rotary_dim:]
        # One cos/sin row per token, shared by every head of that token.
        c = cos.reshape((-1, 1, half))
        s = sin.reshape((-1, 1, half))
        if is_neox:
            first = span[:, :, :half]
            second = span[:, :, half:]
            rotated = jt.concat([first * c - second * s,
                                 second * c + first * s], dim=-1)
        else:
            # GPT-J interleaves the pairs instead of splitting them in half.
            pairs = span.reshape((-1, heads, half, 2))
            first = pairs[:, :, :, 0]
            second = pairs[:, :, :, 1]
            rotated = jt.stack([first * c - second * s,
                                second * c + first * s], dim=-1)
            rotated = rotated.reshape((-1, heads, rotary_dim))
        out = rotated if tail.shape[-1] == 0 else jt.concat([rotated, tail], dim=-1)
        return out.reshape(shape)

    return rotate(query), rotate(key)


def _rotary_embedding_acl(
        positions, query, key, cos_sin_cache,
        head_size, is_neox, rotary_dim):
    if not (
        getattr(jt.compiler, "has_acl", 0)
        and getattr(jt.flags, "use_acl", 0)
        and jt.flags.use_cuda
        and getattr(jt.flags, "no_grad", 0)
        and key is not None
        and is_neox
        and rotary_dim == head_size
        and head_size % 64 == 0
        and str(positions.dtype) in ("int32", "int64")
        and cos_sin_cache.ndim == 2
        and str(query.dtype) == str(key.dtype) == str(cos_sin_cache.dtype)
        and str(query.dtype) in ("float16", "bfloat16", "float32")
    ):
        return None

    token_count = int(positions.numel())
    flat_positions = positions.reshape((token_count,))
    cache = None
    acl_embedding = getattr(jt.nn, "_acl_embedding", None)
    if acl_embedding is not None:
        cache = acl_embedding(flat_positions, cos_sin_cache)
    if cache is None:
        cache_width = int(cos_sin_cache.shape[-1])
        cache_index = flat_positions.reshape((token_count, 1)).broadcast(
            (token_count, cache_width))
        cache = jt.gather(cos_sin_cache, 0, cache_index)
    half = rotary_dim // 2
    cos_half = cache[:, :half]
    sin_half = cache[:, half:rotary_dim]
    cos = jt.concat((cos_half, cos_half), dim=-1).reshape(
        (1, 1, token_count, rotary_dim))
    sin = jt.concat((sin_half, sin_half), dim=-1).reshape(
        (1, 1, token_count, rotary_dim))

    def to_bnsd(packed):
        heads = int(packed.shape[-1]) // head_size
        if token_count == 1:
            return packed.reshape((1, heads, 1, head_size))
        value = packed.reshape((token_count, heads, head_size))
        return value.transpose(0, 1).reshape((1, heads, token_count, head_size))

    query_bnsd = to_bnsd(query)
    key_bnsd = to_bnsd(key)
    query_bnsd, key_bnsd = jt.nn.rotary_emb(
        query_bnsd, key_bnsd, freq_cos=cos, freq_sin=sin)

    def from_bnsd(value, shape):
        heads = int(value.shape[1])
        if token_count == 1:
            return value.reshape(shape)
        return value.reshape((heads, token_count, head_size)).transpose(
            0, 1).reshape(shape)

    return (
        from_bnsd(query_bnsd, query.shape),
        from_bnsd(key_bnsd, key.shape),
    )
