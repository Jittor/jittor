"""vLLM's flash-attention bundle, answered from Jittor's paged attention.

vLLM's FlashAttention backend imports ``vllm.vllm_flash_attn`` -- a compiled
wheel that a source checkout does not carry. What it wants from it is three
entry points: packed variable-length attention for a prefill, single-token
attention against a paged cache for a decode, and the cache write between them.
All three are shapes :mod:`jittor.nn` already implements; the code here is the
vLLM-shaped signature around them.

The bundle also carries pieces this backend never reaches -- an FA3 scheduler,
CUTE kernels, fused rotary layers -- which vLLM imports while deciding what to
use. Those resolve permissively, so a version that reaches for one more of them
does not fail at import.
"""

import sys

import jittor as jt

from ..permissive import PermissiveModule, install_permissive_package

# torch is the shim itself, and this module is imported while the shim is still
# installing, so it cannot be reached at module scope -- only from a call, by
# which time vLLM is importable and the shim is long finished.

_BUNDLE = "vllm.vllm_flash_attn"
_INTERFACE = _BUNDLE + ".flash_attn_interface"


def host_lengths(tensor):
    """Bring a small metadata tensor to the host as plain ints.

    Part of this module's surface because the caller is meant to do it once for
    a whole forward pass: the paged-attention entry points take these lists so
    that walking many layers costs one device-to-host sync, not one per layer.
    """
    try:
        return [int(value) for value in tensor.cpu().numpy().tolist()]
    except Exception:
        return [int(value) for value in tensor]


def flash_attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q=None, max_seqlen_k=None,
                           softmax_scale=None, causal=False, window_size=None,
                           alibi_slopes=None, softcap=0.0, out=None,
                           block_table=None, **kwargs):
    """Packed variable-length attention over a batch with no padding.

    ``query`` is ``[total_q, heads, dim]`` and the two cumulative-length arrays
    say where each sequence starts. Grouped-query attention arrives as fewer
    key/value heads than query heads, repeated to match.
    """
    import torch
    import torch.nn.functional as functional

    out_dtype = query.dtype
    # In fp32 throughout: query arrives fresh while key/value come back from a
    # cache, so the two can differ in dtype, and the attention itself loses
    # accuracy in fp16.
    query, key, value = query.float32(), key.float32(), value.float32()
    query_heads, dim = query.shape[1], query.shape[2]
    if softmax_scale is None:
        softmax_scale = dim ** -0.5
    starts_q = host_lengths(cu_seqlens_q)
    starts_k = host_lengths(cu_seqlens_k)
    repeats = query_heads // key.shape[1]
    results = []
    for index in range(len(starts_q) - 1):
        q_i = query[starts_q[index]:starts_q[index + 1]]
        k_i = key[starts_k[index]:starts_k[index + 1]]
        v_i = value[starts_k[index]:starts_k[index + 1]]
        if repeats > 1:
            k_i = k_i.repeat_interleave(repeats, dim=1)
            v_i = v_i.repeat_interleave(repeats, dim=1)
        attended = functional.scaled_dot_product_attention(
            q_i.transpose(0, 1).unsqueeze(0),
            k_i.transpose(0, 1).unsqueeze(0),
            v_i.transpose(0, 1).unsqueeze(0),
            is_causal=bool(causal), scale=float(softmax_scale))
        results.append(
            attended.squeeze(0).transpose(0, 1).contiguous().cast(out_dtype))
    combined = torch.cat(results, dim=0) if len(results) > 1 else results[0]
    if out is not None:
        out.copy_(combined)
        return out
    return combined


def flash_attn_with_kvcache(query, key_cache, value_cache, key=None, value=None,
                            cache_seqlens=None, block_table=None,
                            softmax_scale=None, causal=False, window_size=None,
                            alibi_slopes=None, softcap=0.0, out=None, **kwargs):
    """One query token per sequence, attending over everything cached so far.

    The caches are paged as ``[blocks, block_size, heads, dim]``; each row of
    ``block_table`` lists the blocks belonging to one sequence, and
    ``cache_seqlens`` says how much of the last one is filled. Attention is not
    causal here -- a single query token may see the whole history.
    """
    import torch
    import torch.nn.functional as functional

    out_dtype = query.dtype
    query = query.float32()
    query_heads, dim = query.shape[-2], query.shape[-1]
    block_size, kv_heads = key_cache.shape[1], key_cache.shape[2]
    if softmax_scale is None:
        softmax_scale = dim ** -0.5
    repeats = query_heads // kv_heads
    lengths = host_lengths(cache_seqlens)
    results = []
    for row in range(query.shape[0]):
        length = lengths[row]
        blocks = block_table[row, :-(-length // block_size)]
        keys = key_cache[blocks].reshape(-1, kv_heads, dim)[:length].float32()
        values = value_cache[blocks].reshape(-1, kv_heads, dim)[:length].float32()
        if repeats > 1:
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        attended = functional.scaled_dot_product_attention(
            query[row].reshape(-1, query_heads, dim).transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0),
            is_causal=False, scale=float(softmax_scale))
        results.append(attended.squeeze(0).transpose(0, 1))
    stacked = torch.stack(results, dim=0).cast(out_dtype)
    if out is not None:
        out.copy_(stacked.reshape(out.shape))
        return out
    return stacked.reshape(query.shape)


def flash_attn_varlen_paged(query, kv_cache, cu_seqlens_q, seqused_k,
                            block_table, softmax_scale, causal, cq=None,
                            sk=None, cur_k=None, cur_v=None,
                            decode_fast_path=False):
    """vLLM's unified V1 attention, in terms of Jittor's paged attention.

    Choosing between the fused decode kernel, a batched path for a uniform
    batch, and a per-request loop is :func:`jittor.nn.paged_attention`'s job.
    What stays here is the vLLM-shaped signature and the host-side metadata the
    caller has already converted once for the whole forward pass.
    """
    return jt.nn.paged_attention(
        query, kv_cache, cu_seqlens_q, seqused_k, block_table,
        scale=softmax_scale, causal=causal, query_lengths=cq, key_lengths=sk)


def reshape_and_cache_kv_v1(key, value, kv_cache, slot_mapping, slots=None,
                            no_sync=False):
    """Write this step's key/value into vLLM's V1 paged cache.

    The write has to be visible to the attention that reads it, which under
    lazy execution means forcing it. ``no_sync`` is for the path where
    attention reads the current key/value directly and this write is only for
    the *next* token -- there the caller's own end-of-layer sync is enough.
    """
    jt.nn.reshape_and_cache(key, value, kv_cache, slot_mapping, slots=slots)
    if no_sync:
        return
    jt.sync([kv_cache])


def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping,
                            kv_cache_dtype="auto", k_scale=None, v_scale=None):
    """The same write against separate key and value caches."""
    block_size = key_cache.shape[1]
    for token, slot in enumerate(host_lengths(slot_mapping)):
        if slot < 0:
            continue
        block, offset = slot // block_size, slot % block_size
        key_cache[block, offset] = key[token]
        value_cache[block, offset] = value[token]


def _no_scheduler_metadata(*args, **kwargs):
    """FA3 plans its work ahead; the path taken here does not."""
    return None


def install():
    """Publish ``vllm.vllm_flash_attn`` and return the module names it owns."""

    published = []
    bundle = PermissiveModule(_BUNDLE)
    bundle.flash_attn_varlen_func = flash_attn_varlen_func
    bundle.flash_attn_with_kvcache = flash_attn_with_kvcache
    bundle.get_scheduler_metadata = _no_scheduler_metadata
    bundle.__version__ = "2.6.1"
    sys.modules[_BUNDLE] = bundle
    published.append(_BUNDLE)
    # V1 reaches the same three entry points by this longer path as well.
    interface = PermissiveModule(_INTERFACE)
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    interface.flash_attn_with_kvcache = flash_attn_with_kvcache
    interface.get_scheduler_metadata = _no_scheduler_metadata
    sys.modules[_INTERFACE] = interface
    published.append(_INTERFACE)
    install_permissive_package(_BUNDLE, sys.meta_path)
    return tuple(published)
