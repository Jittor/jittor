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

import os
import sys
import weakref

import jittor as jt
from jittor._runtime.dispatch import try_dispatch

from jittor.compat.permissive import PermissiveModule, install_permissive_package
from jittor.compat.diagnostics import EXPECTED, swallowed
from jittor.compat.transaction import owned_runtime_hook, active_transaction, set_attr, runtime_owns_module

# torch is the shim itself, and this module is imported while the shim is still
# installing, so it cannot be reached at module scope -- only from a call, by
# which time vLLM is importable and the shim is long finished.

_BUNDLE = "vllm.vllm_flash_attn"
_INTERFACE = _BUNDLE + ".flash_attn_interface"
_F32_FUSED_PREFILL_MODE = os.environ.get(
    "JITTOR_VLLM_F32_FUSED_PREFILL", "auto")
_FLASH_GQA_PREFILL_MODE = os.environ.get(
    "JITTOR_VLLM_FLASH_GQA_PREFILL", "0")
_FA2_PAGE_DECODE_BORROWS = {}


def _cuda_graph_capture_active():
    """Whether the shim is recording the real CUDA launch stream."""
    try:
        from jittor.compat.triton import backend
        return backend.graph_capture_active()
    except (ImportError, AttributeError):
        return False


def _require_explicit_graph_ownership(capturing):
    # vLLM already owns the complete decode CUDA graph. Jittor 2.0's default
    # automatic module replay is a second, independent owner of the same KV
    # state; across requests it can replay stale buffers and return wrong
    # tokens. Fail before recording such a graph instead of silently serving
    # corrupted output. The benchmark harness selects this safe configuration.
    if capturing and jt.flags.auto_graph_replay:
        raise RuntimeError(
            "vLLM CUDA Graph requires JT_AUTO_GRAPH_REPLAY=0; set it before "
            "importing Jittor so the outer graph is the only replay owner")


def _graph_single_request_decode(attn_metadata, tokens):
    """Prove a full one-token request from static Graph metadata only."""
    return (
        tokens == 1
        and int(attn_metadata.max_query_len) == 1
        and int(attn_metadata.block_table.shape[0]) == 1
        and int(attn_metadata.query_start_loc.numel()) == 2
        and int(attn_metadata.seq_lens.numel()) == 1
    )


def _graph_fixed_batch_decode(attn_metadata, tokens):
    """Opt-in decode for an engine restricted to one full batch shape."""
    fixed_batch = os.environ.get(
        "JITTOR_VLLM_EXPERIMENTAL_FIXED_BATCH_FUSED_DECODE", "0")
    return (
        fixed_batch == str(tokens)
        and tokens > 1
        and int(attn_metadata.max_query_len) == 1
        and int(attn_metadata.block_table.shape[0]) == tokens
        and int(attn_metadata.query_start_loc.numel()) == tokens + 1
        and int(attn_metadata.seq_lens.numel()) == tokens
        and int(attn_metadata.slot_mapping.numel()) >= tokens
    )


def _f32_prefill_policy(tokens, dtype, head_dim, mode):
    """Conservatively choose the tiled CUDA GQA kernel for full prefill.

    The automatic path is validated on FP16 full prefill through 512 tokens.
    Larger sequences and BF16 require an explicit experimental opt-in.
    """
    dtype_name = str(dtype).rsplit(".", 1)[-1]
    if mode not in ("auto", "1") or int(head_dim) > 128:
        return False
    if mode == "1":
        return dtype_name in ("float16", "bfloat16")
    return dtype_name == "float16" and 1 < int(tokens) <= 512


def host_lengths(tensor):
    """Bring a small metadata tensor to the host as plain ints.

    Part of this module's surface because the caller is meant to do it once for
    a whole forward pass: the paged-attention entry points take these lists so
    that walking many layers costs one device-to-host sync, not one per layer.
    """
    try:
        return [int(value) for value in tensor.cpu().numpy().tolist()]
    except EXPECTED as exc:
        swallowed("vllm/flash_attn.py host_lengths: return [int(value) for value in tensor.cpu().numpy().to...", exc)
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
                            decode_fast_path=False, out=None):
    """vLLM's unified V1 attention, in terms of Jittor's paged attention.

    Choosing between the fused decode kernel, a batched path for a uniform
    batch, and a per-request loop is :func:`jittor.nn.paged_attention`'s job.
    What stays here is the vLLM-shaped signature and the host-side metadata the
    caller has already converted once for the whole forward pass.
    """
    return jt.nn.paged_attention(
        query, kv_cache, cu_seqlens_q, seqused_k, block_table,
        scale=softmax_scale, causal=causal, query_lengths=cq, key_lengths=sk,
        out=out)


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


def _experimental_fa2_paged_decode(query, key, value, kv_cache,
                                   seq_lens, block_table, scale):
    """Probe official FA2 append+decode with a deliberately 256-token KV page.

    This is opt-in. Upstream FA2 rejects vLLM's usual page-16 cache. CUDA
    Graph capture is a separate, explicit probe; its ownership is not proved.
    """
    capture = _cuda_graph_capture_active()
    if (not jt.flags.use_cuda or getattr(jt.compiler, "has_acl", False)
            or jt.flags.auto_graph_replay
            or (capture and os.environ.get(
                "JITTOR_VLLM_EXPERIMENTAL_FA2_PAGE_GRAPH") != "1")):
        raise RuntimeError("experimental FA2 paged decode requires eager CUDA "
                           "or explicit FA2 Graph probe with "
                           "JT_AUTO_GRAPH_REPLAY=0")
    if (int(query.shape[0]) != 1 or int(key.shape[0]) != 1
            or int(value.shape[0]) != 1 or int(seq_lens.numel()) != 1):
        raise RuntimeError("experimental FA2 paged decode supports batch=1 only")
    page = int(kv_cache.shape[2])
    if page % 256:
        raise RuntimeError("official FA2 paged KV requires page size divisible "
                           "by 256; configure an experimental 256-token cache")
    dim = int(query.shape[-1])
    if (dim not in (64, 128)
            or any(str(t.dtype).rsplit(".", 1)[-1] != "float16"
                   for t in (query, key, value, kv_cache))):
        raise RuntimeError("experimental FA2 paged decode requires FP16 "
                           "and head dimension 64 or 128")
    from jittor.compat.shim.backends import flash_attention
    backend, miss = flash_attention.load_backend_for(dim, "fp16")
    if backend is None:
        raise RuntimeError(f"experimental FA2 backend unavailable: {miss}")
    cache_key = id(kv_cache)
    entry = _FA2_PAGE_DECODE_BORROWS.get(cache_key)
    if entry is None or entry[0] is not kv_cache:
        kcache = kv_cache[:, 0]
        vcache = kv_cache[:, 1]
        kcache._jittor_torch_ext_mutable_borrow = True
        vcache._jittor_torch_ext_mutable_borrow = True
        entry = (kv_cache, kcache, vcache, [])
        _FA2_PAGE_DECODE_BORROWS[cache_key] = entry
    _, kcache, vcache, retained_lengths = entry
    # vLLM seq_lens includes this token; FA2 appends K/V at the old length.
    previous_lengths = seq_lens - 1
    retained_lengths.append(previous_lengths)
    q = query.reshape((1, 1, int(query.shape[1]), dim))
    k = key.reshape((1, 1, int(key.shape[1]), dim))
    v = value.reshape((1, 1, int(value.shape[1]), dim))
    max_window = int(block_table.shape[1]) * page - 1
    result = backend._flashattn_jittor_low_level.fwd_kvcache(
        q, kcache, vcache, k, v, previous_lengths,
        None, None, None, None, block_table, None, None,
        float(scale), False, max_window, -1, 0.0, False, 0)
    return result[0].reshape((1, int(query.shape[1]), dim))


def _experimental_vllm_fa2_page16_decode(query, key, value, kv_cache,
                                          seq_lens, block_table, scale):
    """Probe vLLM-pinned FA2 append+decode against its native page-16 KV.

    The first prototype only supports eager batch=1: Graph capture needs a
    separate buffer-lifetime and replay gate before it can serve requests.
    """
    if (not jt.flags.use_cuda or getattr(jt.compiler, "has_acl", False)
            or jt.flags.auto_graph_replay or _cuda_graph_capture_active()):
        raise RuntimeError("experimental vLLM FA2 page-16 decode requires "
                           "eager CUDA with JT_AUTO_GRAPH_REPLAY=0")
    if (int(query.shape[0]) != 1 or int(key.shape[0]) != 1
            or int(value.shape[0]) != 1 or int(seq_lens.numel()) != 1
            or int(block_table.shape[0]) != 1):
        raise RuntimeError("experimental vLLM FA2 page-16 decode is batch=1 only")
    if int(kv_cache.shape[2]) != 16 or int(query.shape[-1]) != 64:
        raise RuntimeError("experimental vLLM FA2 decode requires page=16, head=64")
    if any(str(t.dtype).rsplit(".", 1)[-1] != "float16"
           for t in (query, key, value, kv_cache)):
        raise RuntimeError("experimental vLLM FA2 decode requires FP16")
    from jittor.compat.shim.backends import flash_attention
    backend, miss = flash_attention.load_backend_for(64, "fp16")
    if backend is None:
        raise RuntimeError(f"experimental vLLM FA2 backend unavailable: {miss}")
    cache_key = id(kv_cache)
    entry = _FA2_PAGE_DECODE_BORROWS.get(cache_key)
    if entry is None or entry[0] is not kv_cache:
        kcache = kv_cache[:, 0]
        vcache = kv_cache[:, 1]
        kcache._jittor_torch_ext_mutable_borrow = True
        vcache._jittor_torch_ext_mutable_borrow = True
        entry = (kv_cache, kcache, vcache, [])
        _FA2_PAGE_DECODE_BORROWS[cache_key] = entry
    _, kcache, vcache, retained_lengths = entry
    previous_lengths = seq_lens - 1
    retained_lengths.append(previous_lengths)
    result = backend._flashattn_jittor_low_level.fwd_kvcache(
        query.reshape((1, 1, int(query.shape[1]), 64)),
        kcache, vcache,
        key.reshape((1, 1, int(key.shape[1]), 64)),
        value.reshape((1, 1, int(value.shape[1]), 64)),
        previous_lengths, None, None, None, None, block_table, None, None,
        float(scale), False, -1, -1, 0.0, False, 0)
    return result[0].reshape((1, int(query.shape[1]), 64))


def _direct_full_prefill(query, key, value, starts, lengths, causal, scale):
    """Use fresh Q/K/V for a complete prefill without re-reading paged KV.

    Chunked prefill and prefix-cache hits still need the paged cache.  This
    case does not: reading back the cache we just wrote adds a gather, dtype
    conversions, GQA expansion and a separate attention graph at every layer.
    """
    if (starts is None or lengths is None or len(lengths) < 1
            or len(starts) != len(lengths) + 1
            or starts[0] != 0
            or starts[-1] != int(query.shape[0])
            or starts[-1] != int(key.shape[0])
            or starts[-1] != int(value.shape[0])
            or any(starts[index + 1] - starts[index] != lengths[index]
                   or lengths[index] <= 1 for index in range(len(lengths)))):
        return None
    tokens = starts[-1]
    if len(lengths) > 1:
        if (not getattr(jt.flags, "use_acl", 0)
                or os.environ.get(
                    "JITTOR_VLLM_NPU_EXPERIMENTAL_BATCHED_FIA_PREFILL") != "1"):
            return None
        fused = try_dispatch(
            "nn.fused_infer_prefill_attention", query, key, value,
            scale=float(scale), causal=bool(causal),
            query_lengths=starts[1:])
        return (None if fused is None else
                fused.reshape((tokens, query.shape[1] * query.shape[2])))
    if (getattr(jt.flags, "use_acl", 0)
            and os.environ.get("JITTOR_VLLM_NPU_EXPERIMENTAL_FIA_PREFILL") == "1"):
        fused = try_dispatch(
            "nn.fused_infer_prefill_attention", query, key, value,
            scale=float(scale), causal=bool(causal))
        if fused is not None:
            return fused.reshape((tokens, query.shape[1] * query.shape[2]))
    query = query.reshape((1, tokens, query.shape[1], query.shape[2]))
    key = key.reshape((1, tokens, key.shape[1], key.shape[2]))
    value = value.reshape((1, tokens, value.shape[1], value.shape[2]))
    # CUDA-only opt-in: use official FlashAttention directly for full FP16 GQA.
    if (_FLASH_GQA_PREFILL_MODE == "1" and bool(jt.flags.use_cuda)
            and not getattr(jt.compiler, "has_acl", False)
            and tokens >= 256 and query.shape[-1] == 128
            and query.shape[2] > key.shape[2]
            and query.shape[2] % key.shape[2] == 0
            and key.shape[2] == value.shape[2]
            and all(str(t.dtype).rsplit(".", 1)[-1] == "float16"
                    for t in (query, key, value))):
        from jittor.compat.shim.backends import flash_attention
        backend, miss = flash_attention.load_backend_for(128, "fp16")
        if backend is None:
            raise RuntimeError(
                "JITTOR_VLLM_FLASH_GQA_PREFILL=1 requires the native "
                f"FlashAttention backend: {miss}")
        result = backend.flash_attn_func(
            query, key, value, 0.0, float(scale), bool(causal))
        if result is None:
            raise RuntimeError("native FlashAttention declined FP16 GQA prefill")
        return result.reshape((tokens, query.shape[2], query.shape[3]))
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output_dtype = query.dtype
    # cuDNN 8 declines the FP16 GQA shape used by Qwen3, leaving SDPA as a
    # multi-operator graph. Auto-select Jittor's tiled FP32 GQA kernel only
    # for validated short FP16 full prefill; never redirect ACL or other dims.
    use_f32_prefill = (
        bool(jt.flags.use_cuda)
        and not getattr(jt.compiler, "has_acl", False)
        and _f32_prefill_policy(
            tokens, output_dtype, query.shape[-1], _F32_FUSED_PREFILL_MODE))
    # Probe whether higher-precision direct ACL attention can keep the
    # correct BF16 TP2 greedy sequence without cache-backed prefill overhead.
    # Off by default until complete-model oracle and A/B tests pass.
    if (getattr(jt.flags, "use_acl", 0)
            and os.environ.get("JITTOR_VLLM_ACL_F32_DIRECT_PREFILL") == "1"
            and str(output_dtype).rsplit(".", 1)[-1] == "bfloat16"):
        use_f32_prefill = True
    if use_f32_prefill:
        query, key, value = query.float32(), key.float32(), value.float32()
    result = jt.nn.scaled_dot_product_attention(
        query, key, value, is_causal=bool(causal), scale=float(scale),
        enable_gqa=True)
    if use_f32_prefill:
        result = result.cast(output_dtype)
    return result.transpose(1, 2).reshape(
        (tokens, query.shape[1], query.shape[3]))


def _plain_decoder_prefill(impl, metadata):
    """Whether fresh Q/K/V have the same attention semantics as the cache.

    The direct path cannot implement sliding windows, soft caps, ALiBi,
    attention sinks, prefix-LM or distributed context parallelism. Decline
    those configurations rather than silently changing a model's output.
    """
    return (
        getattr(getattr(impl, "attn_type", None), "name", None) == "DECODER"
        and getattr(impl, "sliding_window", None) == (-1, -1)
        and getattr(impl, "alibi_slopes", None) is None
        and not getattr(impl, "logits_soft_cap", 0)
        and getattr(impl, "sinks", None) is None
        and getattr(impl, "kv_sharing_target_layer_name", None) is None
        and getattr(impl, "kv_cache_dtype", None) == "auto"
        and getattr(impl, "dcp_world_size", 1) == 1
        and getattr(metadata, "causal", None) is True
        and not getattr(metadata, "use_cascade", False)
        and not getattr(metadata, "common_prefix_len", 0)
        and getattr(metadata, "mm_prefix_range_tensor", None) is None
    )


def _allow_direct_full_prefill(mode, use_acl):
    """Keep the fresh-QKV shortcut off by default on ACL.

    On Qwen2.5-3B BF16 TP2 its SDPA arithmetic changes a near-tie into a
    different greedy token from native vLLM-Ascend. The cache-backed path
    matches the native 16-token regression; explicit opt-in remains available
    for further kernel experiments.
    """
    return mode != "0" and (mode == "1" or not use_acl)


def _no_scheduler_metadata(*args, **kwargs):
    """FA3 plans its work ahead; the path taken here does not."""
    return None


@owned_runtime_hook("vllm.flash_attention")
def install():
    """Publish ``vllm.vllm_flash_attn`` and return the module names it owns."""

    if all(runtime_owns_module("vllm.flash_attention", sys.modules, name)
           for name in (_BUNDLE, _INTERFACE)):
        return (_BUNDLE, _INTERFACE)
    published = []
    bundle = PermissiveModule(_BUNDLE)
    bundle.flash_attn_varlen_func = flash_attn_varlen_func
    bundle.flash_attn_with_kvcache = flash_attn_with_kvcache
    bundle.get_scheduler_metadata = _no_scheduler_metadata
    bundle.__version__ = "2.6.1"
    active_transaction().replace_module(sys.modules, _BUNDLE, bundle)
    published.append(_BUNDLE)
    # V1 reaches the same three entry points by this longer path as well.
    interface = PermissiveModule(_INTERFACE)
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    interface.flash_attn_with_kvcache = flash_attn_with_kvcache
    interface.get_scheduler_metadata = _no_scheduler_metadata
    active_transaction().replace_module(sys.modules, _INTERFACE, interface)
    published.append(_INTERFACE)
    # The whole subtree, unlike torch.fx's named list: `vllm.vllm_flash_attn`
    # is a compiled wheel a source checkout does not carry, so absence is the
    # expected state for every name under it and there is no real analysis
    # underneath that a stub could silently stand in for. The three entry
    # points this module implements are published above and keep their real
    # implementation -- the import machinery consults the module table before
    # any finder. Without this the pieces the docstring calls permissive (the
    # FA3 scheduler, the CUTE kernels, the fused rotary layers) raise
    # ImportError at vLLM's module scope, which is what the allowlist added in
    # 36c88b7a did to this caller.
    install_permissive_package(_BUNDLE, sys.meta_path, allow=(_BUNDLE + ".*",),
                               transaction=active_transaction())
    return tuple(published)


def patch_attention_impl(module):
    """Route V1's FlashAttention backend through Jittor's paged attention.

    V1 does one unified variable-length attention against the paged cache and
    writes this step's keys and values into that cache separately. Both go
    through :mod:`jittor.nn` here.

    Two things about the shape of this function are load-bearing. The metadata
    it needs on the host -- slot mapping, sequence starts, sequence lengths --
    is identical for every decoder layer in one forward pass, so it is
    converted once per metadata object rather than once per layer; that
    conversion is a device-to-host sync, and doing it per layer was the largest
    single cost in a decode step. And the lazy graph has to be broken
    periodically or it grows with the square of the layer count and never
    finishes compiling; the cache write is ordered before the attention read by
    the graph itself, so this sync is only bounding the residual stream.
    """

    impl = getattr(module, "FlashAttentionImpl", None)
    if impl is None or getattr(impl, "_jittor_paged_attention", False):
        return False
    from jittor._runtime.graph_replay import _TRACING, graph_replay_barrier
    layers_between_syncs = _layers_between_syncs()
    reuse_full_token_views = (
        os.environ.get("JITTOR_VLLM_REUSE_FULL_TOKEN_VIEWS", "1") != "0"
    )
    fused_cache_decode_mode = os.environ.get(
        "JITTOR_VLLM_FUSED_CACHE_DECODE", "auto")
    direct_full_prefill_mode = os.environ.get(
        "JITTOR_VLLM_DIRECT_FULL_PREFILL", "auto")
    # Unlike the legacy explicit mode, this ACL opt-in always checks that
    # fresh Q/K/V have the same attention semantics as the paged-cache path.
    safe_acl_direct_prefill = (
        os.environ.get("JITTOR_VLLM_NPU_SAFE_DIRECT_PREFILL") == "1"
    )
    batched_acl_direct_prefill = (
        os.environ.get("JITTOR_VLLM_NPU_EXPERIMENTAL_BATCHED_FIA_PREFILL") == "1"
    )
    counter = [0]
    rebind_hits = [0]
    metadata_cache = {}

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:      # the profiling pass has no cache yet
            return output.fill_(0)
        tokens = attn_metadata.num_actual_tokens
        if (tokens == 1 and getattr(jt.flags, "use_acl", 0)
                and _TRACING[0]):
            # ACL decode still converts the per-step slot/sequence metadata to
            # host lists and aclIntArray attributes. Executor replay would
            # silently reuse the first step's values even if tensor arguments
            # were copied into fixed buffers. Refuse this graph until the ACL
            # serving op consumes mutable device-side metadata end to end.
            graph_replay_barrier(
                "ACL paged decode has dynamic host slot/sequence metadata")
        capturing = _cuda_graph_capture_active()
        _require_explicit_graph_ownership(capturing)
        if capturing:
            # Full decode graphs read all varying metadata directly from the
            # persistent GPU tensors. The CUDA cache-write and paged-decode
            # fast paths below do not consume these host lists; reading them
            # back here would issue an illegal D2H synchronization in capture.
            slots = starts = lengths = None
        else:
            slots, starts, lengths = _host_metadata(
                attn_metadata, metadata_cache)
        full_query = (
            reuse_full_token_views and tokens == int(query.shape[0])
        )
        query_tokens = query if full_query else query[:tokens]
        has_cache_update = (
            key is not None and value is not None
            and kv_cache is not None and kv_cache.numel() > 0
            and attn_metadata.slot_mapping is not None
        )
        full_kv = (
            has_cache_update and reuse_full_token_views
            and tokens == int(key.shape[0])
            and tokens == int(value.shape[0])
        )
        key_tokens = key if full_kv else (key[:tokens] if key is not None else None)
        value_tokens = (
            value if full_kv else (value[:tokens] if value is not None else None)
        )
        direct_out = output if tokens == int(output.shape[0]) else None
        zero_mode = os.environ.get("JITTOR_VLLM_DIAG_ZERO_ATTENTION", "0")
        if (getattr(jt.flags, "use_acl", 0)
                and (zero_mode in ("1", "all")
                     or (zero_mode == "prefill" and tokens > 1)
                     or (zero_mode == "decode" and tokens == 1))):
            if has_cache_update:
                reshape_and_cache_kv_v1(
                    key_tokens, value_tokens, kv_cache,
                    attn_metadata.slot_mapping, slots=slots, no_sync=False)
            output[:tokens] = jt.zeros_like(output[:tokens])
            return output
        # Auto-enable only for a full, single-request CUDA Graph decode.
        # Batch 4 was slower in end-to-end A/B.  Keep the old explicit opt-in
        # for eager decode, and avoid evaluating host metadata on the default
        # eager path at every layer.
        fused_requested = (
            capturing if fused_cache_decode_mode == "auto"
            else fused_cache_decode_mode != "0"
        )
        one_token_decode = False
        if (fused_requested and has_cache_update and direct_out is not None
                and tokens == int(query_tokens.shape[0])
                and tokens == int(key_tokens.shape[0])
                and tokens == int(value_tokens.shape[0])):
            if capturing:
                # Capture cannot read starts/lengths back to the host.  The
                # scalar max_query_len and static tensor sizes are sufficient
                # for batch 1; the kernel reads varying lengths from GPU.
                one_token_decode = (
                    _graph_single_request_decode(attn_metadata, tokens)
                    or _graph_fixed_batch_decode(attn_metadata, tokens)
                )
            else:
                one_token_decode = (
                    lengths is not None and starts is not None
                    and len(lengths) == tokens and len(starts) == tokens + 1
                    and starts[0] == 0 and starts[-1] == tokens
                    and all(starts[index + 1] == starts[index] + 1
                            for index in range(tokens))
                )
        attended = None
        if one_token_decode:
            if os.environ.get("JITTOR_VLLM_EXPERIMENTAL_VLLM_FA2_PAGE16_DECODE") == "1":
                attended = _experimental_vllm_fa2_page16_decode(
                    query_tokens, key_tokens, value_tokens, kv_cache,
                    attn_metadata.seq_lens, attn_metadata.block_table,
                    self.scale)
            elif os.environ.get("JITTOR_VLLM_EXPERIMENTAL_FA2_PAGE_DECODE") == "1":
                attended = _experimental_fa2_paged_decode(
                    query_tokens, key_tokens, value_tokens, kv_cache,
                    attn_metadata.seq_lens, attn_metadata.block_table,
                    self.scale)
            else:
                attended = jt.nn.reshape_cache_and_paged_attention_decode(
                    query_tokens, key_tokens, value_tokens, kv_cache,
                    attn_metadata.slot_mapping, attn_metadata.seq_lens,
                    attn_metadata.block_table, self.scale, direct_out)
        if has_cache_update and attended is None:
            reshape_and_cache_kv_v1(
                key_tokens, value_tokens,
                kv_cache,
                attn_metadata.slot_mapping, slots=slots, no_sync=True)
        use_acl = bool(getattr(jt.flags, "use_acl", 0))
        safe_acl_requested = (
            safe_acl_direct_prefill and use_acl
            and direct_full_prefill_mode != "0"
        )
        batched_acl_requested = (
            batched_acl_direct_prefill and use_acl
            and direct_full_prefill_mode != "0"
            and starts is not None and len(starts) > 2
        )
        direct_full_prefill = (
            (safe_acl_requested or batched_acl_requested
             or _allow_direct_full_prefill(
                direct_full_prefill_mode, use_acl))
            and tokens > 1
            and not capturing
            and ((not safe_acl_requested and direct_full_prefill_mode == "1")
                 or _plain_decoder_prefill(self, attn_metadata))
        )
        if attended is None and has_cache_update and direct_full_prefill:
            attended = _direct_full_prefill(
                query_tokens, key_tokens, value_tokens, starts, lengths,
                attn_metadata.causal, self.scale)
        if attended is None:
            attended = flash_attn_varlen_paged(
                query_tokens, kv_cache, attn_metadata.query_start_loc,
                attn_metadata.seq_lens, attn_metadata.block_table, self.scale,
                attn_metadata.causal, cq=starts, sk=lengths, out=direct_out)
        if attended is not direct_out:
            attended = attended.reshape((tokens,) + tuple(output.shape[1:]))
            if (use_acl and not capturing and direct_out is not None
                    and tokens == 1
                    and starts is not None and len(starts) == 2
                    and starts[0] == 0 and starts[1] == 1
                    and lengths is not None and len(lengths) == 1
                    and lengths[0] > 1
                    and int(attn_metadata.max_query_len) == 1
                    and os.environ.get("JITTOR_NPU_ATTENTION_OUTPUT_REBIND") == "1"):
                # Eager single-request decode only. Do not change prefill,
                # dynamic-batch, or graph buffer ownership semantics.
                # Publish the result Var instead of launching a full-shape
                # StridedSliceAssignV2 copy.
                output.update(attended)
                if (os.environ.get("JITTOR_NPU_ATTENTION_OUTPUT_REBIND_TRACE") == "1"
                        and rebind_hits[0] == 0):
                    print("npu_attention_output_rebind_hit",
                          "seq_len", lengths[0], flush=True)
                rebind_hits[0] += 1
            else:
                output[:tokens] = attended
        counter[0] += 1
        if (layers_between_syncs > 0
                and counter[0] % layers_between_syncs == 0):
            jt.sync_all()
        return output

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Already done: forward writes the cache immediately before reading it.

        V1 calls this once per layer as well, which would double every write
        and pay an unconverted host sync for the slot mapping each time.
        """
        return

    set_attr(impl, "forward", forward)
    set_attr(impl, "do_kv_cache_update", do_kv_cache_update)
    set_attr(impl, "_jittor_paged_attention", True)
    return True


def _layers_between_syncs():
    # Sampling's Triton bridge submits the graph that produces logits once per
    # decode step, so an additional attention-layer sync only cuts that graph
    # in half and drains overlap. Keep an explicit positive override for models
    # that does not reach the ordinary sampling path. The large default is a
    # safety valve for that unusual caller while keeping the validated serving
    # path free of a per-token global barrier; zero remains a full disable.
    raw = os.environ.get("JITTOR_VLLM_LAYERS_PER_SYNC", "1000000")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1000000


def _host_metadata(attn_metadata, cache):
    """Convert per-forward metadata once, with an ACL auto-replay safeguard.

    With Jittor auto replay enabled, ACL V1 incremental decode diverged after
    six tokens when these host lists were reused. A query-start readback at
    each later layer prevented it. With auto replay disabled, TP1/TP2/TP4,
    long-context page crossing and batch-4 decode passed without that fence.
    Keep this small correctness fence ACL-only until the graph lifetime root
    cause is fixed; CUDA retains the once-per-forward fast path.
    """

    key = id(attn_metadata)
    entry = cache.get(key)
    default_readback = ('1' if getattr(jt.flags, 'auto_graph_replay', 1)
                        else '0')
    if (entry is not None and getattr(jt.compiler, 'has_acl', False)
            and getattr(jt.flags, 'use_acl', 0)
            and os.environ.get('JITTOR_VLLM_NPU_SAFE_METADATA_READBACK',
                               default_readback) != '0'):
        if os.environ.get('JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK', '1') != '0':
            # numpy() synchronizes and transfers this tiny ACL tensor directly.
            # Avoid constructing an extra lazy cpu() Var at every layer.
            attn_metadata.query_start_loc.numpy()
        else:
            host_lengths(attn_metadata.query_start_loc)
    if entry is None:
        entry = (host_lengths(attn_metadata.slot_mapping),
                 host_lengths(attn_metadata.query_start_loc),
                 host_lengths(attn_metadata.seq_lens))
        cache[key] = entry
        try:
            # Drop the entry with the metadata object, so a later object
            # reusing its id cannot pick up stale lengths.
            weakref.finalize(attn_metadata, cache.pop, key, None)
        except TypeError:
            if len(cache) > 256:
                cache.clear()
    return entry


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.v1.attention.backends.flash_attn": patch_attention_impl,
}
