"""Attention over a paged key/value cache.

A serving stack keeps the keys and values of every in-flight request in fixed
size blocks and hands attention a table of block ids per request. One function
covers the three shapes that arise -- prefill, decode and chunked prefill -- and
takes a different route for each: a fused CUDA decode kernel, a batched path
when every request has the same lengths, and a per-request loop when they differ.
"""

import jittor as jt

def _host_lengths(value):
    """A tensor of lengths as python ints, or the list it already is."""
    try:
        return [int(v) for v in value.numpy().tolist()]
    except AttributeError:
        return [int(v) for v in value]


def reshape_and_cache(key, value, kv_cache, slot_mapping, slots=None):
    """Write this step's key/value into a paged KV cache.

    ``kv_cache`` is ``[num_blocks, 2, block_size, num_kv_heads, head_dim]`` and
    ``slot_mapping`` gives each token's flat slot. ``slots`` may carry the same
    mapping already on the host, which spares a device-to-host sync when the
    caller walks many layers with one metadata object.
    """
    from .kv_cache_cuda import _reshape_and_cache_cuda
    from .kv_cache_acl import _reshape_and_cache_acl

    if _reshape_and_cache_cuda(key, value, kv_cache, slot_mapping) is not None:
        return kv_cache
    if _reshape_and_cache_acl(
            key, value, kv_cache, slot_mapping, slots=slots) is not None:
        return kv_cache
    block_size = kv_cache.shape[2]
    if slots is None:
        slots = _host_lengths(slot_mapping)
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        block, offset = slot // block_size, slot % block_size
        kv_cache[block, 0, offset] = key[token]
        kv_cache[block, 1, offset] = value[token]
    return kv_cache


def paged_attention(query, kv_cache, cu_seqlens_q, seq_lens, block_table,
                    scale=None, causal=True, query_lengths=None,
                    key_lengths=None):
    """Attention of packed queries against a paged KV cache.

    ``query`` is ``[total_tokens, num_heads, head_dim]``; ``kv_cache`` is
    ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``; each request's keys
    live in the blocks named by its ``block_table`` row. A request's ``Lq`` query
    tokens are the LAST ``Lq`` positions of its ``Lk = seq_lens[i]`` keys, which
    is the bottom-right causal alignment, so prefill (``Lq == Lk``), decode
    (``Lq == 1``) and chunked prefill (``Lq < Lk``) all read the same way.

    ``query_lengths``/``key_lengths`` may carry ``cu_seqlens_q``/``seq_lens``
    already on the host; passing them spares a device-to-host sync per call,
    which matters when every layer of a forward pass repeats it.

    Scores are accumulated in float32 and the result is cast back to the query's
    dtype.
    """
    from .kv_cache_cuda import _paged_attention_decode_cuda

    num_heads, head_dim = int(query.shape[1]), int(query.shape[2])
    block_size, num_kv_heads = int(kv_cache.shape[2]), int(kv_cache.shape[3])
    repeats = num_heads // num_kv_heads
    if scale is None:
        scale = head_dim ** -0.5
    scale = float(scale)
    causal = causal is not False and causal is not None

    if causal:
        decoded = _paged_attention_decode_cuda(
            query, kv_cache, seq_lens, block_table, scale)
        if decoded is not None:
            return decoded
        from .kv_cache_acl import _paged_attention_decode_acl
        decoded = _paged_attention_decode_acl(
            query, kv_cache, block_table, scale, key_lengths=key_lengths)
        if decoded is not None:
            return decoded

    out_dtype = query.dtype
    q = query.float32()
    starts = (list(query_lengths) if query_lengths is not None
              else _host_lengths(cu_seqlens_q))
    lengths = (list(key_lengths) if key_lengths is not None
               else _host_lengths(seq_lens))
    requests = len(starts) - 1

    # One request per python iteration costs about six kernel launches per layer,
    # so a many-sequence prefill spends its time in launch overhead rather than
    # in the attention itself. When every request has the same query and key
    # length -- a same-length prompt batch always does -- the whole batch is one
    # gather and two batched matmuls, with a mask that broadcasts across it.
    uniform = (
        requests > 1
        and len({starts[i + 1] - starts[i] for i in range(requests)}) == 1
        and len(set(lengths[:requests])) == 1
        and (starts[1] - starts[0]) > 0
    )
    if uniform:
        span_q = starts[1] - starts[0]
        span_k = lengths[0]
        used = -(-span_k // block_size)      # blocks holding span_k keys
        rows = _gather_block_table(block_table, requests, used)
        cache = _gather_cache_blocks(kv_cache, rows.reshape(-1)).reshape(
            (requests, used, 2, block_size, num_kv_heads, head_dim))
        keys, values = _split_cache_kv(cache, 2)
        keys = _slice_dim(keys.reshape(
            (requests, used * block_size, num_kv_heads, head_dim)),
            1, 0, span_k)
        values = _slice_dim(values.reshape(
            (requests, used * block_size, num_kv_heads, head_dim)),
            1, 0, span_k)
        keys, values = keys.float32(), values.float32()
        if repeats > 1:
            keys = _repeat_interleave_dim(keys, 2, repeats)
            values = _repeat_interleave_dim(values, 2, repeats)
        queries = _slice_dim(q, 0, 0, requests * span_q).reshape(
            (requests, span_q, num_heads, head_dim)).transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = jt.matmul(queries, keys.transpose(2, 3)) * scale
        if causal:
            scores = scores + _causal_mask(span_q, span_k) * (-1e30)
        weights = jt.nn.softmax(scores, dim=-1)
        packed = jt.matmul(weights, values)
        return packed.transpose(1, 2).reshape(
            (requests * span_q, num_heads, head_dim)).cast(out_dtype)

    outputs = []
    for i in range(requests):
        span_q = starts[i + 1] - starts[i]
        if span_q <= 0:
            continue
        span_k = lengths[i]
        used = -(-span_k // block_size)
        blocks = _gather_block_table(block_table, 1, used, request=i)
        cache = _gather_cache_blocks(kv_cache, blocks.reshape(-1))
        keys, values = _split_cache_kv(cache, 1)
        keys = _slice_dim(
            keys.reshape((-1, num_kv_heads, head_dim)), 0, 0, span_k
        )
        values = _slice_dim(
            values.reshape((-1, num_kv_heads, head_dim)), 0, 0, span_k
        )
        query_tokens = _slice_dim(q, 0, starts[i], span_q)
        decoded = _decode_attention(query_tokens, keys, values, scale)
        if decoded is not None:
            outputs.append(decoded.cast(out_dtype))
            continue
        keys, values = keys.float32(), values.float32()
        if repeats > 1:
            keys = _repeat_interleave_dim(keys, 1, repeats)
            values = _repeat_interleave_dim(values, 1, repeats)
        queries = query_tokens.transpose(0, 1)
        scores = jt.matmul(
            queries, keys.transpose(0, 1).transpose(1, 2)) * scale
        if causal:
            scores = scores + _causal_mask(span_q, span_k) * (-1e30)
        weights = jt.nn.softmax(scores, dim=-1)
        packed = jt.matmul(weights, values.transpose(0, 1))
        outputs.append(packed.transpose(0, 1).cast(out_dtype))
    if len(outputs) == 1:
        return outputs[0]
    return jt.concat(outputs, dim=0)


def _causal_mask(span_q, span_k):
    """1.0 where a query position may not see a key position.

    The queries are the last ``span_q`` of ``span_k`` keys, so query ``j`` sees
    keys up to ``j + span_k - span_q``.
    """
    offset = span_k - span_q
    rows = jt.arange(span_q).reshape(span_q, 1)
    cols = jt.arange(span_k).reshape(1, span_k)
    return (cols > (rows + offset)).float32()


def _gather_cache_blocks(kv_cache, block_ids):
    from .kv_cache_acl import _gather_cache_blocks_acl

    selected = _gather_cache_blocks_acl(kv_cache, block_ids)
    if selected is not None:
        return selected
    return kv_cache[block_ids]


def _gather_block_table(block_table, request_count, block_count, request=None):
    from .kv_cache_acl import _gather_block_table_acl

    selected = _gather_block_table_acl(
        block_table, request_count, block_count, request=request
    )
    if selected is not None:
        return selected
    if request is None:
        return block_table[:request_count, :block_count]
    return block_table[request, :block_count].reshape((1, block_count))


def _split_cache_kv(cache, dim):
    from .kv_cache_acl import _split_cache_kv_acl

    selected = _split_cache_kv_acl(cache, dim)
    if selected is not None:
        return selected
    key_slices = [slice(None)] * cache.ndim
    value_slices = list(key_slices)
    key_slices[dim] = 0
    value_slices[dim] = 1
    return cache[tuple(key_slices)], cache[tuple(value_slices)]


def _slice_dim(value, dim, start, length):
    from .kv_cache_acl import _slice_dim_acl

    selected = _slice_dim_acl(value, dim, start, length)
    if selected is not None:
        return selected
    slices = [slice(None)] * value.ndim
    slices[dim] = slice(start, start + length)
    return value[tuple(slices)]


def _repeat_interleave_dim(value, dim, repeats):
    from .kv_cache_acl import _repeat_interleave_dim_acl

    expanded = _repeat_interleave_dim_acl(value, dim, repeats)
    if expanded is not None:
        return expanded
    return value.repeat_interleave(repeats, dim=dim)


def _decode_attention(query, key, value, scale):
    from .kv_cache_acl import _decode_attention_acl

    return _decode_attention_acl(query, key, value, scale)



__all__ = ["paged_attention", "reshape_and_cache"]
