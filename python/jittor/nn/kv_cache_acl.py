"""Private ACL tensor paths for paged key/value caches."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs


def _on_acl():
    return bool(
        getattr(jt.compiler, "has_acl", 0) and getattr(jt.flags, "use_acl", 0) and jt.flags.use_cuda
    )


def _gather_rows(value, row_ids):
    rows = len(row_ids)
    index = jt.array(row_ids, dtype="int32").reshape((rows,) + (1,) * (value.ndim - 1))
    index = index.broadcast((rows,) + tuple(value.shape[1:]))
    return jt.gather(value, 0, index)


def _reshape_and_cache_acl(key, value, kv_cache, slot_mapping, slots=None):
    tensors = (key, value, kv_cache, slot_mapping)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if not (_on_acl() and not _output_requires_grad(tensors)):
        return None

    key_shape = tuple(int(size) for size in key.shape)
    value_shape = tuple(int(size) for size in value.shape)
    cache_shape = tuple(int(size) for size in kv_cache.shape)
    if len(key_shape) != 3 or value_shape != key_shape or len(cache_shape) != 5:
        return None
    if cache_shape[1] != 2 or cache_shape[3:] != key_shape[1:]:
        return None
    if int(slot_mapping.numel()) < key_shape[0]:
        return None
    value_dtypes = {str(tensor.dtype) for tensor in (key, value, kv_cache)}
    if len(value_dtypes) != 1 or value_dtypes.pop() not in (
        "float16",
        "bfloat16",
        "float32",
    ):
        return None
    if str(slot_mapping.dtype) not in ("int32", "int64"):
        return None

    token_count = key_shape[0]
    if slots is None:
        slots = slot_mapping[:token_count].numpy().tolist()
    slots = [int(slot) for slot in slots[:token_count]]
    capacity = cache_shape[0] * cache_shape[2]
    valid_tokens = [token for token, slot in enumerate(slots) if 0 <= slot < capacity]
    if not valid_tokens:
        return _stop_grad_outputs(kv_cache)

    if token_count <= 16 and len(valid_tokens) == token_count:
        from jittor.extern.acl.aclops.flashattention_op import KVCacheMemcpyACL
        KVCacheMemcpyACL(cache_shape[2], slots)(key, value, kv_cache)
        return _stop_grad_outputs(kv_cache)

    source = jt.stack((key, value), dim=1)
    if len(valid_tokens) != token_count:
        source = _gather_rows(source, valid_tokens)
        valid_slots = [slots[token] for token in valid_tokens]
        slot_ids = jt.array(valid_slots, dtype="int32")
    else:
        slot_ids = slot_mapping[:token_count]

    row_width = 2 * key_shape[1] * key_shape[2]
    flat_cache = kv_cache.transpose(0, 2, 1, 3, 4).reshape((capacity, row_width))
    source = source.reshape((len(valid_tokens), row_width))
    scatter_index = slot_ids.reshape((-1, 1)).broadcast(source.shape)
    updated = jt.scatter(flat_cache, 0, scatter_index, source)
    updated = updated.reshape(
        (cache_shape[0], cache_shape[2], 2, key_shape[1], key_shape[2])
    ).transpose(0, 2, 1, 3, 4)
    kv_cache.update(updated)
    return _stop_grad_outputs(kv_cache)


def _gather_cache_blocks_acl(kv_cache, block_ids):
    if not all(isinstance(tensor, jt.Var) for tensor in (kv_cache, block_ids)):
        return None
    if not (_on_acl() and not _output_requires_grad(kv_cache, block_ids)):
        return None
    cache_shape = tuple(int(size) for size in kv_cache.shape)
    if len(cache_shape) != 5 or cache_shape[1] != 2:
        return None
    if str(block_ids.dtype) not in ("int32", "int64"):
        return None
    block_count = int(block_ids.numel())
    index_shape = (block_count,) + cache_shape[1:]
    gather_index = block_ids.reshape((block_count,) + (1,) * (len(cache_shape) - 1)).broadcast(
        index_shape
    )
    return _stop_grad_outputs(jt.gather(kv_cache, 0, gather_index))


def _gather_block_table_acl(block_table, request_count, block_count, request=None):
    if not isinstance(block_table, jt.Var):
        return None
    if not (_on_acl() and not _output_requires_grad(block_table)):
        return None
    if block_table.ndim != 2 or str(block_table.dtype) not in ("int32", "int64"):
        return None
    if request is None:
        row_ids = jt.index((request_count, block_count), dim=0, dtype="int32")
    else:
        row_ids = jt.full((1, block_count), int(request), dtype="int32")
    return _stop_grad_outputs(jt.gather(block_table, 0, row_ids))


def _split_cache_kv_acl(cache, dim):
    if not isinstance(cache, jt.Var):
        return None
    if not (_on_acl() and not _output_requires_grad(cache)):
        return None
    shape = list(int(size) for size in cache.shape)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape) or shape[dim] != 2:
        return None
    shape[dim] = 1
    key_index = jt.zeros(shape, dtype="int32")
    value_index = jt.ones(shape, dtype="int32")
    result_shape = tuple(shape[:dim] + shape[dim + 1 :])
    key = jt.gather(cache, dim, key_index).reshape(result_shape)
    value = jt.gather(cache, dim, value_index).reshape(result_shape)
    return _stop_grad_outputs((key, value))


def _slice_dim_acl(value, dim, start, length):
    if not isinstance(value, jt.Var):
        return None
    if not (_on_acl() and not _output_requires_grad(value)):
        return None
    shape = list(int(size) for size in value.shape)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        return None
    start = int(start)
    length = int(length)
    if start < 0 or length < 0 or start + length > shape[dim]:
        return None
    shape[dim] = length
    index = jt.index(tuple(shape), dim=dim, dtype="int32")
    if start:
        index = index + start
    return _stop_grad_outputs(jt.gather(value, dim, index))


def _repeat_interleave_dim_acl(value, dim, repeats):
    if not isinstance(value, jt.Var):
        return None
    if not (_on_acl() and not _output_requires_grad(value)):
        return None
    shape = list(int(size) for size in value.shape)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape) or repeats <= 0:
        return None
    reshaped = shape[: dim + 1] + [1] + shape[dim + 1 :]
    expanded = list(reshaped)
    expanded[dim + 1] = int(repeats)
    result = list(shape)
    result[dim] *= int(repeats)
    return _stop_grad_outputs(
        value.reshape(reshaped).broadcast(expanded).reshape(result))


def _decode_attention_acl(query, key, value, scale):
    if not all(isinstance(tensor, jt.Var) for tensor in (query, key, value)):
        return None
    if not (_on_acl() and not _output_requires_grad(query, key, value)):
        return None
    query_shape = tuple(int(size) for size in query.shape)
    key_shape = tuple(int(size) for size in key.shape)
    value_shape = tuple(int(size) for size in value.shape)
    if (
        len(query_shape) != 3
        or len(key_shape) != 3
        or value_shape != key_shape
        or query_shape[0] != 1
        or query_shape[-1] != key_shape[-1]
        or query_shape[1] % key_shape[1] != 0
    ):
        return None
    if str(key.dtype) not in ("bfloat16", "float32") or str(value.dtype) != str(
        key.dtype
    ):
        return None

    from jittor.extern.acl.aclops.flashattention_op import (
        scaled_dot_product_attention_acl,
    )

    output_dtype = query.dtype
    query = query.cast(key.dtype).transpose(0, 1).reshape(
        (1, query_shape[1], 1, query_shape[2])
    )
    key = key.transpose(0, 1).reshape(
        (1, key_shape[1], key_shape[0], key_shape[2])
    )
    value = value.transpose(0, 1).reshape(
        (1, value_shape[1], value_shape[0], value_shape[2])
    )
    output = scaled_dot_product_attention_acl(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=query_shape[1] != key_shape[1],
    )
    if output is None:
        return None
    return _stop_grad_outputs(
        output.reshape((query_shape[1], query_shape[2])).reshape(
            query_shape).cast(output_dtype))


def _paged_attention_decode_acl(query, kv_cache, block_table, scale,
                                key_lengths=None):
    tensors = (query, kv_cache, block_table)
    if not all(isinstance(tensor, jt.Var) for tensor in tensors):
        return None
    if not (_on_acl() and not _output_requires_grad(tensors)):
        return None
    if key_lengths is None:
        return None

    query_shape = tuple(int(size) for size in query.shape)
    cache_shape = tuple(int(size) for size in kv_cache.shape)
    table_shape = tuple(int(size) for size in block_table.shape)
    lengths = [int(length) for length in key_lengths]
    batch = len(lengths)
    if (
        batch != 1
        or len(query_shape) != 3
        or query_shape[0] != batch
        or len(cache_shape) != 5
        or cache_shape[1] != 2
        or cache_shape[2] != 128
        or len(table_shape) != 2
        or table_shape[0] != batch
        or str(kv_cache.dtype) != "bfloat16"
        or str(block_table.dtype) != "int32"
        or query_shape[2] != cache_shape[4]
        or query_shape[1] % cache_shape[3] != 0
    ):
        return None
    if any(
        length <= 0 or -(-length // cache_shape[2]) > table_shape[1]
        for length in lengths
    ):
        return None

    from jittor.extern.acl.aclops.flashattention_op import (
        PagedIncreFlashAttentionACL,
    )

    output_dtype = query.dtype
    packed_query = query.cast(kv_cache.dtype).reshape(
        (batch, query_shape[1], 1, query_shape[2])
    )
    output = PagedIncreFlashAttentionACL(
        query_shape[1], cache_shape[3], cache_shape[2], scale, lengths
    )(packed_query, kv_cache, block_table)
    _paged_attention_decode_acl.backend_name = \
        "acl_paged_incre_flash_attention_v4"
    return _stop_grad_outputs(output.reshape(query_shape).cast(output_dtype))


_paged_attention_decode_acl.backend_name = None


__all__ = []
