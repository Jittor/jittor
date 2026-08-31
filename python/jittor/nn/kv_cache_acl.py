"""Private ACL tensor paths for paged key/value caches."""

import jittor as jt


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
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
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
        return kv_cache

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
    return kv_cache


def _gather_cache_blocks_acl(kv_cache, block_ids):
    if not all(isinstance(tensor, jt.Var) for tensor in (kv_cache, block_ids)):
        return None
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
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
    return jt.gather(kv_cache, 0, gather_index)


def _gather_block_table_acl(block_table, request_count, block_count, request=None):
    if not isinstance(block_table, jt.Var):
        return None
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
        return None
    if block_table.ndim != 2 or str(block_table.dtype) not in ("int32", "int64"):
        return None
    if request is None:
        row_ids = jt.index((request_count, block_count), dim=0, dtype="int32")
    else:
        row_ids = jt.full((1, block_count), int(request), dtype="int32")
    return jt.gather(block_table, 0, row_ids)


def _split_cache_kv_acl(cache, dim):
    if not isinstance(cache, jt.Var):
        return None
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
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
    return key, value


def _slice_dim_acl(value, dim, start, length):
    if not isinstance(value, jt.Var):
        return None
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
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
    return jt.gather(value, dim, index)


def _repeat_interleave_dim_acl(value, dim, repeats):
    if not isinstance(value, jt.Var):
        return None
    if not (_on_acl() and getattr(jt.flags, "no_grad", 0)):
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
    return value.reshape(reshaped).broadcast(expanded).reshape(result)


__all__ = []
