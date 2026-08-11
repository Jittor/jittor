"""Packed variable-length attention utilities."""

from collections import OrderedDict
from threading import RLock

import jittor as jt


_CU_SEQLENS_CACHE = OrderedDict()
_CU_SEQLENS_CACHE_LOCK = RLock()
_CU_SEQLENS_CACHE_LIMIT = 128


class _FactoryIdentity:
    __slots__ = ("factory",)

    def __init__(self, factory):
        self.factory = factory

    def __hash__(self):
        return id(self.factory)

    def __eq__(self, other):
        return isinstance(other, _FactoryIdentity) and self.factory is other.factory


def sequence_lengths(layout):
    """Convert a sequence of slices or ``(start, stop)`` pairs to lengths."""
    lengths = []
    for item in layout:
        if isinstance(item, slice):
            if item.step not in (None, 1) or item.stop is None:
                raise ValueError("layout slices must be contiguous with a stop")
            start = 0 if item.start is None else int(item.start)
            stop = int(item.stop)
        else:
            try:
                start, stop = item
                start, stop = int(start), int(stop)
            except Exception:
                raise TypeError("layout entries must be slices or (start, stop) pairs")
        if start < 0 or stop < start:
            raise ValueError("layout entries must have 0 <= start <= stop")
        lengths.append(stop - start)
    return tuple(lengths)


def cumulative_sequence_lengths(lengths, device=None, tensor_factory=None):
    """Return a bounded-cache int32 cumulative-length tensor.

    ``tensor_factory`` can be supplied by a compatibility layer whose tensor
    constructor accepts an explicit device. The default creates a native Jittor
    tensor under the active device flag.
    """
    normalized = tuple(int(length) for length in lengths)
    if any(length < 0 for length in normalized):
        raise ValueError("sequence lengths must be non-negative")
    device_key = str(device if device is not None else _active_device_key())
    factory_key = None if tensor_factory is None else _FactoryIdentity(tensor_factory)
    key = (device_key, normalized, factory_key)
    with _CU_SEQLENS_CACHE_LOCK:
        cached = _CU_SEQLENS_CACHE.get(key)
        if cached is not None:
            _CU_SEQLENS_CACHE.move_to_end(key)
            return cached

    values = [0]
    for length in normalized:
        values.append(values[-1] + length)
    if tensor_factory is None:
        result = jt.array(values, dtype="int32")
    else:
        result = tensor_factory(values, dtype="int32", device=device)
    with _CU_SEQLENS_CACHE_LOCK:
        existing = _CU_SEQLENS_CACHE.get(key)
        if existing is not None:
            _CU_SEQLENS_CACHE.move_to_end(key)
            return existing
        _CU_SEQLENS_CACHE[key] = result
        while len(_CU_SEQLENS_CACHE) > _CU_SEQLENS_CACHE_LIMIT:
            _CU_SEQLENS_CACHE.popitem(last=False)
    return result


def _active_device_key():
    return "cuda" if jt.flags.use_cuda else "cpu"


def _tensor_device(value):
    try:
        return value.device
    except Exception:
        try:
            return int(value.get_device())
        except Exception:
            return _active_device_key()


def _prepare_varlen(value, lengths, tail_rank, tensor_factory):
    shape = tuple(int(size) for size in value.shape)
    if lengths is None:
        if len(shape) != tail_rank + 2:
            raise ValueError("dense attention inputs must include batch and sequence axes")
        batch, sequence = shape[:2]
        normalized = (sequence,) * batch
        flat = value.reshape((batch * sequence,) + shape[2:])
        restore_shape = (batch, sequence)
    else:
        normalized = tuple(int(length) for length in lengths)
        if len(shape) != tail_rank + 1:
            raise ValueError("packed attention inputs must have one leading token axis")
        if sum(normalized) != shape[0]:
            raise ValueError("packed token count does not match sequence lengths")
        flat = value
        restore_shape = None
    maximum = max(normalized) if normalized else 0
    cu = cumulative_sequence_lengths(
        normalized, device=_tensor_device(value), tensor_factory=tensor_factory
    )
    return flat, normalized, maximum, cu, restore_shape


def _restore_dense(value, shape):
    if shape is None:
        return value
    return value.reshape(shape + tuple(int(size) for size in value.shape[1:]))


def varlen_scaled_dot_product_attention(
    q,
    k=None,
    v=None,
    *,
    q_lengths=None,
    kv_lengths=None,
    varlen_func=None,
    qkvpacked_func=None,
    kvpacked_func=None,
    tensor_factory=None,
):
    """Dispatch packed or dense inputs through FlashAttention varlen callables.

    One input is QKV-packed, two inputs are Q plus KV-packed, and three inputs
    are separate Q/K/V. Packed inputs provide explicit sequence lengths; dense
    inputs omit them and are flattened/restored around the backend call.
    """
    if k is None and v is not None:
        raise ValueError("v cannot be provided without k")
    if k is None:
        if qkvpacked_func is None:
            raise ValueError("qkvpacked_func is required for packed QKV")
        q_flat, _, max_q, cu_q, restore = _prepare_varlen(q, q_lengths, 3, tensor_factory)
        out = qkvpacked_func(q_flat, cu_q, max_q)
        return _restore_dense(out, restore)

    q_flat, q_normalized, max_q, cu_q, restore = _prepare_varlen(q, q_lengths, 2, tensor_factory)
    if v is None:
        if kvpacked_func is None:
            raise ValueError("kvpacked_func is required for packed KV")
        k_flat, kv_normalized, max_kv, cu_kv, _ = _prepare_varlen(k, kv_lengths, 3, tensor_factory)
        if len(q_normalized) != len(kv_normalized):
            raise ValueError("q_lengths and kv_lengths must describe the same number of sequences")
        if q_normalized == kv_normalized:
            cu_kv = cu_q
        out = kvpacked_func(q_flat, k_flat, cu_q, cu_kv, max_q, max_kv)
        return _restore_dense(out, restore)

    if varlen_func is None:
        raise ValueError("varlen_func is required for separate Q/K/V")
    k_flat, kv_normalized, max_kv, cu_kv, _ = _prepare_varlen(k, kv_lengths, 2, tensor_factory)
    if len(q_normalized) != len(kv_normalized):
        raise ValueError("q_lengths and kv_lengths must describe the same number of sequences")
    v_flat, v_normalized, _, _, _ = _prepare_varlen(v, kv_lengths, 2, tensor_factory)
    if kv_normalized != v_normalized:
        raise ValueError("k and v sequence lengths must match")
    if q_normalized == kv_normalized:
        cu_kv = cu_q
    out = varlen_func(q_flat, k_flat, v_flat, cu_q, cu_kv, max_q, max_kv)
    return _restore_dense(out, restore)


__all__ = [
    "cumulative_sequence_lengths",
    "sequence_lengths",
    "varlen_scaled_dot_product_attention",
]
