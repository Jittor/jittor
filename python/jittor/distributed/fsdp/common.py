"""Native FSDP tensor factories and WORLD collective routing."""

from types import SimpleNamespace

import numpy as np
import jittor as jt
from jittor._core.var import placement_scope_like as _factory_scope_like


class StateRecord(SimpleNamespace):
    """Weak-referenceable metadata held by its sharded module."""


def _slice_flat(value, start, size):
    return value[int(start) : int(start) + int(size)]


def _all_gather_shards(value, group=None):
    from jittor import distributed as dist

    return dist.all_gather(value, group=group)


def _reduce_scatter_padded(value, group=None):
    from jittor import distributed as dist

    return dist.reduce_scatter(value, group=group)


def _flatten_var(v):
    return v.reshape((-1,))


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _pad_flat(flat, padded_numel):
    n = int(flat.numel()) if callable(getattr(flat, "numel", None)) else int(np.prod(flat.shape))
    if n == int(padded_numel):
        return flat
    pad = _zeros_like_shape(flat, (int(padded_numel) - n,))
    return jt.concat([flat, pad], dim=0)


def _zeros_like_shape(reference, shape, dtype=None):
    with _factory_scope_like(reference):
        return jt.zeros(shape, dtype=reference.dtype if dtype is None else dtype)


def _param_numel(v):
    return int(np.prod(tuple(int(x) for x in v.shape)))


def _value_requires_grad(value):
    if isinstance(value, jt.Var):
        try:
            return bool(value.requires_grad)
        except (AttributeError, TypeError):
            return not value.is_stop_grad()
    if isinstance(value, dict):
        return any(_value_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_value_requires_grad(item) for item in value)
    return False


def _primary_input_requires_grad(args, kwargs):
    """Whether a module must retain its forward graph for an input gradient."""
    return any(_value_requires_grad(value) for value in args) or any(
        _value_requires_grad(value) for value in kwargs.values()
    )


def _materialize_frozen_output(value):
    """Sever a completed frozen forward graph while preserving its structure."""
    if isinstance(value, jt.Var):
        return jt.Var.copy(value).stop_grad()
    if isinstance(value, tuple):
        values = tuple(_materialize_frozen_output(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*values)
        if type(value) is tuple:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    if isinstance(value, list):
        values = [_materialize_frozen_output(item) for item in value]
        if type(value) is list:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    if isinstance(value, dict):
        values = {key: _materialize_frozen_output(item) for key, item in value.items()}
        if type(value) is dict:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    return value


def _tensor_values(value):
    if isinstance(value, jt.Var):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_values(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_values(item)]
    return []
