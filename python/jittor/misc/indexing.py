"""Public Var indexing behavior and its installation boundary."""

import numpy as np

import jittor as jt
from .._runtime.dispatch import dispatch_context, try_dispatch


_native_var_getitem = jt.Var.getitem
_native_var_setitem = jt.Var.setitem


def _is_cascade_index(slices):
    if isinstance(slices, tuple) and len(slices) == 1:
        slices = slices[0]
    return _is_plain_int(slices)


def _is_plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_basic_index(index):
    """Whether ``x[index]`` is a view of ``x`` rather than a gather from it.

    Torch's rule, and the same one the storage model needs: basic indexing
    selects a sub-range that keeps naming the same elements, so a write through
    the result belongs to ``x``; advanced indexing collects elements into a new
    tensor, so a write to it does not.
    """
    if isinstance(index, tuple):
        return all(_is_basic_index(item) for item in index)
    if index is None or index is Ellipsis:
        return True
    if isinstance(index, slice):
        return all(part is None or _is_plain_int(part)
                   for part in (index.start, index.stop, index.step))
    return _is_plain_int(index)


def _native_bool_coordinates(slices):
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        return tuple(slices.where())
    return slices


def var_getitem(x, slices, return_x=None):
    """Native getitem overloads with optional backend execution."""
    # Integer views retain their native producer so chained assignment can
    # discover the held ancestor without a second Python view graph.
    if return_x is None and not _is_cascade_index(slices):
        result = try_dispatch("tensor.getitem", x, slices, return_x)
        if result is not None:
            return result
    slices = _native_bool_coordinates(slices)
    if return_x is None:
        return _native_var_getitem(x, slices)
    return _native_var_getitem(x, slices, return_x)


def var_setitem(x, slices, value, reduce=None):
    """Return an updated Var; assignment belongs to the public write owner."""
    value = _acl_assignment_value(x, value, reduce)
    if reduce in (None, "void") and not x._needs_cascade_setitem():
        result = try_dispatch("tensor.setitem", x, slices, value, reduce)
        if result is not None:
            return result
    slices = _native_bool_coordinates(slices)
    if reduce is None:
        return _native_var_setitem(x, slices, value)
    return _native_var_setitem(x, slices, value, reduce)


def _acl_assignment_value(x, value, reduce=None):
    if reduce not in (None, "void") or dispatch_context(x).backend != "acl":
        return value
    if not isinstance(value, jt.Var):
        return jt.array(value, dtype=x.dtype).stop_grad()
    if value.dtype != x.dtype:
        return value.cast(x.dtype)
    return value


def _is_torch_0d(value):
    return isinstance(value, jt.Var) and getattr(value, "_torch_0d", False)


def _mark_0d(value):
    try:
        value._torch_0d = True
    except Exception:
        pass
    return value


def _dispatch_slices(slices):
    if isinstance(slices, range):
        return jt.array(list(slices))
    if isinstance(slices, tuple):
        return tuple(
            (item != 0) if isinstance(item, jt.Var) and item.dtype == "uint8"
            else jt.array(list(item)) if isinstance(item, range)
            else int(item.item()) if _is_torch_0d(item)
            else item
            for item in slices
        )
    if _is_torch_0d(slices):
        return int(slices.item())
    return slices


def _maybe_constant_index_gather(x, slices):
    if not isinstance(slices, jt.Var) or slices.ndim != 1 or x.ndim < 1:
        return None
    const_value = getattr(slices, "_jittor_constant_index_value", None)
    if const_value is None:
        return None
    index_length = int(slices.shape[0])
    dim_zero = int(x.shape[0])
    index = int(const_value)
    if index < 0:
        index += dim_zero
    if index < 0 or index >= dim_zero:
        return None
    output_shape = [index_length] + list(x.shape[1:])
    base = x.getitem(
        (slice(index, index + 1),) + (slice(None),) * (x.ndim - 1)
    )
    return base.broadcast(output_shape)


def getitem(x, slices):
    """Apply Jittor indexing, recording a view when the index is a basic one."""

    out = _getitem_result(x, slices)
    if isinstance(out, jt.Var) and _is_basic_index(slices):
        out._set_view_of(x, slices)
    return out


def _getitem_result(x, slices):
    """Apply Jittor indexing with the established Torch-compatible extensions."""

    if isinstance(slices, jt.Var) and slices.dtype == "uint8":
        slices = slices != 0
    slices = _dispatch_slices(slices)
    if not _is_cascade_index(slices):
        result = try_dispatch("tensor.getitem", x, slices, None)
        if result is not None:
            return result
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        return getitem(x, slices.where())
    if isinstance(slices, range):
        slices = jt.array(list(slices))

    constant_gather = _maybe_constant_index_gather(x, slices)
    if constant_gather is not None:
        return constant_gather

    if (
        isinstance(slices, int)
        and not isinstance(slices, bool)
        and x.ndim == 1
    ):
        return _mark_0d(x.getitem(slices))

    if isinstance(slices, tuple):
        normalized = []
        for item in slices:
            if isinstance(item, jt.Var) and item.dtype == "uint8":
                normalized.extend((item != 0).where())
            elif isinstance(item, jt.Var) and item.dtype == "bool":
                normalized.extend(item.where())
            elif isinstance(item, range):
                normalized.append(jt.array(list(item)))
            elif _is_torch_0d(item):
                normalized.append(int(item.item()))
            else:
                normalized.append(item)
        slices = tuple(normalized)
    elif _is_torch_0d(slices):
        slices = int(slices.item())
    return x.getitem(slices)


def setitem(x, slices, value):
    """Apply Jittor assignment with the established mask and complex rules."""

    if x.dtype == "complex64" and isinstance(value, (complex, np.complexfloating)):
        value = jt.array(np.asarray([value], dtype=np.complex64))
    value = _acl_assignment_value(x, value)

    if isinstance(slices, jt.Var) and slices.dtype == "uint8":
        slices = slices != 0
    slices = _dispatch_slices(slices)
    needs_cascade = x._needs_cascade_setitem()
    if not needs_cascade:
        result = try_dispatch("tensor.setitem", x, slices, value, None)
        if result is not None:
            return x.assign(result)
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        if slices.shape == x.shape:
            if isinstance(value, (int, float)):
                value = jt.array(value).broadcast(x.shape)
                return x.assign(slices.ternary(value, x))
            if isinstance(value, jt.Var) and value.shape == [1]:
                value = jt.broadcast(value, x.shape)
                return x.assign(slices.ternary(value, x))
        slices = slices.where()
    elif isinstance(slices, tuple):
        normalized = []
        for item in slices:
            if isinstance(item, jt.Var) and item.dtype == "uint8":
                normalized.extend((item != 0).where())
            elif isinstance(item, jt.Var) and item.dtype == "bool":
                normalized.extend(item.where())
            else:
                normalized.append(item)
        slices = tuple(normalized)
    result = x.setitem(slices, value)
    if x._is_view():
        # A recorded view needs no ancestry walk: `assign` writes through it, at
        # any depth and for any basic index, where `check_cascade_setitem` could
        # only rewrite chains of at most ten single integers. The dispatch
        # decision above is deliberately left on `_needs_cascade_setitem` alone,
        # so which results a backend is allowed to produce does not change here.
        return x.assign(result)
    return x.check_cascade_setitem(result) if needs_cascade else x.assign(result)


def install_var_indexing():
    """Install the native indexing layer before backend and Torch wrappers."""

    jt.Var.getitem = var_getitem
    jt.Var.setitem = var_setitem
    jt.Var.__getitem__ = getitem
    jt.Var.slice_var = getitem
    jt.Var.__setitem__ = setitem


__all__ = ["getitem", "install_var_indexing", "setitem", "var_getitem", "var_setitem"]
