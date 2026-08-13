"""Public Var indexing behavior and its installation boundary."""

import numpy as np

import jittor as jt


def _is_torch_0d(value):
    return isinstance(value, jt.Var) and getattr(value, "_torch_0d", False)


def _mark_0d(value):
    try:
        value._torch_0d = True
    except Exception:
        pass
    return value


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
    """Apply Jittor indexing with the established Torch-compatible extensions."""

    if isinstance(slices, jt.Var) and slices.dtype == "uint8":
        slices = slices != 0
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

    if isinstance(slices, jt.Var) and slices.dtype == "uint8":
        slices = slices != 0
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
    return x.check_cascade_setitem(x.setitem(slices, value))


def install_var_indexing():
    """Install the native indexing layer before backend and Torch wrappers."""

    jt.Var.__getitem__ = getitem
    jt.Var.slice_var = getitem
    jt.Var.__setitem__ = setitem


__all__ = ["getitem", "install_var_indexing", "setitem"]
