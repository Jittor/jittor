"""Public Var indexing behavior and its installation boundary."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
from jittor_core import Var

from .._runtime.dispatch import dispatch_context, try_dispatch


_native_var_getitem = Var.getitem
_native_var_setitem = Var.setitem

#: `jittor` itself, bound on first use. It cannot be imported at module scope
#: -- jittor imports this module -- but the `import jittor as jt` statement the
#: functions below used instead is an IMPORT_NAME on every indexing operation,
#: and one `x[i]` reaches four of them.
_jt = None


def _jittor():
    global _jt
    import jittor
    _jt = jittor
    return jittor


_PYINT = (0).__class__


def _is_cascade_index(slices):
    if isinstance(slices, tuple) and len(slices) == 1:
        slices = slices[0]
    return _is_plain_int(slices)


def _is_plain_int(value):
    # `x[0]` -- the commonest index there is -- settles on the first line.
    if type(value) is _PYINT:
        return True
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


def _is_static_positive_basic_index(index):
    """The no-grad subset shared with the native frontend short path."""
    for item in index if type(index) is tuple else (index,):
        if item is None or item is Ellipsis or type(item) is _PYINT:
            continue
        if (isinstance(item, slice)
                and all(part is None or type(part) is _PYINT
                        for part in (item.start, item.stop, item.step))
                and (item.step is None or item.step > 0)):
            continue
        return False
    return True


def _native_bool_coordinates(slices):
    jt = _jt if _jt is not None else _jittor()
    if isinstance(slices, jt.Var) and _jittor_dtype_name(slices.dtype) == "bool":
        return tuple(slices.where())
    return slices


def _materialize_index_views(slices):
    """Make advanced-index buffers dense for native generated kernels."""
    import jittor as jt
    if isinstance(slices, jt.Var):
        if slices._storage_is_contiguous():
            return slices
        return jt.ops.contiguous(slices)
    if isinstance(slices, tuple):
        return tuple(_materialize_index_views(item) for item in slices)
    return slices


def var_getitem(x, slices, return_x=None):
    """Native getitem overloads with optional backend execution."""
    import jittor as jt
    if not _is_basic_index(slices) and not x._storage_is_contiguous():
        # Advanced indexing returns a copy, so densifying its source preserves
        # aliasing semantics while keeping generated gather kernels away from
        # storage offsets/strides they do not encode in their index expression.
        x = jt.ops.contiguous(x)
    # Integer views retain their native producer so chained assignment can
    # discover the held ancestor without a second Python view graph.
    if return_x is None and not _is_cascade_index(slices):
        result = try_dispatch("tensor.getitem", x, slices, return_x)
        if result is not None:
            return result
    slices = _materialize_index_views(_native_bool_coordinates(slices))
    if return_x is None:
        return _native_var_getitem(x, slices)
    return _native_var_getitem(x, slices, return_x)


def var_setitem(x, slices, value, reduce=None):
    """Return an updated Var; assignment belongs to the public write owner."""
    value = _acl_assignment_value(x, value, reduce)
    if reduce in (None, "void"):
        result = try_dispatch("tensor.setitem", x, slices, value, reduce)
        if result is not None:
            return result
    slices = _materialize_index_views(_native_bool_coordinates(slices))
    if reduce is None:
        return _native_var_setitem(x, slices, value)
    return _native_var_setitem(x, slices, value, reduce)


def _assignment_backend(x):
    # `dispatch_context(x).backend`. Every assignment asks it -- twice, once
    # from `setitem` and once from `var_setitem` -- and reads nothing else, so
    # the native context answers it without building the dtype names and the
    # tuple around it. A replaced `dispatch_context` is how a caller states the
    # device; it has no such shortcut and is asked as before.
    backend_of = getattr(dispatch_context, "_backend_of", None)
    if backend_of is not None:
        return backend_of(x)
    return dispatch_context(x).backend


def _acl_assignment_value(x, value, reduce=None):
    jt = _jt if _jt is not None else _jittor()
    if reduce not in (None, "void") or _assignment_backend(x) != "acl":
        return value
    if not isinstance(value, jt.Var):
        return jt.array(value, dtype=x.dtype).stop_grad()
    if value.dtype != x.dtype:
        return value.cast(x.dtype)
    return value


def _dispatch_slices(slices):
    jt = _jt if _jt is not None else _jittor()
    if isinstance(slices, range):
        return jt.array(list(slices))
    if isinstance(slices, tuple):
        return tuple(
            (item != 0) if isinstance(item, jt.Var) and _jittor_dtype_name(item.dtype) == "uint8"
            else jt.array(list(item)) if isinstance(item, range)
            else item
            for item in slices
        )
    return slices


def _maybe_constant_index_gather(x, slices):
    jt = _jt if _jt is not None else _jittor()
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


def _as_integer_array_index(value):
    import jittor as jt

    if isinstance(value, jt.Var):
        if _jittor_dtype_name(value.dtype) in ("int32", "int64"):
            return value
        return None
    if not isinstance(value, (list, np.ndarray)):
        return None
    array = np.asarray(value)
    if array.size == 0:
        array = array.astype(np.int64)
    elif array.dtype.kind not in ("i", "u"):
        return None
    return jt.array(array)


def _single_integer_array_index(x, index, axis=0):
    """Implement one integer-array index with all other axes unchanged."""
    import jittor as jt

    index = _as_integer_array_index(index)
    if x.ndim < 1 or index is None:
        return None
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        return None
    source_axes = (axis,) + tuple(i for i in range(x.ndim) if i != axis)
    source = x if axis == 0 else x.transpose(source_axes)
    if not source._storage_is_contiguous():
        source = jt.ops.contiguous(source)
    if not index._storage_is_contiguous():
        index = jt.ops.contiguous(index)
    index = jt.where(index < 0, index + int(x.shape[axis]), index)
    row_size = 1
    for size in source.shape[1:]:
        row_size *= int(size)
    flat_index = index.reshape((-1,))
    gather_index = flat_index.reshape((-1, 1)).broadcast(
        (int(flat_index.shape[0]), row_size)
    )
    if not gather_index._storage_is_contiguous():
        gather_index = jt.ops.contiguous(gather_index)
    gathered = jt.gather(
        source.reshape((int(source.shape[0]), row_size)), 0, gather_index
    )
    result = gathered.reshape(tuple(index.shape) + tuple(source.shape[1:]))
    index_rank = index.ndim
    source_positions = {
        original_axis: index_rank + position
        for position, original_axis in enumerate(source_axes[1:])
    }
    output_axes = []
    for original_axis in range(x.ndim):
        if original_axis == axis:
            output_axes.extend(range(index_rank))
        else:
            output_axes.append(source_positions[original_axis])
    if output_axes != list(range(result.ndim)):
        result = result.transpose(tuple(output_axes))
    return result


def _single_integer_array_tuple_index(x, slices):
    if (not isinstance(slices, tuple)
            or sum(item is Ellipsis for item in slices) > 1):
        return None
    consumed = sum(item is not None and item is not Ellipsis for item in slices)
    missing = x.ndim - consumed
    if missing < 0:
        return None
    expanded = []
    found_ellipsis = False
    for item in slices:
        if item is Ellipsis:
            expanded.extend([slice(None)] * missing)
            found_ellipsis = True
        else:
            expanded.append(item)
    if not found_ellipsis:
        expanded.extend([slice(None)] * missing)

    candidate = None
    axis = 0
    for item in expanded:
        if item is None:
            return None
        index = _as_integer_array_index(item)
        if index is not None:
            if candidate is not None:
                return None
            candidate = (axis, index)
        elif not (isinstance(item, slice)
                  and item.start is None and item.stop is None
                  and item.step is None):
            return None
        axis += 1
    if candidate is None:
        return None
    return _single_integer_array_index(x, candidate[1], candidate[0])


#: `_fast_getitem`'s answer when the index needs the full path.
_SLOW = object()
_NONE_TYPE = type(None)
_ELLIPSIS_TYPE = type(Ellipsis)
#: Index dtypes `_getitem_result` rewrites: a mask becomes its coordinates.
_MASK_DTYPES = ("bool", "uint8")


def _plain_slice(index):
    # The `slice` arm of `_is_basic_index`, restricted to exact ints.
    start, stop, step = index.start, index.stop, index.step
    return ((start is None or start.__class__ is _PYINT)
            and (stop is None or stop.__class__ is _PYINT)
            and (step is None or step.__class__ is _PYINT))


def _fast_getitem(x, slices):
    """``x[slices]`` for the index forms the full path passes through
    unchanged, or `_SLOW`.

    While no backend registers ``tensor.getitem`` and ``Var.getitem`` is still
    `var_getitem`, every rewrite in `_getitem_result` and `var_getitem` is a
    no-op for an exact int, a slice of exact ints, None, Ellipsis, a tuple of
    those and of index Vars, and a lone index Var; a lone bool mask becomes
    its ``where()`` coordinates, as the full path does. Index Vars here are
    neither bool nor uint8 and carry no constant-index value. Every other form
    (uint8 masks, ranges, lists, numpy scalars) takes the full path, so the
    result and the recorded view are the same either way. This matters
    because ``x[...]`` is the commonest Var operation: the full path costs
    several microseconds of Python per call on top of the native getitem.
    """
    if x.__class__.getitem is not var_getitem:
        return _SLOW
    cls = slices.__class__
    if (cls is _PYINT or cls is _NONE_TYPE or cls is _ELLIPSIS_TYPE
            or (cls is slice and _plain_slice(slices))):
        basic = True
    elif cls is tuple:
        basic = True
        for item in slices:
            item_cls = item.__class__
            if item_cls is _PYINT or item_cls is _NONE_TYPE or item_cls is _ELLIPSIS_TYPE:
                continue
            if item_cls is slice:
                if _plain_slice(item):
                    continue
                return _SLOW
            if isinstance(item, Var) and _jittor_dtype_name(item.dtype) not in _MASK_DTYPES:
                basic = False
                continue
            return _SLOW
    elif isinstance(slices, Var):
        name = _jittor_dtype_name(slices.dtype)
        if name == "bool":
            return _native_var_getitem(x, slices.where())
        if name == "uint8" or (slices.ndim == 1 and x.ndim >= 1 and getattr(
                slices, "_jittor_constant_index_value", None) is not None):
            return _SLOW
        return _native_var_getitem(x, slices)
    else:
        return _SLOW
    out = _native_var_getitem(x, slices)
    if basic and isinstance(out, Var):
        out._set_view_of(x, slices)
    return out

def getitem(x, slices):
    """Apply Jittor indexing, recording a view when the index is a basic one."""
    # The fast path answers only what `try_dispatch` would not have: with the
    # native dispatcher and nothing registered for `tensor.getitem`. A
    # replaced `try_dispatch` has no `_is_registered` and is always consulted.
    is_registered = getattr(try_dispatch, "_is_registered", None)
    if is_registered is not None and not is_registered("tensor.getitem"):
        # A basic index of a plain Var is answered natively in one call --
        # the getitem op and the view record `_fast_getitem` would make --
        # and so, while `Var.getitem` is still `var_getitem`, are the index
        # Var forms `_fast_getitem` passes through (a bool mask as its
        # `where()` coordinates). It declines (NotImplemented) for anything
        # else, including a subclass, which may own `getitem` itself.
        native = getattr(try_dispatch, "_native_basic_getitem", None)
        if native is not None and x.__class__ is Var:
            out = native(x, slices, Var.getitem is var_getitem)
            if out is not NotImplemented:
                return out
        out = _fast_getitem(x, slices)
        if out is not _SLOW:
            return out
    jt = _jt if _jt is not None else _jittor()

    out = _getitem_result(x, slices)
    if isinstance(out, jt.Var) and _is_basic_index(slices) and not out._is_view():
        # A backend may return an already recorded view of a different base.
        # Keep that owner, as the Torch wrapper does, for later writes.
        out._set_view_of(x, slices)
    return out


def _getitem_result(x, slices):
    """Apply Jittor indexing with the established Torch-compatible extensions."""
    jt = _jt if _jt is not None else _jittor()

    if not _is_basic_index(slices) and not x._storage_is_contiguous():
        x = jt.ops.contiguous(x)
    if isinstance(slices, jt.Var) and _jittor_dtype_name(slices.dtype) == "uint8":
        slices = slices != 0
    slices = _dispatch_slices(slices)
    if not _is_cascade_index(slices):
        result = try_dispatch("tensor.getitem", x, slices, None)
        if result is not None:
            return result
        if jt.flags.no_grad and _is_static_positive_basic_index(slices):
            # Selection has already happened. Calling x.getitem here retries
            # a declining provider, including any callback side effects.
            return _native_var_getitem(x, slices)
    if isinstance(slices, jt.Var) and _jittor_dtype_name(slices.dtype) == "bool":
        return getitem(x, slices.where())
    if isinstance(slices, range):
        slices = jt.array(list(slices))
    integer_result = _single_integer_array_index(x, slices)
    if integer_result is not None:
        return integer_result

    constant_gather = _maybe_constant_index_gather(x, slices)
    if constant_gather is not None:
        return constant_gather

    if isinstance(slices, tuple):
        normalized = []
        for item in slices:
            if isinstance(item, jt.Var) and _jittor_dtype_name(item.dtype) == "uint8":
                normalized.extend((item != 0).where())
            elif isinstance(item, jt.Var) and _jittor_dtype_name(item.dtype) == "bool":
                normalized.extend(item.where())
            elif isinstance(item, range):
                normalized.append(jt.array(list(item)))
            else:
                normalized.append(item)
        slices = tuple(normalized)
        integer_result = _single_integer_array_tuple_index(x, slices)
        if integer_result is not None:
            return integer_result
    return x.getitem(slices)


def setitem(x, slices, value):
    """Apply Jittor assignment with the established mask and complex rules."""
    # The common assignments -- a plain Var, an index of ints, slices and
    # index Vars or a bool mask, a Var or Python number value -- are built
    # natively, op for op what the rest of this function and `var_setitem`
    # build; it declines (NotImplemented) for everything else. Only with
    # nothing registered for `tensor.setitem`, `Var.setitem` still
    # `var_setitem`, and the native `dispatch_context` answering the ACL
    # question below.
    is_registered = getattr(try_dispatch, "_is_registered", None)
    if is_registered is not None and not is_registered("tensor.setitem"):
        native = getattr(try_dispatch, "_native_setitem", None)
        if (native is not None and x.__class__ is Var and Var.setitem is var_setitem
                and getattr(dispatch_context, "_backend_of", None) is not None):
            out = native(x, slices, value)
            if out is not NotImplemented:
                return out
    jt = _jt if _jt is not None else _jittor()
    from .._core.var import _factory_scope_like

    if _jittor_dtype_name(x.dtype) == "complex64" and isinstance(value, (complex, np.complexfloating)):
        # 0-d, so it fits a single indexed element as well as a slice; a
        # [1]-shaped value tripped setitem's ``data_dim <= out dims`` check
        # on ``z[i, j] = 1 + 2j``.
        value = jt.array(np.asarray(value, dtype=np.complex64))
    value = _acl_assignment_value(x, value)

    if isinstance(slices, jt.Var) and _jittor_dtype_name(slices.dtype) == "uint8":
        slices = slices != 0
    slices = _dispatch_slices(slices)
    result = try_dispatch("tensor.setitem", x, slices, value, None)
    if result is not None:
        # assign handles recorded views as well as ordinary tensor holders.
        return x.assign(result)
    if isinstance(slices, jt.Var) and _jittor_dtype_name(slices.dtype) == "bool":
        if slices.shape == x.shape:
            if isinstance(value, (int, float)):
                with _factory_scope_like(x):
                    value = jt.array(value, dtype=x.dtype).broadcast(x.shape)
                return x.assign(slices.ternary(value, x))
            if isinstance(value, jt.Var) and value.shape == [1]:
                value = jt.broadcast(value, x.shape)
                return x.assign(slices.ternary(value, x))
        slices = slices.where()
    elif isinstance(slices, tuple):
        normalized = []
        for item in slices:
            if isinstance(item, jt.Var) and _jittor_dtype_name(item.dtype) == "uint8":
                normalized.extend((item != 0).where())
            elif isinstance(item, jt.Var) and _jittor_dtype_name(item.dtype) == "bool":
                normalized.extend(item.where())
            else:
                normalized.append(item)
        slices = tuple(normalized)
    result = x.setitem(slices, value)
    return x.assign(result)


def install_var_indexing():
    """Install the native indexing layer before backend and Torch wrappers."""
    jt = _jt if _jt is not None else _jittor()

    jt.Var.getitem = var_getitem
    jt.Var.setitem = var_setitem
    jt.Var.__getitem__ = getitem
    jt.Var.slice_var = getitem
    jt.Var.__setitem__ = setitem


__all__ = ["getitem", "install_var_indexing", "setitem", "var_getitem", "var_setitem"]
