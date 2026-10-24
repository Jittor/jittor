"""Torch numerical indexing operations."""

def gather(input, dim, index, *, sparse_grad=False, out=None):
    """Adapt Torch names and frontend scope; native gather owns math/VJP."""
    from . import jt, _native_gather
    from ...context import get_install_context
    from ...frontend import tensor_frontend

    if sparse_grad:
        raise NotImplementedError("gather sparse gradients are not implemented")
    if out is not None:
        raise NotImplementedError("gather(out=...) is not implemented")
    tensor_type = get_install_context(jt).state["Var"]
    with tensor_frontend(tensor_type, like=input, default_placement=False):
        return _native_gather(input, dim, index)


def _tensor_gather(self, dim, index, *, sparse_grad=False):
    return gather(self, dim, index, sparse_grad=sparse_grad)


def take_along_dim(input, indices, dim=None):
    """Gather values after broadcasting indices outside the gather dimension."""
    from . import (
        jt,
    )
    if dim is None:
        return jt.gather(input.reshape(-1), 0, indices.reshape(-1))
    d = dim % input.ndim
    target = list(input.shape)
    target[d] = indices.shape[d]
    if list(indices.shape) != target:
        indices = jt.broadcast(indices, target)
    return jt.gather(input, d, indices)


def masked_fill(input, mask, value, *, out=None):
    """Torch argument/frontend boundary; selection math stays native."""
    from . import jt, _native_masked_fill, _jittor_dtype_name
    from ...context import get_install_context
    from ...frontend import tensor_frontend

    if out is not None:
        raise NotImplementedError("masked_fill(out=...) is not implemented")
    if not jt.core._is_tensor_carrier(input) or not jt.core._is_tensor_carrier(mask):
        raise TypeError("masked_fill expects Tensor input and mask")
    if _jittor_dtype_name(mask.dtype) != "bool":
        raise RuntimeError("masked_fill requires a boolean mask")
    if jt.core.dispatch_context([input]) != jt.core.dispatch_context([mask]):
        raise RuntimeError("masked_fill input and mask must share a device")
    if jt.core._is_tensor_carrier(value):
        if value.ndim != 0:
            raise RuntimeError("masked_fill value Tensor must be zero-dimensional")
        if jt.core.dispatch_context([input]) != jt.core.dispatch_context([value]):
            raise RuntimeError("masked_fill input and value must share a device")
    tensor_type = get_install_context(jt).state["Var"]
    with tensor_frontend(tensor_type, like=input, default_placement=False):
        return _native_masked_fill(input, mask, value)


def _masked_select_impl(input, mask, out=None):
    # Keep the compatibility boundary: ``out`` is accepted for API shape
    # compatibility but is not populated by this approximate fallback.
    return input[mask]


def masked_select(input, mask, out=None):
    """Return the flattened elements selected by a boolean mask."""
    from . import (
        _masked_select_impl,
    )
    return _masked_select_impl(input, mask, out=out)


def split_with_sizes(input, split_sizes, dim=0):
    """Split a tensor into chunks with the requested sizes."""
    return input.split(split_sizes, dim)


def equal(a, b):
    """Return a Python bool for Torch's same-shape, elementwise equality."""
    from . import (
        _NestedTensor,
        jt,
    )
    if isinstance(a, _NestedTensor) or isinstance(b, _NestedTensor):
        return bool(a.equal(b)) if isinstance(a, _NestedTensor) else False
    if not jt.core._is_tensor_carrier(a) or not jt.core._is_tensor_carrier(b):
        raise TypeError("equal(): both inputs must be tensors")
    if jt.core.dispatch_context([a]) != jt.core.dispatch_context([b]):
        raise RuntimeError("equal(): tensors must be on the same device")
    if tuple(a.shape) != tuple(b.shape):
        return False
    if a.numel() == 0:
        return True
    # Torch also accepts identical storage/layout without examining NaN values.
    # The native address query synchronizes in place; it does not copy to host.
    if (a.dtype == b.dtype
            and a._storage_offset() == b._storage_offset()
            and tuple(a._storage_strides()) == tuple(b._storage_strides())
            and a._storage_address == b._storage_address):
        return True
    from ...frontend import tensor_frontend
    with tensor_frontend(type(a), like=a, default_placement=False):
        return bool(jt.all(jt.ops.equal(a, b)).item())


def tensor_split(input, indices_or_sections, dim=0):
    """Split a tensor into uneven chunks using view slices."""
    d = dim % input.ndim
    length = int(input.shape[d])

    def _slice(start, stop):
        index = [slice(None)] * input.ndim
        index[d] = slice(start, stop)
        return input[tuple(index)]

    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        base, rem = divmod(length, n)
        sizes = [base + 1] * rem + [base] * (n - rem)
        result, start = [], 0
        for size in sizes:
            result.append(_slice(start, start + size))
            start += size
        return result
    points, result, previous = list(indices_or_sections), [], 0
    for point in points + [length]:
        result.append(_slice(previous, point))
        previous = point
    return result


def take(input, index):
    """Select flattened elements using Torch's ``take`` semantics."""
    return input.reshape((-1,))[index]


def index_copy(input, dim, index, source):
    """Copy rows/slices into a clone along ``dim`` (Torch non-inplace form)."""
    from . import (
        jt,
    )
    result = input.clone()
    d = dim % result.ndim
    idx = index if isinstance(index, jt.Var) else jt.array(index)
    if d == 0:
        result[idx] = source
    else:
        slices = [slice(None)] * result.ndim
        slices[d] = idx
        result[tuple(slices)] = source
    return result


def index_copy_(input, dim, index, source):
    """Copy rows/slices into ``input`` in place and return it."""
    from . import (
        jt,
    )
    d = dim % input.ndim
    idx = index if isinstance(index, jt.Var) else jt.array(index)
    if d == 0:
        input[idx] = source
    else:
        slices = [slice(None)] * input.ndim
        slices[d] = idx
        input[tuple(slices)] = source
    return input


def index_put(input, indices, values, accumulate=False):
    """Return a clone with indexed values assigned using Torch semantics."""
    from . import (
        jt,
    )
    result = input.clone()
    idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
    if not accumulate:
        result[idx if len(idx) > 1 else idx[0]] = values
        return result
    vals = values if isinstance(values, jt.Var) else jt.array(values)
    if len(idx) == 1:
        index = idx[0] if isinstance(idx[0], jt.Var) else jt.array(idx[0])
        result.assign(result.index_add(0, index.int64().reshape((-1,)), vals))
        return result
    raise NotImplementedError(
        "index_put(accumulate=True) with a partial multi-dim index")


def index_put_(input, indices, values, accumulate=False):
    """Assign indexed values in place using Torch's duplicate-safe path."""
    from . import (
        jt,
    )
    idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
    if not accumulate:
        input[idx if len(idx) > 1 else idx[0]] = values
        return input
    vals = values if isinstance(values, jt.Var) else jt.array(values)
    if len(idx) == input.ndim:
        shape = input.shape
        strides = [1] * input.ndim
        for k in range(input.ndim - 2, -1, -1):
            strides[k] = strides[k + 1] * int(shape[k + 1])
        linear = None
        for k, ind in enumerate(idx):
            term = (ind if isinstance(ind, jt.Var) else jt.array(ind)).int64().reshape((-1,)) * strides[k]
            linear = term if linear is None else linear + term
        flat_values = vals.reshape((-1,))
        if int(flat_values.shape[0]) == 1 and int(linear.shape[0]) > 1:
            flat_values = flat_values.broadcast(linear.shape)
        input.assign(input.reshape((-1,)).index_add(0, linear, flat_values).reshape(shape))
        return input
    if len(idx) == 1:
        index = idx[0] if isinstance(idx[0], jt.Var) else jt.array(idx[0])
        input.assign(input.index_add(0, index.int64().reshape((-1,)), vals))
        return input
    raise NotImplementedError("index_put_(accumulate=True) with a partial multi-dim index")
