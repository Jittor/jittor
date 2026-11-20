"""Torch numerical sparse operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

class _SparseCOO:
    def __init__(self, dense=None, *, coo=None, coalesced=False):
        self._dense = dense
        self._coo = coo
        self._coalesced = coalesced

    def to_dense(self):
        if self._coo is not None:
            return self._coo.to_dense()
        return self._dense

    @property
    def shape(self):
        return (self._coo if self._coo is not None else self._dense).shape

    @property
    def dtype(self):
        return (self._coo if self._coo is not None else self._dense).dtype

    @property
    def device(self):
        return (self._coo.values if self._coo is not None else self._dense).device

    def _scalar_coo(self):
        if self._coo is None:
            raise NotImplementedError("this operation requires a 2-D scalar COO tensor")
        return self._coo

    def _nnz(self):
        return int(self._scalar_coo().nnz)

    def is_coalesced(self):
        return self._coalesced

    def coalesce(self):
        from jittor._core.var import device_scope_like

        coo = self._scalar_coo()
        if self._coalesced:
            return self
        with device_scope_like(coo.values):
            return _SparseCOO(coo=coo.coalesce(), coalesced=True)

    def indices(self):
        if not self._coalesced:
            raise RuntimeError("Cannot get indices on an uncoalesced tensor. Call coalesce() first.")
        return self._scalar_coo().indices

    def values(self):
        if not self._coalesced:
            raise RuntimeError("Cannot get values on an uncoalesced tensor. Call coalesce() first.")
        return self._scalar_coo().values

    # Retain the existing dense arithmetic for consumers such as PEFT's SHiRA.
    def __mul__(self, other):
        return _SparseCOO(self.to_dense() * _as_dense(other))
    __rmul__ = __mul__
    def __add__(self, other):
        return self.to_dense() + _as_dense(other)
    __radd__ = __add__
    def __sub__(self, other):
        return self.to_dense() - _as_dense(other)
    def __rsub__(self, other):
        return _as_dense(other) - self.to_dense()
    def __neg__(self):
        return _SparseCOO(-self.to_dense())
    def __iadd__(self, other):
        self._dense = self.to_dense() + _as_dense(other)
        self._coo = None
        self._coalesced = False
        return self
    def t(self):
        if self._coo is not None:
            return _SparseCOO(coo=self._coo.t())
        return _SparseCOO(self._dense.t())
    def sum(self, dim=None):
        dense = self.to_dense()
        return _SparseCOO(dense.sum(dim) if dim is not None else dense.sum())


def _as_dense(value):
    return value.to_dense() if isinstance(value, _SparseCOO) else value


def sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                      requires_grad=False, is_coalesced=None, **kwargs):
    """Construct a COO compatibility tensor, retaining scalar 2-D sparse storage."""
    from . import (
        _SparseCOO,
        jt,
        np,
    )
    from ...types import _dtype_to_str
    selected = _dtype_to_str(dtype)
    if not isinstance(indices, jt.Var):
        indices = jt.array(indices)
    if not isinstance(values, jt.Var):
        values = jt.array(values, dtype=selected)
    elif selected is not None:
        values = values.cast(selected)
    rank = int(indices.shape[0])
    nnz = int(indices.shape[1]) if indices.ndim == 2 else int(indices.shape[0])
    tail = [int(d) for d in values.shape[1:]]
    if indices.ndim == 2 and rank == 2 and values.ndim == 1 and (size is None or len(size) == 2):
        from jittor.sparse import sparse_array
        from ...tensor_state import compatibility_owner

        owner = compatibility_owner(jt)
        values = owner.as_tensor(values, device=device)
        indices = owner.as_tensor(indices, dtype=owner.int64, device=values.device)
        if requires_grad and not values.requires_grad:
            values = values.detach().requires_grad_(True)
        if size is None:
            size = tuple(int(indices[axis].max().item()) + 1 if nnz else 0 for axis in range(rank))
        coalesced = nnz < 2 if is_coalesced is None else bool(is_coalesced)
        return _SparseCOO(coo=sparse_array(indices, values, jt.NanoVector(size)), coalesced=coalesced)

    idx_np = indices.numpy().astype("int64").reshape(rank, -1)
    if size is not None:
        full = [int(s) for s in size]
    else:
        full = [int(idx_np[s].max()) + 1 if nnz > 0 else 0 for s in range(rank)] + tail
    sparse_shape, tail_shape = full[:rank], full[rank:]
    prod = 1
    for d in sparse_shape:
        prod *= int(d)
    linear = np.zeros(nnz, dtype="int64")
    stride = 1
    for s in range(rank - 1, -1, -1):
        linear = linear + idx_np[s] * stride
        stride *= int(sparse_shape[s])
    flat = jt.zeros([prod] + tail_shape, dtype=_jittor_dtype_name(values.dtype))
    if nnz > 0:
        flat.index_add_(0, jt.array(linear), values.reshape([nnz] + tail_shape))
    return _SparseCOO(flat.reshape(sparse_shape + tail_shape))


def sparse_mm(left, right):
    from jittor._core.var import device_scope_like
    from jittor.sparse import spmm

    coo = left._scalar_coo()
    if left.device != right.device or left.dtype != right.dtype:
        raise RuntimeError("torch.sparse.mm requires matching devices and dtypes")
    with device_scope_like(right):
        output = spmm(coo, right)
        if not coo.nnz:
            # Preserve zero gradients for the empty operand and the dense input.
            output = output + coo.values.sum() + right[:0].sum()
        return output
