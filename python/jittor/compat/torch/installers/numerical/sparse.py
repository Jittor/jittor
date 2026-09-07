"""Torch numerical sparse operations."""

class _SparseCOO:
    def __init__(self, dense):
        self._dense = dense
    def to_dense(self):
        return self._dense
    @property
    def shape(self):
        return self._dense.shape
    @property
    def dtype(self):
        return self._dense.dtype
    def t(self):
        from . import (
            _SparseCOO,
        )
        return _SparseCOO(self._dense.t())
    def sum(self, dim=None):
        from . import (
            _SparseCOO,
        )
        return _SparseCOO(self._dense.sum(dim) if dim is not None else self._dense.sum())


def sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                      requires_grad=False, **kwargs):
    """Materialize a dense-backed COO compatibility tensor."""
    from . import (
        _SparseCOO,
        jt,
        np,
    )
    if not isinstance(indices, jt.Var): indices = jt.array(indices)
    if not isinstance(values, jt.Var): values = jt.array(values)
    rank = int(indices.shape[0])
    nnz = int(indices.shape[1]) if indices.ndim == 2 else int(indices.shape[0])
    tail = [int(d) for d in values.shape[1:]]
    idx_np = indices.numpy().astype("int64").reshape(rank, -1)
    if size is not None:
        full = [int(s) for s in size]
    else:
        full = [int(idx_np[s].max()) + 1 if nnz > 0 else 0 for s in range(rank)] + tail
    sparse_shape, tail_shape = full[:rank], full[rank:]
    prod = 1
    for d in sparse_shape: prod *= int(d)
    linear = np.zeros(nnz, dtype="int64")
    stride = 1
    for s in range(rank - 1, -1, -1):
        linear = linear + idx_np[s] * stride
        stride *= int(sparse_shape[s])
    flat = jt.zeros([prod] + tail_shape, dtype=str(values.dtype))
    if nnz > 0:
        flat.index_add_(0, jt.array(linear), values.reshape([nnz] + tail_shape))
    return _SparseCOO(flat.reshape(sparse_shape + tail_shape))
