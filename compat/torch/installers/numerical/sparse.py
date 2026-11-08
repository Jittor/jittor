"""Torch numerical sparse operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

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
    # Torch's sparse tensors take the elementwise scalar/arithmetic operators
    # dense tensors do; the storage here is dense already, so they are the
    # dense answers. PEFT's SHiRA adapter adds its sparse delta to a dense
    # weight (`weight += torch.sparse_coo_tensor(...)`), which is exactly
    # dense += sparse.
    def __mul__(self, other):
        return _SparseCOO(self._dense * _as_dense(other))
    __rmul__ = __mul__
    def __add__(self, other):
        return self._dense + _as_dense(other)
    __radd__ = __add__
    def __sub__(self, other):
        return self._dense - _as_dense(other)
    def __rsub__(self, other):
        return _as_dense(other) - self._dense
    def __neg__(self):
        return _SparseCOO(-self._dense)
    def __iadd__(self, other):
        self._dense = self._dense + _as_dense(other)
        return self
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


def _as_dense(value):
    return value.to_dense() if isinstance(value, _SparseCOO) else value


def sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                      requires_grad=False, **kwargs):
    """Materialize a dense-backed COO compatibility tensor."""
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
