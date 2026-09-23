"""Restricted Torch COO metadata tensors backed by native coordinate storage.

Only full-dimensional boolean COO tensors created by Tensor.to_sparse are
supported. The object never presents its values as a dense Var payload: native
kernels cannot accidentally interpret its nnz vector as the logical tensor.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor.sparse.coo import SparseVar, dense_to_sparse

from .fidelity import Fidelity, register_fidelity
from .frontend import tensor_frontend
from .tensor_state import compatibility_owner
from .types import _dtype_to_str, device, dtype


class SparseCOOTensor(SparseVar):
    """An explicitly limited frontend owner for a native boolean COO tensor."""

    def __init__(self, coo):
        if not isinstance(coo, SparseVar):
            raise TypeError("COO tensor storage must be a native SparseVar")
        if dtype_name(coo._values().dtype) != "bool":
            raise NotImplementedError("COO metadata tensors currently require bool values")
        if not coo._coalesced:
            raise NotImplementedError("COO metadata tensors require coalesced native storage")
        self._coo = coo

    @property
    def shape(self):
        return compatibility_owner(jt).Size(tuple(self._coo.shape))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._coo._values().dtype

    @property
    def device(self):
        return self._coo._values().device

    @property
    def layout(self):
        return compatibility_owner(jt).sparse_coo

    @property
    def is_sparse(self):
        return True

    @property
    def is_sparse_csr(self):
        return False

    @property
    def is_cpu(self):
        return self.device.type == "cpu"

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def requires_grad(self):
        return False

    @property
    def is_leaf(self):
        return True

    @property
    def grad(self):
        return None

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self):
        result = 1
        for length in self.shape:
            result *= length
        return result

    def __len__(self):
        return self.shape[0]

    def __bool__(self):
        count = self.numel()
        if count == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if count != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        # The one logical element may be implicit zero, or an explicitly stored
        # value that the caller changed. No dense materialization is necessary.
        return bool(self.values().any().item()) if self._nnz() else False

    nelement = numel

    def sparse_dim(self):
        return self.ndim

    def dense_dim(self):
        return 0

    def _nnz(self):
        return self._coo._values().numel()

    def is_coalesced(self):
        # Every supported constructor preserves native nonzero's row-major order.
        return True

    def coalesce(self):
        return self

    def indices(self):
        return self._coo._indices()

    def values(self):
        return self._coo._values()

    def _indices(self):
        return self.indices()

    def _values(self):
        return self.values()

    def to_dense(self, dtype=None, *, masked_grad=True):
        if dtype is not None and _dtype_to_str(dtype) != "bool":
            raise NotImplementedError("COO metadata to_dense currently preserves bool dtype")
        with tensor_frontend(type(self.values()), like=self.values()):
            return self._coo.to_dense()

    def detach(self):
        # Bool values and integer coordinates cannot require gradients. Keep
        # their existing native holders: native Var.detach() clones the graph
        # and would lose Torch's bidirectional storage sharing on mutation.
        # A separate COO container lets copy_ replace one tensor's coordinates
        # without replacing the detached tensor's retained storage.
        return type(self)(SparseVar(
            self.indices(), self.values(), jt.NanoVector(self.shape), coalesced=True))

    def clone(self, *, memory_format=None):
        if memory_format not in (None, "preserve_format"):
            raise NotImplementedError("COO clone supports preserve_format only")
        return type(self)(SparseVar(
            self.indices().clone(), self.values().clone(), jt.NanoVector(self.shape), coalesced=True))

    def to(self, *args, **kwargs):
        unknown = set(kwargs) - {"device", "dtype", "copy", "non_blocking", "memory_format"}
        if unknown:
            raise TypeError("unexpected COO to keywords: %s" % ", ".join(sorted(unknown)))
        if len(args) > 2:
            raise NotImplementedError("COO to supports device and dtype arguments only")
        if kwargs.get("memory_format") not in (None, "preserve_format"):
            raise NotImplementedError("COO to supports preserve_format only")
        selected_device = kwargs.get("device")
        selected_dtype = kwargs.get("dtype")
        for argument in args:
            if isinstance(argument, dtype):
                selected_dtype = argument
            elif isinstance(argument, (str, device)):
                selected_device = argument
            elif isinstance(argument, (jt.Var, SparseCOOTensor)):
                selected_device, selected_dtype = argument.device, argument.dtype
            else:
                raise TypeError("unsupported COO to argument %r" % (argument,))
        if selected_dtype is not None and _dtype_to_str(selected_dtype) != "bool":
            raise NotImplementedError("COO metadata tensors currently support bool dtype only")
        if selected_device is None:
            selected_device = self.device
        selected_device = device(selected_device)
        if selected_device.type not in ("cpu", "cuda", "npu"):
            raise NotImplementedError("unsupported COO device %s" % selected_device)
        copy = bool(kwargs.get("copy", False))
        non_blocking = bool(kwargs.get("non_blocking", False))
        values = self.values().to(device=selected_device, copy=copy, non_blocking=non_blocking)
        indices = self.indices().to(device=selected_device, copy=copy, non_blocking=non_blocking)
        if values is self.values() and indices is self.indices():
            return self
        return type(self)(SparseVar(indices, values, jt.NanoVector(self.shape), coalesced=True))

    def copy_(self, source, non_blocking=False):
        """Copy supported COO storage while retaining this buffer's identity."""
        if not isinstance(source, SparseCOOTensor):
            raise NotImplementedError("COO copy_ requires a boolean COO source; dense copying is unsupported")
        if tuple(source.shape) != tuple(self.shape):
            raise RuntimeError("COO copy_ requires identical logical shapes")
        if source.dtype != self.dtype:
            raise NotImplementedError("COO copy_ requires matching bool dtype")
        if source.device != self.device:
            raise NotImplementedError("COO copy_ currently requires the same device")
        if source is self:
            return self
        indices = source.indices().to(copy=True, non_blocking=bool(non_blocking))
        values = source.values().to(copy=True, non_blocking=bool(non_blocking))
        self._coo = SparseVar(indices, values, jt.NanoVector(self.shape), coalesced=True)
        return self

    def cpu(self, memory_format=None):
        return self.to(device="cpu", memory_format=memory_format)

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        target = "cuda" if device is None else "cuda:%d" % device if isinstance(device, int) else device
        return self.to(device=target, non_blocking=non_blocking, memory_format=memory_format)

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise RuntimeError("only floating point tensors can require gradients")
        return self

    def numpy(self, *args, **kwargs):
        raise TypeError("can't convert sparse COO tensor to numpy; use Tensor.to_dense() first")

    def to_sparse(self, sparse_dim=None, *, layout=None, blocksize=None, dense_dim=None):
        _check_sparse_request(self.ndim, sparse_dim, layout, blocksize, dense_dim)
        return self

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("this operation is not supported for boolean COO metadata tensors")

    # Do not inherit dense or sparse algebra that is outside the supported surface.
    t = transpose = permute = sum = __getitem__ = __setitem__ = _unsupported
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _unsupported
    __matmul__ = __rmatmul__ = __truediv__ = __rtruediv__ = _unsupported
    __eq__ = __ne__ = _unsupported
    __hash__ = object.__hash__

    def __repr__(self):
        return "SparseCOOTensor(size=%r, nnz=%d, dtype=%s, device=%s)" % (
            tuple(self.shape), self._nnz(), self.dtype, self.device)


def _check_sparse_request(ndim, sparse_dim, layout, blocksize, dense_dim):
    owner = compatibility_owner(jt)
    if layout is not None and layout is not owner.sparse_coo:
        raise NotImplementedError("Tensor.to_sparse supports COO layout only")
    if blocksize is not None:
        raise NotImplementedError("block sparse layouts are not supported")
    if dense_dim not in (None, 0) or sparse_dim not in (None, ndim):
        raise NotImplementedError("hybrid sparse/dense dimensions are not supported")
    if ndim == 0:
        raise NotImplementedError("scalar COO conversion is not supported")


def to_sparse(input, sparse_dim=None, *, layout=None, blocksize=None, dense_dim=None):
    _check_sparse_request(input.ndim, sparse_dim, layout, blocksize, dense_dim)
    if dtype_name(input.dtype) != "bool":
        raise NotImplementedError("Tensor.to_sparse currently supports bool metadata only")
    with tensor_frontend(type(input), like=input):
        return SparseCOOTensor(dense_to_sparse(input))


register_fidelity(
    "torch.Tensor.to_sparse", to_sparse, Fidelity.APPROXIMATE,
    "full-dimensional boolean COO metadata over native nonzero/gather storage; "
    "CPU bool metadata validated, accelerators unverified; arithmetic, floating gradients, "
    "hybrid/block layouts and persistent sparse buffers are unsupported",
)
