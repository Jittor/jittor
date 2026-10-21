"""Coordinate-format sparse tensors and sparse-dense multiplication."""

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Xiangli Li <190569238@qq.com>
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import numpy as np
import jittor as jt


class _CoalesceCOO(jt.Function):
    """Merge duplicate ``(row, col)`` entries by summing their values.

    nnz can shrink, which is a genuinely data-dependent output shape, so this
    goes through ``jt.Function`` (whose Python-level ``execute()`` can return
    a freshly-shaped Var). The work stays on the operands' device: the
    linearised keys are sorted, run starts are flagged and prefix-summed into
    each entry's output slot, and values are summed into those slots with
    ``reindex_reduce``. Only the merged count -- the output shape -- is read
    back to the host. The output is in ascending key (row-major) order, the
    order ``np.unique`` gave the previous host implementation.
    """

    def execute(self, row, col, values, N):
        nnz = int(row.shape[0])
        trailing = list(values.shape[1:])
        if nnz == 0:
            self.inverse = jt.zeros((0,), dtype="int64")
            return row.clone(), col.clone(), values.clone()
        # N as an int64 Var: a Python int operand is narrowed to int32, and
        # keys past 2^31 then wrap.
        n = jt.array(np.array(N, dtype=np.int64), dtype="int64")
        key = row.cast("int64") * n + col.cast("int64")
        order, sorted_key = jt.argsort(key, 0)
        new_run = (sorted_key[1:] != sorted_key[:-1]).cast("int64")
        starts = jt.concat([jt.ones((1,), dtype="int64"), new_run], 0)
        slot_sorted = jt.cumsum(starts, 0) - 1
        count = int(starts.sum().item())
        # order is a permutation: each original entry is written exactly once.
        inverse = slot_sorted.reindex_reduce("add", [nnz], ["@e0(i0)"], extras=[order])
        # every entry of a run carries the same key
        merged_key = sorted_key.reindex_reduce(
            "maximum", [count], ["@e0(i0)"], extras=[slot_sorted])
        index = ["@e0(i0)"] + ["i%d" % (d + 1) for d in range(len(trailing))]
        merged = values.reindex_reduce("add", [count] + trailing, index, extras=[inverse])
        self.inverse = inverse
        return ((merged_key // n).cast(row.dtype),
                (merged_key % n).cast(col.dtype),
                merged)

    def grad(self, d_row, d_col, d_values):
        # each original (possibly-duplicate) entry receives the gradient of
        # the merged entry it was summed into -- a pure gather by the stored
        # forward `inverse` map.
        if d_values is None:
            return None, None, None, None
        return None, None, jt.index_select(d_values, 0, self.inverse), None


class SparseVar:
    def __init__(self,indices,values,shape):
        if not (
            isinstance(indices, jt.Var)
            and isinstance(values, jt.Var)
            and isinstance(shape, jt.NanoVector)
        ):
            raise TypeError("SparseVar requires Var indices/values and a NanoVector shape")
        if indices.ndim != 2 or indices.shape[0] != len(shape):
            raise ValueError(
                f"indices must be shape (ndim={len(shape)}, nnz), "
                f"got {tuple(indices.shape)}")
        if values.ndim == 0 or values.shape[0] != indices.shape[1]:
            raise ValueError("values must have a leading nnz dimension matching indices")
        if "int" not in str(indices.dtype):
            raise TypeError("indices must be an integer dtype")
        self.indices = indices
        self.values = values
        self.shape = shape
        self.ndim = len(shape)

    @property
    def nnz(self):
        return self.indices.shape[1]

    @property
    def dtype(self):
        return self.values.dtype

    def is_sparse(self):
        return True
        
    def _indices(self):
        return self.indices
    
    def _values(self):
        return self.values
    
    def t(self):
        indices = list(self.indices.split(1,dim=0))
        indices[-1],indices[-2] = indices[-2],indices[-1]
        indices = jt.concat(indices,dim=0)
        shape = list(self.shape)
        shape[-1],shape[-2] = shape[-2],shape[-1]
        shape = jt.NanoVector(shape)
        return SparseVar(indices,self.values,shape)
        
    def _index_exprs(self):
        """Index expressions picking coordinate ``d`` of nonzero ``i0``."""
        return ["@e0(%d, i0)" % d for d in range(self.ndim)]

    def to_dense(self):
        """Densify, *summing* values that share a coordinate.

        COO tensors are uncoalesced by definition: the same coordinate may
        appear more than once and its value is the sum of the duplicates
        (this is what torch's ``to_dense`` and scipy's ``toarray`` do).
        Scattering with an assignment instead made the result depend on which
        duplicate happened to be written last.
        """
        return self.values.reindex_reduce(
            "add", list(self.shape), self._index_exprs(),
            extras=[self.indices])

    def coalesce(self):
        """Merge duplicate ``(row, col)`` entries by summing their values.

        Uncoalesced is the default (see :meth:`to_dense`); call this to get
        a normalized, duplicate-free COO tensor explicitly.
        """
        assert self.ndim == 2, "coalesce() only supports a plain 2-D sparse matrix"
        N = int(self.shape[1])
        row, col, values = _CoalesceCOO()(self.indices[0], self.indices[1],
                                           self.values, N)
        indices = jt.stack([row, col], dim=0)
        return SparseVar(indices, values, self.shape)

    def to_csr(self):
        """Convert to :class:`jittor.sparse.csr.SparseCSR` (row-sorted;
        duplicates are kept -- call :meth:`coalesce` first if that's wanted).
        """
        assert self.ndim == 2, "to_csr() only supports a plain 2-D sparse matrix"
        from .csr import SparseCSR, _coo_row_to_csr
        row, col = self.indices[0], self.indices[1]
        perm, crow = _coo_row_to_csr(row, int(self.shape[0]))
        sorted_col = jt.index_select(col, 0, perm)
        sorted_values = jt.index_select(self.values, 0, perm)
        return SparseCSR(crow, sorted_col, sorted_values, self.shape)

def sparse_array(indices,values,shape):
    return SparseVar(indices,values,shape)

def spmm(spase_x,y):
    """Sparse-dense matrix product, without materialising the sparse operand.

    Gathering the rows of ``y`` that each nonzero needs and scattering the
    scaled rows back costs O(nnz * y.shape[1]); densifying first cost
    O(rows * cols) memory and a dense matmul, which is what made this useless
    on any sparse matrix worth the name.

    ``spase_x`` may be a :class:`SparseVar` (COO) or a
    :class:`jittor.sparse.csr.SparseCSR`; CSR is converted to COO first (a
    cheap, lazy row-index expansion -- see ``csr.py``) and runs through the
    same code below.
    """
    if hasattr(spase_x, "to_coo") and not isinstance(spase_x, SparseVar):
        spase_x = spase_x.to_coo()
    if not isinstance(spase_x, SparseVar) or not isinstance(y, jt.Var):
        raise TypeError("spmm requires a SparseVar/SparseCSR and a dense Var")
    if spase_x.ndim != 2 or y.ndim != 2 or spase_x.shape[-1] != y.shape[0]:
        raise ValueError("spmm expects 2-D operands with matching inner dimensions")

    indices = spase_x.indices
    nnz = indices.shape[1]
    n_cols = y.shape[1]
    out_shape = [spase_x.shape[0], n_cols]
    if nnz == 0:
        return jt.zeros(out_shape, y.dtype)
    # rows of y selected by the column index of each nonzero
    gathered = y.reindex([nnz, n_cols], ["@e0(1, i0)", "i1"], extras=[indices])
    scaled = gathered * spase_x.values.broadcast([nnz, n_cols], dims=[1])
    # accumulate into the row index of each nonzero (duplicates add up, which
    # is also what makes an uncoalesced operand come out right)
    return scaled.reindex_reduce(
        "add", out_shape, ["@e0(0, i0)", "i1"], extras=[indices])


__all__ = ["SparseVar", "sparse_array", "spmm"]
    
