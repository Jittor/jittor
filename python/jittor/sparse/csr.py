"""Compressed-row (CSR) sparse matrices, and conversion to/from COO."""

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
#
# ``coo.py``'s native COO layer builds ``to_dense``/``spmm`` out of ordinary
# differentiable ``reindex``/``reindex_reduce`` calls instead of a host round
# trip through scipy/cupy. CSR support follows the same idiom: COO<->CSR
# conversion is a pure row-sort that keeps nnz fixed, so it is expressed with
# ``jt.argsort``/``jt.searchsorted`` and stays fully lazy/symbolic and
# differentiable -- no new C++ binding, no host sync, no scipy/cupy
# dependency. Only ``coalesce()`` (nnz can shrink -- a genuinely
# data-dependent output shape) has to leave the lazy graph, exactly like
# ``coo.py``'s own ``coalesce`` does.
#
# ``spmm()`` on a ``SparseCSR`` (see ``coo.py``) converts to COO first (a
# cheap, lazy permutation) and reuses the existing COO spmm rather than
# calling the native ``cusparse_spmmcsr`` custom op directly: that op writes
# into a caller-supplied output buffer as a side effect (``OpFlags::
# _manual_set_vnbb``) and has no autograd of its own, and every place it is
# exercised today calls ``.fetch_sync()`` immediately afterwards -- there is
# no precedent in this tree for composing it lazily inside a larger graph or
# threading a custom backward through it. Wiring it in as a CUDA fast path is
# a real, worthwhile follow-up, but needs its own verification first rather
# than reusing an assumption from a different op.

import jittor as jt

from .coo import SparseVar


def _csr_row_indices(crow_indices, nnz):
    """Expand a CSR row-pointer array (length ``rows+1``) into a per-nonzero
    row index (length ``nnz``).

    For each stored-element position, count how many row-pointer boundaries
    from ``crow_indices[1:]`` are ``<=`` that position -- this lands a
    position in row ``r`` exactly when ``crow[r] <= position < crow[r+1]``,
    which also handles empty rows for free (no boundary in an empty row ever
    gets counted).
    """
    if nnz == 0:
        return jt.zeros((0,), crow_indices.dtype)
    out_int32 = str(crow_indices.dtype) == "int32"
    positions = jt.arange(nnz, dtype=crow_indices.dtype)
    return jt.searchsorted(crow_indices[1:], positions, out_int32=out_int32,
                            side="right")


def _coo_row_to_csr(row, num_rows):
    """Row-sort permutation and CSR row pointer for a COO ``row`` array.

    Duplicates are kept (not coalesced), so nnz is preserved and this stays a
    static-shape permutation -- unlike :func:`coo._CoalesceCOO`, it never
    needs to leave the lazy graph.
    """
    nnz = row.shape[0]
    out_int32 = str(row.dtype) == "int32"
    if nnz == 0:
        return jt.zeros((0,), row.dtype), jt.zeros((num_rows + 1,), row.dtype)
    perm, sorted_row = jt.argsort(row)
    boundaries = jt.arange(num_rows + 1, dtype=row.dtype)
    crow = jt.searchsorted(sorted_row, boundaries, out_int32=out_int32,
                            side="left")
    return perm, crow


class SparseCSR:
    """CSR sparse matrix (2-D only). ``crow_indices`` is ``(rows+1,)``,
    ``col_indices``/``values`` are ``(nnz,)``. Mirrors
    ``torch.sparse_csr_tensor``.
    """

    def __init__(self, crow_indices, col_indices, values, shape):
        if not (
            isinstance(crow_indices, jt.Var)
            and isinstance(col_indices, jt.Var)
            and isinstance(values, jt.Var)
            and isinstance(shape, jt.NanoVector)
        ):
            raise TypeError(
                "SparseCSR requires Var crow_indices/col_indices/values and "
                "a NanoVector shape")
        if len(shape) != 2:
            raise ValueError("SparseCSR only supports a 2-D matrix")
        if int(crow_indices.shape[0]) != int(shape[0]) + 1:
            raise ValueError("crow_indices must have length rows+1")
        if int(col_indices.shape[0]) != int(values.shape[0]):
            raise ValueError("col_indices/values nnz mismatch")
        if "int" not in str(crow_indices.dtype) or "int" not in str(col_indices.dtype):
            raise TypeError("crow_indices/col_indices must be an integer dtype")
        self.crow_indices = crow_indices
        self.col_indices = col_indices
        self.values = values
        self.shape = shape
        self.ndim = 2

    @property
    def nnz(self):
        return self.col_indices.shape[0]

    @property
    def dtype(self):
        return self.values.dtype

    def is_sparse(self):
        return True

    def to_coo(self):
        """Expand to :class:`jittor.sparse.SparseVar` (COO)."""
        row = _csr_row_indices(self.crow_indices, self.nnz)
        indices = jt.stack([row, self.col_indices], dim=0)
        return SparseVar(indices, self.values, self.shape)

    def to_dense(self):
        return self.to_coo().to_dense()

    def coalesce(self):
        """Merge duplicate ``(row, col)`` entries by summing their values."""
        return self.to_coo().coalesce().to_csr()

    def t(self):
        return self.to_coo().t().to_csr()


def sparse_csr_array(crow_indices, col_indices, values, shape):
    return SparseCSR(crow_indices, col_indices, values, shape)


__all__ = ["SparseCSR", "sparse_csr_array"]
