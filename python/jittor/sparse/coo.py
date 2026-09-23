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

import jittor as jt
class SparseVar:
    def __init__(self,indices,values,shape, *, coalesced=False):
        if not (
            isinstance(indices, jt.Var)
            and isinstance(values, jt.Var)
            and isinstance(shape, jt.NanoVector)
        ):
            raise TypeError("SparseVar requires Var indices/values and a NanoVector shape")
        self.indices = indices
        self.values = values
        self.shape = shape
        self.ndim = len(shape)
        self._coalesced = bool(coalesced)
        
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

def dense_to_sparse(input):
    """Build full-dimensional COO storage using native nonzero and gather.

    Native nonzero enumerates coordinates in row-major order, making this
    construction coalesced without a host copy or a second sort.
    """
    if not isinstance(input, jt.Var):
        raise TypeError("dense_to_sparse requires a Var")
    if input.ndim == 0:
        raise NotImplementedError("scalar COO conversion is not supported")
    coordinates = jt.nonzero(input).transpose(0, 1).cast("int64")
    values = input.reindex(
        [coordinates.shape[1]],
        ["@e0(%d,i0)" % dimension for dimension in range(input.ndim)],
        extras=[coordinates],
    )
    return SparseVar(coordinates, values, jt.NanoVector(input.shape), coalesced=True)


def sparse_array(indices,values,shape):
    return SparseVar(indices,values,shape)

def spmm(spase_x,y):
    """Sparse-dense matrix product, without materialising the sparse operand.

    Gathering the rows of ``y`` that each nonzero needs and scattering the
    scaled rows back costs O(nnz * y.shape[1]); densifying first cost
    O(rows * cols) memory and a dense matmul, which is what made this useless
    on any sparse matrix worth the name.
    """
    if not isinstance(spase_x, SparseVar) or not isinstance(y, jt.Var):
        raise TypeError("spmm requires a SparseVar and a dense Var")
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


__all__ = ["SparseVar", "dense_to_sparse", "sparse_array", "spmm"]
    
