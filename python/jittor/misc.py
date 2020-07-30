# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#   Dun Liang <randonlang@gmail.com>.
#   Wenyang Zhou <576825820@qq.com>
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from collections.abc import Sequence

def repeat(x, *shape):
    r'''
    Repeats this var along the specified dimensions.

    Args:

        [in] x (var): jittor var.

        [in] shape (tuple): int or tuple. The number of times to repeat this var along each dimension.
 
    Example:

        >>> x = jt.array([1, 2, 3])

        >>> x.repeat(4, 2)
        [[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]]

        >>> x.repeat(4, 2, 1).size()
        [4, 2, 3,]
    '''
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    len_x_shape = len(x.shape)
    len_shape = len(shape)
    x_shape = x.shape
    rep_shape = shape
    if len_x_shape < len_shape:
        x_shape = (len_shape - len_x_shape) * [1] + x.shape
        x = x.broadcast(x_shape)
    elif len_x_shape > len_shape:
        rep_shape = (len_x_shape - len_shape) * [1] + shape
    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()
    dims = []
    for i in range(len(tar_shape)): dims.append(f"i{i}%{x_shape[i]}")
    return x.reindex(tar_shape, dims)
jt.Var.repeat = repeat

def chunk(x, chunks, dim=0):
    r'''
    Splits a var into a specific number of chunks. Each chunk is a view of the input var.

    Last chunk will be smaller if the var size along the given dimension dim is not divisible by chunks.

    Args:

        [in] input (var) – the var to split.

        [in] chunks (int) – number of chunks to return.

        [in] dim (int) – dimension along which to split the var.

    Example:

        >>> x = jt.random((10,3,3))

        >>> res = jt.chunk(x, 2, 0)

        >>> print(res[0].shape, res[1].shape)
        [5,3,3,] [5,3,3,]
    '''
    l = x.shape[dim]
    res = []
    if l <= chunks:
        for i in range(l):
            res.append(x[(slice(None,),)*dim+([i,],)])
    else:
        nums = (l-1) // chunks + 1
        for i in range(chunks-1):
            res.append(x[(slice(None,),)*dim+(slice(i*nums,(i+1)*nums),)])
        if (i+1)*nums < l:
            res.append(x[(slice(None,),)*dim+(slice((i+1)*nums,None),)])
    return res
jt.Var.chunk = chunk

def stack(x, dim=0):
    r'''
    Concatenates sequence of vars along a new dimension.

    All vars need to be of the same size.

    Args:

        [in] x (sequence of vars) – sequence of vars to concatenate.

        [in] dim (int) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated vars (inclusive).

    Example:

        >>> a1 = jt.array([[1,2,3]])

        >>> a2 = jt.array([[4,5,6]])

        >>> jt.stack([a1, a2], 0)
        [[[1 2 3]
        [[4 5 6]]]
    '''
    assert isinstance(x, list)
    assert len(x) >= 2
    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.contrib.concat(res, dim=dim)
jt.Var.stack = stack

def flip(x, dim=0):
    r'''
    Reverse the order of a n-D var along given axis in dims.

    Args:

        [in] input (var) – the input var.
 
        [in] dims (a list or tuple) – axis to flip on.

    Example:

        >>> x = jt.array([[1,2,3,4]])

        >>> x.flip(1)
        [[4 3 2 1]]
    '''
    assert isinstance(dim, int)
    tar_dims = []
    for i in range(len(x.shape)):
        if i == dim:
            tar_dims.append(f"{x.shape[dim]-1}-i{i}")
        else:
            tar_dims.append(f"i{i}")
    return x.reindex(x.shape, tar_dims)
jt.Var.flip = flip