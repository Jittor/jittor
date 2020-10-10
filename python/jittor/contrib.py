# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from jittor import pool

def argmax_pool(x, size, stride, padding=0):
    return pool.pool(x, size, 'maximum', padding, stride)

def concat(arr, dim):
    '''Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * [out] out:  concat result

Example::

        jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        # return [[1],[2],[2],[2]]
    '''
    # TODO: low performance when concat lots of vars
    total_dim = 0
    if dim < 0: dim += len(arr[0].shape)
    for a in arr:
        total_dim += a.shape[dim]
    cdim = 0
    s = None
    indexes = [ f"i{i}" for i in range(len(a.shape)) ]
    for a in arr:
        shape = list(a.shape)
        shape[dim] = total_dim
        indexes[dim] = f"i{dim}-{cdim}"
        b = a.reindex(shape, indexes)
        # ugly fix for preventing large fused op 
        if len(arr)>=100:
            b.stop_fuse()
        if s is None:
            s = b
        else:
            s += b
        cdim += a.shape[dim]
    return s

def check(bc):
    bc = np.array(bc)
    if ((bc != 1) * (bc != bc.max(0))).sum() > 0:
        raise Exception(f"Shape not match.")
    else:
        return bc.max(0)

def slice_var_index(x, slices):
    if not isinstance(slices, tuple):
        slices = (slices,)
    if isinstance(slices[0], jt.Var):
        if len(slices) == 1 and slices[0].dtype == "bool":
            return slice_var_index(x, tuple(slices[0].where()))
    bc = []
    ml = -1
    for idx, s in enumerate(slices):
        if isinstance(s, jt.Var):
            shape = s.shape
        elif isinstance(s, np.ndarray):
            shape = list(s.shape)
        elif isinstance(s, list):
            shape = list(np.array(s).shape)
        else:
            continue
        if len(shape) >= ml:
            ml = len(shape)
        bc.append(shape)
    for idx, shape in enumerate(bc):
        if len(shape) < ml:
            shape = (ml - len(shape)) * [1] + shape
            bc[idx] = shape
    if len(bc) >= 1: 
        bc_shape = check(bc)
        ss = []
        for idx, s in enumerate(slices):
            if isinstance(s, np.ndarray) or isinstance(s, list):
                ss.append(jt.array(s).broadcast(bc_shape.tolist()))
            elif isinstance(s, jt.Var):
                ss.append(s.broadcast(bc_shape.tolist()))
            else:
                ss.append(s)
        slices = ss
    out_shape = []
    out_index = []
    shape = x.shape
    cnt_list = 0
    extras_idx = []
    extras = []
    has_ellipse = 0
    ellipse_index = 0
    for s,i in zip(slices,range(len(slices))):
        if isinstance(s,type(...)):
            has_ellipse+=1
            ellipse_index = i
    if has_ellipse>1:
        raise Exception(f"There are more than one ...")
    elif has_ellipse==1:
        slices = list(slices)
        del slices[ellipse_index]
        while len(slices)<len(shape):
            slices.insert(ellipse_index,slice(None))

    for i in range(len(shape)):
        if i>=len(slices):
            s = slice(None)
        else:
            s = slices[i]
        sp = shape[i]
        j = len(out_shape)
        if isinstance(s, int):
            if s<0: s += sp
            out_index.append(str(s))
        elif isinstance(s, slice):
            if s == slice(None):
                out_shape.append(sp)
                out_index.append(f"i{j}")
                continue
            start = 0 if s.start is None else s.start
            stop = sp if s.stop is None else s.stop
            step = 1 if s.step is None else s.step
            if start<0: start += sp
            if stop<0: stop += sp
            if stop>sp+1: stop = sp
            out_shape.append(1+int(max(0, (stop-start-1)//step)))
            out_index.append(f"{start}+i{j}*{step}")
        elif isinstance(s, jt.Var):
            if cnt_list == 0:
                for idx in range(len(bc_shape)):
                    extras_idx.append(f"i{len(out_shape) + idx}")
                out_shape += bc_shape.tolist()
            out_index.append(f"@e{cnt_list}("+ ",".join(extras_idx) + ")")
            cnt_list += 1
            extras.append(s)
        else:
            raise Exception(f"Not support slice {s}")
    if len(out_shape)==0:
        out_shape = [1]
    # Stop fuse both input and output, prevent recompile
    x.stop_fuse()
    return (out_shape, out_index, 0, [], extras)

def slice_var(x, slices):
    reindex_args = slice_var_index(x, slices)
    x.stop_fuse()
    return x.reindex(*reindex_args).stop_fuse()

def setitem(x, slices, value):
    reindex_args = slice_var_index(x, slices)
    reindex_reduce_args = (x.shape, reindex_args[1]) + reindex_args[3:]
    xslice = x.stop_fuse().reindex(*reindex_args).stop_fuse()
    value = jt.broadcast(value, xslice)
    value = value.cast(x.dtype)
    one = jt.broadcast(1, xslice)
    if not isinstance(reindex_args[0][0], jt.Var):
        reindex_args = (x.shape,) + reindex_args[1:]
    mask = one.reindex_reduce("add", *reindex_reduce_args)
    data = value.reindex_reduce("add", *reindex_reduce_args)
    # Stop fuse both input and output, prevent recompile
    out = mask.ternary(data, x).stop_fuse()
    x.assign(out)
    return x

jt.Var.__getitem__ = jt.Var.slice_var = slice_var
jt.Var.__setitem__ = setitem

# PATCH
def getitem(x, slices):
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        return getitem(x, slices.where())
    if isinstance(slices, list):
        slices = tuple(slices)
    return x.getitem(slices)

def setitem(x, slices, value):
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        mask = jt.broadcast(slices, x)
        value = jt.broadcast(value, x)
        return mask.ternary(value, mask)
    if isinstance(slices, list):
        slices = tuple(slices)
    return x.assign(x.setitem(slices, value))

jt.Var.__getitem__ = jt.Var.slice_var = getitem
jt.Var.__setitem__ = setitem

def concat(arr, dim):
    '''Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * [out] out:  concat result

Example::

        jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        # return [[1],[2],[2],[2]]
    '''
    # TODO: low performance when concat lots of vars
    total_dim = 0
    if dim < 0: dim += len(arr[0].shape)
    for a in arr:
        total_dim += a.shape[dim]
    cdim = 0
    shape = list(a.shape)
    shape[dim] = total_dim
    s = jt.empty(shape, a.dtype)
    slices = [slice(None)]*len(a.shape)
    for a in arr:
        if a.shape[dim] == 0:
            continue
        slices[dim] = slice(cdim, cdim+a.shape[dim])
        # print(slices, type(a))
        s = s.setitem(tuple(slices), a)
        # s = jt.setitem(s, tuple(slices), a)
        cdim += a.shape[dim]
    return s
