# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
from jittor import pool
from collections.abc import Sequence


def argmax_pool(x, size, stride, padding=0):
    return pool.pool(x, size, 'maximum', padding, stride)

def concat(arr, dim):
    '''Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * [out] out:  concat result

Example::

        >>> jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        jt.Var([[1 2]
                [2 2]], dtype=int32)
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

def _slice_var_old(x, slices):
    reindex_args = slice_var_index(x, slices)
    x.stop_fuse()
    return x.reindex(*reindex_args).stop_fuse()

def _setitem_old(x, slices, value):
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

# PATCH
def getitem(x, slices):
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        return getitem(x, slices.where())
    if isinstance(slices, tuple):
        ss = []
        for s in slices:
            if isinstance(s, jt.Var) and s.dtype == "bool":
                ss.extend(s.where())
            else:
                ss.append(s)
        slices = tuple(ss)
    return x.getitem(slices)

def setitem(x, slices, value):
    if isinstance(slices, jt.Var) and slices.dtype == "bool":
        if slices.shape == x.shape:
            if isinstance(value, (int, float)):
                value = jt.array(value).broadcast(x.shape)
                return x.assign(slices.ternary(value, x))
            elif isinstance(value, jt.Var) and value.shape == [1,]:
                value = jt.broadcast(value, x.shape)
                return x.assign(slices.ternary(value, x))
        slices = slices.where()
    elif isinstance(slices, tuple):
        ss = []
        for s in slices:
            if isinstance(s, jt.Var) and s.dtype == "bool":
                ss.extend(s.where())
            else:
                ss.append(s)
        slices = tuple(ss)
    return x.assign(x.setitem(slices, value))

jt.Var.__getitem__ = jt.Var.slice_var = getitem
jt.Var.__setitem__ = setitem


def _merge_dtypes(dtypes):
    s = -1
    e = -1
    names = ["bool","uint","int","float"]
    dbytes = ["8","16","32","64"]
    for d in dtypes:
        for name in names:
            if d.startswith(name):
                s = max(s,names.index(name))
        for db in dbytes:
            if d.endswith(db):
                e = max(e,dbytes.index(db))
    assert s>=0 and s<4 and e<4
    dtype = names[s]+("" if e ==-1 else dbytes[e])
    return dtype 

@jt.flag_scope(amp_reg=4) # _custom_flag
def concat(arr, dim=0):
    '''Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * return:  concat result

Example::

        jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        # return jt.Var([[1,2],[2,2]],dtype=int32)
    '''
    if not isinstance(arr, Sequence):
        raise TypeError("concat arr needs to be a tuple or list")
    if len(arr) == 0:
        raise ValueError("need at least one array to concat")
    total_dim = 0
    if dim < 0: dim += len(arr[0].shape)
    dtypes = []
    for a in arr:
        total_dim += a.shape[dim]
        dtypes.append(str(a.dtype))
    cdim = 0
    shape = list(a.shape)
    shape[dim] = total_dim
    s = jt.empty(shape, dtype = _merge_dtypes(dtypes))
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
