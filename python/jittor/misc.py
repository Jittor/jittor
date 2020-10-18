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
import math
from collections.abc import Sequence,Iterable


def repeat(x, *shape):
    r'''
    Repeats this var along the specified dimensions.

    Args:

        x (var): jittor var.

        shape (tuple): int or tuple. The number of times to repeat this var along each dimension.
 
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

        input (var) – the var to split.

        chunks (int) – number of chunks to return.

        dim (int) – dimension along which to split the var.

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


def expand(x, shape):
    return x.broadcast(shape)
jt.Var.expand = expand


def median(x,dim=None,keepdim=False):
    if dim is None:
        x = x.reshape(-1)
        dim=0
    _,x = x.argsort(dim)
    slices = [slice(None) for i in range(dim-1)]
    k = (x.shape[dim]-1)//2
    if keepdim:
        slices.append(slice(k,k+1))
    else:
        slices.append(k)
    return x[tuple(slices)]

jt.Var.median = median

def stack(x, dim=0):
    r'''
    Concatenates sequence of vars along a new dimension.

    All vars need to be of the same size.

    Args:

        x (sequence of vars) – sequence of vars to concatenate.

        dim (int) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated vars (inclusive).

    Example:

        >>> a1 = jt.array([[1,2,3]])

        >>> a2 = jt.array([[4,5,6]])

        >>> jt.stack([a1, a2], 0)
        [[[1 2 3]
        [[4 5 6]]]
    '''
    assert isinstance(x, Sequence)
    if len(x) < 2:
        return x[0].unsqueeze(dim)

    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.contrib.concat(res, dim=dim)
jt.Var.stack = stack

def flip(x, dim=0):
    r'''
    Reverse the order of a n-D var along given axis in dims.

    Args:

        input (var) – the input var.
 
        dims (a list or tuple) – axis to flip on.

    Example:

        >>> x = jt.array([[1,2,3,4]])

        >>> x.flip(1)
        [[4 3 2 1]]
    '''
    assert isinstance(dim, int)
    if dim<0:
        dim+=x.ndim
    assert dim>=0 and dim<len(x.shape)

    tar_dims = []
    for i in range(len(x.shape)):
        if i == dim:
            tar_dims.append(f"{x.shape[dim]-1}-i{i}")
        else:
            tar_dims.append(f"i{i}")
    return x.reindex(x.shape, tar_dims)
jt.Var.flip = flip

def cross(input, other, dim=-1):
    r'''
    Returns the cross product of vectors in dimension dim of input and other.

    the cross product can be calculated by (a1,a2,a3) x (b1,b2,b3) = (a2b3-a3b2, a3b1-a1b3, a1b2-a2b1)

    input and other must have the same size, and the size of their dim dimension should be 3.

    If dim is not given, it defaults to the first dimension found with the size 3.

    Args:

        input (Tensor) – the input tensor.

        other (Tensor) – the second input tensor

        dim (int, optional) – the dimension to take the cross-product in.

        out (Tensor, optional) – the output tensor.
    
    Example:

        >>> input = jt.random((6,3))

        >>> other = jt.random((6,3))

        >>> jt.cross(input, other, dim=1)
        [[-0.42732686  0.6827885  -0.49206433]
        [ 0.4651107   0.27036983 -0.5580432 ]
        [-0.31933784  0.10543461  0.09676848]
        [-0.58346975 -0.21417202  0.55176204]
        [-0.40861478  0.01496297  0.38638002]
        [ 0.18393655 -0.04907863 -0.17928357]]

        >>> jt.cross(input, other)
        [[-0.42732686  0.6827885  -0.49206433]
        [ 0.4651107   0.27036983 -0.5580432 ]
        [-0.31933784  0.10543461  0.09676848]
        [-0.58346975 -0.21417202  0.55176204]
        [-0.40861478  0.01496297  0.38638002]
        [ 0.18393655 -0.04907863 -0.17928357]]
    '''
    assert input.shape==other.shape, "input shape and other shape must be same"
    if dim < 0: dim += len(input.shape)
    assert input.shape[dim] == 3, "input dim shape must be 3"
    a1 = input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(2,)]-input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(1,)]
    a2 = input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(0,)]-input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(2,)]
    a3 = input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(1,)]-input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(0,)]
    return jt.contrib.concat([a1.unsqueeze(dim),a2.unsqueeze(dim),a3.unsqueeze(dim)], dim=dim)
jt.Var.cross = cross

def normalize(input, p=2, dim=1, eps=1e-12):
    r'''        
    Performs L_p normalization of inputs over specified dimension.

    Args:

        input – input array of any shape

        p (float) – the exponent value in the norm formulation. Default: 2

        dim (int) – the dimension to reduce. Default: 1

        eps (float) – small value to avoid division by zero. Default: 1e-12

    Example:

        >>> x = jt.random((6,3))
        [[0.18777736 0.9739261  0.77647036]
        [0.13710196 0.27282116 0.30533272]
        [0.7272278  0.5174613  0.9719775 ]
        [0.02566639 0.37504175 0.32676998]
        [0.0231761  0.5207773  0.70337296]
        [0.58966476 0.49547017 0.36724383]]

        >>> jt.normalize(x)
        [[0.14907198 0.7731768  0.61642134]
        [0.31750825 0.63181424 0.7071063 ]
        [0.5510936  0.39213243 0.736565  ]
        [0.05152962 0.7529597  0.656046  ]
        [0.02647221 0.59484214 0.80340654]
        [0.6910677  0.58067477 0.4303977 ]]
    '''
    assert p == 2
    if p == 2:
        return input / jt.maximum(input.sqr().sum(dim,True).sqrt(), eps)
jt.Var.normalize = normalize

def unbind(x, dim=0):
    r'''
    Removes a var dimension.

    Returns a tuple of all slices along a given dimension, already without it.

    Args:

        input (var) – the var to unbind

        dim (int) – dimension to remove

    Example:

        a = jt.random((3,3))
        b = jt.unbind(a, 0)

    '''
    if dim < 0: dim += len(x.shape)
    return [x[(slice(None),)*dim+(i,)] for i in range(x.shape[dim])]

jt.Var.unbind = unbind

def make_grid(x, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    assert range == None
    assert scale_each == False
    if isinstance(x, list): x = jt.stack(x)
    if normalize: x = (x - x.min()) / (x.max() - x.min())
    b,c,h,w = x.shape
    ncol = math.ceil(b / nrow)
    return x.reindex([c, h*ncol+(ncol+1)*padding, w*nrow+(nrow+1)*padding], 
                     [f"i1/{padding+h}*{nrow}+i2/{padding+w}", "i0", 
                      f"i1-i1/{padding+h}*{padding+h}-{padding}", f"i2-i2/{padding+w}*{padding+w}-{padding}"], overflow_value=pad_value)


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple([x]*n)
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def unique(x):
    r'''
    Returns the unique elements of the input tensor.

    Args:

        x– the input tensor.
    '''
    x = x.reshape(-1)
    _,x = jt.argsort(x)
    index2 = [i for i in range(1,x.shape[0])]
    index1 = [i for i in range(x.shape[0]-1)]
    y = x[1:][x[index2] != x[index1]]
    x = jt.contrib.concat([x[:1],y],dim=0)
    return x

jt.Var.unique = unique


def hypot(a,b):
    return jt.sqrt(a.sqr()+b.sqr())

def rad2deg(x):
    return 180 * x / np.pi

jt.Var.rad2deg = rad2deg

def deg2rad(x):
    return x * np.pi / 180.

jt.Var.deg2rad = deg2rad

def arctan2(y,x):
    angle = jt.zeros(x.shape,dtype=x.dtype)
    mask = x!=0.0
    if angle[mask].numel()>0:
        angle[mask] = jt.arctan(y[mask]/x[mask])
        
    mask = (y<0) & (x<0)
    if angle[mask].numel()>0:
        angle[mask] -= np.pi
        
    mask = (y>0) &(x<0)
    if angle[mask].numel()>0:
        angle[mask] +=np.pi
    return angle



def nonzero(x):
    r'''
    Return the index of the elements of input tensor which are not equal to zero.
    '''
    x = jt.where(x)
    x = [xx.unsqueeze(1) for xx in x]
    if len(x)<2:
        return x[0]
    x = jt.contrib.concat(x,dim=1)
    return x

jt.Var.nonzero = nonzero


def arange(start=0, end=None, step=1,dtype=None):
    if end is None:
        end,start = start,0
    l = round((end-start)//step)+1
    if (l-1)*step+start>=end:
        l-=1
    x = jt.index((l,),0)
    x = x*step+start
    if dtype is not None:
        x= x.cast(dtype)
    return x

def randperm(n, dtype="int64"):
    x = np.arange(n)
    np.random.shuffle(x)
    return jt.array(x).cast(dtype)

def log2(x):
    return jt.log(x)/math.log(2.0)

jt.Var.log2 = log2

def item(x):
    assert x.ndim==1 and x.shape[0]==1
    return x.data[0]

jt.Var.item  = item

def meshgrid(*tensors):
    r'''
    Take N tensors, each of which can be 1-dimensional vector, and create N n-dimensional grids, 
    where the i th grid is defined by expanding the i th input over dimensions defined by other inputs.
    '''
    size = len(tensors)
    shape = []
    for i in range(size):
        assert isinstance(tensors[i],jt.Var) and tensors[i].ndim==1
        shape.append(tensors[i].shape[0])
    grids = []
    view_shape = [1]*size
    for i in range(size):
        vs = view_shape[:]
        vs[i]=-1
        grids.append(tensors[i].reshape(vs).expand(shape))

    return grids


def split(d,split_size,dim):
    r'''
    Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If  split_size is an integer type, then tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

    If split_size is a list, then tensor will be split into len(split_size) chunks with sizes in dim according to split_size_or_sections.
   
    Args:
        d (Tensor) – tensor to split.

        split_size (int) or (list(int)) – size of a single chunk or list of sizes for each chunk

        dim (int) – dimension along which to split the tensor.
    '''
    if isinstance(split_size,int):
        shape = d.shape[dim]
        if shape % split_size == 0:
            split_size = [split_size]*(shape//split_size)
        else:
            split_size = [split_size]*(shape//split_size)+[shape%split_size]
    if isinstance(split_size, Iterable):
        assert sum(split_size)==d.shape[dim]

    if dim<0:
        dim+=d.ndim
        
    ans = []
    last = 0
    for i in split_size:
        if i==0:
            shape = list(d.shape)
            shape[dim]=0
            new_d = jt.zeros(tuple(shape),dtype=d.dtype)
            ans.append(new_d)
            continue

        ss = (slice(None),)*dim+(slice(last,last+i),)
        new_d = d[ss]
        last +=i
        ans.append(new_d)
    return tuple(ans)

jt.Var.split = split

def tolist(x):
    return x.numpy().tolist()
jt.Var.tolist = tolist

def topk(input, k, dim=None, largest=True, sorted=True):
    if input.numel()==0:
        return jt.array([],dtype=input.dtype),jt.array([],dtype='int32')
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    
    index,values = jt.argsort(input,dim=dim,descending=largest)
    dims = (slice(None),)*dim+(slice(0,k),)
    indices = index[dims]
    values = values[dims]
    return values,indices

jt.Var.topk = topk

def kthvalue(input, k, dim=None, keepdim=False):
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    index,values = jt.argsort(input,dim=dim)
    dims = (slice(None),)*dim+(slice(k-1,k),)
    indices = index[dims]
    values = values[dims]
    if not keepdim and indices.ndim>1:
        indices = indices.squeeze(dim)
        values = values.squeeze(dim)
    return values,indices

jt.Var.kthvalue = kthvalue


def gather(x,dim,index):
    if dim<0:
        dim+=index.ndim
    x_shape = list(x.shape )
    i_shape = list(index.shape)
    assert i_shape[dim]>0
    assert x.ndim == index.ndim
    i_shape[dim]=x_shape[dim]
    assert i_shape == x_shape
    ins = []
    for i in range(index.ndim):
        ins.append(jt.index(index.shape,dim=i))
    ins[dim]=index
    return x.reindex(ins)


def prod(x,dim=0):
    x = jt.log(x)
    x = x.sum(dim=dim)
    return jt.exp(x)

jt.Var.prod = prod

def cumsum_forward(np, data):
    a = data['inputs'][0]
    b = data['outputs'][0]
    np.cumsum(a, axis=1, out=b)

def cumsum_backward(np, data):
    dout = data['dout']
    out = data['outputs'][0]
    np.cumsum(dout[:, ::-1], axis=1, out=out)
    np.copyto(out, out[:, ::-1])

def cumsum(x, dim=None):
    '''
    Parameters:
    -----------
    x: [batch_size, N], jt.var

    Returns:
    --------
    the cumulative sum of x
    '''
    return jt.numpy_code(x.shape, x.dtype, [x], cumsum_forward, [cumsum_backward])

jt.Var.cumsum = cumsum

def cumprod(x,dim=0):
    x = jt.log(x)
    x = cumsum(x,dim=dim)
    return jt.exp(x)

jt.Var.cumprod=cumprod

def nms(dets,thresh):
    '''
      dets jt.array [x1,y1,x2,y2,score]
      x(:,0)->x1,x(:,1)->y1,x(:,2)->x2,x(:,3)->y2,x(:,4)->score
    '''
    threshold = str(thresh)
    order = jt.argsort(dets[:,4],descending=True)[0]
    dets = dets[order]
    s_1 = '(@x(j,2)-@x(j,0)+1)*(@x(j,3)-@x(j,1)+1)'
    s_2 = '(@x(i,2)-@x(i,0)+1)*(@x(i,3)-@x(i,1)+1)'
    s_inter_w = 'max((Tx)0,min(@x(j,2),@x(i,2))-max(@x(j,0),@x(i,0))+1)'
    s_inter_h = 'max((Tx)0,min(@x(j,3),@x(i,3))-max(@x(j,1),@x(i,1))+1)'
    s_inter = s_inter_h+'*'+s_inter_w
    iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
    fail_cond = iou+'>'+threshold
    selected = jt.candidate(dets, fail_cond)
    return order[selected]


jt.Var.expand = jt.Var.broadcast
jt.Var.expand_as = jt.Var.broadcast_var


def index_fill_(x,dim,indexs,val):
    r'''
    Fills the elements of the input tensor with value val by selecting the indices in the order given in index.

    Args:
        x - the input tensor
        dim - dimension along which to index
        index – indices of input tensor to fill in
        val – the value to fill with
    '''
    overflow_conditions = [f'i{dim}=={i}'for i in indexs]
    indexs = [f'i{i}' for i in range(len(x.shape))]
    return x.reindex(shape = x.shape,indexes = indexs,overflow_conditions=overflow_conditions,overflow_value=val)

def triu_(x,diagonal=0):
    r'''
    Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    Args:
        x – the input tensor.

        diagonal – the diagonal to consider,default =0
    '''
    l = len(x.shape)
    assert l>1
    overflow_conditions=[f'i{l-1}<i{l-2}+{diagonal}']
    indexs = [f'i{i}' for i in range(l)]
    return x.reindex(x.shape,indexs,overflow_conditions=overflow_conditions,overflow_value=0)

jt.Var.triu_ = triu_
