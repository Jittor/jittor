# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Wenyang Zhou <576825820@qq.com>
#   Guoye Yang <498731903@qq.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import math
from collections.abc import Sequence,Iterable

def index_add_(x, dim, index, tensor):
    """ Take out each index subscript vector of the dim dimension and add the corresponding tensor variable.
    
    Example:

        x = jt.ones((5,3))
        tensor = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index = jt.array([0,4,2])
        x.index_add_(0, index, tensor)
        print(x)

        >>> jt.Var([[  2.,   3.,   4.],
        [  1.,   1.,   1.],
        [  8.,   9.,  10.],
        [  1.,   1.,   1.],
        [  5.,   6.,   7.]])
    """
    assert len(index.shape) == 1
    assert tensor.shape[0] == index.shape[0]
    x[(slice(None,),)*dim+(index,)] += tensor
jt.Var.index_add_ = index_add_

def __copy__(x):
    return x.copy().detach()
jt.Var.__copy__ = __copy__

def __deepcopy__(x,memo):
    result = x.copy().detach()
    memo[id(x)]=result
    return result
jt.Var.__deepcopy__ = __deepcopy__

def __len__(x):
    return x.shape[0]
jt.Var.__len__ = __len__

def __iter__(x):
    result = []
    for i in range(x.shape[0]):
        result.append(x[i])
    return result.__iter__()
jt.Var.__iter__ = __iter__

def all(x, dim=[]):
    return x.all_(dim).bool()
jt.Var.all = all

def any(x,dim):
    return x.any_(dim).bool()
jt.Var.any = any
    
def bernoulli(input):
    return (input>jt.rand_like(input)).cast(input.dtype)

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
    #TODO if input.shape[i]=1, no add [1]
    reshape_shape = []
    broadcast_shape = []
    for x_s,r_s in zip(x_shape,rep_shape):
        reshape_shape.append(1)
        reshape_shape.append(x_s)

        broadcast_shape.append(r_s)
        broadcast_shape.append(1)

    x = x.reshape(reshape_shape)
    x = x.broadcast(broadcast_shape)

    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()

    x = x.reshape(tar_shape)
    return x

jt.Var.repeat = repeat

def repeat_interleave(x,repeats,dim=None):
    # TODO repeats is jt.Var
    assert isinstance(repeats,int)
    if dim == None:
        x = x.reshape(-1)
        dim=0
    if dim<0: dim+=x.ndim
    
    tar_shape = list(x.shape)
    x_shape = list(x.shape)
    tar_shape[dim] = tar_shape[dim]*repeats 
    dims = []
    for i in range(len(tar_shape)):
        if dim==i:
            dims.append(f"i{i}/{repeats}")
        else:
            dims.append(f"i{i}")
    return x.reindex(tar_shape,dims)

jt.Var.repeat_interleave = repeat_interleave

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
    if dim<0:
        dim += x.ndim
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


def t(x):
    pose = [i for i in range(x.ndim)]
    pose[-1], pose[-2] = pose[-2], pose[-1]
    return x.transpose(*pose)
jt.Var.t = t 

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
    if isinstance(dim, int):
        dim = [dim]
    for i in range(len(dim)):
        if dim[i]<0:
            dim[i] += x.ndim
        assert dim[i]>=0 and dim[i]<x.ndim
    dim = set(dim)

    tar_dims = []
    for i in range(len(x.shape)):
        if i in dim:
            tar_dims.append(f"xshape{i}-1-i{i}")
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
    assert isinstance(range, tuple) or range is None
    assert scale_each == False
    if isinstance(x, list): x = jt.stack(x)
    assert isinstance(x, jt.Var)
    if x.ndim < 4: return x
    if x.ndim == 4 and x.shape[0] <= 1: return x
    nrow = min(nrow, x.shape[0])
    if normalize: 
        if range is None: x = (x - x.min()) / (x.max() - x.min())
        else: x = (x - range[0]) / (range[1] - range[0])
    b,c,h,w = x.shape
    ncol = math.ceil(b / nrow)
    return x.reindex([c, h*ncol+(ncol+1)*padding, w*nrow+(nrow+1)*padding], 
                     [f"i1/{padding+h}*{nrow}+i2/{padding+w}", "i0", 
                      f"i1-i1/{padding+h}*{padding+h}-{padding}", f"i2-i2/{padding+w}*{padding+w}-{padding}"], overflow_value=pad_value)

def save_image(
    x,
    filepath,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each = False,
    pad_value = 0,
    format = None
):
    from PIL import Image
    grid = make_grid(x, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = (grid*255+0.5).clamp(0, 255).permute(1, 2, 0).uint8().numpy()
    im = Image.fromarray(ndarr)
    im.save(filepath, format=format)


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
    index,= jt.index((x.shape[0],))
    y = x[1:][x[index[1:]] != x[index[:-1]]]
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

def log2(x):
    return jt.log(x)/math.log(2.0)

jt.Var.log2 = log2

def meshgrid(*tensors):
    r'''
    Take N tensors, each of which can be 1-dimensional vector, and create N n-dimensional grids, 
    where the i th grid is defined by expanding the i th input over dimensions defined by other inputs.
    '''
    if len(tensors)==1 and isinstance(tensors[0], list):
        tensors = tensors[0]
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

def view_as(x,y):
    return x.reshape(y.shape)
jt.Var.view_as = view_as

def diag(x,diagonal=0):
    assert x.ndim==1 or (x.ndim==2 and x.shape[0]==x.shape[1])
    d = diagonal if diagonal>=0 else -diagonal
    d_str = f'+{diagonal}' if diagonal>=0 else f'{diagonal}'

    if x.ndim==1:
        output_shape = (x.shape[0]+d,)*2
        return x.reindex(output_shape,[f'i1-{d}' if diagonal>=0 else f'i0-{d}'],overflow_conditions=[f'i0{d_str}!=i1'])
    else:
        output_shape = (x.shape[0]-d,)
        return x.reindex(output_shape,[f'i0+{d}' if diagonal<=0 else 'i0',f'i0+{d}' if diagonal>=0 else 'i0'])

jt.Var.diag = diag
        

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
jt.Var.gather = gather

def _prod(x,dim=0):
    x = jt.log(x)
    x = x.sum(dim=dim)
    return jt.exp(x)


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

def print_tree(now, max_memory_size, prefix1, prefix2, build_by):
    def format_size(s):
        if (s < 1024):
            s = str(s)
            return s + ' B'

        if (s < 1024*1024):
            s = format(s/1024, '.2f')
            return s + ' KB'

        if (s < 1024*1024*1024):
            s = format(s/1024/1024, '.2f')
            return s + ' MB'

        s = format(s/1024/1024/1024, '.2f')
        return s + ' GB'

    out = ''
    tab = '   '
    out += prefix1+now['name']+'('+now['type']+')\n'
    out += prefix2+'['+format_size(now['size'])+'; '+format(now['size']/max_memory_size*100, '.2f')+'%]\n'
    if (build_by == 0):
        for p in now['path']:
            out += prefix2+p+'\n'
    else:
        out += prefix2+now['path'] + '\n'
    if (len(now['children']) > 0):
        out += prefix2 + tab + '| ' + '\n'
    else:
        out += prefix2 + '\n'
    for i in range(len(now['children'])):
        c = now['children'][i]
        if i < len(now['children']) - 1:
            prefix1_ = prefix2 + tab + '├─'
            prefix2_ = prefix2 + tab + '| '
        else:
            prefix1_ = prefix2 + tab + '└─'
            prefix2_ = prefix2 + tab + '  '
        out += print_tree(c, max_memory_size, prefix1_, prefix2_, build_by)
    return out

def get_max_memory_treemap(build_by=0, do_print=True):
    '''show treemap of max memory consumption

Example::

        net = jt.models.resnet18()
        with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
            imgs = jt.randn((1,3,224,224))
            net(imgs).sync()
            jt.get_max_memory_treemap()

Output::

    | 
    ├─./python/jittor/test/test_memory_profiler.py:100(test_sample)
    | [19.03 MB; 29.67%]
    | ./python/jittor/test/test_memory_profiler.py:100
    |    | 
    |    └─./python/jittor/__init__.py:730(__call__)
    |      [19.03 MB; 29.67%]
    |      ./python/jittor/__init__.py:730
    |         | 
    |         └─./python/jittor/models/resnet.py:152(execute)
    |           [19.03 MB; 29.67%]
    |           ./python/jittor/models/resnet.py:152
    |              | 
    |              ├─./python/jittor/models/resnet.py:142(_forward_impl)
    |              | [6.13 MB; 9.55%]
    |              | ./python/jittor/models/resnet.py:142
    |              |    | 



    '''
    div1 = "[!@#div1!@#]"
    div2 = "[!@#div2!@#]"
    div3 = "[!@#div3!@#]"
    info = jt.get_max_memory_info()

    vars = []
    vars_ = info.split(div1)
    max_memory_size = int(vars_[0])
    vars_ = vars_[1:]
    for v_ in vars_:
        v__ = v_.split(div2)
        var = {'size':int(v__[1]), 'stack':[]}
        v__ = v__[2:-1]
        for s_ in v__:
            s__ = s_.split(div3)
            s = {'path':s__[0], 'name':s__[1], 'type':s__[2]}
            var['stack'].append(s)
        vars.append(var)
    if (build_by == 0): # build tree by name
        tree = {'name':'root', "children":[], 'size':0, 'path':[], 'type':''}

        def find_child(now, key):
            for c in now['children']:
                if (c['name'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            for s in v['stack']:
                ch = find_child(now, s['name'])
                if (ch is not None):
                    if (not s['path'] in ch['path']):
                        ch['path'].append(s['path'])
                    assert(ch['type']==s['type'])
                    now = ch
                    now['size'] += v['size']
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'], 'path':[s['path']], 'type':s['type']}
                    now['children'].append(now_)
                    now = now_
    elif (build_by == 1): # build tree by path
        tree = {'name':'root', "children":[], 'size':0, 'path':'_root_', 'type':''}

        def find_child(now, key):
            for c in now['children']:
                if (c['path'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            for s in v['stack']:
                ch = find_child(now, s['path'])
                if (ch is not None):
                    now = ch
                    now['size'] += v['size']
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'], 'path':s['path'], 'type':s['type']}
                    now['children'].append(now_)
                    now = now_
    else:
        assert(False)
        
    def sort_tree(now):
        def takeSize(elem):
            return elem['size']
        now['children'].sort(key=takeSize, reverse=True)
        for c in now['children']:
            sort_tree(c)
    sort_tree(tree)
    out = print_tree(tree, max_memory_size, '', '', build_by)
    if (do_print):
        print(out)
    return tree, out
    
def python_pass_warper(mod_func, args, kw):
    import importlib
    mod, func = mod_func.rsplit(".", 1)
    mod = importlib.import_module(mod)
    func = getattr(mod, func)
    args = args + ("**kw",)
    args = ",".join(args)
    return eval(f"func({args})")

def auto_parallel(n, src, **kw):
    """
    auto parallel(CPU and GPU) n-d for loop function like below:

    Before:

    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    for (int i0=0; i0<n0; i0++)
        for (int i1=0; i1<n1; i1++)
            inner_func(n0, i0, n1, i1, ...);

    After:

    @python.jittor.auto_parallel(2)
    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    inner_func(n0, 0, n1, 0, ...);


    """
    # src = prev_func func_name(args)code
    a, b = src.split('(', 1)
    prev_func, func_name = a.rsplit(None, 1)
    args, code = b.split(')', 1)
    args = args.split(',')
    assert len(args) >= n*2, (args, n)
    oargs = args[n*2:]
    pargs = args[:n*2]
    piargs = pargs[1::2]
    pnargs = pargs[0::2]
    pnargs2 = [ a.split()[-1] for a in pnargs ]
    oargs2 = [ a.split()[-1] for a in oargs ]
    entry_func_args_def = ",".join(["int tn"+str(i) for i in range(n)]
        + pnargs + oargs)
    entry_func_args = ",".join(["tn"+str(i) for i in range(n)]
        + pnargs2 + oargs2)
    tid_def = ""
    tid_loop = ""
    call_args = []
    for i in reversed(range(n)):
        tid_def += f"\nauto tid{i} = tid & ((1<<tn{i})-1);"
        tid_def += f"\nauto tnum{i} = 1<<tn{i};"
        tid_def += f"\ntid = tid>>tn{i};"
    for i in range(n):
        tid_loop += f"\nfor (int i{i}=tid{i}; i{i}<{pnargs2[i]}; i{i}+=tnum{i})"
        call_args.append(pnargs2[i])
        call_args.append(f"i{i}")
    call_args += oargs2
    call_args = ",".join(call_args)
    xn = '\n'
    new_src = f"""
#ifdef JIT_cuda
__device__
#endif
{src.replace(func_name, func_name+"_inner", 1)}

#ifdef JIT_cuda
__global__ static void {func_name}_entry({entry_func_args_def}) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    {tid_def}
    {tid_loop}
    {func_name}_inner({call_args});
}}
#endif

inline static void {func_name}({",".join(pargs+oargs)}) {{
#ifdef JIT_cuda
    int thread_num = 256*1024;
    {xn.join([f"int tn{i} = NanoVector::get_nbits(std::min(thread_num, {pnargs2[i]})) - 2;thread_num >>= tn{i};" for i in reversed(range(n))])}
    thread_num = 1<<({"+".join([f"tn{i}" for i in range(n)])});
    int p1 = std::max(thread_num/1024, 1);
    int p2 = std::min(thread_num, 1024);
    {func_name}_entry<<<p1,p2>>>({entry_func_args});
#else
    {xn.join([f"for (int i{i}=0; i{i}<{pnargs2[i]}; i{i}++)" for i in range(n)])}
    {func_name}_inner({call_args});
#endif
}}
"""
    return new_src


def cumprod(a, dim):
    class CumprodFunc(jt.Function):
        def forward_code(self, np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            out = np.cumprod(a, self.dim)
            np.copyto(b, out)

        def backward_code(self, np, data):
            a, b, dout = data["inputs"]
            out = data["outputs"][0]

            sdim = a.shape[self.dim]
            dim = (len(a.shape)+1)*[1]
            dim[self.dim+1] = sdim
            res = np.tile(np.expand_dims(b, self.dim+1), dim)
            dout = np.tile(np.expand_dims(dout, self.dim+1), dim)

            dim[self.dim]=sdim
            dim[self.dim+1]=1
            a = np.tile(np.expand_dims(a, self.dim), dim)
            res = res/a
            
            mask = np.tril(np.ones((sdim, sdim)))
            for i in range(self.dim):
                mask = np.expand_dims(mask, 0)
            for i in range(len(a.shape)-self.dim-2):
                mask = np.expand_dims(mask, -1)
            res = np.sum(mask*res*dout, self.dim)
            
            np.copyto(out, res)

        def execute(self, a, dim):
            self.save_vars = a
            self.dim = dim
            self.res = jt.numpy_code(
                a.shape,
                a.dtype,
                [a],
                self.forward_code,
            )
            return self.res

        def grad(self, grad_a):
            a = self.save_vars
            b = self.res
            return jt.numpy_code(
                a.shape,
                a.dtype,
                [a, b, grad_a],
                self.backward_code,
            )

    func = CumprodFunc()
    if dim<0:
        dim+=len(a.shape)
    return func(a, dim)

def linspace(start, end, steps):
    res = jt.index((steps,))[0]
    res = res*(end-start)/float(steps-1)+start
    return res

def randperm(n, dtype="int32"):
    key = jt.random((n,))
    index, _ = jt.argsort(key)
    return index.cast(dtype)

def set_global_seed(seed):
    import random
    random.seed(seed)
    jt.set_seed(seed)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass

def searchsorted(sorted, values, right=False):
    """
    Find the indices from the innermost dimension of `sorted` for each `values`.

Example::

    sorted = jt.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = jt.array([[3, 6, 9], [3, 6, 9]])
    ret = jt.searchsorted(sorted, values)
    assert (ret == [[1, 3, 4], [1, 2, 4]]).all(), ret

    ret = jt.searchsorted(sorted, values, right=True)
    assert (ret == [[2, 3, 5], [1, 3, 4]]).all(), ret
    
    sorted_1d = jt.array([1, 3, 5, 7, 9])
    ret = jt.searchsorted(sorted_1d, values)
    assert (ret == [[1, 3, 4], [1, 3, 4]]).all(), ret


    """
    _searchsorted_header = f"""
namespace jittor {{

@python.jittor.auto_parallel(2)
inline static void searchsorted(
    int batch_num, int batch_id, int value_num, int value_id,
    int sorted_num, int batch_stride,
    {sorted.dtype}* __restrict__  sort_p, {values.dtype}* __restrict__  value_p, 
    int32* __restrict__ index_p) {{
    int32 l = batch_id * batch_stride;
    int32 r = l + sorted_num;
    auto v = value_p[batch_id * value_num + value_id];
    while (l<r) {{
        int32 m = (l+r)/2;
        if (sort_p[m] {"<=" if right else "<"} v)
            l = m+1;
        else
            r = m;
    }}
    index_p[batch_id * value_num + value_id] = l - batch_id * batch_stride;
}}

}}
"""
    _searchsorted_src = """
    int value_num = in1->shape[in1->shape.size()-1];
    int sorted_num = in0->shape[in0->shape.size()-1];
    int32 batch_num = in0->num / sorted_num;
    int32 batch_num2 = in1->num / value_num;
    int32 batch_stride = batch_num == 1 ? 0 : sorted_num;
    CHECK(batch_num == batch_num2 || batch_num == 1);

    searchsorted(batch_num2, 0, value_num, 0, sorted_num, batch_stride, in0_p, in1_p, out0_p);
"""
    return jt.code(values.shape, "int32", [sorted, values], 
        cpu_header=_searchsorted_header,
        cpu_src=_searchsorted_src,
        cuda_header=_searchsorted_header,
        cuda_src=_searchsorted_src)


def scatter(x:jt.Var, dim:int, index:jt.Var, src:jt.Var, reduce='void'):
    ''' if x is a 3-D array, rewrite x like:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

Parameters::

    * x (jt.Var) – input array
    * dim (int) – the axis along which to index
    * index (jt.Var) – the indices of elements to scatter, can be either empty or of the same dimensionality as src. When empty, the operation returns self unchanged.
    * src (jt.Var) – the source element(s) to scatter.
    * reduce (str, optional) – reduction operation to apply, can be either 'add' or 'multiply'.

Example::

    src = jt.arange(1, 11).reshape((2, 5))
    index = jt.array([[0, 1, 2, 0]])
    x = jt.zeros((3, 5), dtype=src.dtype).scatter_(0, index, src)
    assert (x.data == 
        [[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]]).all()
    index = jt.array([[0, 1, 2], [0, 1, 4]])
    x = jt.zeros((3, 5), dtype=src.dtype).scatter_(1, index, src)
    assert (x.data ==
        [[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]]).all()
    x = jt.full((2, 4), 2.).scatter_(1, jt.array([[2], [3]]),
            jt.array(1.23), reduce='multiply')
    assert np.allclose(x.data, 
        [[2.0000, 2.0000, 2.4600, 2.0000],
        [2.0000, 2.0000, 2.0000, 2.4600]]), x
    x = jt.full((2, 4), 2.).scatter_(1, jt.array([[2], [3]]),
            jt.array(1.23), reduce='add')
    assert np.allclose(x.data,
        [[2.0000, 2.0000, 3.2300, 2.0000],
        [2.0000, 2.0000, 2.0000, 3.2300]])

    '''
    shape = index.shape
    if src.shape != shape and src.numel() != 1:
        src = src[tuple( slice(None,s) for s in shape )]
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.setitem(tuple(indexes), src, reduce)

def scatter_(x, dim, index, src, reduce='void'):
    return x.assign(x.scatter(dim, index, src, reduce))

jt.Var.scatter = scatter
jt.Var.scatter_ = scatter_

def gather(x, dim, index):
    ''' if x is a 3-D array, reindex x like:

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2


Parameters::

    * x (jt.Var) – the source array
    * dim (int) – the axis along which to index
    * index (jt.Var) – the indices of elements to gather

Example::

    t = jt.array([[1, 2], [3, 4]])
    data = t.gather(1, jt.array([[0, 0], [1, 0]]))
    assert (data.data == [[ 1,  1], [ 4,  3]]).all()
    data = t.gather(0, jt.array([[0, 0], [1, 0]]))
    assert (data.data == [[ 1,  2], [ 3,  2]]).all()

    '''
    shape = index.shape
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.getitem(tuple(indexes))

jt.Var.gather = gather

def roll(x, shifts, dims=None):
    '''Roll the tensor along the given dimension(s).

Parameters::

    * x (jt.Var) – the source array
    * shifts (int or tuple) – shift offset of dims
    * dims (int or tuple) – shift dims

Examples::

        x = jt.array([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        y = x.roll(1, 0)
        assert (y.numpy() == [[7,8],[1,2],[3,4],[5,6]]).all()
        y = x.roll(-1, 0)
        assert (y.numpy() == [[3,4],[5,6],[7,8],[1,2]]).all()
        y = x.roll(shifts=(2, 1), dims=(0, 1))
        assert (y.numpy() == [[6,5],[8,7],[2,1],[4,3]]).all()

    '''
    if isinstance(shifts, int):
        shifts = (shifts,)
    if dims is None:
        dims = tuple(range(len(shifts)))
    elif isinstance(dims, int):
        dims = (dims,)
    assert len(dims) == len(shifts)
    ids = [ f'i{i}' for i in range(x.ndim) ]
    for i in range(len(dims)):
        shift = shifts[i]
        d = dims[i]
        size = x.shape[d]
        shift = shift % size
        if shift<0: shift += size
        ids[d] = f'(i{d}<{shift}?i{d}+{size-shift}:(i{d}-{shift}))'
    return x.reindex(x.shape, ids)

jt.Var.roll = roll

def safe_log(x):
    return jt.safe_clip(x, 1e-30, 1e30).log()
jt.Var.safe_log = safe_log
