# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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

def knn(unknown, known, k):
    ''' find k neighbors for unknown array from known array

    Args:
        
        unknown (var): shape [b, n, c]
        known (var): shape [b, m, c]
        k (int)

    '''
    b, n, c = unknown.shape
    _, m, _ = known.shape
    dists2 = jt.empty((b, n, k), dtype="float")
    idx = jt.empty((b, n, k), dtype="int")
    src = '''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void knn_kernel(int b, int batch_index, int n, int index, int m,
                        const float *__restrict__ unknown,
                        const float *__restrict__ known,
                        float *__restrict__ dist2,
                        int *__restrict__ idx) {

#define K %s
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * K;
    idx += batch_index * n * K;
    int j = index;
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        float tmp_dist[K];
        int tmp_idx[K];
        #pragma unroll
        for (int i=0; i<K; i++) tmp_dist[i] = 1e30;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            
            int first = -1;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (first == -1 && d<tmp_dist[i])
                    first = i;
            if (first == -1) continue;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (K-1-i > first) {
                    tmp_dist[K-1-i] = tmp_dist[K-2-i];
                    tmp_idx[K-1-i] = tmp_idx[K-2-i];
                }
            tmp_dist[first] = d;
            tmp_idx[first] = k;
        }
        #pragma unroll
        for (int i=0; i<K; i++) {
            dist2[j * K + i] = tmp_dist[i];
            idx[j * K + i] = tmp_idx[i];
        }
    }
}
    knn_kernel(in0->shape[0], 0, in0->shape[1], 0, in1->shape[1], in0_p, in1_p, out0_p, out1_p);
    ''' % k
    return jt.code([unknown, known], [dists2, idx],
    cpu_src=src,
    cuda_src=src)

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

def __contains__(x, key):
    return bool((x == key).any())
jt.Var.__contains__ = __contains__

def new(x, *args):
    if len(args) != 1 or isinstance(args[0], int):
        return jt.empty(args, x.dtype)
    return jt.array(args[0]).cast(x.dtype)
jt.Var.new = new

def __index__(x):
    return int(x.item())
jt.Var.__index__ = __index__

def sort(input, dim=-1, descending=False, stable=False):
    index, value = jt.argsort(input, dim, descending)
    return value, index
jt.Var.sort = sort

def all(x, dim=()):
    return x.all_(dim).bool()
jt.Var.all = all

def any(x,dim=()):
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
        rep_shape = (len_x_shape - len_shape) * [1] + list(shape)

    reshape_shape = []
    broadcast_shape = []
    for x_s,r_s in zip(x_shape,rep_shape):
        if r_s != 1:
            reshape_shape.append(1)
            broadcast_shape.append(r_s)
        reshape_shape.append(x_s)
        broadcast_shape.append(1)

    x = x.reshape(reshape_shape)
    x = x.broadcast(broadcast_shape)

    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()

    x = x.reshape(tar_shape)
    return x

jt.Var.repeat = repeat
# tile = jt.Var.tile = repeat
ne = jt.Var.ne = jt.Var.not_equal

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


def expand(x, *shape):
    ''' Expand and broadcast this array, -1 represents this dimension is not changed.

Example::

    a = jt.zeros((3,1))
    b = a.expand(3, 4)
    assert b.shape == (3,4)
    b = a.expand(-1, 4)
    assert b.shape == (3,4)
    b = a.expand((3, 4))
    assert b.shape == (3,4)
    b = a.expand((-1, 4))
    assert b.shape == (3,4)

    '''
    if len(shape) == 1 and isinstance(shape[0], (tuple,list,jt.NanoVector)):
        shape = shape[0]
    shape = list(shape)
    offset = len(shape) - len(x.shape)
    for i in range(len(x.shape)):
        if shape[offset + i] == -1:
            shape[offset + i] = x.shape[i]
    return x.broadcast(shape)
jt.Var.expand = expand


def t(x):
    pose = [i for i in range(x.ndim)]
    pose[-1], pose[-2] = pose[-2], pose[-1]
    return x.transpose(*pose)
jt.Var.t = t 

def median(x,dim=None,keepdim=False, keepdims=False):
    keepdim = keepdim or keepdims
    if dim is None:
        x = x.reshape(-1)
        dim=0
    _,x = jt.argsort(x, dim)
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
    for i,x_ in enumerate(x):
            x[i] = jt.array(x_)
    if len(x) < 2:
        return x[0].unsqueeze(dim)

    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.concat(res, dim=dim)
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
    return jt.concat([a1.unsqueeze(dim),a2.unsqueeze(dim),a3.unsqueeze(dim)], dim=dim)
jt.Var.cross = cross

def normalize(input, p=2, dim=1, eps=1e-30):
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
    return input / input.norm(p, dim, True, eps)
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
    if normalize: 
        if range is None: x = (x - x.min()) / (x.max() - x.min())
        else: x = (x - range[0]) / (range[1] - range[0])
    if x.ndim < 4: return x
    if x.ndim == 4 and x.shape[0] <= 1: return x
    nrow = min(nrow, x.shape[0])
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


def unique(
    input: jt.Var, 
    return_inverse: bool=False, 
    return_counts: bool=False, 
    dim: int=None):

    r'''
    Returns the unique elements of the input tensor.

    Args:

        input (var) – the input var

        return_inverse (bool) – Whether to also return the indices for where elements in the original input ended up in the returned unique list. default: False

        return_counts (bool) – Whether to also return the counts for each unique element. default: False

        dim (int) – the dimension to apply unique. If None, the unique of the flattened input is returned. default: None

    Example:

        >>> jittor.unique(jittor.array([1, 3, 2, 3]))
        jt.Var([1 2 3], dtype=int32)

        >>> jittor.unique(jittor.array([1, 3, 2, 3, 2]), return_inverse=True, return_counts=True)
        (jt.Var([1 2 3], dtype=int32), jt.Var([0 2 1 2 1], dtype=int32), jt.Var([1 2 2], dtype=int32))

        >>> jittor.unique(jittor.array([[1, 3], [2, 3]]), return_inverse=True)
            (jt.Var([1 2 3], dtype=int32), jt.Var([[0 2]
                                                   [1 2]], dtype=int32))

        >>> jittor.unique(jittor.array([[1, 3], [1, 3]]), dim=0)
            jt.Var([[1 3]], dtype=int32)
    '''

    temp_shape = None
    if dim == None:
        temp_shape = list(input.shape)
        input_flatten = input.flatten()
        dim = 0
    else:
        input_flatten = input

    input_flatten = input_flatten.transpose(dim, 0)
    orig_shape = input_flatten.shape
    input_flatten = input_flatten.view(orig_shape[0], -1)
    
    with jt.flag_scope(compile_options = {"FLAGS:  --extended-lambda ": 1} if jt.flags.use_cuda else {}):
        indice = jt.code((input_flatten.shape[0], ), 'int32', [input_flatten],
            cpu_header='''
            #include <algorithm>
            ''',
            cpu_src='''
            @alias(input_flatten, in0)
            @alias(indice, out)

            int dimlen = input_flatten_shape0, dimsize = input_flatten_shape1;
            for(int i = 0; i < dimlen; ++i) @indice(i) = i;
            std::sort(&@indice(0), &@indice(dimlen), [&](int a, int b){
                for(int i = 0; i < dimsize; ++i) {
                    int lhs = @input_flatten(a, i), rhs = @input_flatten(b, i);
                    if (lhs != rhs) return lhs < rhs;
                }
                return false;
            });
            ''',
            cuda_header='''
            #undef out
            #include <thrust/extrema.h>
            #include <thrust/device_ptr.h>
            #include <thrust/execution_policy.h>
            #include <thrust/device_vector.h>
            #include <thrust/sequence.h>
    
            #include <thrust/sequence.h>
            #include <thrust/sort.h>
            #include <thrust/unique.h>

            #include <cub/cub.cuh> 
            #include <executor.h>
            ''',
            cuda_src=
            '''
                @alias(input_flatten, in0)
                @alias(indice, out)
                int dimlen = indice_shape0, dimsize = input_flatten_shape1;

                if (dimsize == 1) {
                    size_t raw_allocation, d_allocation, temp_storage_bytes = 0;
                    void *d_temp_storage = NULL;
                    int32_t* raw_ptr = (int32_t*)exe.allocator->alloc(dimlen * (sizeof(int32_t) + sizeof(input_flatten_type)), raw_allocation);

                    thrust::device_ptr<int32_t> arange_ptr = thrust::device_pointer_cast(raw_ptr);
                    thrust::sequence(arange_ptr, arange_ptr + dimlen);

                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p, 
                                                    (input_flatten_type*)(raw_ptr + dimlen), thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);
                    d_temp_storage = exe.allocator->alloc(temp_storage_bytes, d_allocation);
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p,
                                                    (input_flatten_type*)(raw_ptr + dimlen), thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);

                    exe.allocator->free(raw_ptr, dimlen * (sizeof(int) + sizeof(input_flatten_type)), raw_allocation);
                    exe.allocator->free(d_temp_storage, temp_storage_bytes, d_allocation);
                } else {
                    thrust::device_ptr<input_flatten_type> input_ptr = thrust::device_pointer_cast(input_flatten_p);
                    thrust::device_ptr<int32_t> indice_ptr = thrust::device_pointer_cast(indice_p);

                    thrust::sequence(indice_ptr, indice_ptr + dimlen);
                    thrust::sort(thrust::device, indice_ptr, indice_ptr + dimlen,
                        [=] __device__ (int32_t a, int32_t b)->bool {
                            for(int i = 0; i < dimsize; ++i) {
                                input_flatten_type lhs = input_ptr[i + a * dimsize],
                                                rhs = input_ptr[i + b * dimsize];
                                if (lhs != rhs) return lhs < rhs;
                            }
                            return false;
                        });
                }
            '''
        )
    input_sorted = input_flatten[indice][:]
    
    dimlen = indice.shape[0]

    diff = jt.logical_not(jt.all(input_sorted[1:] == input_sorted[: -1], 1))
    diff = jt.concat([jt.Var([False]), diff], 0)
    diff = jt.array(diff, dtype = jt.int32)
  
    with jt.flag_scope(compile_options = {"FLAGS:  --extended-lambda ": 1} if jt.flags.use_cuda else {}):
        output, inverse = jt.code(
            [(-input_sorted.shape[0], ), (indice.shape)],
            [input_sorted.dtype, indice.dtype],
            [input_sorted, diff, indice],
            cpu_header='''
                #include <algorithm>
                @alias(input_sorted, in0)
                @alias(diff, in1)
                @alias(indice, in2)
                @alias(output, out0)
                @alias(inverse, out1)
            ''',
            cpu_src=
            f"bool return_inverse = {int(return_inverse)};" +
            '''
                int tot = -1;
                for (int i = 0; i < input_sorted_shape0; ++i) {
                    if (i == 0 || @diff(i)) {
                        ++tot; @output(tot) = i;
                    }
                    if (return_inverse)
                        @inverse(@indice(i)) = tot;
                }
                output->set_shape({tot + 1});
            ''',
            cuda_header='''
                #undef out

                #include <thrust/extrema.h>
                #include <thrust/device_ptr.h>
                #include <thrust/execution_policy.h>

                #include <thrust/sequence.h>
                #include <thrust/unique.h>
                #include <thrust/sort.h>

                #include <thrust/scan.h>
                #include <executor.h>

                @alias(input_sorted, in0)
                @alias(diff, in1)
                @alias(indice, in2)
                @alias(output, out0)
                @alias(inverse, out1)
            ''',
            cuda_src=
            f"bool return_inverse = {int(return_inverse)};" +
            '''
                int dimlen = input_sorted_shape0, dimsize = input_sorted_shape1;
                size_t raw_allocation;
                int32_t* raw_ptr = (int32_t*)exe.allocator->alloc(2 * dimlen * sizeof(int), raw_allocation);

                thrust::device_ptr<int32_t> diff_ptr = thrust::device_pointer_cast(diff_p),
                                            inverse_ptr = thrust::device_pointer_cast(inverse_p),
                                            array_ptr = thrust::device_pointer_cast(raw_ptr),
                                            sum_ptr = thrust::device_pointer_cast(raw_ptr + dimlen),
                                            indice_ptr = thrust::device_pointer_cast(indice_p);
                thrust::device_ptr<input_sorted_type> input_ptr = thrust::device_pointer_cast(input_sorted_p);

                if (return_inverse) {
                    thrust::inclusive_scan(diff_ptr, diff_ptr + dimlen, sum_ptr);
                    thrust::scatter(sum_ptr, sum_ptr + dimlen, indice_ptr, inverse_ptr);
                }

                thrust::sequence(array_ptr, array_ptr + dimlen);
                int32_t num = thrust::unique(array_ptr, array_ptr + dimlen,
                    [=] __device__ (int32_t a, int32_t b)->bool {
                        for(int i = 0; i < dimsize; ++i) {
                            input_sorted_type lhs = input_ptr[i + a * dimsize],
                                            rhs = input_ptr[i + b * dimsize];
                            if (lhs != rhs) return false;
                        }
                        return true;
                    }) - array_ptr;

                cudaMemcpy(output_p, raw_ptr, sizeof(int32_t) * num, cudaMemcpyDeviceToDevice);
                exe.allocator->free(raw_ptr, 2 * dimlen * sizeof(int32_t), raw_allocation);
                output->set_shape({ num });
            '''
        )
    indice_shape = (output.shape[0], )
    output = input_sorted[output][:]

    new_shape = list(orig_shape[1:])
    new_shape.insert(0, -1)
    output = output.view(new_shape).transpose(dim, 0)
    if temp_shape != None:
        inverse = inverse.view(temp_shape).transpose(dim, 0)

    if return_inverse:
        if return_counts:
            counts = jt.zeros(indice_shape, dtype=jt.int32)
            jt.scatter_(counts, 0, inverse.flatten(), jt.ones(dimlen), reduce='add')
            return output, inverse, counts
        else:
            return output, inverse
    else:
        return output

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
    x = (x!=0.0).ternary(x, 1e-30)
    angle = (y/x).arctan()
    mask = (x<0)&(y<0)
    angle = angle - mask*np.pi
    mask = (x<0)&(y>=0)
    angle = angle + mask*np.pi
    return angle
atan2 = arctan2


def nonzero(x):
    r'''
    Return the index of the elements of input tensor which are not equal to zero.
    '''
    x = jt.where(x)
    x = [xx.unsqueeze(1) for xx in x]
    if len(x)<2:
        return x[0]
    x = jt.concat(x,dim=1)
    return x

jt.Var.nonzero = nonzero


def arange(start=0, end=None, step=1,dtype=None):
    if isinstance(start, jt.Var): start = start.item()
    if isinstance(end, jt.Var): end = end.item()
    if isinstance(step, jt.Var): step = step.item()
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


def split(d, split_size, dim=0):
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
    s_last = len(split_size)-1
    gopt_disable = jt.flags.gopt_disable or jt.flags.use_acl
    for j, i in enumerate(split_size):
        if i==0:
            shape = list(d.shape)
            shape[dim]=0
            new_d = jt.zeros(tuple(shape),dtype=d.dtype)
            ans.append(new_d)
            continue

        ss = (slice(None),)*dim+(slice(last,last+i),)
        if gopt_disable:
            new_d = d.getitem(ss)
        else:
            new_d, d = d.getitem(ss, int(j==s_last))

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

# reference: https://github.com/pytorch/pytorch/blob/25d5a815f74db80ef19a3f714709b55b05675245/torch/_refs/__init__.py
def diagonal(x, offset=0, dim1=0, dim2=1):
    def __normalize_dim(d, rank):
        if d < 0:
            d += rank
        if d < 0 or d >= rank:
            msg = f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {d})"
            raise IndexError(msg)
        return d
    assert x.ndim >= 2, f"diagonal dimensions requires ndim larger than 2, but got {x.ndim}"
    dim1 = __normalize_dim(dim1, x.ndim)
    dim2 = __normalize_dim(dim2, x.ndim)
    assert dim1 != dim2, f"diagonal dimensions cannot be identical {dim1}, {dim2}"

    if offset >= 0:
        diag_size = max(min(x.shape[dim1], x.shape[dim2] - offset), 0)
    else:
        diag_size = max(min(x.shape[dim1] + offset, x.shape[dim2]), 0)

    sizes = []
    indices = []
    lsizes = 0
    dim_diag = x.ndim - 2
    abs_offset = offset if offset >= 0 else -offset
    for i, s in enumerate(x.shape):
        if i == dim1:
            if offset >= 0:
                indices.append(f"i{dim_diag}")
            else:
                indices.append(f"i{dim_diag}+{abs_offset}")
        elif i == dim2:
            if offset >= 0:
                indices.append(f"i{dim_diag}+{abs_offset}")
            else:
                indices.append(f"i{dim_diag}")
        else:
            indices.append(f"i{lsizes}")
            sizes.append(s)
            lsizes += 1
    out_shape = tuple(sizes + [diag_size])
    return x.reindex(out_shape, indices)

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

def kthvalue(input, k, dim=None, keepdim=False, keepdims=False):
    keepdim = keepdim or keepdims
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

def _prod(x,dim=0):
    x = jt.log(x)
    x = x.sum(dim=dim)
    return jt.exp(x)


def numpy_cumsum(x, dim=None):
    ''' cumsum implemented with numpy or cupy.
    
        This function should not be called directly. Instead, jittor.misc.cumsum is recommended.
    '''
    def cumsum_forward(np, data):
        a = data['inputs'][0]
        b = data['outputs'][0]
        np.cumsum(a, axis=dim, out=b)

    def cumsum_backward(np, data):
        dout = data['dout']
        out = data['outputs'][0]
        np.cumsum(np.flip(dout, dim), axis=dim, out=out)
        np.copyto(out, np.flip(out, dim))
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    return jt.numpy_code(x.shape, x.dtype, [x], cumsum_forward, [cumsum_backward])

def cub_cumsum(x, dim=None):
    ''' cumsum implemented with CUB.
    
        This function should not be called directly. Instead, jittor.misc.cumsum is recommended.
    '''
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    shape = list(x.shape)
    if (dim != -1 and dim != len(shape) - 1):
        order = list(range(len(shape)))
        order[dim], order[-1] = order[-1], order[dim]
        shape[dim], shape[-1] = shape[-1], shape[dim]
        x = x.permute(order)
    if (len(shape) > 2):
        x = x.reshape([-1, shape[-1]])
    x = jt.compile_extern.cub_ops.cub_cumsum(x)
    if (len(shape) > 2):
        x = x.reshape(shape)
    if (dim != -1 and dim != len(shape) - 1):
        x = x.permute(order)
    return x

def cumsum(x, dim=None):
    '''
    Parameters:
    -----------
    x: jt.var
    dim: int

    Returns:
    --------
    the cumulative sum in dim of x
    '''
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    if jt.flags.use_cuda:
        return cub_cumsum(x, dim)
    else:
        return numpy_cumsum(x, dim)

jt.Var.cumsum = cumsum

def cumprod(x,dim=None):
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

# def triu_(x,diagonal=0):
#     r'''
#     Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

#     The upper triangular part of the matrix is defined as the elements on and above the diagonal.

#     Args:
#         x – the input tensor.

#         diagonal – the diagonal to consider,default =0
#     '''
#     l = len(x.shape)
#     assert l>1
#     overflow_conditions=[f'i{l-1}<i{l-2}+{diagonal}']
#     indexs = [f'i{i}' for i in range(l)]
#     return x.reindex(x.shape,indexs,overflow_conditions=overflow_conditions,overflow_value=0)

# jt.Var.triu_ = triu_

def print_tree(now, max_memory_size, prefix1, prefix2, build_by):
    def format_size(s, end='B'):
        if (s < 1024):
            s = str(s)
            return s + ' '+end

        if (s < 1024*1024):
            s = format(s/1024, '.2f')
            return s + ' K'+end

        if (s < 1024*1024*1024):
            s = format(s/1024/1024, '.2f')
            return s + ' M'+end

        s = format(s/1024/1024/1024, '.2f')
        return s + ' G'+end

    out = ''
    tab = '   '
    out += prefix1+now['name']+'('+now['type']+')\n'
    out += prefix2+'['+format_size(now['size'])+'; '+format(now['size']/max_memory_size*100, '.2f')+'%; cnt:'+format_size(now['cnt'],'') + ']\n'
    if len(now['children']) == 0 and len(now['vinfo']):
        out += prefix2+now['vinfo'][0]
        if len(now['vinfo']) > 1: out += "..."
        out += '\n'
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
        vinfo = v__[0].split("{")[0]
        var = {'size':int(v__[1]), 'stack':[], 'cnt':1, "vinfo":vinfo}
        v__ = v__[2:-1]
        for s_ in v__:
            s__ = s_.split(div3)
            s = {'path':s__[0], 'name':s__[1], 'type':s__[2]}
            var['stack'].append(s)
        vars.append(var)
    if (build_by == 0): # build tree by name
        tree = {'name':'root', "children":[], 'size':0, 'cnt':1, 'path':[], 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['name'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['name'])
                if (ch is not None):
                    if (not s['path'] in ch['path']):
                        ch['path'].append(s['path'])
                    assert(ch['type']==s['type'])
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'], 'cnt':v['cnt'], 'path':[s['path']], 'type':s['type'], 'vinfo':[v['vinfo']]}
                    now['children'].append(now_)
                    now = now_
    elif (build_by == 1): # build tree by path
        tree = {'name':'root', "children":[], 'size':0, 'cnt':0, 'path':'_root_', 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['path'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['path'])
                if (ch is not None):
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'],  'cnt':v['cnt'], 'path':s['path'], 'type':s['type'], 'vinfo':[v['vinfo']]}
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
    
def python_pass_wrapper(mod_func, args, kw):
    import importlib
    mod, func = mod_func.rsplit(".", 1)
    mod = importlib.import_module(mod)
    func = getattr(mod, func)
    args = args + ("**kw",)
    args = ",".join(args)
    return eval(f"func({args})")

def auto_parallel(n, src, block_num=1024, **kw):
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
    int thread_num = 256*{block_num};
    {xn.join([f"int tn{i} = NanoVector::get_nbits(std::min(thread_num, {pnargs2[i]})) - 2;thread_num >>= tn{i};" for i in reversed(range(n))])}
    thread_num = 1<<({"+".join([f"tn{i}" for i in range(n)])});
    int p1 = std::max(thread_num/{block_num}, 1);
    int p2 = std::min(thread_num, {block_num});
    {func_name}_entry<<<p1,p2>>>({entry_func_args});
#else
    {xn.join([f"for (int i{i}=0; i{i}<{pnargs2[i]}; i{i}++)" for i in range(n)])}
    {func_name}_inner({call_args});
#endif
}}
"""
    return new_src


def numpy_cumprod(a, dim):
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
    if steps > 1:
        res = jt.index((steps,))[0]
        res = res*float((end-start)/(steps-1))+start
    else:
        res = jt.array([start])
    return res

def randperm(n, dtype="int32"):
    key = jt.random((n,))
    index, _ = jt.argsort(key)
    return index.cast(dtype)

def set_global_seed(seed, different_seed_for_mpi=True):
    ''' Sets the seeds of the random number generators of Python, numpy and jittor,
    simultaneously.

    .. note::
    Jittor also gurantees each worker of jittor.dataset.Dataset to hold a different seed,
    also gurantees each process hold a different seed which using mpi,
    which is (global_seed ^ (worker_id*1167)) ^ 1234 + jt.rank * 2591
    '''
    if (different_seed_for_mpi):
        seed = seed + jt.rank * 2591
    import random
    random.seed(seed)
    jt.set_seed(seed)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass

import time
set_global_seed(int(time.time() * 1000000) % 100000007)

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

class _CTCLossFunction(jt.Function):
    def execute(self, log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
        self.blank = blank
        T, N, C = log_probs.shape
        _N, S = targets.shape
        assert _N == N
        log_alpha = jt.full([T,N,S*2+1], -1e30)
        result = jt.empty((N,))
        jt.code([log_probs, targets, input_lengths, target_lengths], [log_alpha, result], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        @out0(j,i,k) = std::log(
                            std::exp(l1-m) +
                            std::exp(l2-m) +
                            std::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        """, cuda_src=f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        @out0(j,i,k) = ::log(
                            ::exp(l1-m) +
                            ::exp(l2-m) +
                            ::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                 __syncthreads();
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = ::max(l1, l2);
                    out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """)
        self.saved_var = [log_probs, targets, input_lengths, target_lengths, log_alpha, result]
        return result

    def grad(self, dout):
        blank = self.blank
        inputs = self.saved_var + [dout]
        dlog_probs = jt.zeros_like(inputs[0])
        dlog_alpha = jt.zeros_like(inputs[4])
        jt.code(inputs, [dlog_probs, dlog_alpha], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                if (input_len==0)
                    // write out1 --> read in6
                    // out1(i) = out0(0,i,0);
                    @out1(0,i,0) = @in6(i);
                else {{
                    out_type l1 = @in4(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @in4(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    // out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = std::exp(l1-m);
                    out_type l2_exp = std::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        out_type l1_exp = std::exp(l1-m);
                        out_type l2_exp = std::exp(l2-m);
                        out_type l3_exp = std::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        @out0(j,i,target) += dalpha;

                        @out1(j-1,i,k) += dalpha * l1_exp / sumexp;
                        if (k>0)
                            @out1(j-1,i,k-1) += dalpha * l2_exp / sumexp;
                        if (k>1 && target_2 != target)
                            @out1(j-1,i,k-2) += dalpha * l3_exp / sumexp;
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                @out0(0,i,blank) += @out1(0,i,0);
                if (target_len)
                    @out0(0,i,@in1(i,0)) += @out1(0,i,1);
            }}
        """, cuda_src=f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                if (input_len==0)
                    // write out1 --> read in6
                    // out1(i) = out0(0,i,0);
                    @out1(0,i,0) = @in6(i);
                else {{
                    out_type l1 = @in4(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @in4(input_len-1, i, target_len*2-1);
                    out_type m = ::max(l1, l2);
                    // out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = ::exp(l1-m);
                    out_type l2_exp = ::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        out_type l1_exp = ::exp(l1-m);
                        out_type l2_exp = ::exp(l2-m);
                        out_type l3_exp = ::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        atomicAdd(&@out0(j,i,target), dalpha);

                        atomicAdd(&@out1(j-1,i,k), dalpha * l1_exp / sumexp);
                        if (k>0)
                            atomicAdd(&@out1(j-1,i,k-1), dalpha * l2_exp / sumexp);
                        if (k>1 && target_2 != target)
                            atomicAdd(&@out1(j-1,i,k-2), dalpha * l3_exp / sumexp);
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                __syncthreads();
                if (threadIdx.x==0) {{
                    @out0(0,i,blank) += @out1(0,i,0);
                    if (target_len)
                        @out0(0,i,@in1(i,0)) += @out1(0,i,1);
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """)
        return (dlog_probs,)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    '''The Connectionist Temporal Classification loss.


    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Input:

        log_probs: shape is [T, N, C], T is the sequence length, N is the batch size, C is the class number.
        targets: shape is [N, S], N is the batch size, S is the target sequence length, element should between [0,C).
        input_lengths: shape is [N], which represents the length of input, element should between [0,T].
        target_lengths: shape is N, which represents the length of target, element should between [0,S].
        blank (int, default 0): blank label index
        reduction (string): reduce batch loss,
            if reduction is none, it will return (N,) array,
            if reduction is mean or sum, it will return one scalar
        zero_infinity (bool, default False):
            zero_infinity for grad

    Example:

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        loss = jt.ctc_loss(input, target, input_lengths, target_lengths)

        dinput = jt.grad(loss, input)

    '''
    result = _CTCLossFunction.apply(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity)
    if reduction=="mean":
        return result.mean()
    elif reduction=="sum":
        return result.sum()
    assert reduction=="none"
    return result


class CTCLoss(jt.Module):
    '''The Connectionist Temporal Classification loss.


    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf


    Args:

        blank (int, default 0): blank label index
        reduction (string): reduce batch loss,
            if reduction is none, it will return (N,) array,
            if reduction is mean or sum, it will return one scalar
        zero_infinity (bool, default False):
            zero_infinity for grad

    Input:

        log_probs: shape is [T, N, C], T is the sequence length, N is the batch size, C is the class number.
        targets: shape is [N, S], N is the batch size, S is the target sequence length, element should between [0,C).
        input_lengths: shape is [N], which represents the length of input, element should between [0,T].
        target_lengths: shape is N, which represents the length of target, element should between [0,S].

    Example:

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        ctc_loss = jt.CTCLoss()
        loss = ctc_loss(input, target, input_lengths, target_lengths)

        dinput = jt.grad(loss, input)

    '''
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def execute(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction, self.zero_infinity)

def _simple_for(x, func):
    with jt.flag_scope(compile_options={"FLAGS: -O2 ":1}):
        src = f'''
        __inline_static__
        @python.jittor.auto_parallel(1)
        void kernel(int n0, int i0, in0_type* _x, out0_type* y) {{
            using namespace std;
            auto x = _x[i0];
            y[i0] = {func};
        }}
        kernel(in0->num, 0, in0_p, out0_p);
        '''
        return jt.code(x.shape, "bool", [x], cpu_src=src, cuda_src=src)

def isnan(x): return _simple_for(x, "isnan(float(x))")
jt.Var.isnan = isnan
def isfinite(x): return _simple_for(x, "!isnan(float(x)) && !isinf(float(x))")
jt.Var.isfinite = isfinite
def isinf(x): return _simple_for(x, "isinf(float(x))")
jt.Var.isinf = isinf
def isneginf(x): return _simple_for(x, "x<0 && isinf(float(x))")
jt.Var.isneginf = isneginf
def isposinf(x): return _simple_for(x, "x>0 && isinf(float(x))")
jt.Var.isposinf = isposinf

# fake torch interface
def contiguous(x): return x.clone()
jt.Var.contiguous = contiguous
def cpu(x): return x.clone()
jt.Var.cpu = cpu
def to(x, *args, **kargs):
    args += tuple(kargs.values())
    if len(args) >= 1:
        s = args[0]
        if isinstance(s, jt.NanoString) or callable(s):
            return x.cast(s)
        s = str(s)
        if "cuda" in s:
            jt.flags.use_cuda = 1
        elif "cpu" in s:
            jt.flags.use_cuda = 0
    return x.clone()
jt.Var.to = to

def rsqrt(x):
    return 1/jt.sqrt(x)
jt.Var.rsqrt = rsqrt

def from_torch(x):
    '''
    Convert torch Tensor to Jittor Var
    '''
    return jt.Var(x.cpu().numpy())

def triu(input: jt.Var, diagonal:int=0) -> jt.Var:
    ''' Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    :param input: the input tensor.
    :param diagonal:  the diagonal to consider(int).

    Example::

        a = jt.ones(3, 3)
        b = jt.triu(a)
        assert jt.all_equal(b, [[1,1,1],[0,1,1],[0,0,1]])
        
        b = jt.triu(a, diagonal=1)
        assert jt.all_equal(b, [[0,1,1],[0,0,1],[0,0,0]])
        
        b = jt.triu(a, diagonal=-1)
        assert jt.all_equal(b, [[1,1,1],[1,1,1],[0,1,1]])

    '''
    index = input.index()
    mask = index[-2] <= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))
jt.Var.triu = triu
jt.Var.triu_ = lambda x,diagonal=0: x.assign(x.triu(diagonal))

def tril(input: jt.Var, diagonal:int=0) -> jt.Var:
    ''' Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    :param input: the input tensor.
    :param diagonal:  the diagonal to consider(int).

    Example::

        a = jt.ones(3, 3)
        b = jt.tril(a)
        assert jt.all_equal(b, [[1,0,0],[1,1,0],[1,1,1]])
        
        b = jt.tril(a, diagonal=1)
        assert jt.all_equal(b, [[1,1,0],[1,1,1],[1,1,1]])
        
        b = jt.tril(a, diagonal=-1)
        assert jt.all_equal(b, [[0,0,0],[1,0,0],[1,1,0]])

    '''
    index = input.index()
    mask = index[-2] >= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))
jt.Var.tril = tril
jt.Var.tril_ = lambda x: x.assign(x.tril())

def all_equal(a: jt.Var, b: jt.Var) -> bool:
    return (a == b).all().item()
jt.all_equal = all_equal

def _to_float(x: jt.Var) -> jt.Var:
    if x.dtype != "float64": x = x.float()
    return x
jt.Var._to_float = _to_float

def index_select(x: jt.Var, dim:int, index: jt.Var) -> jt.Var:
    '''Returns a new var which indexes the x var along dimension dim using the entries in index.

The returned var has the same number of dimensions as the original var (x). The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.

    :param x: the input tensor.
    :param dim:  the dimension to index.
    :param index:  the 1-D tensor containing the indices to index.

    Example::

        x = jt.randn(3, 4)
        indices = torch.tensor([2, 1])
        y = jt.index_select(x, 0, indices)
        assert jt.all_equal(y, x[indices])
        y = jt.index_select(x, 1, indices)
        assert jt.all_equal(y, x[:, indices])


    '''
    return x.getitem(((slice(None),)*dim)+(index,))
jt.index_select = index_select

def multinomial(weights: jt.Var, num_samples: int, replacement: bool=False) -> jt.Var:
    ''' Returns a var where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of input weights.

    :param weights: the input probability.
    :param num_samples: number of samples.
    :param replacement: whether to draw with replacement or not.


    Example::

        weights = jt.float32([0, 10, 3, 0])
        x = jt.multinomial(weights, 2)
        assert jt.all_equal(x, [1, 2]) or jt.all_equal(x, [2, 1])
        x = jt.multinomial(weights, 4, replacement=True)
        assert x.shape == (4, )

        weights = jt.float32([[0,0,2],[0,1,0], [0.5,0,0]])
        x = jt.multinomial(weights, 1)
        assert jt.all_equal(x, [[2],[1],[0]])

    '''
    if replacement:
        cum_probs = jt.cumsum(weights)[..., None, :]
        cum_probs_l = cum_probs[..., :-1]
        cum_probs_r = cum_probs[..., 1:]
        shape = weights.shape[:-1] + (num_samples, 1)
        rand = jt.rand(shape) * cum_probs[..., :1, -1:]
        one_hot = jt.logical_and(cum_probs_l < rand, rand <= cum_probs_r)
        index = one_hot.index(one_hot.ndim - 1) + 1
        return (one_hot * index).sum(-1)
    else:
        # A-Res algorithm
        # Pavlos S. Efraimidis and Paul G. Spirakis, 2006, Weighted random sampling with a reservoir
        assert num_samples <= weights.shape[-1], "num_samples larger than the input"
        # prevent rand generate 1, 1^inf = 1, with override other result
        a = jt.rand(weights.shape).minimum(0.999999)
        rand = a ** (1/weights)
        _, indices = jt.topk(rand, num_samples)
        return indices

def histc(input, bins, min=0., max=0.):
    ''' Return the histogram of the input N-d array.

    :param input: the input array.
    :param bins: number of bins.
    :param min: min of the range.
    :param max: max of the range.

    Example::

        inputs = jt.randn((40,40))
        joup = jt.histc(x, bins=10)
        
    '''
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    if bins <= 0:
        raise RuntimeError(f"bins must be > 0, but got {bins}")
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input <= max)] - min) / bin_length).int().reshape(-1)
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins,], ["@e0(i0)"], extras=[histc])
    if hist.sum() != histc.shape[0]:
        hist[-1] += 1
    return hist

def peek_s(x):
    if isinstance(x, jt.Var):
        return x.peek()
    if isinstance(x, (list, tuple)):
        res = "["
        for a in x:
            res += peek_s(a)
            res += ", "
        res += "]"
        return res
    if isinstance(x, dict):
        res = "{"
        for a in x:
            res += a
            res += ":"
            res += peek_s(x[a])
            res += ", "
        res += "}"
        return res
    if isinstance(x, str):
        return x
    return x.__class__.__name__

def peek(x):
    print(peek_s(x))

class Finfo:
    pass
bfloat16_finfo = Finfo()
bfloat16_finfo.min = -1e38
bfloat16_finfo.max = 1e38

def finfo(dtype):
    if dtype == "bfloat16":
        return bfloat16_finfo
    return np.finfo(str(dtype).split('.')[-1])

def iinfo(dtype):
    return np.iinfo(str(dtype).split('.')[-1])


def index_select(input,dim,indices):
    return input[(None,)*dim+(indices,)]

jt.Var.index_select = index_select

def cuda(x):
    jt.flags.use_cuda = 1
    return x
jt.Var.cuda = cuda
jt.Var.npu = cuda

def expm1(x):
    return jt.exp(x) - 1


def isin(elements, test_elements, assume_unique=False, invert=False):
    
    elements = elements.unsqueeze(-1)
    test_elements = test_elements.unsqueeze(0)
    comparison = elements == test_elements
    result = comparison.any(dim=-1)

    if invert:
        result = jt.logical_not(result)
    
    return result