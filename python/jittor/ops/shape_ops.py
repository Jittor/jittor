"""Shape ops tensor operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from collections.abc import Sequence, Iterable
from jittor_core import Var
from .._runtime.dispatch import select_kernel, try_dispatch

def _repeat_interleave_cpu_source():
    return ('''
        @alias(x, in0)
        @alias(offsets, in1)
        @alias(out, out0)
        int64_t total = out->num;
        int n = x_shape0;
        int64_t inner = out->num / out_shape0;
        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t out_row = linear / inner;
            int lo = 0, hi = n - 1;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if ((int64_t)offsets_p[mid] > out_row) hi = mid;
                else lo = mid + 1;
            }
            out_p[linear] = (out_type)x_p[(int64_t)lo * inner + (linear % inner)];
        }
        '''
    )


def _stack_cpu_source(suffix, n, write_lines):
    return (f"""
    const index_t suffix = {suffix};
    for (index_t iid=0; iid<in0_shape0; ++iid) {{
        index_t prefix = iid / suffix;
        index_t rem = iid - prefix * suffix;
        index_t base_out = prefix * ({n} * suffix);
{write_lines}
    }}
    """
    )


def _unbind_cpu_source(suffix, n, write_lines):
    return (f"""
    const index_t suffix = {suffix};
    const index_t full_stride = suffix * {n};
    for (index_t oid=0; oid<out0_shape0; ++oid) {{
        index_t prefix = oid / suffix;
        index_t rem = oid - prefix * suffix;
        index_t base_in = prefix * full_stride + rem;
{write_lines}
    }}
    """
    )


def repeat_interleave(x,repeats,dim=None,output_size=None):
    # torch-compatible: `repeats` may be a python int (every element repeated the
    # same number of times) OR a 1-D Var/list giving a per-element repeat count
    # (len == x.shape[dim]). The per-element form is required by e.g. the Qwen-VL
    # vision tower (`repeat_interleave(grid_thw[:,1]*grid_thw[:,2], grid_thw[:,0])`).
    from jittor.backends.cuda.kernels.misc.tensor_ops import (
        _repeat_interleave_dim0_cuda, _stack_no_grad_cuda_fast,
        _unbind_no_grad_cuda_fast, _unique_code_cuda, _scan_2d_cuda,
    )
    import jittor as jt
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    if dim < 0:
        dim += x.ndim

    if isinstance(repeats, int):
        tar_shape = list(x.shape)
        tar_shape[dim] = tar_shape[dim]*repeats
        dims = []
        for i in range(len(tar_shape)):
            if dim==i:
                dims.append(f"i{i}/{repeats}")
            else:
                dims.append(f"i{i}")
        return x.reindex(tar_shape,dims)

    result = _repeat_interleave_dim0_cuda(x, repeats, dim, output_size, _repeat_interleave_cpu_source)
    if result is not None:
        return result

    # per-element repeats: build a gather index along `dim` then index_select.
    if isinstance(repeats, Var):
        rep_list = [int(c) for c in repeats.numpy().reshape(-1)]
    else:
        rep_list = [int(c) for c in repeats]
    n = x.shape[dim]
    if len(rep_list) == 1 and n != 1:
        rep_list = rep_list * n
    assert len(rep_list) == n, \
        f"repeat_interleave: repeats length {len(rep_list)} != dim size {n}"
    idx = []
    for i, c in enumerate(rep_list):
        idx.extend([i] * c)
    index = jt.array(idx).int64()
    if index.shape[0] == 0:
        new_shape = list(x.shape); new_shape[dim] = 0
        return jt.zeros(new_shape, x.dtype)
    if dim == 0 and n == 1 and x.ndim == 1 and _jittor_dtype_name(x.dtype) in (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
    ):
        value = int(x.item())
        out = jt.full((index.shape[0],), value, x.dtype)
        try:
            out._jittor_constant_index_value = value
        except Exception:
            pass
        return out
    return x.getitem(tuple(slice(None) if d != dim else index for d in range(x.ndim)))


def t(x):
    pose = [i for i in range(x.ndim)]
    pose[-1], pose[-2] = pose[-2], pose[-1]
    return x.transpose(*pose)


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
    import jittor as jt
    assert isinstance(x, Sequence)
    if isinstance(x, tuple):
        x = list(x)
    for i,x_ in enumerate(x):
            x[i] = jt.array(x_)
    if len(x) < 2:
        return x[0].unsqueeze(dim)

    fast = jt.misc._stack_no_grad_cuda_fast(x, dim, _stack_cpu_source)
    if fast is not None:
        return fast

    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.concat(res, dim=dim)


def flip(x, dim=0, dims=None):
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
    if dims is not None:          # torch spells the flip axis kwarg `dims`
        dim = dims
    if isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)           # copy: the loop below mutates dim in place
    for i in range(len(dim)):
        if dim[i]<0:
            dim[i] += x.ndim
        if dim[i] < 0 or dim[i] >= x.ndim:
            raise ValueError(
                "flip: dim {} out of range for tensor with {} dimensions".format(
                    dim[i], x.ndim
                )
            )
    result = try_dispatch("tensor.flip", x, dim)
    if result is not None:
        return result
    dim = set(dim)

    tar_dims = []
    for i in range(len(x.shape)):
        if i in dim:
            tar_dims.append(f"xshape{i}-1-i{i}")
        else:
            tar_dims.append(f"i{i}")
    return x.reindex(x.shape, tar_dims)


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
    import jittor as jt
    if dim < 0: dim += len(x.shape)
    fast = jt.misc._unbind_no_grad_cuda_fast(x, dim, _unbind_cpu_source)
    if fast is not None:
        return fast
    return [x[(slice(None),)*dim+(i,)] for i in range(x.shape[dim])]


def meshgrid(*tensors, indexing=None):
    r'''
    Take N tensors, each of which can be 1-dimensional vector, and create N n-dimensional grids,
    where the i th grid is defined by expanding the i th input over dimensions defined by other inputs.

    `indexing` matches torch.meshgrid: 'ij' (default, matrix indexing — jittor's
    native behavior) or 'xy' (Cartesian, which swaps the first two axes). swin and
    many vision models call torch.meshgrid(..., indexing='ij').
    '''
    if len(tensors)==1 and isinstance(tensors[0], list):
        tensors = tensors[0]
    size = len(tensors)
    shape = []
    for i in range(size):
        assert isinstance(tensors[i],Var) and tensors[i].ndim==1
        shape.append(tensors[i].shape[0])
    grids = []
    view_shape = [1]*size
    for i in range(size):
        vs = view_shape[:]
        vs[i]=-1
        grids.append(tensors[i].reshape(vs).expand(shape))

    if indexing == "xy" and size >= 2:
        grids = [g.transpose(0, 1) for g in grids]
    return grids


def _split_slice(d, selection, is_last, gopt_disable):
    if gopt_disable:
        return d.getitem(selection), d
    return d.getitem(selection, int(is_last))


def _split_slice_acl(d, selection, is_last, gopt_disable):
    return d.getitem(selection), d


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
    import jittor as jt
    result = try_dispatch("tensor.split", d, split_size, dim)
    if result is not None:
        return result
    if isinstance(split_size,int):
        shape = d.shape[dim]
        if shape % split_size == 0:
            split_size = [split_size]*(shape//split_size)
        else:
            split_size = [split_size]*(shape//split_size)+[shape%split_size]
    if dim<0:
        dim+=d.ndim
    if dim < 0 or dim >= d.ndim:
        raise ValueError("split: dim {} out of range for tensor with {} dimensions".format(dim, d.ndim))
    if isinstance(split_size, Iterable):
        if sum(split_size) != d.shape[dim]:
            raise ValueError(
                "split: split sizes must sum to dimension {}, got {}".format(
                    d.shape[dim], sum(split_size)
                )
            )

    ans = []
    last = 0
    s_last = len(split_size)-1
    gopt_disable = jt.flags.gopt_disable
    slice_kernel = select_kernel("misc.split_slice", d)
    for j, i in enumerate(split_size):
        if i==0:
            shape = list(d.shape)
            shape[dim]=0
            new_d = jt.zeros(tuple(shape),dtype=d.dtype)
            ans.append(new_d)
            continue

        ss = (slice(None),)*dim+(slice(last,last+i),)
        new_d, d = slice_kernel(d, ss, j == s_last, gopt_disable)

        last +=i
        ans.append(new_d)
    return tuple(ans)


def view_as(x,y):
    return x.reshape(y.shape)


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
    import jittor as jt
    result = try_dispatch("tensor.roll", x, shifts, dims)
    if result is not None:
        return result
    if dims is None:
        # torch: when dims is None the tensor is FLATTENED, rolled by the (scalar)
        # shift, then restored to the original shape (NOT rolled along dim 0).
        s = shifts[0] if isinstance(shifts, (tuple, list)) else shifts
        return jt.misc.roll(x.reshape((-1,)), s, 0).reshape(x.shape)
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    assert len(dims) == len(shifts)
    ids = [ f'i{i}' for i in range(x.ndim) ]
    for i in range(len(dims)):
        shift = shifts[i]
        # normalize negative dims: f'i{d}' with d=-1 emits the literal 'i-1' (an
        # undeclared codegen variable -> "'op0_i' was not declared" compile error).
        # torch allows dims=-1; map it to a real axis index for the reindex expression.
        d = dims[i] % x.ndim
        size = x.shape[d]
        if size == 0:
            # torch.roll is a no-op on an empty dim; avoid modulo-by-zero.
            continue
        shift = shift % size
        if shift<0: shift += size
        ids[d] = f'(i{d}<{shift}?i{d}+{size-shift}:(i{d}-{shift}))'
    return x.reindex(x.shape, ids)


def triu(input: Var, diagonal:int=0) -> Var:
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
    import jittor as jt
    result = try_dispatch("tensor.triu", input, diagonal)
    if result is not None:
        return result
    index = input.index()
    mask = index[-2] <= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))


def tril(input: Var, diagonal:int=0) -> Var:
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
    import jittor as jt
    index = input.index()
    mask = index[-2] >= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))
