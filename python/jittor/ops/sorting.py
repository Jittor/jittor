"""Sorting tensor operations."""

from jittor_core import Var
from .. import _arg_policy
from .._runtime.dispatch import select_kernel

def sort(input, dim=-1, descending=False, stable=False):
    ''' Sort along ``dim``, returning ``(values, indices)`` like ``torch.sort``.

    ``stable=True`` is not implemented: jittor's ``argsort`` is not a stable sort
    on CPU (verified against ``numpy.argsort(kind="stable")`` -- equal keys come
    back permuted), so accepting the flag would promise an ordering the sort does
    not deliver. It happens to be stable on the current CUDA backend, which is
    exactly why the difference has to be refused rather than assumed.
    '''
    import jittor as jt
    if stable:
        _arg_policy.unsupported(
            "jittor.sort", "stable", stable,
            "jittor's argsort is not stable on CPU, so equal elements come back "
            "in an arbitrary order and the indices of tied keys differ from "
            "torch's")
    index, value = jt.argsort(input, dim, descending)
    return value, index


def median(x, dim=None, keepdim=False, keepdims=False):
    import jittor as jt
    keepdim = keepdim or keepdims
    if dim is None:
        x = x.reshape(-1)
        dim = 0
        requested_dim = dim
    else:
        requested_dim = dim
        if dim < 0:
            dim += x.ndim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(
            f"median(): dimension out of range (expected to be in range of "
            f"[-{x.ndim}, {x.ndim - 1}], but got {requested_dim})"
        )

    sorted_result = jt.argsort(x, dim)
    if isinstance(sorted_result, (tuple, list)):
        _, values = sorted_result
    else:
        values = jt.gather(x, dim, sorted_result)

    slices = [slice(None)] * x.ndim
    k = (x.shape[dim] - 1) // 2
    if keepdim:
        slices[dim] = slice(k, k + 1)
    else:
        slices[dim] = k
    return values[tuple(slices)]


def _unique_code_generic(*args, **kwargs):
    import jittor as jt
    with jt.flag_scope(compile_options={}):
        return jt.code(*args, **kwargs)


def unique(
    input: Var,
    sorted: bool=True,          # torch kwarg; jittor's unique is always sorted
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
    from jittor.backends.cuda.kernels.misc import tensor_ops as _cuda_tensor_ops
    import jittor as jt

    # One implementation, every dtype, both devices. There used to be four
    # arms here: the native int32 CUDA kernel; "cast to int32 and recurse" for
    # integers that fit; `flag_scope(use_cuda=0)` -- compute the whole thing on
    # the CPU and move it back -- for everything else; and the CPU path itself.
    # Choosing between them read a reduction back to the host with a Python
    # truth test, once per call, in the middle of a lazy graph.
    #
    # The reason given for the detour was that the CUDA kernel "only sorts
    # correctly for 32-bit int keys". cub sorts any key type; what was wrong was
    # the *index* var the kernels write their answer into -- see the second
    # jt.code below -- plus a hand-carved scratch buffer that misaligned 64-bit
    # keys. Both are fixed here, so the arms that worked around them are gone,
    # along with the CPU detour they sent float inputs down -- whose comparator
    # truncated the sort key to int.
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

    code = select_kernel("misc.unique_code", input_flatten)
    indice = code((input_flatten.shape[0], ), 'int32', [input_flatten],
        cpu_header='''
        #include <algorithm>
        ''',
        cpu_src='''
        @alias(input_flatten, in0)
        @alias(indice, out)

        int dimlen = input_flatten_shape0, dimsize = input_flatten_shape1;
        for(int i = 0; i < dimlen; ++i) @indice(i) = i;
        // input_flatten_type, not int. Truncating the key made 1.5 and 1.2
        // compare equal, so the duplicate-dropping pass -- which only
        // merges *neighbours* -- left both in the output, unsorted.
        std::sort(&@indice(0), &@indice(dimlen), [&](int a, int b){
            for(int i = 0; i < dimsize; ++i) {
                input_flatten_type lhs = @input_flatten(a, i),
                                   rhs = @input_flatten(b, i);
                if (lhs != rhs) return lhs < rhs;
            }
            return false;
        });
        ''',
        cuda_header=_cuda_tensor_ops.unique_sort_header(),
        cuda_src=
        _cuda_tensor_ops.unique_sort_source()
    )
    input_sorted = input_flatten[indice][:]

    dimlen = indice.shape[0]

    # counts are derived from inverse, so the kernel has to fill inverse in
    # whenever either of the two is asked for.
    need_inverse = return_inverse or return_counts

    diff = jt.logical_not(jt.all(input_sorted[1:] == input_sorted[: -1], 1))
    diff = jt.concat([Var([False]), diff], 0)
    diff = jt.array(diff, dtype = jt.int32)

    # `output` holds *positions* in input_sorted, so it is an index var --
    # both kernels below write indices into it and the caller immediately
    # gathers with it. It used to be created with `input_sorted.dtype`, and
    # the CUDA body memcpy's raw int32 indices into it: with an int32 input
    # that reinterpretation is a no-op and everything worked, with any other
    # dtype the indices came back as garbage. That, not cub, is what "the
    # CUDA unique kernel only sorts correctly for 32-bit int keys" was.
    output, inverse = code(
        [(-input_sorted.shape[0], ), (indice.shape)],
        [indice.dtype, indice.dtype],
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
        f"bool return_inverse = {int(need_inverse)};" +
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
        cuda_header=_cuda_tensor_ops.unique_compact_header(),
        cuda_src=
        _cuda_tensor_ops.unique_compact_source(need_inverse)
    )
    indice_shape = (output.shape[0], )
    output = input_sorted[output][:]

    new_shape = list(orig_shape[1:])
    new_shape.insert(0, -1)
    output = output.view(new_shape).transpose(dim, 0)
    if temp_shape != None:
        inverse = inverse.view(temp_shape).transpose(dim, 0)

    if return_counts:
        counts = jt.zeros(indice_shape, dtype=jt.int32)
        jt.scatter_(counts, 0, inverse.flatten(), jt.ones(dimlen), reduce='add')

    if return_inverse and return_counts:
        return output, inverse, counts
    if return_inverse:
        return output, inverse
    if return_counts:
        return output, counts
    return output


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    r'''Eliminates all but the FIRST element from every consecutive group of
    equivalent elements (torch.unique_consecutive). Unlike ``unique`` this does NOT
    sort and only collapses runs, so ``[1,1,2,2,1]`` -> ``[1,2,1]``. Needed by the
    Qwen2.5-VL window-attention index computation.
    '''
    import jittor as jt
    if not isinstance(input, Var):
        input = jt.array(input)
    if dim is None:
        flat = input.reshape(-1)
        n = flat.shape[0]
        if n == 0:
            out = flat
            group = jt.zeros([0], dtype='int64')
        else:
            # boundary[i] == True where element i starts a new run.
            if n == 1:
                keep = jt.ones([1], dtype='bool')
            else:
                diff = flat[1:] != flat[:-1]
                keep = jt.concat([jt.ones([1], dtype='bool'), diff], dim=0)
            # group id of each input element = cumulative count of run-starts - 1
            group = keep.int32().cumsum(0) - 1   # 0-based group index per element
            out = flat[keep]
        ret = [out]
        if return_inverse:
            ret.append(group.int64() if n else group)
        if return_counts:
            if n == 0:
                counts = jt.zeros([0], dtype='int64')
            else:
                num_groups = int(out.shape[0])
                # scatter-add in int32 (int64 scatter-add fails to compile on this
                # CUDA build, same atomic limitation as int64 reduce), then widen.
                counts = jt.zeros([num_groups], dtype='int32')
                jt.scatter_(counts, 0, group.int32(), jt.ones([n], dtype='int32'), reduce='add')
                counts = counts.int64()
            ret.append(counts)
        return ret[0] if len(ret) == 1 else tuple(ret)
    # dim-wise: collapse consecutive equal slices along `dim`.
    if dim < 0:
        dim += input.ndim
    moved = input.transpose(0, dim) if dim != 0 else input
    m = moved.shape[0]
    flat2 = moved.reshape(m, -1)
    if m <= 1:
        keep = jt.ones([m], dtype='bool')
    else:
        eq = (flat2[1:] == flat2[:-1]).all(dim=1)
        keep = jt.concat([jt.ones([1], dtype='bool'), eq.logical_not()], dim=0)
    out = moved[keep]
    out = out.transpose(0, dim) if dim != 0 else out
    if not (return_inverse or return_counts):
        return out
    group = keep.int32().cumsum(0) - 1
    ret = [out]
    if return_inverse:
        ret.append(group.int64())
    if return_counts:
        num_groups = int(keep.int32().sum().item())
        counts = jt.zeros([num_groups], dtype='int32')
        jt.scatter_(counts, 0, group.int32(), jt.ones([m], dtype='int32'), reduce='add')
        ret.append(counts.int64())
    return tuple(ret)


def topk(input, k, dim=None, largest=True, sorted=True):
    ''' Top-``k`` values and indices along ``dim``, like ``torch.topk``.

    ``sorted=False`` is accepted and has no effect: torch leaves the ordering
    *unspecified* in that case, and this implementation always returns the k
    elements in sorted order, which satisfies the weaker contract. Nothing the
    caller asked for is withheld, so this one is not routed through
    ``_arg_policy`` -- see ``tests/ops/test_ignored_arguments.py``.
    '''
    import jittor as jt
    if input.numel()==0:
        return jt.array([],dtype=input.dtype),jt.array([],dtype='int64')
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim

    index,values = jt.argsort(input,dim=dim,descending=largest)
    dims = (slice(None),)*dim+(slice(0,k),)
    # int64 like torch, and the same dtype on both branches: the empty case
    # builds its own array and used to be the only one anybody looked at.
    # jt.argsort itself still hands back int32 (a C++ op default with a CUB
    # path under it, out of scope here), so the cast is what makes topk agree
    # with itself.
    indices = index[dims].int64()
    values = values[dims]
    return values,indices


_kthvalue_native_argsort = None  # Captured by the ordered facade.

def _kthvalue_argsort(input, dim):
    import jittor as jt
    return jt.argsort(input, dim=dim)


def kthvalue(input, k, dim=None, keepdim=False, keepdims=False):
    import jittor as jt
    keepdim = keepdim or keepdims
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    # native jt.argsort returns (index, values); the torch_compat layer overrides
    # the module-level argsort to torch's indices-only. Handle both.
    sorter = select_kernel("misc.kthvalue_argsort", input)
    _srt = sorter(input, dim=dim)
    if isinstance(_srt, tuple):
        index, values = _srt
    else:
        index = _srt
        values = jt.gather(input, dim, index)
    dims = (slice(None),)*dim+(slice(k-1,k),)
    indices = index[dims]
    values = values[dims]
    if not keepdim and indices.ndim>1:
        indices = indices.squeeze(dim)
        values = values.squeeze(dim)
    return values,indices
