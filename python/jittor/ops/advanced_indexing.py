"""Advanced indexing tensor operations."""

import numbers as _numbers
from jittor_core import Var
from .._runtime.dispatch import try_dispatch

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
    if len(index.shape) != 1:
        raise ValueError("index_add_: index must be one-dimensional")
    if tensor.shape[0] != index.shape[0]:
        raise ValueError(
            "index_add_: tensor and index lengths differ ({} vs {})".format(
                tensor.shape[0], index.shape[0]
            )
        )
    # torch parity: index_add_ ACCUMULATES all contributions at DUPLICATE indices
    # (e.g. index=[0,0] adds both rows to row 0). The old impl used `x[adv_idx] += t`,
    # which compiles to a read-add-write and is LAST-WRITE-WINS for dups (drops the
    # earlier contribution). Route through the dup-correct out-of-place index_add
    # (scatter_add path) and assign back in place. See test_torch_compat_scatter.py.
    x.assign(x.index_add(dim, index, tensor))


def nonzero(x):
    r'''
    Return the index of the elements of input tensor which are not equal to zero.
    '''
    import jittor as jt
    result = try_dispatch("tensor.nonzero", x)
    if result is not None:
        return result
    x = jt.where(x)
    x = [xx.unsqueeze(1) for xx in x]
    if len(x)<2:
        return x[0]
    x = jt.concat(x,dim=1)
    return x


def index_fill_(x,dim,indexs,val):
    r'''
    Fills the elements of the input tensor with value val by selecting the indices in the order given in index.

    Args:
        x - the input tensor
        dim - dimension along which to index
        index – indices of input tensor to fill in
        val – the value to fill with
    '''
    import jittor as jt
    # NOTE: the old impl (`overflow_conditions=[f'i{dim}=={i}' for i in indexs]`) was
    # broken three ways: f'i{dim}' crashed JIT compile for negative dim (emits 'i-1'),
    # it iterated the index TENSOR into an f-string (only worked for a python int list),
    # and it overwrote `indexs`. Rewrite mask-based: build a 1-D membership mask along
    # `dim` and blend. Matches torch.index_fill_ (in-place); index may be a tensor/list.
    res = jt.misc.index_fill(x, dim, indexs, val)
    return x.assign(res)


def index_fill(x, dim, index, val):
    ''' Out-of-place torch.index_fill: fill x along `dim` at the given `index`
    positions with scalar `val`. '''
    import jittor as jt
    d = dim % x.ndim
    size = x.shape[d]
    idx = index.reshape((-1,)) if isinstance(index, Var) else jt.array(index).reshape((-1,))
    ar = jt.arange(size).cast(idx.dtype)
    mask1d = (ar.reshape((-1, 1)) == idx.reshape((1, -1))).any(1)      # (size,) bool
    shp = [1] * x.ndim; shp[d] = size
    mask_f = mask1d.reshape(shp).broadcast(x.shape).float32()
    return x * (1 - mask_f) + float(val) * mask_f


def _indexing_dim(op, x, dim):
    """The ``dim`` argument of gather/scatter, normalised against ``x``'s rank.

    Without this the only thing standing between a bad ``dim`` and the user was
    ``indexes[dim] = index`` -- a list assignment, which answered
    ``IndexError: list assignment index out of range``: no operation, no rank,
    no bound.
    """
    ndim = x.ndim
    if not isinstance(dim, _numbers.Integral):
        raise TypeError("%s: dim must be an integer, got %s"
                        % (op, type(dim).__name__))
    dim = int(dim)
    if not -ndim <= dim < ndim:
        raise IndexError(
            "%s: dim %d is out of range for a %d-D input of shape %s "
            "(expected a dim in [%d, %d])"
            % (op, dim, ndim, list(x.shape), -ndim, ndim - 1))
    return dim + ndim if dim < 0 else dim


def _indexing_index(op, x, dim, index, bounded):
    """The ``index`` argument of gather/scatter: a Var of ``x``'s rank.

    ``gather``/``scatter`` build a reindex expression out of ``index.shape``
    and one ``i{k}`` per remaining axis. When ``index`` has a different rank
    from ``x`` that expression is still well formed -- it just indexes
    something the caller did not ask for -- so ``jt.ones((3,4)).gather(0,
    jt.array([0,1,2]))`` returned a [3,4] var with no complaint, and
    ``.gather(0, jt.zeros((3,6)))`` read four columns into six. torch rejects
    both; so does this.
    """
    if not isinstance(index, Var):
        raise TypeError("%s: index must be a jt.Var, got %s"
                        % (op, type(index).__name__))
    if index.dtype.is_float() or index.dtype.is_complex():
        raise TypeError("%s: index must have an integer dtype, got %s"
                        % (op, index.dtype))
    if index.ndim != x.ndim:
        raise RuntimeError(
            "%s: index must have the same number of dims as the input, but "
            "index is %d-D with shape %s and the input is %d-D with shape %s"
            % (op, index.ndim, list(index.shape), x.ndim, list(x.shape)))
    if not bounded:
        return
    for axis in range(x.ndim):
        if axis != dim and index.shape[axis] > x.shape[axis]:
            raise RuntimeError(
                "%s: index shape %s is larger than the input shape %s at dim "
                "%d; apart from dim %d every index dim must be no larger than "
                "the input's"
                % (op, list(index.shape), list(x.shape), axis, dim))


def _scatter_into(x, dim, index, src, reduce='void'):
    '''The in-place core shared by ``scatter`` and ``scatter_``: writes into ``x``.'''
    import jittor as jt
    dim = _indexing_dim("scatter", x, dim)
    _indexing_index("scatter", x, dim, index, bounded=False)
    result = try_dispatch("tensor.scatter", x, dim, index, src, reduce)
    if result is not None:
        return result
    shape = index.shape
    # torch allows a SCALAR src: scatter(x, dim, index, value) fills the indexed
    # positions with a constant (e.g. phimoe masks logits with torch.scatter(.., -inf)).
    if not isinstance(src, Var):
        src = jt.array(src).cast(x.dtype).broadcast(shape)
    if src.shape != shape and src.numel() != 1:
        src = src[tuple( slice(None,s) for s in shape )]
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.setitem(tuple(indexes), src, reduce)


def scatter(x:Var, dim:int, index:Var, src:Var, reduce='void'):
    ''' Out-of-place scatter, matching ``torch.Tensor.scatter``: ``x`` is left
    unchanged and a new array is returned. Use :func:`scatter_` for the in-place
    form.

    if x is a 3-D array, the RESULT looks like x with:

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
    # Clone first: the write below goes through setitem, which mutates its
    # target. Without the clone `y = x.scatter(...)` silently rewrote x -- the
    # opposite of torch, and the reason scatter_add had to clone by hand.
    return _scatter_into(x.clone(), dim, index, src, reduce)


def scatter_(x, dim, index, src, reduce='void'):
    ''' In-place scatter, matching ``torch.Tensor.scatter_``: writes into ``x``
    and returns it. '''
    return x.assign(_scatter_into(x, dim, index, src, reduce))


def scatter_add(x, dim, index, src):
    ''' torch's Tensor.scatter_add (out-of-place): accumulate src into a COPY of x
    at `index` along `dim`. `scatter` is itself out-of-place, so no extra clone. '''
    return x.scatter(dim, index, src, reduce='add')


def scatter_add_(x, dim, index, src):
    return x.scatter_(dim, index, src, reduce='add')


_SCATTER_REDUCE_JT = {'sum': 'add', 'add': 'add', 'prod': 'multiply',
                      'multiply': 'multiply', 'amax': 'maximum', 'max': 'maximum',
                      'maximum': 'maximum', 'amin': 'minimum', 'min': 'minimum',
                      'minimum': 'minimum'}

def _segment_reduce(x, dim, index, src, jt_op):
    ''' contrib[out_cell] = jt_op-reduce over the src elements that scatter into it
    (cells receiving nothing get the reduce identity: 0/1/-inf/+inf). Uses
    reindex_reduce (PULL-based, race-free) rather than scatter's setitem-reduce,
    because jittor's CUDA min/max setitem-reduce is buggy for multi-column index
    patterns (deterministically drops contributions; tracked core bug). '''
    from .shape_ops import t
    d = dim if dim >= 0 else dim + x.ndim
    nd = src.ndim
    coords = ",".join(f"i{t}" for t in range(nd))
    exprs = [("@e0(" + coords + ")") if k == d else f"i{k}" for k in range(nd)]
    return src.reindex_reduce(jt_op, list(x.shape), exprs, extras=[index])


def scatter_reduce(x, dim, index, src, reduce, include_self=True):
    ''' torch's Tensor.scatter_reduce(dim, index, src, reduce, include_self=True).
    Supports reduce = sum/prod/mean/amax/amin and BOTH include_self values, out-of-place
    and DUAL-CARD correct (Ascend + CUDA). include_self=False excludes the original
    `self` at receiving positions while leaving non-receiving positions untouched. '''
    import jittor as jt
    if reduce != 'mean' and reduce not in jt.misc._SCATTER_REDUCE_JT:
        raise NotImplementedError(f"scatter_reduce reduce='{reduce}' not supported (tracked)")
    # count of src elements landing in each output cell (race-free reindex_reduce)
    ones_like_src = jt.ones(src.shape, x.dtype)
    count = jt.misc._segment_reduce(x, dim, index, ones_like_src, "add")
    hit = count > 0
    if reduce == 'mean':
        s = jt.misc._segment_reduce(
            x, dim, index, src, "add",
        )                                                            # sum of src per cell
        if include_self:
            return (x + s) / (count + jt.ones(x.shape, x.dtype))
        return jt.ternary(hit, s / count.maximum(jt.ones(x.shape, x.dtype)), x)
    jr = jt.misc._SCATTER_REDUCE_JT[reduce]
    contrib = jt.misc._segment_reduce(
        x, dim, index, src, jr,
    )                                                               # identity where unreceived
    if include_self:
        if jr == 'add':       return x + contrib
        if jr == 'multiply':  return x * contrib
        if jr == 'maximum':   return jt.maximum(x, contrib)
        return jt.minimum(x, contrib)                              # minimum
    # include_self=False: receiving cells use contrib, non-receiving keep self
    return jt.ternary(hit, contrib, x)


def index_add(x, dim, index, source, alpha=1):
    ''' torch's Tensor.index_add (out-of-place): out[..,index[k],..] += alpha*source[..,k,..],
    ACCUMULATING duplicate indices. jittor's native index_add_ uses `+=` on an advanced
    index, which is last-write-wins (does NOT accumulate dups), so route through
    scatter_add (the proper reduce='add' path) with the 1-D index broadcast to source. '''
    d = dim if dim >= 0 else dim + x.ndim
    src = source if alpha == 1 else source * alpha
    shp = [1] * source.ndim
    shp[d] = index.shape[0]
    full_idx = index.reshape(shp).broadcast(source.shape)
    return x.scatter_add(d, full_idx.int32(), src)


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
    dim = _indexing_dim("gather", x, dim)
    _indexing_index("gather", x, dim, index, bounded=True)
    result = try_dispatch("tensor.gather", x, dim, index)
    if result is not None:
        return result
    shape = index.shape
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.getitem(tuple(indexes))


def isin(elements, test_elements, assume_unique=False, invert=False):

    import jittor as jt
    elements = elements.unsqueeze(-1)
    test_elements = test_elements.unsqueeze(0)
    comparison = elements == test_elements
    result = comparison.any(dim=-1)

    if invert:
        result = jt.logical_not(result)

    return result
