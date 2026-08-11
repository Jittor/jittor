"""Tensor repetition, chunking, and broadcast shape transforms."""

from .runtime import jt, preserve_facade_origins


def repeat(x, *shape):
    '''
    Repeats this var along the specified dimensions.

    Args:

        x (var): jittor var.

        shape (tuple): int or tuple. The number of times to repeat this var along each dimension.
\x20
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
    if len(shape) == 1 and isinstance(shape[0], jt.misc.Sequence):
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

    tar_shape = (jt.misc.np.array(x_shape) * jt.misc.np.array(rep_shape)).tolist()

    x = x.reshape(tar_shape)
    return x


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
        # ceil(l/chunks) per chunk, last may be shorter -- matches torch.chunk.
        # NB: iterate by start offset, not `range(chunks-1)`; the latter left `i`
        # unbound (UnboundLocalError) for chunks==1 (e.g. single-GPU dispatch).
        nums = (l-1) // chunks + 1
        for start in range(0, l, nums):
            res.append(x[(slice(None,),)*dim+(slice(start, min(start+nums, l)),)])
    return res


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


_FACADE_SYMBOLS = (repeat, chunk, expand)
preserve_facade_origins(_FACADE_SYMBOLS)
