"""Selection tensor operations."""

from jittor_core import Var
from .._runtime.dispatch import optional_kernel

@optional_kernel("misc.index_select", "acl")
def _index_select_acl(input, dim, indices):
    import jittor as jt
    ndim = input.ndim
    output_shape = list(input.shape)
    output_shape[dim] = indices.shape[0]
    index_shape = [1] * ndim
    index_shape[dim] = indices.shape[0]
    index = indices.reshape(index_shape).broadcast(output_shape)
    return jt.gather(input, dim, index)


def index_select(input: Var, dim: int, indices: Var) -> Var:
    '''Returns a new var which indexes the x var along dimension dim using the entries in index.

The returned var has the same number of dimensions as the original var (x). The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.

    :param input: the input tensor.
    :param dim:  the dimension to index.
    :param indices:  the 1-D tensor containing the indices to index.

    Example::

        x = jt.randn(3, 4)
        indices = jt.array([2, 1])
        y = jt.index_select(x, 0, indices)
        assert jt.all_equal(y, x[indices])
        y = jt.index_select(x, 1, indices)
        assert jt.all_equal(y, x[:, indices])


    '''
    ndim = input.ndim
    original_dim = dim
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {original_dim})"
        )
    if not isinstance(indices, Var):
        raise TypeError(
            "index_select: index must be a jt.Var, got %s"
            % type(indices).__name__)
    if indices.ndim != 1:
        # ``input[..., indices]`` happily takes an index of any rank and folds
        # it into the output shape, so a 2-D index against a [3,4] input used
        # to return a [2,2,4] var instead of being rejected. torch requires a
        # vector here and raises IndexError; so does this.
        raise IndexError(
            "index_select: index is supposed to be a vector, but got a %d-D "
            "index of shape %s" % (indices.ndim, list(indices.shape)))
    result = _index_select_acl(input, dim, indices)
    if result is not None:
        return result
    return input[(slice(None),) * dim + (indices,)]
