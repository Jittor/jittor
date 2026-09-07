"""Search tensor operations."""

import numpy as np
from jittor_core import Var
from .._runtime.dispatch import optional_kernel

@optional_kernel("misc.searchsorted", "acl")
def _searchsorted_acl(sorted, values, right, out_dtype, out):
    if sorted.ndim == 1:
        sorted_view = sorted.reshape(
            (1,) * values.ndim + (sorted.shape[-1],)
        )
    else:
        if sorted.ndim != values.ndim:
            raise ValueError(
                "batched sorted and values must have the same rank"
            )
        sorted_view = sorted.unsqueeze(-2)
    values_view = values.unsqueeze(-1)
    before = sorted_view <= values_view if right else sorted_view < values_view
    ret = before.int32().sum(dim=-1).cast(out_dtype)
    if out is not None:
        out.assign(ret)
        return out
    return ret


def searchsorted(sorted, values, right=False, out_int32=False, side=None, sorter=None, out=None):
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
    import jittor as jt
    if side is not None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        right = side == "right"
    if sorter is not None:
        raise NotImplementedError("searchsorted sorter is not supported")
    scalar_value = not isinstance(values, Var) and np.isscalar(values)
    if not isinstance(values, Var):
        values = jt.array(values, dtype=sorted.dtype)
    elif values.dtype != sorted.dtype:
        values = values.cast(sorted.dtype)
    if scalar_value or values.ndim == 0:
        values = values.reshape((1,))
    out_dtype = "int32" if out_int32 else "int64"
    result = _searchsorted_acl(sorted, values, right, out_dtype, out)
    if result is not None:
        return result
    out_ctype = "int32" if out_int32 else "int64"
    _searchsorted_header = f"""
namespace jittor {{

@python.jittor.auto_parallel(2)
inline static void searchsorted(
    int batch_num, int batch_id, int value_num, int value_id,
    int sorted_num, int batch_stride,
    {sorted.dtype}* __restrict__  sort_p, {values.dtype}* __restrict__  value_p,
    {out_ctype}* __restrict__ index_p) {{
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
    index_p[batch_id * value_num + value_id] = ({out_ctype})(l - batch_id * batch_stride);
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
    ret = jt.code(values.shape, out_dtype, [sorted, values],
        cpu_header=_searchsorted_header,
        cpu_src=_searchsorted_src,
        cuda_header=_searchsorted_header,
        cuda_src=_searchsorted_src)
    if out is not None:
        out.assign(ret)
        return out
    return ret
