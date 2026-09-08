"""ACL tensor adapters selected by the shared runtime dispatch table."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from collections.abc import Iterable
from typing import Union
import numpy as np
import jittor as jt
from .ops.clamp_op import ClampACL
from .ops.triu_op import TriuACL
from .ops.flip_op import FlipACL
from .ops.concat_op import ConcatACL, SplitWithSizeACL
from .ops.gather_scatter_op import GatherACL, ScatterACL
from .ops.embedding_op import EmbeddingACL
from .ops.truth_reduce_op import truth_reduce
from .ops.cumsum_op import CumsumACL
from .ops.index_op import IndexACL
from .ops.arg_reduce_op import ArgReduceACL
from .ops.where_op import WhereACL, NonzeroACL
from .ops.floor_op import FloorIntACL
from .ops.getitem_op import GetItemACL, basic_slice_acl
from .ops.setitem_op import SetItemACL
from .ops.transpose_op import TransPoseACL
from .ops.roll_op import RollACL
from .ops.sigmoid_op import SigmoidACL


def _clamp_acl(input, min_value, max_value):
    if (
        isinstance(input, jt.Var)
        and isinstance(min_value, jt.Var)
        and isinstance(max_value, jt.Var)
        and (_jittor_dtype_name(input.dtype) == "float32")
        and (_jittor_dtype_name(min_value.dtype) == "float32")
        and (_jittor_dtype_name(max_value.dtype) == "float32")
        and (min_value.numel() == 1)
        and (max_value.numel() == 1)
        and min_value.is_stop_grad()
        and max_value.is_stop_grad()
    ):
        return ClampACL()(input, min_value, max_value)
    return None


def triu_acl(x, diagonal=0):
    return TriuACL()(x, diagonal)


def flip_acl(x, dim):
    return FlipACL()(x, dim)


def concat(x, dim=0):
    return ConcatACL()(x, dim)


def constant_pad_acl(x, amounts, value):
    if not all((isinstance(width, (int, np.integer)) for width in amounts)) or any(
        (int(width) < 0 for width in amounts)
    ):
        return None
    result = x
    fill_value = float(value)
    for pair_index in range(len(amounts) // 2):
        before = int(amounts[2 * pair_index])
        after = int(amounts[2 * pair_index + 1])
        if before == 0 and after == 0:
            continue
        axis = result.ndim - pair_index - 1
        parts = []
        fill_shape = list(result.shape)
        if before:
            fill_shape[axis] = before
            parts.append(jt.full(fill_shape, fill_value, dtype=result.dtype))
        parts.append(result)
        if after:
            fill_shape[axis] = after
            parts.append(jt.full(fill_shape, fill_value, dtype=result.dtype))
        result = concat(parts, axis)
    return result


def gather_acl(input, dim, index):
    return GatherACL()(input, dim, index)


def embedding_acl(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    del norm_type
    if (
        not isinstance(input, jt.Var)
        or not isinstance(weight, jt.Var)
        or input.ndim < 1
        or (weight.ndim != 2)
        or (_jittor_dtype_name(input.dtype) not in ("int32", "int64"))
        or (_jittor_dtype_name(weight.dtype) not in ("float32", "bfloat16"))
        or (max_norm is not None)
        or sparse
        or (not isinstance(scale_grad_by_freq, (bool, np.bool_)))
        or bool(scale_grad_by_freq)
    ):
        return None
    try:
        num_embeddings = int(weight.shape[0])
    except (TypeError, ValueError):
        return None
    if padding_idx is not None:
        if not isinstance(padding_idx, (int, np.integer)):
            return None
        padding_idx = int(padding_idx)
        if padding_idx < 0:
            return None
        if padding_idx >= num_embeddings:
            return None
    return EmbeddingACL(padding_idx, scale_grad_by_freq)(input, weight)


def any_acl(input, dim=None):
    return truth_reduce(input, dim, reduce_all=False)


def all_acl(input, dim=()):
    return truth_reduce(input, dim, reduce_all=True)


def cumsum_acl(input, dim=-1):
    return CumsumACL()(input, dim)


def index_acl(inshape: Union[jt.Var, list], dim=None, dtype="int32"):
    if isinstance(inshape, jt.Var):
        inshape = inshape.shape
    return IndexACL()(inshape, dim, dtype)


def scatter_acl(input, dim, index, src, reduce="void"):
    if not isinstance(src, jt.Var):
        src = jt.full(index.shape, src, input.dtype)
    return ScatterACL()(input, dim, index, src, reduce)


def arg_reduce_acl(input, op, dim, keepdims=False):
    if _jittor_dtype_name(input.dtype) not in ("float16", "float32"):
        return None
    return ArgReduceACL(jt.ops.arg_reduce)(input, op, dim, keepdims)


def where_acl(condition, x=None, y=None):
    return WhereACL()(condition, x, y)


def nonzero_acl(x):
    return NonzeroACL()(x)


def floor_int_acl(x):
    return FloorIntACL()(x)


def getitem_acl(x, slices, return_x=None):
    if return_x is not None:
        return None
    if isinstance(slices, jt.Var):
        return GetItemACL()(x, slices, return_x)
    if isinstance(slices, list):
        return GetItemACL()(x, jt.array(slices), return_x)
    if isinstance(slices, (np.int8, np.int16, np.int32, np.int64)):
        slices = int(slices)
    if hasattr(np, "int128") and isinstance(slices, np.int128):
        slices = int(slices)
    if hasattr(np, "int256") and isinstance(slices, np.int256):
        slices = int(slices)

    if slices is not None and (not isinstance(slices, Iterable) or isinstance(slices, str)):
        return _getitem_without_none(x, slices)
    if isinstance(slices, int) or isinstance(slices, slice):
        slices = (slices,)
    if not isinstance(slices, tuple):
        raise TypeError("ACL getitem slices must be a tuple, integer, slice, or tensor")

    insert_positions = _get_insert_positions(x, slices)
    slices_without_none = tuple((s for s in slices if s is not None))
    result = _getitem_without_none(x, slices_without_none)
    for i in insert_positions:
        result = result.unsqueeze(i)
    return result


def _getitem_without_none(x, items):
    result = basic_slice_acl(x, items)
    if result is not None:
        return result
    return GetItemACL()(x, items)


def _get_insert_positions(x, slices):
    result = []
    pos = 0
    not_none_cnt = sum(1 for item in slices if item is not None)
    for item in slices:
        if isinstance(item, jt.Var):
            pos += 1
        elif isinstance(item, int):
            continue
        elif item is None:
            result.append(pos)
            pos += 1
        elif item is Ellipsis:
            pos += 1 + x.ndim - not_none_cnt
        else:
            pos += 1
    return result


def setitem_acl(x, slices, value, reduce=None):
    if reduce not in (None, "void"):
        return None
    return SetItemACL()(x, slices, value)


def transpose_acl(x, *dim):
    return TransPoseACL()(x, *dim)


def _roll_acl(x, shifts, dims=None):
    if not (
        isinstance(x, jt.Var)
        and _jittor_dtype_name(x.dtype)
        in ("bfloat16", "float16", "float32", "int8", "uint8", "int32", "uint32", "bool")
    ):
        return None
    if dims is None:
        shift = shifts[0] if isinstance(shifts, (tuple, list)) else shifts
        if not isinstance(shift, int):
            return None
        return RollACL()(x.reshape((-1,)), (shift,), (0,)).reshape(x.shape)
    normalized_shifts = shifts if isinstance(shifts, (tuple, list)) else (shifts,)
    normalized_dims = dims if isinstance(dims, (tuple, list)) else (dims,)
    if (
        len(normalized_shifts) != len(normalized_dims)
        or not all((isinstance(value, int) for value in normalized_shifts))
        or (not all((isinstance(value, int) for value in normalized_dims)))
    ):
        return None
    rank = int(x.ndim)
    if rank == 0 or any((dim < -rank or dim >= rank for dim in normalized_dims)):
        return None
    normalized_dims = tuple((dim % rank for dim in normalized_dims))
    return RollACL()(x, tuple(normalized_shifts), normalized_dims)


def _split_acl(x, split_size, dim=0):
    if (
        not getattr(jt.flags, "no_grad", 0)
        or not isinstance(x, jt.Var)
        or _jittor_dtype_name(x.dtype) not in ("float16", "bfloat16", "float32")
        or (not isinstance(dim, (int, np.integer)))
        or isinstance(dim, (bool, np.bool_))
    ):
        return None
    axis = int(dim)
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        return None
    if isinstance(split_size, (int, np.integer)) and (not isinstance(split_size, (bool, np.bool_))):
        size = int(split_size)
        if size <= 0:
            return None
        extent = int(x.shape[axis])
        split_sizes = [size] * (extent // size)
        if extent % size:
            split_sizes.append(extent % size)
    elif isinstance(split_size, Iterable):
        split_sizes = list(split_size)
        if not split_sizes or not all(
            (
                isinstance(size, (int, np.integer))
                and (not isinstance(size, (bool, np.bool_)))
                and (int(size) > 0)
                for size in split_sizes
            )
        ):
            return None
        split_sizes = [int(size) for size in split_sizes]
    else:
        return None
    if not split_sizes:
        return None
    if sum(split_sizes) != int(x.shape[axis]):
        return None
    return SplitWithSizeACL()(x, split_sizes, axis)


def sigmoid_acl(x):
    return SigmoidACL()(x)
