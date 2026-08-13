"""Compatibility surface for the historical :mod:`jittor.contrib` module."""

from collections.abc import Sequence

import jittor as jt
import numpy as np
from jittor import pool
from jittor.misc.concatenation import cat, concat
from jittor.misc.indexing import getitem, setitem
from jittor.pool import argmax_pool


def check(bc):
    """Return the common broadcast shape used by legacy advanced indexing."""

    bc = np.array(bc)
    if ((bc != 1) * (bc != bc.max(0))).sum() > 0:
        raise Exception("Shape not match.")
    return bc.max(0)


def slice_var_index(x, slices):
    """Build the historical reindex arguments for an indexing expression."""

    if not isinstance(slices, tuple):
        slices = (slices,)
    if isinstance(slices[0], jt.Var):
        if len(slices) == 1 and slices[0].dtype == "bool":
            return slice_var_index(x, tuple(slices[0].where()))

    broadcast_shapes = []
    max_rank = -1
    for item in slices:
        if isinstance(item, jt.Var):
            shape = item.shape
        elif isinstance(item, np.ndarray):
            shape = list(item.shape)
        elif isinstance(item, list):
            shape = list(np.array(item).shape)
        else:
            continue
        max_rank = max(max_rank, len(shape))
        broadcast_shapes.append(shape)
    for index, shape in enumerate(broadcast_shapes):
        if len(shape) < max_rank:
            broadcast_shapes[index] = [1] * (max_rank - len(shape)) + list(shape)
    if broadcast_shapes:
        broadcast_shape = check(broadcast_shapes)
        normalized = []
        for item in slices:
            if isinstance(item, (np.ndarray, list)):
                normalized.append(jt.array(item).broadcast(broadcast_shape.tolist()))
            elif isinstance(item, jt.Var):
                normalized.append(item.broadcast(broadcast_shape.tolist()))
            else:
                normalized.append(item)
        slices = normalized

    output_shape = []
    output_indices = []
    shape = x.shape
    tensor_index_count = 0
    extra_indices = []
    extras = []
    ellipsis_positions = [
        index for index, item in enumerate(slices) if item is Ellipsis
    ]
    if len(ellipsis_positions) > 1:
        raise Exception("There are more than one ...")
    if ellipsis_positions:
        ellipsis_index = ellipsis_positions[0]
        slices = list(slices)
        del slices[ellipsis_index]
        while len(slices) < len(shape):
            slices.insert(ellipsis_index, slice(None))

    for axis in range(len(shape)):
        item = slice(None) if axis >= len(slices) else slices[axis]
        dimension = shape[axis]
        output_axis = len(output_shape)
        if isinstance(item, int):
            if item < 0:
                item += dimension
            output_indices.append(str(item))
        elif isinstance(item, slice):
            if item == slice(None):
                output_shape.append(dimension)
                output_indices.append("i{}".format(output_axis))
                continue
            start = 0 if item.start is None else item.start
            stop = dimension if item.stop is None else item.stop
            step = 1 if item.step is None else item.step
            if start < 0:
                start += dimension
            if stop < 0:
                stop += dimension
            if stop > dimension + 1:
                stop = dimension
            output_shape.append(1 + int(max(0, (stop - start - 1) // step)))
            output_indices.append("{}+i{}*{}".format(start, output_axis, step))
        elif isinstance(item, jt.Var):
            if tensor_index_count == 0:
                extra_indices = [
                    "i{}".format(len(output_shape) + index)
                    for index in range(len(broadcast_shape))
                ]
                output_shape += broadcast_shape.tolist()
            output_indices.append(
                "@e{}({})".format(tensor_index_count, ",".join(extra_indices))
            )
            tensor_index_count += 1
            extras.append(item)
        else:
            raise Exception("Not support slice {}".format(item))
    if not output_shape:
        output_shape = [1]
    x.stop_fuse()
    return output_shape, output_indices, 0, [], extras


__all__ = [
    "Sequence",
    "argmax_pool",
    "cat",
    "check",
    "concat",
    "getitem",
    "jt",
    "np",
    "pool",
    "setitem",
    "slice_var_index",
]
