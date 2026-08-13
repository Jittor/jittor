"""Tensor concatenation operations."""

from collections.abc import Sequence

import jittor as jt


def _merge_dtypes(dtypes):
    dtype = dtypes[0]
    for item in dtypes[1:]:
        dtype = jt.binary_dtype_infer("add", dtype, item)
    return dtype


def concat(arr, dim=0):
    """Concatenate a sequence of Vars along ``dim``."""

    with jt.flag_scope(amp_reg=4):
        if not isinstance(arr, Sequence):
            raise TypeError("concat arr needs to be a tuple or list")
        if len(arr) == 0:
            raise ValueError("need at least one array to concat")

        base_shape = list(arr[0].shape)
        base_dim = len(base_shape)
        if dim < 0:
            dim += base_dim
        if dim < 0 or dim >= base_dim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                "[{}, {}], but got {})".format(-base_dim, base_dim - 1, dim)
            )

        total_dim = 0
        dtypes = []
        for value in arr:
            if len(value.shape) != base_dim:
                raise RuntimeError(
                    "get different number of dimensions of {} and {}".format(
                        base_dim, len(value.shape)
                    )
                )
            for axis in range(base_dim):
                if axis != dim and value.shape[axis] != base_shape[axis]:
                    raise RuntimeError(
                        "Sizes of vars must match except in dimension {}. "
                        "Expected size {} but got size {} for dimension number "
                        "{} in the list.".format(
                            dim,
                            base_shape[axis],
                            value.shape[axis],
                            axis,
                        )
                    )
            total_dim += value.shape[dim]
            dtypes.append(str(value.dtype))

        output_shape = base_shape
        output_shape[dim] = total_dim
        output = jt.empty(output_shape, dtype=_merge_dtypes(dtypes))
        slices = [slice(None)] * base_dim
        offset = 0
        for value in arr:
            if value.shape[dim] == 0:
                continue
            slices[dim] = slice(offset, offset + value.shape[dim])
            output = output.setitem(tuple(slices), value)
            offset += value.shape[dim]
        return output


cat = concat

__all__ = ["cat", "concat"]
