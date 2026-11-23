"""Storage order for Torch's five-dimensional channels-last format."""

import jittor as jt


CHANNELS_LAST_3D = "channels_last_3d"
_TO_NDHWC = (0, 2, 3, 4, 1)
_FROM_NDHWC = (0, 4, 1, 2, 3)


def _require_5d(shape):
    if len(shape) != 5:
        raise RuntimeError("required rank 5 tensor to use channels_last_3d format")


def is_channels_last_3d(tensor):
    shape = tuple(tensor.shape)
    if len(shape) != 5:
        return False
    strides = tensor._storage_strides()
    expected = 1
    for axis in reversed(_TO_NDHWC):
        size = shape[axis]
        if size != 1 and strides[axis] != expected:
            return False
        expected *= size
    return True


def suggests_channels_last_3d(tensor):
    """Whether a layout-aware conversion can retain this tensor's strides."""
    shape = tuple(tensor.shape)
    if len(shape) != 5 or 0 in shape:
        return False
    strides = tensor._storage_strides()
    channel_stride = strides[1]
    if channel_stride == 0:
        return False
    minimum = 0
    for axis in reversed(_TO_NDHWC):
        if strides[axis] < minimum or (axis == 0 and minimum == channel_stride):
            return False
        minimum = strides[axis] * shape[axis]
    return True


def copy_channels_last_3d(tensor):
    _require_5d(tensor.shape)
    physical = tensor._storage_permute(_TO_NDHWC)
    copied = jt.Var.copy(physical, False)
    return copied._storage_permute(_FROM_NDHWC)


def empty_channels_last_3d(*dimensions, dtype="float32"):
    shape = dimensions
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, jt.NanoVector)):
        shape = shape[0]
    _require_5d(shape)
    physical = jt.empty(tuple(shape[axis] for axis in _TO_NDHWC), dtype=dtype)
    return physical._storage_permute(_FROM_NDHWC)
