"""Stateless unpool pooling implementation."""

import math
import jittor as jt
from .core_3d import _triple


def _max_unpool2d_parameters(kernel_size, stride=None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if stride is None: stride = kernel_size
    p_kernel_size = kernel_size
    p_stride = stride
    if p_kernel_size[0] <= 0 or p_kernel_size[1] <= 0:
        raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
    if p_stride[0] <= 0 or p_stride[1] <= 0:
        raise RuntimeError(f"stride must be greater than zero, but got {stride}")
    return {
        "kernel_size": p_kernel_size,
        "stride": p_stride,
    }


def _max_unpool2d(x, id, output_size=None, *, kernel_size, stride):
    b, c, ph, pw = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    if output_size:
        h, w = output_size[-2:]
    else:
        # The index in ``id`` was encoded with the *original* row width, and
        # the decode below uses the reconstructed width, so a default that
        # does not reproduce the original shape silently relocates (and
        # drops) values.  Use torch's convention, which inverts the pooling
        # formula: it agrees with ``ph * sh`` whenever stride == kernel_size.
        h, w = (ph - 1) * sh + kh, (pw - 1) * sw + kw
    if stride == kernel_size:
        x = x.reindex(shape=[b, c, h, w],
            indexes=['i0', 'i1', f'i2/{kh}', f'i3/{kw}'],
            extras=[id],
            overflow_conditions=[
                f'(i2*yshape3+i3) != @e0(i0,i1,i2/{kh},i3/{kw})'],
            overflow_value=0)
    else:
        x = x.reindex_reduce(
            op="add",
            shape=[b, c, h, w],
            indexes=['i0', 'i1',
                f'@e0(i0,i1,i2,i3)/xshape3',
                f'@e0(i0,i1,i2,i3)%xshape3'],
            extras=[id],
        )
    return x


def _max_unpool3d_parameters(kernel_size, stride=None):
    if stride is None: stride = kernel_size
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    p_kernel_size = kernel_size
    p_stride = stride
    if p_kernel_size[0] <= 0 or p_kernel_size[1] <= 0 or p_kernel_size[2] <= 0:
        raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
    if p_stride[0] <= 0 or p_stride[1] <= 0 or p_stride[2] <= 0:
        raise RuntimeError(f"stride must be greater than zero, but got {stride}")
    return {
        "kernel_size": p_kernel_size,
        "stride": p_stride,
    }


def _max_unpool3d(x, id, output_size=None, *, kernel_size, stride):
    b, c, pd, ph, pw = x.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    if output_size:
        d, h, w = output_size[-3:]
    else:
        # Same inversion as MaxUnpool2d; see the note there.
        d, h, w = (pd - 1) * sd + kd, (ph - 1) * sh + kh, (pw - 1) * sw + kw
    if stride == kernel_size:
        x = x.reindex(shape=[b, c, d, h, w],
            indexes=['i0', 'i1', f'i2/{kd}', f'i3/{kh}', f'i4/{kw}'],
            extras=[id],
            overflow_conditions=[
                f'(i2*yshape3*yshape4+i3*yshape4+i4) != @e0(i0,i1,i2/{kd},i3/{kh},i4/{kw})'],
            overflow_value=0)
    else:
        x = x.reindex_reduce(
            op="add",
            shape=[b, c, d, h, w],
            indexes=['i0', 'i1',
                f'@e0(i0,i1,i2,i3,i4)/(xshape4*xshape3)',
                f'@e0(i0,i1,i2,i3,i4)/xshape4%xshape3',
                f'@e0(i0,i1,i2,i3,i4)%xshape4'],
            extras=[id],
        )
    return x
