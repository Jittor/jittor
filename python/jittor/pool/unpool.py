"""Legacy max-unpooling modules."""

import jittor as jt


class MaxUnpool2d(jt.Module):
    ''' MaxUnpool2d is the invert version of MaxPool2d with indices.
    It takes the output index of MaxPool2d as input.
    The element will be zero if it is not the max pooled value.

    Example::

    >>> import jittor as jt
    >>> from jittor import nn

    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = jt.array([[[[ 1.,  2,  3,  4,0],
                            [ 5,  6,  7,  8,0],
                            [ 9, 10, 11, 12,0],
                            [13, 14, 15, 16,0],
                            [0,  0,  0,  0, 0]]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.shape)
    jt.array([[[[   0.,  0.,   0.,   0.,   0.],
                [   0.,  6.,   0.,   8.,   0.],
                [   0.,  0.,   0.,   0.,   0.],
                [   0., 14.,   0.,  16.,   0.],
                [   0.,  0.,   0.,   0.,   0.]]]])
    '''
    def __init__(self, kernel_size, stride=None):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if stride is None: stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
            raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
        if self.stride[0] <= 0 or self.stride[1] <= 0:
            raise RuntimeError(f"stride must be greater than zero, but got {stride}")

    def execute(self, x, id, output_size=None):
        b, c, ph, pw = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if output_size:
            h, w = output_size[-2:]
        else:
            # The index in ``id`` was encoded with the *original* row width, and
            # the decode below uses the reconstructed width, so a default that
            # does not reproduce the original shape silently relocates (and
            # drops) values.  Use torch's convention, which inverts the pooling
            # formula: it agrees with ``ph * sh`` whenever stride == kernel_size.
            h, w = (ph - 1) * sh + kh, (pw - 1) * sw + kw
        if self.stride == self.kernel_size:
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


class MaxUnpool3d(jt.Module):
    ''' MaxUnpool3d is the invert version of MaxPool3d with indices.
    It takes the output index of MaxPool3d as input.
    The element will be zero if it is not the max pooled value.
    '''
    def __init__(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        kernel_size = jt.pool._triple(kernel_size)
        stride = jt.pool._triple(stride)
        self.kernel_size = kernel_size
        self.stride = stride
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0 or self.kernel_size[2] <= 0:
            raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
        if self.stride[0] <= 0 or self.stride[1] <= 0 or self.stride[2] <= 0:
            raise RuntimeError(f"stride must be greater than zero, but got {stride}")

    def execute(self, x, id, output_size=None):
        b, c, pd, ph, pw = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        if output_size:
            d, h, w = output_size[-3:]
        else:
            # Same inversion as MaxUnpool2d; see the note there.
            d, h, w = (pd - 1) * sd + kd, (ph - 1) * sh + kh, (pw - 1) * sw + kw
        if self.stride == self.kernel_size:
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


_PUBLIC_SYMBOLS = (MaxUnpool2d, MaxUnpool3d)
