"""Legacy one-dimensional pooling modules."""

from .runtime import jt, preserve_facade_origins


class AdaptiveAvgPool1d(jt.Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        # x: (N, C, L) -> (N, C, output_size); mirrors AdaptiveAvgPool2d for 1d.
        ol = self.output_size[0] if isinstance(self.output_size, (tuple, list)) else self.output_size
        if ol is None:
            ol = x.shape[2]
        if ol == 1:
            return x.reduce("mean", [2], keepdims=True)
        N, C, L = x.shape
        s = jt.pool.math.floor(L / ol)
        ks = L - (ol - 1) * s
        l = (L - ks) // s + 1
        xx = x.reindex([N, C, l, ks], [
            "i0",          # Nid
            "i1",          # Cid
            f"i2*{s}+i3",  # Lid
        ])
        return xx.reduce("mean", [3])


class MaxPool1d(jt.Module):
    '''1D max pooling, (N,C,L) -> (N,C,Lout). torch-compatible.

    Implemented with reindex+reduce rather than the 2D Pool because Pool rejects a
    size-1 spatial dim. Padding positions map out-of-bounds -> -inf so they never win
    a max (matches torch, which pads with -inf for max pooling).'''
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=None, ceil_mode=False):
        assert dilation == 1, "MaxPool1d: dilation>1 not supported"
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def execute(self, x):
        N, C, L = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        if self.ceil_mode:
            lo = (L + 2 * p - k + s - 1) // s + 1
        else:
            lo = (L + 2 * p - k) // s + 1
        xx = x.reindex([N, C, lo, k], ["i0", "i1", f"i2*{s}+i3-{p}"],
                       overflow_value=float("-inf"))
        return xx.reduce("maximum", [3])


class AvgPool1d(jt.Module):
    '''1D average pooling, (N,C,L) -> (N,C,Lout). torch-compatible.

    count_include_pad=True (torch default) divides every window by kernel_size, so
    padded (out-of-bounds) positions contribute 0 to the sum but still count in the
    denominator; =False divides by the number of real (non-pad) elements.'''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        N, C, L = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        if self.ceil_mode:
            lo = (L + 2 * p - k + s - 1) // s + 1
        else:
            lo = (L + 2 * p - k) // s + 1
        idx = ["i0", "i1", f"i2*{s}+i3-{p}"]
        summed = x.reindex([N, C, lo, k], idx, overflow_value=0.0).reduce("add", [3])
        if self.count_include_pad:
            return summed / k
        # denominator = count of real (non-pad) elements per window
        ones = jt.ones([N, C, L]).reindex([N, C, lo, k], idx, overflow_value=0.0)
        return summed / ones.reduce("add", [3]).maximum(1.0)


_FACADE_SYMBOLS = (AdaptiveAvgPool1d, MaxPool1d, AvgPool1d)
preserve_facade_origins(_FACADE_SYMBOLS)
