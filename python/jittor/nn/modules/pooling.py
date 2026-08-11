"""Torch-compatible average-pooling overrides for :mod:`jittor.nn`."""

import jittor as jt


def adaptive_avg_pool2d(input, output_size):
    ''' Applies a 2D adaptive average pooling over an input signal composed of
    several input planes. Torch-compatible functional interface that reuses the
    :class:`AdaptiveAvgPool2d` module implementation.

    :param input: the input var of shape ``(N, C, H, W)``
    :type input: jt.Var

    :param output_size: the target output size ``(H_out, W_out)``. A single int
        ``H_out`` is interpreted as ``(H_out, H_out)``; ``None`` keeps that
        dimension unchanged.
    :type output_size: int or tuple

    Example:
        >>> x = jt.randn(2, 3, 10, 12)
        >>> y = nn.adaptive_avg_pool2d(x, (5, 6))
    '''
    return jt.nn.AdaptiveAvgPool2d(output_size)(input)


# ``jittor.pool`` ships correct ``MaxPool*`` and average-pooling implementations
# that match PyTorch in the easy cases. These overrides preserve two additional
# torch behaviours on the ``nn`` surface:
#
# 1. ``avg_pool2d(..., count_include_pad=False)`` divides padded border windows
#    by the number of real input elements.
# 2. ``AdaptiveAvgPool2d`` uses variable-width overlapping bins when the output
#    size does not divide the input size.
#
# Both paths remain differentiable device operations built from reindex/reduce.
class AvgPool2d(jt.Module):
    '''2D average pooling, torch-compatible (N,C,H,W) -> (N,C,Hout,Wout).

    Unlike ``jittor.pool.AvgPool2d`` this honours ``count_include_pad`` exactly as
    PyTorch documents it: when ``True`` (default) padded zeros are counted in the
    averaging denominator; when ``False`` only real input elements are.  ``ceil_mode``
    overshoot beyond the input is never counted as padding (matches torch).
    '''
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        kh, kw = jt.nn._pair(self.kernel_size)
        sh, sw = jt.nn._pair(self.stride)
        ph, pw = jt.nn._pair(self.padding)
        N, C, H, W = x.shape
        if self.ceil_mode:
            Ho = (H + 2 * ph - kh + sh - 1) // sh + 1
            Wo = (W + 2 * pw - kw + sw - 1) // sw + 1
            # torch drops the last window if it would start in the right padding.
            if (Ho - 1) * sh >= H + ph:
                Ho -= 1
            if (Wo - 1) * sw >= W + pw:
                Wo -= 1
        else:
            Ho = (H + 2 * ph - kh) // sh + 1
            Wo = (W + 2 * pw - kw) // sw + 1
        idx = ["i0", "i1", f"i2*{sh}+i4-{ph}", f"i3*{sw}+i5-{pw}"]
        summed = x.reindex([N, C, Ho, Wo, kh, kw], idx,
                           overflow_value=0.0).reduce("add", [4, 5])
        # Fast path: no padding and no ceil overshoot -> every window is full kh*kw.
        if self.count_include_pad and ph == 0 and pw == 0 and not self.ceil_mode:
            return summed / (kh * kw)
        i2 = jt.index((Ho,), dim=0).reshape(Ho, 1).float32()
        i3 = jt.index((Wo,), dim=0).reshape(1, Wo).float32()
        if self.count_include_pad:
            # Divisor = window area clamped to the *padded* input [-pad, dim+pad);
            # ceil_mode overshoot past dim+pad is excluded (torch semantics).
            h_lo = (i2 * sh - ph).maximum(-float(ph))
            h_hi = (i2 * sh - ph + kh).minimum(float(H + ph))
            w_lo = (i3 * sw - pw).maximum(-float(pw))
            w_hi = (i3 * sw - pw + kw).minimum(float(W + pw))
        else:
            # Divisor = window area clamped to the *real* input [0, dim).
            h_lo = (i2 * sh - ph).maximum(0.0)
            h_hi = (i2 * sh - ph + kh).minimum(float(H))
            w_lo = (i3 * sw - pw).maximum(0.0)
            w_hi = (i3 * sw - pw + kw).minimum(float(W))
        denom = ((h_hi - h_lo) * (w_hi - w_lo)).reshape(1, 1, Ho, Wo)
        return summed / denom


def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    '''Functional 2D average pooling, torch-compatible (see :class:`AvgPool2d`).'''
    return jt.nn.AvgPool2d(
        kernel_size, stride, padding, ceil_mode, count_include_pad,
    )(x)


class AdaptiveAvgPool2d(jt.Module):
    '''2D adaptive average pooling, torch-compatible (N,C,H,W) -> (N,C,Oh,Ow).

    Uses torch's variable-width overlapping bins
    ``hstart=floor(i*H/Oh)``, ``hend=ceil((i+1)*H/Oh)`` (and likewise for W) and
    divides by the real bin size, so it matches PyTorch even when the output size
    does not divide the input size (the common diffusers / classifier-head case).
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = ow = self.output_size
        elif hasattr(self.output_size, "__len__") and not isinstance(self.output_size, str):
            # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
            oh = x.shape[2] if self.output_size[0] is None else int(self.output_size[0])
            ow = x.shape[3] if self.output_size[1] is None else int(self.output_size[1])
        else:
            raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list "
                            f"input. Not support {type(self.output_size)} yet.")
        N, C, H, W = x.shape
        if oh == 1 and ow == 1:
            return x.reduce("mean", [2, 3], keepdims=True)
        yy, xx = jt.meshgrid(jt.arange(0, oh, 1), jt.arange(0, ow, 1))   # (oh, ow)
        startH = jt.floor(yy * H / oh).int32()
        endH = jt.ceil((yy + 1) * H / oh).int32()
        startW = jt.floor(xx * W / ow).int32()
        endW = jt.ceil((xx + 1) * W / ow).int32()
        maxH = int(jt.max(endH - startH).data)
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        out = x.reindex(
            [N, C, oh, ow, maxH, maxW],
            ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"],
            extras=[startH, endH, startW, endW],
            overflow_conditions=["i4 >= @e1(i2, i3) - @e0(i2, i3)",
                                 "i5 >= @e3(i2, i3) - @e2(i2, i3)"],
            overflow_value=0)
        return out.reduce("sum", [4, 5]) / pixel_count[None, None, ...]
