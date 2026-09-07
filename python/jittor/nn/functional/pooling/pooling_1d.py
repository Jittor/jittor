"""Stateless pooling 1d pooling implementation."""

import math
import jittor as jt


def _adaptive_avg_pool1d(x, *, output_size):
    # x: (N, C, L) -> (N, C, output_size); mirrors AdaptiveAvgPool2d for 1d.
    ol = output_size[0] if isinstance(output_size, (tuple, list)) else output_size
    if ol is None:
        ol = x.shape[2]
    if ol == 1:
        return x.reduce("mean", [2], keepdims=True)
    N, C, L = x.shape
    s = math.floor(L / ol)
    ks = L - (ol - 1) * s
    l = (L - ks) // s + 1
    xx = x.reindex([N, C, l, ks], [
        "i0",          # Nid
        "i1",          # Cid
        f"i2*{s}+i3",  # Lid
    ])
    return xx.reduce("mean", [3])


def _max_pool1d_parameters(kernel_size, stride=None, padding=0, dilation=1, return_indices=None, ceil_mode=False):
    assert dilation == 1, "MaxPool1d: dilation>1 not supported"
    p_kernel_size = kernel_size
    p_stride = stride if stride else kernel_size
    p_padding = padding
    p_ceil_mode = ceil_mode
    p_return_indices = return_indices
    return {
        "ceil_mode": p_ceil_mode,
        "kernel_size": p_kernel_size,
        "padding": p_padding,
        "return_indices": p_return_indices,
        "stride": p_stride,
    }


def _max_pool1d(x, *, ceil_mode, kernel_size, padding, return_indices, stride):
    N, C, L = x.shape
    k, s, p = kernel_size, stride, padding
    if ceil_mode:
        lo = (L + 2 * p - k + s - 1) // s + 1
    else:
        lo = (L + 2 * p - k) // s + 1
    xx = x.reindex([N, C, lo, k], ["i0", "i1", f"i2*{s}+i3-{p}"],
                   overflow_value=float("-inf"))
    return xx.reduce("maximum", [3])


def _avg_pool1d_parameters(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    p_kernel_size = kernel_size
    p_stride = stride if stride else kernel_size
    p_padding = padding
    p_ceil_mode = ceil_mode
    p_count_include_pad = count_include_pad
    return {
        "ceil_mode": p_ceil_mode,
        "count_include_pad": p_count_include_pad,
        "kernel_size": p_kernel_size,
        "padding": p_padding,
        "stride": p_stride,
    }


def _avg_pool1d(x, *, ceil_mode, count_include_pad, kernel_size, padding, stride):
    N, C, L = x.shape
    k, s, p = kernel_size, stride, padding
    if ceil_mode:
        lo = (L + 2 * p - k + s - 1) // s + 1
    else:
        lo = (L + 2 * p - k) // s + 1
    idx = ["i0", "i1", f"i2*{s}+i3-{p}"]
    summed = x.reindex([N, C, lo, k], idx, overflow_value=0.0).reduce("add", [3])
    if count_include_pad:
        return summed / k
    # denominator = count of real (non-pad) elements per window
    ones = jt.ones([N, C, L]).reindex([N, C, lo, k], idx, overflow_value=0.0)
    return summed / ones.reduce("add", [3]).maximum(1.0)
