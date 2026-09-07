"""Stateless adaptive pooling implementation."""

import math
import jittor as jt
from .core_3d import _triple
from .entrypoints import max_pool2d, max_pool3d


def _legacy_adaptive_avg_pool2d_parameters(output_size):
    p_output_size = output_size
    return {
        "output_size": p_output_size,
    }


def _legacy_adaptive_avg_pool2d(x, *, output_size):
    if isinstance(output_size, int):
        oh = output_size
        ow = output_size
    elif hasattr(output_size, "__len__") and not isinstance(output_size, str):
        # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
        oh = x.shape[2] if output_size[0] is None else int(output_size[0])
        ow = x.shape[3] if output_size[1] is None else int(output_size[1])
    else:
        raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(output_size)} yet.")
    if oh == 1 and ow == 1:
        return x.reduce("mean", [2,3], keepdims=True)
    N,C,H,W = x.shape
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    h = (H-ksh)//sh+1
    w = (W-ksw)//sw+1
    xx = x.reindex([N,C,h,w,ksh,ksw], [
        "i0", # Nid
        "i1", # Cid
        f"i2*{sh}+i4", # Hid
        f"i3*{sw}+i5", # Wid
    ])
    return xx.reduce("mean", [4,5])


def _adaptive_max_pool2d_parameters(output_size, return_indices=False):
    p_output_size = output_size
    p_return_indices = return_indices
    return {
        "output_size": p_output_size,
        "return_indices": p_return_indices,
    }


def _adaptive_max_pool2d(x, *, output_size, return_indices):
    if isinstance(output_size, int):
        oh = output_size
        ow = output_size
    elif hasattr(output_size, "__len__") and not isinstance(output_size, str):
        # tuple / list / jittor NanoVector (e.g. x.shape[2:] from a semantic head)
        oh = x.shape[2] if output_size[0] is None else int(output_size[0])
        ow = x.shape[3] if output_size[1] is None else int(output_size[1])
    else:
        raise TypeError(f"AdaptiveMaxPool2d only support int, tuple or list input. Not support {type(output_size)} yet.")
    if oh == 1 and ow == 1:
        return x.reduce("maximum", [2,3], keepdims=True)
    N,C,H,W = x.shape
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    if return_indices:
        return max_pool2d(
            kernel_size=(ksh, ksw),
            stride=(sh, sw), return_indices=True, x=x)
    h = (H-ksh)//sh+1
    w = (W-ksw)//sw+1
    xx = x.reindex([N,C,h,w,ksh,ksw], [
        "i0", # Nid
        "i1", # Cid
        f"i2*{sh}+i4", # Hid
        f"i3*{sw}+i5", # Wid
    ])
    return xx.reduce("maximum", [4,5])


def _adaptive_avg_pool3d_parameters(output_size):
    p_output_size = _triple(output_size)
    return {
        "output_size": p_output_size,
    }


def _adaptive_avg_pool3d(x, *, output_size):
    od, oh, ow = output_size
    if od == 1 and oh == 1 and ow == 1:
        return x.reduce("mean", [2,3,4], keepdims=True)
    N,C,D,H,W = x.shape
    sd = math.floor(D / od)
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksd = D - (od - 1) * sd
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    d = (D-ksd)//sd+1
    h = (H-ksh)//sh+1
    w = (W-ksw)//sw+1
    xx = x.reindex([N,C,d,h,w,ksd,ksh,ksw], [
        "i0", # Nid
        "i1", # Cid
        f"i2*{sd}+i5", # Did
        f"i3*{sh}+i6", # Hid
        f"i4*{sw}+i7", # Wid
    ])
    return xx.reduce("mean", [5,6,7])


def _adaptive_max_pool3d_parameters(output_size, return_indices=False):
    p_output_size = _triple(output_size)
    p_return_indices = return_indices
    return {
        "output_size": p_output_size,
        "return_indices": p_return_indices,
    }


def _adaptive_max_pool3d(x, *, output_size, return_indices):
    od, oh, ow = output_size
    if od == 1 and oh == 1 and ow == 1 and not return_indices:
        return x.reduce("maximum", [2,3,4], keepdims=True)
    N,C,D,H,W = x.shape
    sd = math.floor(D / od)
    sh = math.floor(H / oh)
    sw = math.floor(W / ow)
    ksd = D - (od - 1) * sd
    ksh = H - (oh - 1) * sh
    ksw = W - (ow - 1) * sw
    if return_indices:
        return max_pool3d(
            kernel_size=(ksd, ksh, ksw),
            stride=(sd, sh, sw), return_indices=True, x=x)
    d = (D-ksd)//sd+1
    h = (H-ksh)//sh+1
    w = (W-ksw)//sw+1
    xx = x.reindex([N,C,d,h,w,ksd,ksh,ksw], [
        "i0", # Nid
        "i1", # Cid
        f"i2*{sd}+i5", # Did
        f"i3*{sh}+i6", # Hid
        f"i4*{sw}+i7", # Wid
    ])
    return xx.reduce("maximum", [5,6,7])
