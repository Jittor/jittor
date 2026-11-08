"""Channels-last activations: an NCHW view of dense NHWC storage.

A half-precision convolution that records no gradient hands its result out
this way (see ``jittor.nn.backends.cudnn``), and the elementwise operators
keep it (``propagate_storage_layout``). An operator with a kernel of its own
serves such an input from the NHWC storage directly and hands its result back
the same way, instead of having it copied dense -- which would cost the copy
and undo the layout for the next convolution too.
"""
import jittor as jt


def channels_last_source(x):
    """The dense NHWC tensor `x` is an NCHW view of, or None."""
    if len(x.shape) != 4 or x._storage_is_contiguous() or x._storage_offset():
        return None
    n, c, h, w = (int(size) for size in x.shape)
    if min(c, h, w) <= 1 or tuple(x._storage_strides()) != (h * w * c, 1, w * c, c):
        return None
    return x._storage_permute((0, 2, 3, 1))


def channels_last_view(y):
    """`y`, dense [N, H, W, C], as the NCHW view of its storage."""
    return y._storage_permute((0, 3, 1, 2))


def records_no_grad(*values):
    return jt.flags.no_grad or all(
        v.is_stop_grad() for v in values if isinstance(v, jt.Var))


def offer_channels_last(y, build):
    """Say that ``build()`` computes ``y`` as a channels-last view instead.

    A convolution that records a gradient hands out NCHW: whether NHWC pays
    depends on what reads it -- a batch norm with NHWC kernels keeps the whole
    chain NHWC, a group norm or a reshape would convert it straight back. So
    the reader decides (`take_channels_last`); `y` itself stays a graph node
    nobody runs unless something else reads it.
    """
    y.__dict__["_channels_last_offer"] = (y.id, build)


def take_channels_last(x):
    """``x`` as a channels-last view, if the pass that makes it offered one."""
    entry = getattr(x, "__dict__", {}).get("_channels_last_offer")
    if entry is not None and entry[0] == x.id and not x.is_finished:
        return entry[1]()
    return None
