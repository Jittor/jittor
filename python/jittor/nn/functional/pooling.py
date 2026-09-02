"""Torch-compatible functional average pooling.

Average pooling has exactly one implementation in this package.  Everything
that averages a window -- ``nn.functional.avg_pool2d`` / ``avg_pool3d``,
``nn.AvgPool2d`` / ``nn.AvgPool3d``, ``jt.pool.AvgPool2d`` / ``AvgPool3d`` /
``avg_pool2d``, and ``jt.pool.Pool(op="mean")`` / ``Pool3d(op="mean")`` --
ends in :func:`_avg_pool_nd`.  There used to be three (a corrected 2-D one
here, an uncorrected 2-D one in ``jt.pool``, and a third in the 3-D pooling
kernel), which is how ``jt.nn.AvgPool2d`` and ``jt.pool.AvgPool2d`` came to
return different numbers for the same arguments.
"""

import jittor as jt


def _pool_output_size(size, kernel, stride, padding, ceil_mode):
    """One spatial extent of a pooling output, exactly as torch computes it.

    ``ceil_mode`` rounds up, and then torch *drops a trailing window that would
    start inside the right padding* -- ``pooling_output_shape_pad_lr``'s
    ``if ((out - 1) * stride >= size + padding) --out;``.  Omitting that
    correction is why the legacy ``Pool``/``Pool3d`` emit one extra plane for
    ``ceil_mode=True`` with non-zero padding.
    """
    if not ceil_mode:
        return (size + 2 * padding - kernel) // stride + 1
    out_size = (size + 2 * padding - kernel + stride - 1) // stride + 1
    if (out_size - 1) * stride >= size + padding:
        out_size -= 1
    return out_size


def _window_counts(size, kernel, stride, padding, out_size, count_include_pad):
    """The averaging divisor along one axis, per output position.

    torch clamps each window to the *padded* extent when ``count_include_pad``
    is true and to the real input when it is false.  The divisor separates per
    axis, so the N-d divisor is the outer product of these vectors -- and it
    depends only on the geometry, so it is a constant rather than a graph node.
    """
    counts = []
    for index in range(out_size):
        start = index * stride - padding
        end = start + kernel
        if count_include_pad:
            low, high = start, min(end, size + padding)
        else:
            low, high = max(start, 0), min(end, size)
        counts.append(max(high - low, 1))
    return counts


def _avg_pool_nd(x, rank, kernel_size, stride, padding, ceil_mode,
                 count_include_pad, api):
    """Average pooling over the trailing ``rank`` dimensions.

    2-D and 3-D differ only in the rank, so they share this body; keeping two
    copies is what let the two of them drift apart in the first place.
    """
    if x.ndim != rank + 2:
        raise ValueError(
            "{}: expected a {}-D input (N, C and {} spatial dims), but got a "
            "{}-D input of shape {}.".format(
                api, rank + 2, rank, x.ndim, tuple(x.shape)))
    as_tuple = jt.nn._pair if rank == 2 else jt.nn._triple
    kernel = as_tuple(kernel_size)
    strides = kernel if stride is None else as_tuple(stride)
    pads = as_tuple(padding)
    sizes = [int(s) for s in x.shape[2:]]
    for axis in range(rank):
        if kernel[axis] <= 0:
            raise RuntimeError(
                "{}: kernel_size must be greater than zero, but got {}".format(
                    api, kernel_size))
        if strides[axis] <= 0:
            raise RuntimeError(
                "{}: stride must be greater than zero, but got {}".format(
                    api, stride))
        if pads[axis] < 0:
            raise RuntimeError(
                "{}: padding must be non-negative, but got {}".format(
                    api, padding))
        if pads[axis] * 2 > kernel[axis]:
            raise RuntimeError(
                "{}: padding should be at most half of kernel size, but got "
                "padding={} and kernel_size={}".format(api, padding,
                                                       kernel_size))
    out_sizes = [
        _pool_output_size(sizes[a], kernel[a], strides[a], pads[a], ceil_mode)
        for a in range(rank)
    ]
    if min(out_sizes) <= 0:
        raise RuntimeError(
            "{}: output size is non-positive ({}): input {} is too small for "
            "kernel {}, stride {}, padding {}.".format(
                api, tuple(out_sizes), tuple(x.shape), kernel, strides, pads))
    window = ["i{}*{}-{}+i{}".format(2 + a, strides[a], pads[a], 2 + rank + a)
              for a in range(rank)]
    summed = x.reindex(
        [x.shape[0], x.shape[1]] + out_sizes + list(kernel),
        ["i0", "i1"] + window,
        overflow_value=0.0,
    ).reduce("add", list(range(2 + rank, 2 + 2 * rank)))
    if count_include_pad and not any(pads) and not ceil_mode:
        # Every window is full, so the divisor is a scalar and the whole
        # per-position table below collapses.
        volume = 1
        for extent in kernel:
            volume *= extent
        return summed / volume
    divisor = None
    for axis in range(rank):
        counts = _window_counts(sizes[axis], kernel[axis], strides[axis],
                                pads[axis], out_sizes[axis], count_include_pad)
        view = [1, 1] + [1] * rank
        view[2 + axis] = out_sizes[axis]
        part = jt.array(counts).reshape(view)
        divisor = part if divisor is None else divisor * part
    # The counts are integers; casting them to the input dtype (rather than
    # letting an int32/float16 division promote) keeps float16 in, float16 out.
    return summed / divisor.cast(summed.dtype)


def avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Apply two-dimensional average pooling with Torch divisor semantics."""
    return _avg_pool_nd(x, 2, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, "avg_pool2d")


def avg_pool3d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Apply three-dimensional average pooling with Torch divisor semantics."""
    return _avg_pool_nd(x, 3, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, "avg_pool3d")


def adaptive_avg_pool2d(input, output_size):
    """Apply two-dimensional adaptive average pooling with overlapping bins."""
    if isinstance(output_size, int):
        out_height = out_width = output_size
    elif hasattr(output_size, "__len__") and not isinstance(output_size, str):
        out_height = input.shape[2] if output_size[0] is None else int(output_size[0])
        out_width = input.shape[3] if output_size[1] is None else int(output_size[1])
    else:
        raise TypeError(
            "AdaptiveAvgPool2d only support int, tuple or list input. Not support {} yet.".format(
                type(output_size)
            )
        )
    n, channels, height, width = input.shape
    if out_height == 1 and out_width == 1:
        return input.reduce("mean", [2, 3], keepdims=True)

    yy, xx = jt.meshgrid(jt.arange(0, out_height, 1), jt.arange(0, out_width, 1))
    start_height = jt.floor(yy * height / out_height).int32()
    end_height = jt.ceil((yy + 1) * height / out_height).int32()
    start_width = jt.floor(xx * width / out_width).int32()
    end_width = jt.ceil((xx + 1) * width / out_width).int32()
    max_height = int(jt.max(end_height - start_height).data)
    max_width = int(jt.max(end_width - start_width).data)
    pixel_count = (end_height - start_height) * (end_width - start_width)
    output = input.reindex(
        [n, channels, out_height, out_width, max_height, max_width],
        ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"],
        extras=[start_height, end_height, start_width, end_width],
        overflow_conditions=[
            "i4 >= @e1(i2, i3) - @e0(i2, i3)",
            "i5 >= @e3(i2, i3) - @e2(i2, i3)",
        ],
        overflow_value=0,
    )
    return output.reduce("sum", [4, 5]) / pixel_count[None, None, ...]


__all__ = ["adaptive_avg_pool2d", "avg_pool2d", "avg_pool3d"]
