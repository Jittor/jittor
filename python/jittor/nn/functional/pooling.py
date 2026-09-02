"""Torch-compatible functional average pooling."""

import jittor as jt


def avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Apply two-dimensional average pooling with Torch divisor semantics."""
    stride = kernel_size if stride is None else stride
    kh, kw = jt.nn._pair(kernel_size)
    sh, sw = jt.nn._pair(stride)
    ph, pw = jt.nn._pair(padding)
    n, channels, height, width = x.shape
    if ceil_mode:
        out_height = (height + 2 * ph - kh + sh - 1) // sh + 1
        out_width = (width + 2 * pw - kw + sw - 1) // sw + 1
        if (out_height - 1) * sh >= height + ph:
            out_height -= 1
        if (out_width - 1) * sw >= width + pw:
            out_width -= 1
    else:
        out_height = (height + 2 * ph - kh) // sh + 1
        out_width = (width + 2 * pw - kw) // sw + 1
    indexes = ["i0", "i1", "i2*{}+i4-{}".format(sh, ph), "i3*{}+i5-{}".format(sw, pw)]
    summed = x.reindex(
        [n, channels, out_height, out_width, kh, kw],
        indexes,
        overflow_value=0.0,
    ).reduce("add", [4, 5])
    if count_include_pad and ph == 0 and pw == 0 and not ceil_mode:
        return summed / (kh * kw)

    out_y = jt.index((out_height,), dim=0).reshape(out_height, 1).float32()
    out_x = jt.index((out_width,), dim=0).reshape(1, out_width).float32()
    if count_include_pad:
        height_low = (out_y * sh - ph).maximum(-float(ph))
        height_high = (out_y * sh - ph + kh).minimum(float(height + ph))
        width_low = (out_x * sw - pw).maximum(-float(pw))
        width_high = (out_x * sw - pw + kw).minimum(float(width + pw))
    else:
        height_low = (out_y * sh - ph).maximum(0.0)
        height_high = (out_y * sh - ph + kh).minimum(float(height))
        width_low = (out_x * sw - pw).maximum(0.0)
        width_high = (out_x * sw - pw + kw).minimum(float(width))
    divisor = ((height_high - height_low) * (width_high - width_low)).reshape(
        1, 1, out_height, out_width
    )
    return summed / divisor


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


def adaptive_avg_pool1d(input, output_size):
    """Apply one-dimensional adaptive average pooling with overlapping bins.

    Routed through the two-dimensional kernel over a singleton height so both spellings place
    bin boundaries identically: output `i` averages `[floor(i * L / out), ceil((i + 1) * L / out))`,
    a window whose width varies when `out` does not divide `L`.
    """
    if isinstance(output_size, (tuple, list)):
        length = output_size[0]
    else:
        length = output_size
    if length is None:
        length = input.shape[-1]

    batched = input.ndim == 3
    if not batched:
        if input.ndim != 2:
            raise ValueError(
                "adaptive_avg_pool1d expects a (N, C, L) or (C, L) input, got shape "
                f"{tuple(input.shape)}"
            )
        input = input.unsqueeze(0)
    pooled = adaptive_avg_pool2d(input.unsqueeze(2), (1, int(length))).squeeze(2)
    return pooled if batched else pooled.squeeze(0)


__all__ = ["adaptive_avg_pool1d", "adaptive_avg_pool2d", "avg_pool2d"]
