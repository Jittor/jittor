"""Fold and unfold operations exposed through :mod:`jittor.nn`."""

import jittor as jt  # noqa: F401


def unfold(X, kernel_size, dilation=1, padding=0, stride=1):
    assert X.ndim == 4
    # Accept int OR (tuple/list) pairs -- torch passes lists, e.g. convbert's
    # nn.functional.unfold(kernel_size=[k, 1], padding=[(k-1)//2, 0]).
    _pair = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)  # noqa: E731
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    dilation = _pair(dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    padding = _pair(padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    stride = _pair(stride)
    assert stride[0] > 0 and stride[1] > 0, "stride must be positive"
    n, c, h, w = X.shape
    shape = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2, 4):
        block_nums.append(
            (shape[i] + 2 * padding[i - 2] - dilation[i - 2] * (kernel_size[i - 2] - 1) - 1)
            // stride[i - 2]
            + 1
        )
    if padding[0] != 0 or padding[1] != 0:
        X = X.reindex(
            [n, c, h + padding[0] * 2, w + padding[1] * 2],
            ["i0", "i1", f"i2-{padding[0]}", f"i3-{padding[1]}"],
        )
    return X.reindex(
        [n, c * area, block_nums[0] * block_nums[1]],
        [
            "i0",
            f"i1/{area}",
            f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",
            f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}",
        ],
    )


def fold(X, output_size, kernel_size, dilation=1, padding=0, stride=1):
    assert X.ndim == 3
    assert output_size[0] > 0 and output_size[1] > 0, "output size must be positive."
    _pair = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)  # noqa: E731
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] > 0 and kernel_size[1] > 0, "kernel size must be positive"
    dilation = _pair(dilation)
    assert dilation[0] > 0 and dilation[1] > 0, "dilation must be positive"
    padding = _pair(padding)
    assert padding[0] >= 0 and padding[1] >= 0, "padding must be non-negative"
    stride = _pair(stride)
    assert stride[0] > 0 and stride[1] > 0, "stride must be positive"
    n, cl, num = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2, 4):
        block_nums.append(
            (
                output_size[i - 2]
                + 2 * padding[i - 2]
                - dilation[i - 2] * (kernel_size[i - 2] - 1)
                - 1
            )
            // stride[i - 2]
            + 1
        )
    output = X.reindex_reduce(
        "add",
        [
            n,
            cl // area,
            output_size[0] + 2 * padding[0],
            output_size[1] + 2 * padding[1],
        ],
        [
            "i0",
            f"i1/{area}",
            f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",
            f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}",
        ],
    )
    return output[
        :,
        :,
        padding[0] : padding[0] + output_size[0],
        padding[1] : padding[1] + output_size[1],
    ]


__all__ = ["fold", "unfold"]
