"""Fold and unfold module wrappers."""

import jittor as jt


class Unfold(jt.Module):
    """torch's nn.Unfold (im2col): extract sliding local blocks from a batched
    (N, C, H, W) input into (N, C*prod(kernel_size), L). Wraps the functional unfold.
    (convbert builds its span-based conv with nn.Unfold.)"""

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return jt.nn.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)


class Fold(jt.Module):
    """torch's nn.Fold: the inverse of Unfold, combining sliding local blocks back
    into (N, C, output_size). Wraps the functional fold."""

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def execute(self, x):
        return jt.nn.fold(
            x,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )


__all__ = ["Fold", "Unfold"]
