"""Stateful shape modules exposed through :mod:`jittor.nn`."""

import jittor as jt


class Identity(jt.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def execute(self, input):
        return input


class Flatten(jt.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def execute(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class PixelShuffle(jt.Module):
    def __init__(self, upscale_factor):
        assert upscale_factor > 0, "upscale_factor must be greater than zero,got {}".format(
            upscale_factor
        )
        self.upscale_factor = upscale_factor

    def execute(self, x):
        n, c, h, w = x.shape
        r = self.upscale_factor
        assert c % (r * r) == 0, (
            "input channel needs to be divided by upscale_factor's square in PixelShuffle"
        )
        if r <= 0:
            raise RuntimeError(
                "pixel_shuffle expects a positive upscale_factor, but got {}".format(r)
            )
        return x.reindex(
            [n, int(c / r**2), h * r, w * r],
            [
                "i0",
                "i1*{0}+i2%{1}*{1}+i3%{1}".format(r * r, r),
                "i2/{}".format(r),
                "i3/{}".format(r),
            ],
        )


__all__ = ["Flatten", "Identity", "PixelShuffle"]
