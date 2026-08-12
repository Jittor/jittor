"""Image resize and upsampling modules."""

import jittor as jt


class Resize(jt.Module):
    def __init__(self, size, mode="nearest", align_corners=False):
        super().__init__()
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"sizes must be positive, got {size}")
        elif isinstance(size, tuple) or isinstance(size, list):
            for item in size:
                if item <= 0:
                    raise ValueError(f"sizes must be positive, got {item}")
        else:
            raise ValueError("size must be int or tuple")
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def execute(self, x):
        return jt.nn.resize(x, self.size, self.mode, self.align_corners)


class Upsample(jt.Module):
    def __init__(self, scale_factor=None, mode="nearest", align_corners=False):
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def execute(self, x):
        if self.scale_factor is None:
            raise ValueError("scale_factor should be defined")
        if isinstance(self.scale_factor, float):
            return jt.nn.upsample(
                x,
                size=(
                    int(x.shape[2] * self.scale_factor),
                    int(x.shape[3] * self.scale_factor),
                ),
                mode=self.mode,
                align_corners=self.align_corners,
            )
        return jt.nn.upsample(
            x,
            size=(
                int(x.shape[2] * self.scale_factor[0]),
                int(x.shape[3] * self.scale_factor[1]),
            ),
            mode=self.mode,
            align_corners=self.align_corners,
        )


class UpsamplingBilinear2d(Upsample):
    def __init__(self, scale_factor=None):
        # torch.nn.UpsamplingBilinear2d is documented as equivalent to
        # Upsample(mode='bilinear', align_corners=True) (it predates the 0.3.1
        # default flip to align_corners=False). The base Upsample defaults
        # align_corners=False, so it must be set True here for torch parity.
        Upsample.__init__(self, scale_factor, "bilinear", align_corners=True)


class UpsamplingNearest2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, "nearest")


__all__ = ["Resize", "Upsample", "UpsamplingBilinear2d", "UpsamplingNearest2d"]
