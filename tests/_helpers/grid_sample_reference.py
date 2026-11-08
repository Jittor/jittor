"""Independent finite-coordinate bilinear grid geometry in NumPy float64."""
import numpy as np


def bilinear_grid_sample(values, grid, padding="zeros", align=False):
    values, grid = values.astype(np.float64), grid.astype(np.float64)
    n, c, h, w = values.shape
    def coordinate(raw, size):
        position = ((raw+1)/2)*(size-1) if align else ((raw+1)*size-1)/2
        if padding == "reflection":
            low, high = (0., size-1.) if align else (-.5, size-.5)
            span = high-low
            if span == 0:
                position = np.zeros_like(position)
            else:
                distance = np.abs(position-low)
                extra = np.fmod(distance, span)
                parity = np.floor(distance/span).astype(np.int64) % 2
                position = low + np.where(parity == 0, extra, span-extra)
        if padding in ("border", "reflection"):
            position = np.clip(position, 0, size-1)
        return position
    xx, yy = coordinate(grid[..., 0], w), coordinate(grid[..., 1], h)
    x0, y0 = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    result = np.zeros((n, c) + grid.shape[1:3], dtype=np.float64)
    batch, channels = np.arange(n)[:, None, None, None], np.arange(c)[None, :, None, None]
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix, iy = x0+ox, y0+oy
        weight = (xx-x0 if ox else 1-(xx-x0)) * (yy-y0 if oy else 1-(yy-y0))
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        gathered = values[batch, channels, np.clip(iy, 0, h-1)[:, None], np.clip(ix, 0, w-1)[:, None]]
        result += gathered * (weight*valid)[:, None]
    return result
