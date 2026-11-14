"""Image tensor operations."""

import math
from jittor_core import Var

def make_grid(x, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    import jittor as jt
    if not (isinstance(range, tuple) or range is None):
        raise TypeError("make_grid: range must be a tuple or None")
    if scale_each is not False:
        raise ValueError("make_grid: scale_each=True is not supported")
    if isinstance(x, list): x = jt.stack(x)
    if not isinstance(x, Var):
        raise TypeError("make_grid: x must be a Var or list of Vars")
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("make_grid: expected a 2D, 3D or 4D image tensor")
    if x.shape[1] == 1:
        x = jt.concat((x, x, x), dim=1)
    if normalize:
        low, high = (x.min(), x.max()) if range is None else range
        x = (x.clamp(low, high) - low) / jt.maximum(high - low, 1e-5)
    if x.shape[0] == 1:
        return x.squeeze(0)
    nrow = min(nrow, x.shape[0])
    b,c,h,w = x.shape
    ncol = math.ceil(b / nrow)
    return x.reindex([c, h*ncol+(ncol+1)*padding, w*nrow+(nrow+1)*padding],
                     [f"i1/{padding+h}*{nrow}+i2/{padding+w}", "i0",
                      f"i1-i1/{padding+h}*{padding+h}-{padding}", f"i2-i2/{padding+w}*{padding+w}-{padding}"], overflow_value=pad_value)


def save_image(
    x,
    filepath,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each = False,
    pad_value = 0,
    format = None
):
    import jittor as jt
    from PIL import Image
    grid = jt.misc.make_grid(
        x, nrow=nrow, padding=padding, pad_value=pad_value,
        normalize=normalize, range=range, scale_each=scale_each,
    )

    ndarr = (grid*255+0.5).clamp(0, 255).permute(1, 2, 0).uint8().numpy()
    im = Image.fromarray(ndarr)
    im.save(filepath, format=format)
