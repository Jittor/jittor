"""Image tensor operations."""

import math
from jittor_core import Var

def make_grid(x, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    import jittor as jt
    from .numerical import normalize
    assert isinstance(range, tuple) or range is None
    assert scale_each == False
    if isinstance(x, list): x = jt.stack(x)
    assert isinstance(x, Var)
    if normalize:
        if range is None: x = (x - x.min()) / (x.max() - x.min())
        else: x = (x - range[0]) / (range[1] - range[0])
    if x.ndim < 4: return x
    if x.ndim == 4 and x.shape[0] <= 1: return x
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
    from .numerical import normalize
    from PIL import Image
    grid = jt.misc.make_grid(
        x, nrow=nrow, padding=padding, pad_value=pad_value,
        normalize=normalize, range=range, scale_each=scale_each,
    )

    ndarr = (grid*255+0.5).clamp(0, 255).permute(1, 2, 0).uint8().numpy()
    im = Image.fromarray(ndarr)
    im.save(filepath, format=format)
