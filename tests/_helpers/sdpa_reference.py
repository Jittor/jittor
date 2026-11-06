"""Independent NumPy float64 attention output and gradient reference."""

import numpy as np


def reference(q, k, v, mask, causal, dout, return_mask_gradient=False):
    q, k, v, dout = (t.astype(np.float64) for t in (q, k, v, dout))
    scale = 1 / np.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if causal:
        s = np.where(np.triu(np.ones(s.shape[-2:], bool), 1), -np.inf, s)
    if mask is not None:
        s = np.where(mask, s, -np.inf) if mask.dtype == bool else s + mask
    top = s.max(-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0)
    e = np.exp(s - top)
    total = e.sum(-1, keepdims=True)
    p = np.where(total > 0, e / np.where(total > 0, total, 1), 0)
    out = p @ v
    dv = np.swapaxes(p, -1, -2) @ dout
    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdims=True))
    result = (out, ds @ k * scale, np.swapaxes(ds, -1, -2) @ q * scale, dv)
    if return_mask_gradient:
        # An additive floating mask enters the scores, so dL/dmask = dS in
        # the score layout; callers reduce it when the mask was broadcast.
        return result + (ds,)
    return result
