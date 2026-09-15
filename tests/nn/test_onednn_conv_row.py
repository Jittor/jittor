"""The CPU row of the conv2d kernel table, against the path it stands in front of.

This row did not exist. A CPU convolution reached oneDNN only by ``ConvTuner``
relaying the fused ``reindex * broadcast -> reduce`` subgraph to ``mkl_conv``,
and that relay stopped firing when ``BroadcastToOp`` became a storage
descriptor -- silently, because nothing compared the two paths. What it cost,
measured on ``8x64x56x56`` against ``64x64x3x3``: 0.1079 s against torch's
0.0031 s for the same convolution, 35x.

So the guard is not "does it run" but "does it compute what the fallback
computes", gradients included -- the library op carried no gradient at all
while it was reachable solely as a relay, because autograd ran on the meta-op
subgraph the relay stood in for.
"""
import numpy as np
import pytest

import jittor as jt
from jittor import nn

from _helpers.onednn import requires_onednn


def _reference_conv(x, w, stride, padding, dilation, groups):
    """A convolution written from the definition, in NumPy. No Jittor in it."""
    N, C, H, W = x.shape
    O, Cg, Kh, Kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    oh = (H + 2*ph - dh*(Kh-1) - 1)//sh + 1
    ow = (W + 2*pw - dw*(Kw-1) - 1)//sw + 1
    y = np.zeros((N, O, oh, ow), dtype="float32")
    per_group = O // groups
    for g in range(groups):
        for o in range(g*per_group, (g+1)*per_group):
            for i in range(Cg):
                c = g*Cg + i
                for kh in range(Kh):
                    for kw in range(Kw):
                        y[:, o] += padded[:, c,
                                          kh*dh:kh*dh + oh*sh:sh,
                                          kw*dw:kw*dw + ow*sw:sw] * w[o, i, kh, kw]
    return y


#: Shapes chosen so each geometry oneDNN has to describe appears once:
#: padding, a stride, an asymmetric stride/padding/dilation, groups, and 1x1.
GEOMETRIES = [
    ((2, 4, 9, 9), (6, 4, 3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((2, 4, 9, 9), (6, 4, 3, 3), (2, 2), (0, 0), (1, 1), 1),
    ((1, 4, 11, 13), (4, 4, 3, 3), (1, 2), (2, 1), (2, 1), 1),
    ((2, 6, 8, 8), (6, 3, 3, 3), (1, 1), (1, 1), (1, 1), 2),
    ((1, 3, 7, 7), (5, 3, 1, 1), (1, 1), (0, 0), (1, 1), 1),
]


@pytest.mark.parametrize("xshape,wshape,stride,padding,dilation,groups", GEOMETRIES)
def test_the_cpu_row_computes_a_convolution(xshape, wshape, stride, padding,
                                            dilation, groups):
    requires_onednn()
    with jt.flag_scope(use_cuda=0):
        x = jt.random(list(xshape))
        w = jt.random(list(wshape))
        y = nn.conv2d(x, w, None, stride, padding, dilation, groups)
        want = _reference_conv(x.data, w.data, stride, padding, dilation, groups)
        assert tuple(y.shape) == want.shape, (tuple(y.shape), want.shape)
        np.testing.assert_allclose(y.data, want, rtol=2e-5, atol=2e-5)

        cotangent = jt.random(y.shape)
        gx, gw = jt.grad((y*cotangent).sum(), [x, w])
        assert tuple(gx.shape) == xshape
        assert tuple(gw.shape) == wshape
        # A gradient that is merely the right shape is what a missing
        # ``grad()`` produces; require it to carry something.
        assert float(np.abs(gw.data).max()) > 0


def test_the_cpu_row_agrees_with_the_generic_path_gradients_included():
    """Same inputs, both rows, one comparison -- the relay's own contract."""
    requires_onednn()
    from jittor._runtime.dispatch import override_kernel

    rs = np.random.RandomState(0)
    xn = rs.rand(2, 4, 7, 7).astype("float32")
    wn = rs.rand(6, 4, 3, 3).astype("float32")
    cotangent = rs.rand(2, 6, 7, 7).astype("float32")

    def run():
        x = jt.array(xn)
        w = jt.array(wn)
        y = nn.conv2d(x, w, None, 1, 1, 1, 1)
        gx, gw = jt.grad((y*jt.array(cotangent)).sum(), [x, w])
        return y.data.copy(), gx.data.copy(), gw.data.copy()

    with jt.flag_scope(use_cuda=0):
        library = run()
        # Declining the row is how the generic reindex path is reached; there
        # is no flag for it.
        with override_kernel("conv2d", "cpu", lambda *a, **k: None,
                             supports=lambda *a, **k: False):
            generic = run()

    for got, want, name in zip(library, generic, ("output", "grad_x", "grad_w")):
        np.testing.assert_allclose(
            got, want, rtol=2e-5, atol=2e-5,
            err_msg="the oneDNN row and the generic path disagree on " + name)
