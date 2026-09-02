# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reference tests for the pure-Jittor ``conv3d`` fallback.

``jt.nn.conv3d`` only reaches cuDNN when CUDA *and* cuDNN are both available;
every other configuration (all CPU runs, and CUDA builds without cuDNN) falls
through to the reindex implementation in
``jittor/nn/functional/convolution.py``.  The grouped branch of that fallback
had never been executed, so this module pins it against a direct numpy
convolution.
"""

import unittest

import numpy as np

import jittor as jt


def _reference_conv3d(x, weight, stride, padding, dilation, groups):
    """Textbook 3D convolution (cross-correlation), float64, no shortcuts."""
    n = x.shape[0]
    depth, height, width = x.shape[2:]
    out_channels, in_per_group, kd, kh, kw = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    padded = np.pad(x, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)))
    od = (depth + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = np.zeros((n, out_channels, od, oh, ow), dtype=np.float64)
    out_per_group = out_channels // groups
    for batch in range(n):
        for oc in range(out_channels):
            group = oc // out_per_group
            for i in range(od):
                for j in range(oh):
                    for k in range(ow):
                        total = 0.0
                        for c in range(in_per_group):
                            for a in range(kd):
                                for b in range(kh):
                                    for e in range(kw):
                                        total += (
                                            padded[
                                                batch,
                                                group * in_per_group + c,
                                                i * sd + a * dd,
                                                j * sh + b * dh,
                                                k * sw + e * dw,
                                            ]
                                            * weight[oc, c, a, b, e]
                                        )
                        out[batch, oc, i, j, k] = total
    return out


class TestConv3dReference(unittest.TestCase):
    def _check(self, shape, weight_shape, stride, padding, dilation, groups, bias=False):
        rng = np.random.default_rng(1234)
        x = rng.standard_normal(shape).astype("float32")
        w = rng.standard_normal(weight_shape).astype("float32")
        b = rng.standard_normal(weight_shape[0]).astype("float32") if bias else None
        expected = _reference_conv3d(
            x.astype(np.float64), w.astype(np.float64), stride, padding, dilation, groups
        )
        if b is not None:
            expected = expected + b.astype(np.float64).reshape(1, -1, 1, 1, 1)
        got = jt.nn.conv3d(
            jt.array(x),
            jt.array(w),
            None if b is None else jt.array(b),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ).numpy()
        self.assertEqual(tuple(got.shape), expected.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_groups_one(self):
        self._check((2, 3, 5, 6, 7), (4, 3, 2, 3, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)

    def test_grouped_asymmetric_kernel(self):
        # Regression: the grouped branch broadcast the weight into
        # [...,oh,ow,od,Kh,Kw,Kd] while the input used [...,od,oh,ow,Kd,Kh,Kw],
        # so this used to raise "Check failed xshape(3) == yshape(2)".
        self._check((2, 4, 5, 6, 7), (6, 2, 2, 3, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 2)

    def test_grouped_strided_padded(self):
        self._check((1, 4, 6, 7, 8), (4, 2, 3, 2, 3), (2, 1, 2), (1, 2, 0), (1, 1, 1), 2)

    def test_grouped_dilated_with_bias(self):
        self._check(
            (2, 6, 7, 6, 6), (3, 2, 2, 2, 3), (1, 2, 1), (1, 0, 1), (2, 1, 2), 3, bias=True
        )

    def test_grouped_matches_per_group_conv1(self):
        """A grouped conv must equal independent convolutions on each group."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, 6, 5, 5, 6)).astype("float32")
        w = rng.standard_normal((6, 2, 2, 3, 2)).astype("float32")
        grouped = jt.nn.conv3d(jt.array(x), jt.array(w), groups=3).numpy()
        parts = []
        for g in range(3):
            xs = x[:, g * 2:(g + 1) * 2]
            ws = w[g * 2:(g + 1) * 2]
            parts.append(jt.nn.conv3d(jt.array(xs), jt.array(ws), groups=1).numpy())
        np.testing.assert_allclose(grouped, np.concatenate(parts, axis=1), rtol=1e-4, atol=1e-4)

    def test_grouped_backward_matches_finite_difference(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((1, 4, 4, 4, 5)).astype("float64")
        w = rng.standard_normal((4, 2, 2, 2, 2)).astype("float64")
        seed = rng.standard_normal((1, 4, 3, 3, 4)).astype("float64")

        def value(xv, wv):
            out = _reference_conv3d(xv, wv, (1, 1, 1), (0, 0, 0), (1, 1, 1), 2)
            return float((out * seed).sum())

        xj = jt.array(x.astype("float32"))
        wj = jt.array(w.astype("float32"))
        loss = (jt.nn.conv3d(xj, wj, groups=2) * jt.array(seed.astype("float32"))).sum()
        gx, gw = jt.grad(loss, [xj, wj])
        eps = 1e-4
        for idx in [(0, 0, 1, 1, 2), (0, 3, 2, 0, 3)]:
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            fd = (value(xp, w) - value(xm, w)) / (2 * eps)
            np.testing.assert_allclose(gx.numpy()[idx], fd, rtol=2e-3, atol=2e-3)
        for idx in [(0, 1, 1, 0, 1), (3, 0, 0, 1, 1)]:
            wp = w.copy(); wp[idx] += eps
            wm = w.copy(); wm[idx] -= eps
            fd = (value(x, wp) - value(x, wm)) / (2 * eps)
            np.testing.assert_allclose(gw.numpy()[idx], fd, rtol=2e-3, atol=2e-3)

    def test_groups_must_be_positive(self):
        x = jt.zeros((1, 2, 4, 4, 4))
        w = jt.zeros((2, 2, 2, 2, 2))
        with self.assertRaises(ValueError):
            jt.nn.conv3d(x, w, groups=0)


if __name__ == "__main__":
    unittest.main()
