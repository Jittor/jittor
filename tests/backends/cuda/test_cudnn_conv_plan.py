# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The cuDNN convolution ops execute through cached backend-API plans.

Each configuration is checked against the CPU convolution (an independent
implementation) for the forward value and both gradients, across layouts,
groups, strides, dilations and half precision, with tensor-op numerics both
allowed and forbidden.
"""
import unittest

import numpy as np

import jittor as jt


def _reference(x, w, stride, padding, dilation, groups):
    with jt.flag_scope(use_cuda=0):
        xs = jt.array(x); ws = jt.array(w)
        y = jt.nn.conv2d(xs, ws, None, stride, padding, dilation, groups)
        r = jt.array(np.random.RandomState(7).randn(*y.shape).astype("float32"))
        loss = (y.float32() * r).sum()
        gx, gw = jt.grad(loss, [xs, ws])
        return y.numpy(), gx.numpy(), gw.numpy(), r.numpy()


@unittest.skipIf(not jt.has_cuda, "No cuda found")
class TestCudnnConvPlan(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.cuda_allow_cudnn_tf32)
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda, jt.flags.cuda_allow_cudnn_tf32 = self._saved

    def _check(self, n, c, h, k, r, stride=1, padding=0, dilation=1, groups=1,
               dtype="float32", nhwc=False, tf32=False, tol=None):
        rng = np.random.RandomState(0)
        x = rng.randn(n, c, h, h).astype("float32")
        w = rng.randn(k, c // groups, r, r).astype("float32")
        y_ref, gx_ref, gw_ref, dout = _reference(x, w, stride, padding, dilation, groups)
        jt.flags.cuda_allow_cudnn_tf32 = int(tf32)
        xd = jt.array(x).cast(dtype); wd = jt.array(w).cast(dtype)
        if nhwc:
            xd = xd.transpose(0, 2, 3, 1); wd = wd.transpose(0, 2, 3, 1)
            y = jt.cudnn.ops.cudnn_conv(xd, wd, stride, stride, padding, padding,
                                        dilation, dilation, groups, "acdb", "ohwi", "")
            y_cmp = y.transpose(0, 3, 1, 2)
        else:
            y = jt.cudnn.ops.cudnn_conv(xd, wd, stride, stride, padding, padding,
                                        dilation, dilation, groups, "abcd", "oihw", "")
            y_cmp = y
        loss = (y_cmp.float32() * jt.array(dout)).sum()
        gx, gw = jt.grad(loss, [xd, wd])
        if nhwc:
            gx = gx.transpose(0, 3, 1, 2); gw = gw.transpose(0, 3, 1, 2)
        if tol is None:
            tol = 2e-2 if (tf32 or dtype != "float32") else 1e-4
        for name, got, ref in (("y", y_cmp, y_ref), ("dx", gx, gx_ref), ("dw", gw, gw_ref)):
            got = got.float32().numpy()
            scale = max(1.0, float(np.abs(ref).max()))
            err = float(np.abs(got - ref).max()) / scale
            self.assertLess(err, tol, "%s mismatch %.3g (nhwc=%s tf32=%s dtype=%s)" % (name, err, nhwc, tf32, dtype))

    def test_plain_fp32(self):
        self._check(2, 8, 16, 4, 3, padding=1)

    def test_stride_dilation(self):
        self._check(2, 6, 20, 5, 3, stride=2, padding=2, dilation=2)

    def test_groups(self):
        self._check(2, 8, 12, 8, 3, padding=1, groups=4)

    def test_nhwc(self):
        # NHWC fp32 is only served by tensor-core engines, so it needs the
        # tensor-op permission the layout exists for.
        self._check(2, 8, 16, 4, 3, padding=1, nhwc=True, tf32=True)

    def test_tf32_allowed(self):
        self._check(2, 16, 16, 16, 3, padding=1, tf32=True)

    def test_half(self):
        self._check(2, 8, 16, 8, 3, padding=1, dtype="float16")

    def test_repeated_shapes_reuse_plan(self):
        # The same configuration executed many times must stay correct and
        # not accumulate state; this is the cache hit path.
        for _ in range(3):
            self._check(2, 8, 16, 4, 3, padding=1)


if __name__ == "__main__":
    unittest.main()
