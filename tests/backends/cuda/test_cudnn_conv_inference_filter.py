# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A half-precision convolution without a backward stores its weight as OHWI.

cuDNN runs NHWC kernels for half precision and converted an OIHW filter on
every call: 51 of the 54 ms of layout conversions in a 20-step SD1.5 sample.
The weight is moved into OHWI storage once -- same values, same shape, read
through OIHW strides -- and cuDNN gets its bytes as OHWI. These pin that no
answer changes, that the weight is moved once and holds one
copy, that a replaced weight is moved again, that a backward neither moves it
nor minds that it was moved, and that the move can be turned off.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt
from jittor.nn.backends import cudnn as _cudnn


def _reference(x, w, b, padding):
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh, ow = h + 2 * padding - kh + 1, wd + 2 * padding - kw + 1
    y = np.zeros((n, o, oh, ow))
    for i in range(kh):
        for j in range(kw):
            y += np.einsum("nchw,oc->nohw", xp[:, :, i:i + oh, j:j + ow], w[:, :, i, j])
    return y + b[None, :, None, None]


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
@_test_capability.library_required('cudnn', backend=jt)
class TestCudnnConvInferenceFilter(unittest.TestCase):
    def setUp(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(jt.runtime.scope(use_cuda=1))

    def _conv(self, kernel, dtype="float16"):
        rng = np.random.RandomState(kernel)
        x = rng.randn(2, 16, 12, 12).astype("float32")
        w = (rng.randn(24, 16, kernel, kernel) / (4 * kernel)).astype("float32")
        b = rng.randn(24).astype("float32")
        conv = jt.nn.Conv2d(16, 24, kernel, padding=kernel // 2)
        conv.weight.assign(jt.array(w).cast(dtype))
        conv.bias.assign(jt.array(b).cast(dtype))
        return conv, jt.array(x).cast(dtype), _reference(
            *(jt.array(t).cast(dtype).float32().numpy() for t in (x, w, b)), kernel // 2)

    def _run(self, conv, x):
        with jt.no_grad():
            return conv(x).float32().numpy()

    def test_every_call_answers_the_same(self):
        for kernel in (3, 1):
            conv, x, want = self._conv(kernel)
            for call in range(3):
                np.testing.assert_allclose(self._run(conv, x), want, rtol=1e-2,
                                           atol=1e-2 * np.abs(want).max(),
                                           err_msg="kernel %d call %d" % (kernel, call))

    def test_the_weight_moves_and_keeps_one_copy(self):
        conv, x, _ = self._conv(3)
        before = conv.weight.numpy()
        self._run(conv, x)
        self.assertTrue(_cudnn._is_ohwi_storage(conv.weight))
        np.testing.assert_array_equal(conv.weight.numpy(), before)
        self.assertEqual(conv.weight.__dict__[_cudnn._FILTER_OHWI][1]._storage_address,
                         conv.weight._storage_address)

    def test_a_replaced_weight_is_moved_again(self):
        conv, x, _ = self._conv(3)
        self._run(conv, x)
        conv.weight.assign(jt.array(conv.weight.numpy() * 2))
        conv.weight.sync()
        self.assertFalse(_cudnn._is_ohwi_storage(conv.weight))
        doubled = [self._run(conv, x) for _ in range(2)]
        self.assertTrue(_cudnn._is_ohwi_storage(conv.weight))
        np.testing.assert_allclose(doubled[1], doubled[0], rtol=1e-3, atol=1e-3)

    def test_a_backward_does_not_move_the_weight_nor_mind_a_moved_one(self):
        conv, x, _ = self._conv(3)
        jt.grad(conv(x).float32().sum(), [conv.weight])
        self.assertTrue(conv.weight._storage_is_contiguous())
        dense = jt.grad(conv(x).float32().sum(), conv.weight).numpy()
        self._run(conv, x)
        self._run(conv, x)
        self.assertTrue(_cudnn._is_ohwi_storage(conv.weight))
        moved = jt.grad(conv(x).float32().sum(), conv.weight).numpy()
        np.testing.assert_allclose(moved.astype("float32"), dense.astype("float32"),
                                   rtol=1e-2, atol=1e-2)

    def test_the_move_can_be_turned_off(self):
        conv, x, want = self._conv(3)
        _cudnn.channels_last_filters = False
        try:
            for _ in range(2):
                got = self._run(conv, x)
        finally:
            _cudnn.channels_last_filters = True
        self.assertTrue(conv.weight._storage_is_contiguous())
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2 * np.abs(want).max())

    def test_float32_is_left_alone(self):
        conv, x, _ = self._conv(3, "float32")
        for _ in range(2):
            self._run(conv, x)
        self.assertTrue(conv.weight._storage_is_contiguous())


if __name__ == "__main__":
    unittest.main()
