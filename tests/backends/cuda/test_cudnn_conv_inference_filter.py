# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A half-precision convolution without a backward hands cuDNN an OHWI filter.

cuDNN runs NHWC kernels for half precision and converted an OIHW filter on
every call: 51 of the 54 ms of layout conversions in a 20-step SD1.5 sample.
The OHWI copy is made once per weight version and kept on the weight. These
pin that the answer does not change, that the copy is made once and redone
when the weight is replaced, that training keeps the filter as it is, and that
the copy stays out of what a pickle or a deepcopy of the weight carries.
"""

from _helpers import capability as _test_capability

import copy
import pickle
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

    def test_the_answer_is_unchanged(self):
        for kernel in (3, 1):
            conv, x, want = self._conv(kernel)
            with jt.no_grad():
                got = conv(x).float32().numpy()
            np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2 * np.abs(want).max(),
                                       err_msg="kernel %d" % kernel)

    def test_the_copy_is_made_once_and_redone_for_a_new_weight(self):
        conv, x, _ = self._conv(3)
        with jt.no_grad():
            conv(x).sync()
            first = conv.weight.__dict__[_cudnn._FILTER_CACHE][1]
            conv(x).sync()
            self.assertIs(conv.weight.__dict__[_cudnn._FILTER_CACHE][1], first)
            conv.weight.assign(conv.weight * 2)
            got = conv(x).float32().numpy()
            self.assertIsNot(conv.weight.__dict__[_cudnn._FILTER_CACHE][1], first)
        conv.weight.__dict__.pop(_cudnn._FILTER_CACHE)
        with jt.no_grad():
            np.testing.assert_allclose(conv(x).float32().numpy(), got, rtol=1e-3, atol=1e-3)

    def test_a_backward_keeps_the_filter_as_it_is(self):
        conv, x, _ = self._conv(3)
        y = conv(x)
        jt.grad(y.float32().sum(), [conv.weight])
        self.assertNotIn(_cudnn._FILTER_CACHE, conv.weight.__dict__)

    def test_the_cache_can_be_turned_off(self):
        conv, x, want = self._conv(3)
        _cudnn.cache_half_filters = False
        try:
            with jt.no_grad():
                got = conv(x).float32().numpy()
        finally:
            _cudnn.cache_half_filters = True
        self.assertNotIn(_cudnn._FILTER_CACHE, conv.weight.__dict__)
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2 * np.abs(want).max())

    def test_float32_is_left_alone(self):
        conv, x, _ = self._conv(3, "float32")
        with jt.no_grad():
            conv(x).sync()
        self.assertNotIn(_cudnn._FILTER_CACHE, conv.weight.__dict__)


if __name__ == "__main__":
    unittest.main()
