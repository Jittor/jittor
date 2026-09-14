# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`nn.Linear` through cuBLASLt: bias folded in, algorithm picked by timing.

The fast path answers for a forward under no_grad on dense float32; every
other shape of the problem has to reach the portable path instead, because
this op has no backward and silently losing a gradient would be worse than
any speedup. So most of what is pinned here is the falling back.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor.backends.cuda.kernels.cublas.lt_linear_cuda import lt_linear_cuda


def _reference(x, w, b):
    cin = w.shape[1]
    flat = x.reshape(-1, cin) @ w.T + b
    return flat.reshape(list(x.shape[:-1]) + [w.shape[0]])


@unittest.skipIf(not _test_capability.machine_has_accelerator("cuda"), "no CUDA device")
class TestLtLinearCuda(unittest.TestCase):

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        self.rs = np.random.RandomState(0)

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    def _arrays(self, shape, cin, cout):
        x = (self.rs.randn(*shape) * 0.1).astype("float32")
        w = (self.rs.randn(cout, cin) * 0.1).astype("float32")
        b = (self.rs.randn(cout) * 0.1).astype("float32")
        return x, w, b

    def test_it_matches_the_portable_path_at_every_rank(self):
        for shape, cin, cout in (((2048, 512), 512, 1536),
                                 ((1, 1, 512), 512, 1536),
                                 ((8, 256, 512), 512, 1536),
                                 ((2048, 2048), 2048, 512)):
            with self.subTest(shape=shape):
                xn, wn, bn = self._arrays(shape, cin, cout)
                x, w, b = jt.array(xn), jt.array(wn), jt.array(bn)
                with jt.no_grad():
                    got = lt_linear_cuda(x, w, b)
                self.assertIsNotNone(got, "should be served")
                want = _reference(xn, wn, bn)
                self.assertEqual(tuple(got.shape), want.shape)
                np.testing.assert_allclose(got.numpy(), want, rtol=2e-5, atol=2e-5)

    def test_repeated_calls_answer_the_same(self):
        # The algorithm is chosen once by timing and then fixed; the answer
        # must not move afterwards.
        xn, wn, bn = self._arrays((512, 512), 512, 512)
        x, w, b = jt.array(xn), jt.array(wn), jt.array(bn)
        with jt.no_grad():
            first = lt_linear_cuda(x, w, b).numpy().copy()
            for _ in range(4):
                np.testing.assert_array_equal(lt_linear_cuda(x, w, b).numpy(), first)

    def test_a_gradient_falls_back_and_still_learns(self):
        model = nn.Linear(8, 4)
        opt = nn.SGD(model.parameters(), lr=0.1)
        x = jt.array(np.arange(16, dtype="float32").reshape(2, 8) * 0.1)
        before = model.weight.numpy().copy()
        for _ in range(3):
            opt.step((model(x) ** 2).mean())
        self.assertFalse(np.array_equal(model.weight.numpy(), before))
        grad = jt.grad((model(x) ** 2).mean(), [model.weight])[0].numpy()
        self.assertTrue(np.isfinite(grad).all())
        self.assertGreater(np.abs(grad).max(), 0)

    def test_what_it_refuses(self):
        xn, wn, bn = self._arrays((512, 512), 512, 512)
        x, w, b = jt.array(xn), jt.array(wn), jt.array(bn)
        with jt.no_grad():
            # float16
            self.assertIsNone(lt_linear_cuda(x.float16(), w.float16(), b.float16()))
            # a problem too small to be worth timing
            small = jt.array(self.rs.randn(2, 4).astype("float32"))
            sw = jt.array(self.rs.randn(3, 4).astype("float32"))
            sb = jt.array(self.rs.randn(3).astype("float32"))
            self.assertIsNone(lt_linear_cuda(small, sw, sb))
            # no bias
            self.assertIsNone(lt_linear_cuda(x, w, None))
        # on CPU
        jt.flags.use_cuda = 0
        try:
            with jt.no_grad():
                self.assertIsNone(lt_linear_cuda(x, w, b))
        finally:
            jt.flags.use_cuda = 1

    def test_the_module_agrees_with_the_portable_module(self):
        jt.set_global_seed(7)
        model = nn.Linear(512, 256)
        x = jt.array((self.rs.randn(64, 512) * 0.1).astype("float32"))
        with jt.no_grad():
            fast = model(x).numpy().copy()
        w, b = model.weight.numpy(), model.bias.numpy()
        np.testing.assert_allclose(fast, x.numpy() @ w.T + b, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
