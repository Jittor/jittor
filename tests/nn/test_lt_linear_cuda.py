# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`nn.Linear` through cuBLASLt: bias folded in, algorithm picked by timing.

The fast path answers for a forward under no_grad on dense float16/float32; every
other shape of the problem has to reach the portable path instead, because
this op has no backward and silently losing a gradient would be worse than
any speedup. So most of what is pinned here is the falling back.

The compute dtype is `dtype_infer(x, weight)`'s answer and not a property of
this module, so an autocast scope is a case of the same op rather than a second
one. Two things about it are pinned deliberately, because both were wrong at
some point: the result has to come out in the dtype the *portable* path would
produce (float32 out of a float16 graph is silently wrong downstream), and the
scale type handed to cuBLASLt has to stay `CUDA_R_32F` -- the operand type there
makes every fp16 algorithm query fail, and the fallback hides it.
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
        self._amp_reg = jt.flags.amp_reg
        jt.flags.use_cuda = 1
        self.rs = np.random.RandomState(0)

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda
        jt.flags.amp_reg = self._amp_reg

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

    def test_it_matches_the_portable_path_in_float16(self):
        # float16 operands are served now, and the reference is the same
        # product in float32 -- so the tolerance is float16's, not float32's.
        for shape, cin, cout in (((2048, 512), 512, 1536),
                                 ((8, 256, 512), 512, 1536)):
            with self.subTest(shape=shape):
                xn, wn, bn = self._arrays(shape, cin, cout)
                x = jt.array(xn).cast("float16")
                w = jt.array(wn).cast("float16")
                b = jt.array(bn).cast("float16")
                with jt.no_grad():
                    got = lt_linear_cuda(x, w, b)
                self.assertIsNotNone(got, "float16 should be served")
                self.assertEqual(str(got.dtype), "float16")
                want = _reference(xn, wn, bn)
                np.testing.assert_allclose(got.numpy().astype("float32"), want,
                                           rtol=5e-3, atol=5e-3)

    def test_an_autocast_scope_keeps_the_portable_result_dtype(self):
        # `jt.nn.linear` computes float16 under `amp_prefer16` even though both
        # operands are still float32. If this op answers float32 instead, one
        # linear layer at a time lifts a float16 graph back to float32.
        xn, wn, bn = self._arrays((256, 512), 512, 512)
        x, w, b = jt.array(xn), jt.array(wn), jt.array(bn)
        jt.flags.amp_reg = jt.amp_flags.prefer16
        with jt.no_grad():
            portable = jt.nn.linear(x, w, b)
            fused = lt_linear_cuda(x, w, b)
            self.assertIsNotNone(fused, "should be served under the amp register")
            self.assertEqual(str(portable.dtype), str(fused.dtype),
                             "the fused route must not change the result dtype")
            np.testing.assert_allclose(fused.numpy().astype("float32"),
                                       portable.numpy().astype("float32"),
                                       rtol=2e-3, atol=2e-3)

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
            # bfloat16 needs its own descriptors and its own accumulate rule
            self.assertIsNone(lt_linear_cuda(x.cast("bfloat16"),
                                             w.cast("bfloat16"),
                                             b.cast("bfloat16")))
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
