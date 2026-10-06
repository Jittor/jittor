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
from jittor.backends.cuda.kernels.cublas import lt_linear_cuda as _lt
from jittor.backends.cuda.kernels.cublas.lt_linear_cuda import lt_linear_cuda


def _reference(x, w, b):
    cin = w.shape[1]
    flat = x.reshape(-1, cin) @ w.T + b
    return flat.reshape(list(x.shape[:-1]) + [w.shape[0]])


# `check_accelerator(...).enabled`, not `machine_has_accelerator`: the
# question a gate asks is whether CUDA is usable in *this build*, not
# whether the machine has a card. The helper's own docstring says so --
# it is "the question to ask before reporting that something is
# unverifiable here". A CPU-only build on a GPU box answered yes, the
# class ran, and `jt.flags.use_cuda = 1` raised `No CUDA found`. It also
# keeps the loud path: a CUDA build that FAILED still asserts rather
# than skipping (see tests/_helpers/capability.py).
@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
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

    def test_float32_follows_the_tf32_policy(self):
        # The float32 route used to compute in full float32 whatever the
        # float32 matmul policy said, so `allow_tf32` left every linear layer
        # on the SIMT kernels. Under the policy it now rounds like the
        # portable path does -- visibly away from float64, the way TF32 does --
        # and without it the result is still full float32.
        xn, wn, bn = self._arrays((2048, 1024), 1024, 1024)
        want = xn.astype("float64") @ wn.astype("float64").T + bn
        x, w, b = jt.array(xn), jt.array(wn), jt.array(bn)
        with jt.no_grad():
            full = lt_linear_cuda(x, w, b).numpy()
            with jt.flag_scope(cuda_allow_tf32=1):
                tf32 = lt_linear_cuda(x, w, b).numpy()
                portable = (jt.matmul(x, w.transpose()) + b).numpy()
        self.assertLess(np.abs(full - want).max(), 1e-4)
        self.assertGreater(np.abs(tf32 - want).max(), 1e-4)
        np.testing.assert_allclose(tf32, want, rtol=0, atol=5e-2)
        # Same precision as the portable path under the same policy.
        self.assertLess(np.abs(tf32 - want).max(), 4 * np.abs(portable - want).max() + 1e-6)

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


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestLtLinearTraining(unittest.TestCase):
    """The training route: the portable forward, and a backward whose bias
    gradient comes out of the weight gradient's GEMM (cuBLASLt's BGRADB).
    Checked against the same step on the CPU in float64."""

    def setUp(self):
        self.rs = np.random.RandomState(1)
        self.calls = []
        # Small problems keep the float64 CPU reference cheap; the size floor
        # the route applies is its own case below.
        for name, value in (("_MIN_ROWS", 1), ("_MIN_PRODUCT", 1)):
            self.addCleanup(setattr, _lt, name, getattr(_lt, name))
            setattr(_lt, name, value)
        original = _lt._LtLinearProduct.grad

        def counted(ctx, grad):
            self.calls.append(1)
            return original(ctx, grad)

        _lt._LtLinearProduct.grad = counted
        self.addCleanup(setattr, _lt._LtLinearProduct, "grad", original)

    def _arrays(self, shape, cin, cout):
        x = (self.rs.randn(*shape, cin) * 0.5).astype("float32")
        w = (self.rs.randn(cout, cin) * 0.5).astype("float32")
        b = (self.rs.randn(cout) * 0.5).astype("float32")
        return x, w, b

    def _step(self, use_cuda, x, w, b, dout, dtype, x_grad=True, bias_grad=True,
              transposed=False, no_bias=False):
        with jt.flag_scope(use_cuda=use_cuda, cuda_allow_tf32=0):
            jx, jw, jb = (jt.array(t).cast(dtype) for t in (x, w, b))
            if not x_grad:
                jx = jx.stop_grad()
            if not bias_grad:
                jb = jb.stop_grad()
            out = nn.linear(jx, jw, None if no_bias else jb)
            jd = jt.array(dout).cast(dtype)
            # A transposed product hands the backward a strided gradient.
            loss = (out.transpose() * jd.transpose()).sum() if transposed else (out * jd).sum()
            wants = [v for v, keep in ((jx, x_grad), (jw, True), (jb, bias_grad and not no_bias))
                     if keep]
            grads = jt.grad(loss, wants)
            return [out.float64().numpy()] + [g.float64().numpy() for g in grads]

    def _check(self, shape, cin, cout, dtype, tol, **kw):
        x, w, b = self._arrays(shape, cin, cout)
        dout = self.rs.randn(*shape, cout).astype("float32")
        want = self._step(0, x, w, b, dout, "float64", **kw)
        self.calls.clear()
        got = self._step(1, x, w, b, dout, dtype, **kw)
        self.assertEqual(self.calls, [1], "the training route should have served it")
        self.assertEqual(len(got), len(want))
        for i, (g, r) in enumerate(zip(got, want)):
            np.testing.assert_allclose(g, r, rtol=tol, atol=tol * np.abs(r).max(),
                                       err_msg=f"{dtype} {shape}: output, then gradients; #{i}")

    def test_gradients_match_the_cpu_in_every_dtype(self):
        for dtype, tol in (("float32", 1e-4), ("float16", 2e-2), ("bfloat16", 6e-2)):
            for shape, cin, cout in (((96,), 64, 160), ((3, 40), 72, 48)):
                with self.subTest(dtype=dtype, shape=shape):
                    self._check(shape, cin, cout, dtype, tol)

    def test_without_a_bias(self):
        for dtype, tol in (("float32", 1e-4), ("float16", 2e-2)):
            with self.subTest(dtype=dtype):
                self._check((3, 40), 72, 48, dtype, tol, no_bias=True)
        self._check((96,), 64, 160, "float32", 1e-4, no_bias=True, transposed=True)

    def test_a_module_without_a_bias_takes_the_route(self):
        # `nn.Linear(bias=False)` has a native fast path; a large product that
        # records gradients leaves it for the timed GEMMs.
        _lt._MIN_ROWS, _lt._MIN_PRODUCT = 1024, 1 << 28
        with jt.flag_scope(use_cuda=1):
            model = nn.Linear(768, 768, bias=False)
            x = jt.array(self.rs.randn(2048, 768).astype("float32"))
            out = model(x)
            grad = jt.grad((out * out).sum(), [model.weight])[0]
            grad.sync()
        self.assertEqual(self.calls, [1])

    def test_without_an_input_or_a_bias_gradient(self):
        self._check((3, 40), 72, 48, "float32", 1e-4, x_grad=False)
        self._check((3, 40), 72, 48, "float32", 1e-4, bias_grad=False)

    def test_a_gradient_too_large_to_fold_sums_the_bias_gradient(self):
        # The fold is taken for a gradient the L2 cache holds; with none to
        # hold it, dW still comes from cuBLASLt and db from a plain sum.
        saved = dict(_lt._L2_BYTES)
        self.addCleanup(lambda: (_lt._L2_BYTES.clear(), _lt._L2_BYTES.update(saved)))
        with jt.flag_scope(use_cuda=1):
            _lt._L2_BYTES[int(jt.core.current_device())] = 0
        self._check((3, 40), 72, 48, "float32", 1e-4)
        self._check((96,), 64, 160, "float16", 2e-2)

    def test_a_strided_output_gradient_takes_the_portable_products(self):
        self._check((96,), 64, 160, "float32", 1e-4, transposed=True)

    def test_what_it_leaves_to_the_portable_route(self):
        with jt.flag_scope(use_cuda=1):
            x, w, b = (jt.array(t) for t in self._arrays((64,), 96, 80))
            self.assertIsNone(_lt.lt_linear_train_cuda(x, w.stop_grad(), b))
            with jt.no_grad():
                self.assertIsNone(_lt.lt_linear_train_cuda(x, w, b))
            self.assertIsNone(_lt.lt_linear_train_cuda(x.float64(), w.float64(), b.float64()))
            self.assertIsNone(_lt.lt_linear_train_cuda(x[:, ::2], w[:, ::2], b))
            # Too small for a better GEMM to repay the Function's host time:
            # a diffusion UNet's 16-row time-embedding projection.
            _lt._MIN_ROWS, _lt._MIN_PRODUCT = 1024, 1 << 28
            self.assertIsNone(_lt.lt_linear_train_cuda(
                jt.array(self.rs.randn(16, 512).astype("float32")),
                jt.array(self.rs.randn(512, 512).astype("float32")),
                jt.array(self.rs.randn(512).astype("float32"))))
            self.assertIsNotNone(_lt.lt_linear_train_cuda(
                jt.array(self.rs.randn(4096, 768).astype("float32")),
                jt.array(self.rs.randn(768, 768).astype("float32")),
                jt.array(self.rs.randn(768).astype("float32"))))
        with jt.flag_scope(use_cuda=0):
            self.assertIsNone(_lt.lt_linear_train_cuda(x, w, b))

if __name__ == "__main__":
    unittest.main()
