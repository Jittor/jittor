# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Real (non-complex) jt.linalg.qr, verified against a numpy oracle directly
# (no torch dependency, unlike tests/ops/test_linalg.py's TestLinalgOp.test_qr,
# which requires an installed independent torch and skips without one).
#
# qr()'s backward used to be a plain jt.numpy_code op: an opaque backward
# callback with no backward of its own, so a second derivative raised
# ("numpy_code has no backward callback"), and the m<n (wide) case raised
# NotImplementedError in backward outright (forward already handled it).
# qr() is now a jt.Function whose grad() is built from ordinary
# differentiable jittor ops (matmul/transpose/solve/tril) plus a live
# recursive call back into qr() itself -- so it composes to any order, and
# the wide case gets a real (not just forward-only) implementation.
import unittest

import numpy as np

import jittor as jt
from jittor import linalg


def _recon_close(q, r, a, **kw):
    np.testing.assert_allclose(np.matmul(q, r), a, **kw)


class TestQR(unittest.TestCase):
    def test_square_forward_and_backward(self):
        rng = np.random.RandomState(0)
        a_np = rng.randn(4, 4).astype("float32")
        a = jt.array(a_np)
        a.requires_grad = True
        q, r = linalg.qr(a)
        self.assertEqual(tuple(q.shape), (4, 4))
        self.assertEqual(tuple(r.shape), (4, 4))
        _recon_close(q.numpy(), r.numpy(), a_np, atol=1e-3, rtol=1e-3)
        qtq = q.numpy().T @ q.numpy()
        np.testing.assert_allclose(qtq, np.eye(4), atol=1e-3, rtol=1e-3)

        loss = q.sum() + r.sum()
        g = jt.grad(loss, [a])[0]
        g.sync()
        self._gradcheck((4, 4), seed=0)

    def test_wide_forward_and_backward(self):
        # m < n: forward already supported this; backward used to raise
        # NotImplementedError unconditionally.
        rng = np.random.RandomState(1)
        a_np = rng.randn(2, 5).astype("float32")
        a = jt.array(a_np)
        a.requires_grad = True
        q, r = linalg.qr(a)
        self.assertEqual(tuple(q.shape), (2, 2))
        self.assertEqual(tuple(r.shape), (2, 5))
        _recon_close(q.numpy(), r.numpy(), a_np, atol=1e-3, rtol=1e-3)
        qtq = q.numpy().T @ q.numpy()
        np.testing.assert_allclose(qtq, np.eye(2), atol=1e-3, rtol=1e-3)

        loss = q.sum() + r.sum()
        g = jt.grad(loss, [a])[0]
        g.sync()  # used to raise NotImplementedError here
        self._gradcheck((2, 5), seed=1)

    def test_tall_forward_and_backward(self):
        self._gradcheck((5, 3), seed=2)

    def test_second_order_grad(self):
        # Second derivative through a numpy_code-backed op used to raise
        # "numpy_code has no backward callback for input N" -- qr's grad()
        # is now itself a real op graph (matmul/transpose/solve), so
        # differentiating it again works like any composed jittor op.
        a = jt.array(np.random.RandomState(3).randn(4, 4).astype("float32"))
        a.requires_grad = True
        q, r = linalg.qr(a)
        loss = q.sum() + r.sum()
        g = jt.grad(loss, [a], retain_graph=True)[0]
        g2 = jt.grad(g.sum(), [a])[0]
        g2.sync()
        self.assertEqual(tuple(g2.shape), (4, 4))

    def _gradcheck(self, shape, seed, eps=1e-3):
        # Random-linear-projection loss + central finite differences: avoids
        # the Q/R sign ambiguity a raw reconstruction-only check would hit,
        # and (unlike abs().sum()) has no non-smooth point to produce a
        # spurious NaN/mismatch at.
        rng = np.random.RandomState(seed)
        a_np = rng.randn(*shape).astype("float32")
        m, n = shape
        k = min(m, n)
        p_q = (rng.randn(m, k) * 0.1).astype("float32")
        p_r = (rng.randn(k, n) * 0.1).astype("float32")

        def loss_np(a_val):
            q_np, r_np = np.linalg.qr(a_val)
            return float(np.sum(q_np * p_q) + np.sum(r_np * p_r))

        g_np = np.zeros_like(a_np)
        for idx in np.ndindex(a_np.shape):
            plus = a_np.copy(); plus[idx] += eps
            minus = a_np.copy(); minus[idx] -= eps
            g_np[idx] = (loss_np(plus) - loss_np(minus)) / (2 * eps)

        a = jt.array(a_np)
        a.requires_grad = True
        q, r = linalg.qr(a)
        loss = (q * jt.array(p_q)).sum() + (r * jt.array(p_r)).sum()
        g = jt.grad(loss, [a])[0]
        np.testing.assert_allclose(g.numpy(), g_np, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
