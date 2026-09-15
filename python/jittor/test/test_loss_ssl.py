# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# Maintainers:
#     Vrinda12-tech 
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import unittest
import numpy as np

import jittor as jt
from jittor.loss_ssl import coding_rate, TotalCodingRate

jt.flags.use_cuda = 0


def reference_numpy(x_np, eps=0.01):
    """Literal numpy transcription of the reference PyTorch
    TotalCodingRate.compute_discrimn_loss (tsb0601/EMP-SSL, loss.py)."""
    W = x_np.T
    p, m = W.shape
    I = np.eye(p)
    scalar = p / (m * eps)
    _, logdet = np.linalg.slogdet(I + scalar * (W @ W.T))
    return float(logdet / 2.0)


class TestLossSSL(unittest.TestCase):

    def setUp(self):
        jt.set_global_seed(0)
        np.random.seed(0)

    def test_matches_literal_reference_transcription(self):
        x_np = np.random.randn(50, 20).astype(np.float32)
        for eps in (0.01, 0.1, 1.0, 5.0):
            got = coding_rate(jt.array(x_np), eps=eps, normalize=False).item()
            want = reference_numpy(x_np, eps=eps)
            self.assertAlmostEqual(got, want, places=2, msg=f"eps={eps}")

    def test_default_eps_matches_reference_default(self):
        self.assertEqual(TotalCodingRate().eps, 0.01)

    def test_no_normalization_by_default(self):
        self.assertFalse(TotalCodingRate().normalize)

    def test_loss_is_negative_of_rate(self):
        x_np = np.random.randn(30, 12).astype(np.float32)
        x = jt.array(x_np)
        rate = coding_rate(x, eps=0.01).item()
        loss = TotalCodingRate(eps=0.01)(x).item()
        self.assertAlmostEqual(loss, -rate, places=5)

    def test_raises_on_non_2d_input(self):
        with self.assertRaises(ValueError):
            coding_rate(jt.randn(4, 4, 4))

    def test_raises_on_too_few_samples(self):
        with self.assertRaises(ValueError):
            coding_rate(jt.randn(1, 8))

    def test_raises_on_nonpositive_eps(self):
        z = jt.randn(8, 4)
        with self.assertRaises(ValueError):
            coding_rate(z, eps=0.0)
        with self.assertRaises(ValueError):
            coding_rate(z, eps=-1.0)
        with self.assertRaises(ValueError):
            TotalCodingRate(eps=-0.5)

    def test_finite_even_at_exact_collapse(self):
        v = np.random.randn(1, 12).astype(np.float32)
        x_np = np.repeat(v, 50, axis=0)
        val = coding_rate(jt.array(x_np), eps=0.01).item()
        self.assertTrue(np.isfinite(val))

    def test_collapsed_batch_has_lower_rate_than_diverse_batch(self):
        n, d = 64, 16
        v = np.random.randn(d).astype(np.float32)
        collapsed_np = np.tile(v, (n, 1)) + np.random.normal(0, 1e-4, size=(n, d)).astype(np.float32)
        diverse_np = np.random.randn(n, d).astype(np.float32)
        rate_collapsed = coding_rate(jt.array(collapsed_np), eps=0.01).item()
        rate_diverse = coding_rate(jt.array(diverse_np), eps=0.01).item()
        self.assertLess(rate_collapsed, rate_diverse)

    def test_gradient_flows_and_is_finite(self):
        z = jt.randn(32, 16)
        z.requires_grad = True
        loss = TotalCodingRate(eps=0.01)(z)
        grad = jt.grad(loss, z)
        self.assertEqual(tuple(grad.shape), tuple(z.shape))
        self.assertFalse(bool(jt.isnan(grad).sum().item()))
        self.assertGreater(float((grad * grad).sum().item()), 0.0)

    def test_gradient_pushes_collapsed_batch_apart(self):
        v = np.random.randn(8).astype(np.float32)
        z_np = np.tile(v, (16, 1)) + np.random.normal(0, 1e-3, size=(16, 8)).astype(np.float32)
        z = jt.array(z_np)
        z.requires_grad = True
        std_before = float(z.numpy().std(axis=0).mean())
        loss = TotalCodingRate(eps=0.01)(z)
        grad = jt.grad(loss, z)
        z_after = jt.array(z_np) - 0.01 * grad.numpy()
        std_after = float(z_after.numpy().std(axis=0).mean())
        self.assertGreater(std_after, std_before)

    def test_normalize_true_differs_from_default_and_is_opt_in(self):
        z = jt.randn(20, 10)
        without = coding_rate(z, eps=0.5, normalize=False).item()
        with_norm = coding_rate(z, eps=0.5, normalize=True).item()
        self.assertNotAlmostEqual(without, with_norm, places=2)

    def test_combined_training_step_runs_without_error(self):
        import jittor.nn as nn
        encoder = nn.Linear(8, 4)
        criterion_tcr = TotalCodingRate(eps=0.01)
        optimizer = nn.SGD(encoder.parameters(), lr=0.01)
        x1 = jt.randn(16, 8)
        x2 = x1 + jt.randn(16, 8) * 0.01
        z1, z2 = encoder(x1), encoder(x2)
        z1n = z1 / (jt.norm(z1, dim=1, keepdim=True) + 1e-8)
        z2n = z2 / (jt.norm(z2, dim=1, keepdim=True) + 1e-8)
        pointwise = -(z1n * z2n).sum(dim=1).mean()
        total = pointwise + 1.0 * criterion_tcr(jt.concat([z1, z2], dim=0))
        optimizer.step(total)


if __name__ == '__main__':
    unittest.main()