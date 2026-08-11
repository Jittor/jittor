# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reparameterized-sample (``rsample``) gradient correctness for the distributions.

``rsample`` is what makes VAEs / variational inference trainable: the sample must carry a
gradient back to the distribution parameters. The dangerous failure is silent -- a
``rsample`` that secretly DETACHES returns a perfectly valid sample whose gradient is zero
(or crashes), and every forward / log_prob test still passes. This project has hit exactly
that (Normal missing rsample; Gamma/Beta/Dirichlet rsample crashing or returning a 0
gradient -- see CONTEXT history). This module locks the gradients.

Two kinds of oracle:

  * **Exact** (location-scale families) -- ``Normal``/``LogNormal`` reparameterize as a
    closed form in the realized noise ``eps``, so the gradient is known exactly:
    ``Normal``: ``d/dmu = 1``, ``d/dsigma = eps``; ``LogNormal``: ``d/dloc = x``,
    ``d/dscale = x*eps``. Checked on CPU and the accelerator.
  * **Unbiased Monte-Carlo** (implicit reparam) -- for ``Gamma`` the pathwise estimator of
    ``d/dconcentration E[X]`` must converge to the analytic ``1/rate`` (``E[X]=conc/rate``).
    Plus a guard that ``Gamma``/``Beta``/``Dirichlet`` rsample gradients are finite and
    non-zero (the detach/zero-grad bug class).

Run::  python -m pytest tests/core/test_distributions_grad.py
"""
import unittest

import numpy as np
import jittor as jt
import jittor.distributions as D

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for, to_numpy,
)


class TestReparamExact(JittorTestCase):
    """Location-scale rsample: analytic gradient in the realized noise."""

    def _devices(self, body):
        for dev in get_all_device_types():
            with self.subTest(device=dev):
                with jt.flag_scope(use_cuda=use_cuda_for(dev)):
                    body(dev)

    def test_normal_rsample_grad(self):
        def body(dev):
            mu = jt.array(np.array([0.5, 1.0, -0.3, 2.0], "float32")); mu.requires_grad = True
            sigma = jt.array(np.array([1.0, 2.0, 0.5, 1.5], "float32")); sigma.requires_grad = True
            x = D.Normal(mu, sigma).rsample()
            eps = to_numpy((x - mu) / sigma)                 # realized noise
            gmu, gsig = jt.grad(x.sum(), [mu, sigma])
            self.assertEqual(to_numpy(gmu), np.ones(4, "float32"), atol=1e-5,
                             msg=f"Normal d/dmu == 1 [{dev}]")
            self.assertEqual(to_numpy(gsig), eps, atol=1e-4,
                             msg=f"Normal d/dsigma == eps [{dev}]")
        self._devices(body)

    def test_lognormal_rsample_grad(self):
        def body(dev):
            loc = jt.array(np.array([0.1, -0.2, 0.3], "float32")); loc.requires_grad = True
            scale = jt.array(np.array([0.5, 1.0, 0.8], "float32")); scale.requires_grad = True
            x = D.LogNormal(loc, scale).rsample()
            xn = to_numpy(x)
            eps = (np.log(xn) - to_numpy(loc)) / to_numpy(scale)
            gloc, gscale = jt.grad(x.sum(), [loc, scale])
            # d/dloc exp(loc+scale*eps) = x ; d/dscale = x*eps
            self.assertEqual(to_numpy(gloc), xn, atol=1e-3, rtol=1e-3,
                             msg=f"LogNormal d/dloc == x [{dev}]")
            self.assertEqual(to_numpy(gscale), xn * eps, atol=1e-3, rtol=1e-3,
                             msg=f"LogNormal d/dscale == x*eps [{dev}]")
        self._devices(body)


class TestReparamImplicit(JittorTestCase):
    """Implicit-reparam rsample: unbiased MC gradient + a finite/non-zero guard."""

    def test_gamma_pathwise_expectation_grad(self):
        # E[X] = concentration / rate ; so d/dconcentration E[X] = 1/rate. The pathwise
        # (rsample) estimator of grad(mean(samples)) must converge to that. CPU, fixed seed.
        with jt.flag_scope(use_cuda=0):
            jt.set_global_seed(20240628)
            N = 20000
            c = jt.array(np.array([2.0], "float32")); c.requires_grad = True
            rate = 1.5
            conc = c.broadcast([N])
            x = D.Gamma(conc, jt.ones([N]) * rate).rsample()
            g = to_numpy(jt.grad(x.mean(), c))[0]
            self.assertAlmostEqual(g, 1.0 / rate, delta=0.05,
                                   msg=f"Gamma pathwise d/dconc E[X] ~ 1/rate (got {g:.4f})")

    def test_implicit_rsample_grads_finite_nonzero(self):
        # the detach/zero-grad bug class: Gamma/Beta/Dirichlet rsample must produce a
        # finite, non-zero gradient w.r.t. concentration. NB: the objective is a WEIGHTED
        # sum, not a plain sum -- a Dirichlet sample is constrained to sum to 1, so a plain
        # sum() is the constant 1 and its gradient is legitimately 0 (not a detach bug).
        with jt.flag_scope(use_cuda=0):
            jt.set_global_seed(7)
            checks = []
            conc = jt.array(np.array([1.5, 2.0, 3.0], "float32")); conc.requires_grad = True
            w3 = jt.array(np.array([1.0, 2.0, 3.0], "float32"))
            g = jt.grad((D.Gamma(conc, jt.ones([3])).rsample() * w3).sum(), conc)
            checks.append(("Gamma", to_numpy(g)))
            a = jt.array(np.array([2.0, 3.0], "float32")); a.requires_grad = True
            w2 = jt.array(np.array([1.0, 2.0], "float32"))
            g = jt.grad((D.Beta(a, jt.array(np.array([2.0, 2.0], "float32"))).rsample() * w2).sum(), a)
            checks.append(("Beta", to_numpy(g)))
            dc = jt.array(np.array([1.5, 2.0, 2.5], "float32")); dc.requires_grad = True
            g = jt.grad((D.Dirichlet(dc).rsample() * w3).sum(), dc)   # weighted: sum(x)=1 is degenerate
            checks.append(("Dirichlet", to_numpy(g)))
            for name, gv in checks:
                self.assertTrue(np.all(np.isfinite(gv)), f"{name} rsample grad must be finite")
                self.assertGreater(float(np.abs(gv).max()), 1e-6,
                                   f"{name} rsample grad must be non-zero (detach bug)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
