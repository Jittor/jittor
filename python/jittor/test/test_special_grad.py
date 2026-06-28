# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Analytic backward checks for the special functions whose hand-written gradient
kernels cannot be validated by finite-difference gradcheck.

``erfinv``, ``lgamma`` and ``digamma`` ship limited-precision (~1e-7) forward kernels,
so the float64 finite-difference numerical gradient in ``test_ops``'s gradcheck is
round-off dominated and unreliable for them (digamma is skipped there for exactly this
reason). Their derivatives, however, have exact CLOSED FORMS, so the right oracle is
SciPy -- an independent implementation of those very special functions:

  * ``d/dx erfinv(x)   = sqrt(pi)/2 * exp(erfinv(x)^2)``
  * ``d/dx lgamma(x)   = digamma(x)        == scipy.special.digamma(x)``
  * ``d/dx digamma(x)  = polygamma(1, x)   == scipy.special.polygamma(1, x)``

This module differentiates each op with ``jt.grad`` and asserts the result matches the
SciPy closed form, on every device (CPU and CUDA), at float32 tolerance. Together with
``test_device_parity`` (which pins CUDA-vs-CPU agreement) this gives the special-function
backward kernels real, independent coverage despite the FD gap.

Run::  python -m jittor.test.test_special_grad
"""
import unittest

import numpy as np
import jittor as jt

try:
    import scipy.special as sp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from jittor.test._internal.common_utils import (
    JittorTestCase, get_all_device_types, use_cuda_for, to_numpy,
)


@unittest.skipUnless(HAS_SCIPY, "scipy required for special-function backward oracle")
class TestSpecialGrad(JittorTestCase):
    """Backward of erfinv / lgamma / digamma vs the SciPy closed form, per device."""

    # float32 kernels: a real backward bug is orders of magnitude past this.
    TOL = 2e-4

    def _devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def _check_grad(self, op, x0, ref_grad, label):
        def body(dev):
            x = jt.array(x0)
            g = jt.grad(op(x).sum(), x)
            self.assertEqual(to_numpy(g), ref_grad, atol=self.TOL, rtol=self.TOL,
                             msg=f"{label} backward vs scipy closed form [{dev}]")
        self._devices(body)

    def test_erfinv_grad(self):
        x0 = np.array([0.1, 0.4, -0.3, 0.6, -0.8], dtype="float32")
        ref = (np.sqrt(np.pi) / 2.0 * np.exp(sp.erfinv(x0) ** 2)).astype("float32")
        self._check_grad(jt.erfinv, x0, ref, "erfinv")

    def test_lgamma_grad(self):
        x0 = np.array([0.5, 1.2, 2.3, 3.7, 5.1], dtype="float32")
        ref = sp.digamma(x0).astype("float32")          # d/dx lgamma == digamma
        self._check_grad(lambda x: jt.lgamma.apply(x), x0, ref, "lgamma")

    def test_digamma_grad(self):
        x0 = np.array([0.5, 1.2, 2.3, 3.7, 5.1], dtype="float32")
        ref = sp.polygamma(1, x0).astype("float32")     # d/dx digamma == trigamma
        self._check_grad(lambda x: jt.digamma.apply(x), x0, ref, "digamma")

    def test_erfinv_grad_second_order(self):
        # erfinv is smooth: its 2nd derivative exists. d2/dx2 erfinv = g(x)*(2*erfinv*g(x)),
        # g = sqrt(pi)/2*exp(erfinv^2). Check jt's gradgrad matches that closed form.
        x0 = np.array([0.1, 0.4, -0.3, 0.6], dtype="float32")
        e = sp.erfinv(x0)
        g = np.sqrt(np.pi) / 2.0 * np.exp(e ** 2)
        ref2 = (g * (2.0 * e * g)).astype("float32")

        def body(dev):
            x = jt.array(x0)
            g1 = jt.grad(jt.erfinv(x).sum(), x)
            g2 = jt.grad(g1.sum(), x)
            self.assertEqual(to_numpy(g2), ref2, atol=2e-3, rtol=2e-3,
                             msg=f"erfinv 2nd-order vs closed form [{dev}]")
        self._devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
