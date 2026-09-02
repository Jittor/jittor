# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Seed/output alignment in ``jittor.gradfunctional``'s ``_autograd_grad``.

``_autograd_grad`` drops the outputs that are ``None`` or do not require
gradients, collecting the survivors in ``new_outputs`` and their seeds in
``new_grad_outputs``.  It then built the seeded loss by zipping ``new_outputs``
with the *unfiltered* ``grad_outputs``: every seed after a dropped output was
attached to the wrong output, and any seed past the end of the shortened list
was discarded by ``zip``.  ``new_grad_outputs`` was constructed and never read.

The expected values here are plain calculus and were also confirmed once
against ``torch.autograd.functional.vjp`` from a binary PyTorch 2.12 build in a
separate process; the tests need only numpy.
"""

import unittest

import numpy as np

import jittor as jt
from jittor.gradfunctional import vjp, jvp


X = np.array([1.0, 2.0, 3.0], dtype="float32")


def _scalar(value):
    return jt.array(np.array(value, dtype="float32"))


class TestAutogradGradSeedAlignment(unittest.TestCase):
    def test_vjp_skips_non_differentiable_first_output(self):
        """The seed of a dropped output must not slide onto the next one."""

        def func(x):
            return (x * 3.0).stop_grad(), (x * x).sum()

        _, grad = vjp(
            func, jt.array(X), (jt.ones((3,)), _scalar(5.0)),
        )
        # d/dx of 5 * sum(x^2).  The misaligned version used the ones-seed of
        # the stop_grad output instead, giving 2 * 3 * x.
        np.testing.assert_allclose(grad.numpy(), 5.0 * 2.0 * X, rtol=1e-5, atol=1e-5)

    def test_vjp_skips_non_differentiable_middle_output(self):
        def func(x):
            return (x * x).sum(), (x * 3.0).stop_grad(), (x ** 3).sum()

        _, grad = vjp(
            func, jt.array(X), (_scalar(5.0), jt.ones((3,)), _scalar(2.0)),
        )
        expected = 5.0 * 2.0 * X + 2.0 * 3.0 * X ** 2
        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_vjp_last_seed_is_not_dropped(self):
        """``zip`` truncating to the shortened list silently lost the last seed."""

        def func(x):
            return (
                (x * 5.0).stop_grad(),
                (x * 7.0).stop_grad(),
                (x * x).sum(),
            )

        _, grad = vjp(
            func,
            jt.array(X),
            (jt.zeros((3,)), jt.zeros((3,)), _scalar(4.0)),
        )
        np.testing.assert_allclose(grad.numpy(), 4.0 * 2.0 * X, rtol=1e-5, atol=1e-5)

    def test_vjp_all_outputs_differentiable_is_unchanged(self):
        def func(x):
            return (x * x).sum(), (x ** 3).sum()

        _, grad = vjp(func, jt.array(X), (_scalar(5.0), _scalar(2.0)))
        expected = 5.0 * 2.0 * X + 2.0 * 3.0 * X ** 2
        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_vjp_seed_actually_reaches_the_kept_output(self):
        """Changing only the surviving output's seed must change the result."""

        def func(x):
            return (x * 3.0).stop_grad(), (x * x).sum()

        results = []
        for seed in (1.0, 5.0):
            _, grad = vjp(func, jt.array(X), (jt.ones((3,)), _scalar(seed)))
            results.append(grad.numpy())
        np.testing.assert_allclose(
            results[1], results[0] * 5.0, rtol=1e-5, atol=1e-5)

    def test_jvp_multi_input_is_unchanged(self):
        def func(a, b):
            return (a * a).sum() + (b * b * b).sum()

        _, tangent = jvp(
            func,
            (jt.array(X), jt.array(X * 2)),
            (jt.ones((3,)), jt.ones((3,)) * 2),
        )
        expected = (2 * X).sum() + (3 * (2 * X) ** 2 * 2).sum()
        np.testing.assert_allclose(
            tangent.numpy().reshape(-1), [expected], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
