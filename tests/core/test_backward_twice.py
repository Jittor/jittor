# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A second backward through a released graph must say so, not return zeros.

``grad(..., retain_graph=false)`` releases the backward graph by calling
``set_stop_grad()`` on every var of the closure that no Python holder keeps
alive.  ``stop_grad`` is *permanent* (see the comment on
``VarHolder::set_requires_grad``) and, on its own, indistinguishable from a var
the user stop_grad'ed on purpose -- so a later backward walked into one of those
vars, stopped there, and handed the target a zero gradient in silence.

The pattern that hits it in real code is a shared trunk with two heads: the
first ``optimizer.step(loss_a)`` (which defaults to ``retain_graph=False``)
releases the trunk, and the second head's backward then reports zeros for every
trunk parameter.  PyTorch raises "Trying to backward through the graph a second
time" here; this file pins the same contract.

Note the two different shapes this takes, and why the tests are ordered the way
they are.  Jittor is lazy: as long as nothing has been materialized the forward
graph is still standing and only ``stop_grad`` blocks the walk -- that is the
case asserted here.  Once a value *has* been fetched the executor tears the
finished forward ops down entirely, and the second backward then finds the
target with no successors at all; that path is covered by the missing-gradient
report (``jt.flags.missing_grad_error``), asserted at the end of this file.

Run::  python -m pytest tests/core/test_backward_twice.py
"""

import unittest

import numpy as np

import jittor as jt


def trunk_and_heads():
    """A shared intermediate that no Python name keeps alive inside the graph."""
    weight = jt.array(np.array([2.0, 3.0], dtype="float32"), dtype="float32")
    x = jt.array(np.array([1.0, 1.0], dtype="float32"), dtype="float32")
    hidden = (x * weight) * 2.0          # the trunk
    return weight, (hidden * 3.0).sum(), (hidden * 5.0).sum()


class TestBackwardTwice(unittest.TestCase):
    def test_second_backward_through_released_graph_raises(self):
        weight, loss_a, loss_b = trunk_and_heads()
        # Do not materialize between the two backwards: fetching a value would
        # tear the finished forward ops down and turn this into the "target has
        # no successors" case instead.
        first = jt.grad(loss_a, weight, retain_graph=False)
        with self.assertRaises(Exception) as caught:
            jt.grad(loss_b, weight, retain_graph=False)
        self.assertIn("backward through the graph a second time",
                      str(caught.exception), str(caught.exception)[:2000])
        # The first gradient is still correct and still computable.
        np.testing.assert_allclose(first.numpy(), [6.0, 6.0], rtol=1e-6)

    def test_released_var_is_never_silently_zero(self):
        # The exact silent-wrong-value shape of the bug: no exception, no
        # warning, and a gradient of all zeros.
        weight, loss_a, loss_b = trunk_and_heads()
        jt.grad(loss_a, weight, retain_graph=False)
        try:
            second = jt.grad(loss_b, weight, retain_graph=False)
        except Exception:
            return
        self.fail("second backward returned %s instead of reporting the "
                  "released graph" % (second.numpy(),))

    def test_retain_graph_allows_a_second_backward(self):
        weight, loss_a, loss_b = trunk_and_heads()
        first = jt.grad(loss_a, weight, retain_graph=True)
        second = jt.grad(loss_b, weight, retain_graph=True)
        np.testing.assert_allclose(first.numpy(), [6.0, 6.0], rtol=1e-6)
        np.testing.assert_allclose(second.numpy(), [10.0, 10.0], rtol=1e-6)

    def test_same_loss_twice_with_retain_graph_is_stable(self):
        weight, loss_a, _ = trunk_and_heads()
        a = jt.grad(loss_a, weight, retain_graph=True)
        b = jt.grad(loss_a, weight, retain_graph=True)
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6)

    def test_rebuilding_the_forward_graph_always_works(self):
        # The ordinary training loop: a fresh forward pass every iteration.
        # Releasing iteration N's graph must not affect iteration N+1.
        for _ in range(3):
            weight, loss_a, _ = trunk_and_heads()
            g = jt.grad(loss_a, weight, retain_graph=False)
            np.testing.assert_allclose(g.numpy(), [6.0, 6.0], rtol=1e-6)

    def test_torn_down_graph_is_reported_by_the_missing_grad_check(self):
        # Once a value has been fetched, the executor frees the finished
        # forward ops, so the second backward finds no path at all rather than
        # a stop_grad'ed one. That must be reported too, not silently zeroed.
        weight, loss_a, loss_b = trunk_and_heads()
        first = jt.grad(loss_a, weight, retain_graph=False)
        np.testing.assert_allclose(first.numpy(), [6.0, 6.0], rtol=1e-6)
        before = jt.flags.missing_grad_error
        try:
            jt.flags.missing_grad_error = 1
            with self.assertRaises(Exception) as caught:
                jt.grad(loss_b, weight, retain_graph=False)
            self.assertIn("doesn't have gradient", str(caught.exception),
                          str(caught.exception)[:2000])
        finally:
            jt.flags.missing_grad_error = before


if __name__ == "__main__":
    unittest.main()
