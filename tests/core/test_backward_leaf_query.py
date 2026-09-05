# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Whether a Var is a leaf of the backward graph has to be asked of the graph.

Two large tasks were both waiting on this one missing answer. ``7.11`` wants
``torch.Tensor.is_leaf`` and ``.grad_fn`` to mean something; ``7.12`` wants the
backward leaves to fall out of ``requires_grad`` plus graph connectivity
instead of three process-global dictionaries keyed on ``id()``. Neither can be
built on a guess, and until this landed the shipped answer *was* a guess::

    Var.is_leaf  = property(lambda self: True)     # compat/torch/installers
    Var.grad_fn  = property(lambda self: None)

Constant ``True`` and constant ``None`` are not merely imprecise: they are wrong
in the direction that reads as success. ``if param.is_leaf:`` guards pass for
every tensor in the program, so code that means "this is a parameter I own"
silently accepts an intermediate activation. The first two cases below are the
ones that refute those constants, and they are marked as such.

The rule (``jittor::backward_grad_fn`` in ``src/grad.h``) is the conjunction of
the two things a gradient needs -- ``requires_grad`` on the var, and an edge
into its producer that can carry a gradient -- using the same three edge filters
``grad()``'s own ``bfs_backward`` applies. So "not a leaf" means ``grad()``
really would walk through that op, and the two spellings (``is_backward_leaf``
and ``grad_fn_node_id``) are one query and cannot drift apart.

Where Jittor and torch differ, they differ on ``requires_grad``, not on this:
see ``tests/core/test_backward_leaf_torch_parity.py``.

Run::  python -m pytest tests/core/test_backward_leaf_query.py
"""

import unittest

import numpy as np
import jittor as jt



def float_var(n=4):
    return jt.array(np.arange(n, dtype="float32") + 1.0)


class LeafAssertions(unittest.TestCase):
    """All four spellings, every time: they are one query and must agree."""

    def assert_leaf(self, var):
        self.assertTrue(var.is_backward_leaf)
        self.assertEqual(var.grad_fn_node_id, -1)
        self.assertEqual(var.grad_fn_op_id, -1)
        self.assertEqual(var.grad_fn_name, "")

    def assert_not_leaf(self, var):
        self.assertFalse(var.is_backward_leaf)
        self.assertNotEqual(var.grad_fn_node_id, -1)
        self.assertGreater(var.grad_fn_op_id, 0)
        self.assertNotEqual(var.grad_fn_name, "")


class TestBackwardLeafQuery(LeafAssertions):
    def test_a_var_the_user_made_is_a_leaf(self):
        self.assert_leaf(float_var())

    def test_an_op_output_is_not_a_leaf(self):
        """The case constant ``is_leaf = True`` gets wrong."""
        self.assert_not_leaf(float_var() * 2)

    def test_the_producing_op_is_named(self):
        """The case constant ``grad_fn = None`` gets wrong."""
        x = float_var()
        y = x * 2
        self.assertEqual(y.grad_fn_name, "binary.multiply")
        self.assertEqual(y.sum().grad_fn_name, "reduce.add")

    def test_the_two_spellings_are_one_query(self):
        """torch's invariant: ``t.is_leaf == (t.grad_fn is None)``."""
        x = float_var()
        for var in (x, x * 2, (x * 2).sum(), x.detach(), x * x):
            self.assertEqual(
                var.is_backward_leaf, var.grad_fn_node_id == -1, var)
            self.assertEqual(
                var.is_backward_leaf, var.grad_fn_op_id == -1, var)
            self.assertEqual(
                var.is_backward_leaf, var.grad_fn_name == "", var)

    def test_grad_fn_identity_is_the_op_not_the_kind(self):
        x = float_var()
        first, second = x * 2, x * 3
        # Same kind of op, two instances: the kind id agrees, the identity does
        # not. A `grad_fn` object built on this can be compared for identity.
        self.assertEqual(first.grad_fn_op_id, second.grad_fn_op_id)
        self.assertNotEqual(first.grad_fn_node_id, second.grad_fn_node_id)
        self.assertNotEqual(first.grad_fn_op_id, first.sum().grad_fn_op_id)
        # Asking twice is asking the same question.
        self.assertEqual(first.grad_fn_node_id, first.grad_fn_node_id)

    def test_stop_grad_makes_a_leaf_of_a_non_leaf(self):
        x = float_var()
        y = x * 2
        self.assert_not_leaf(y)
        y.stop_grad()
        self.assert_leaf(y)

    def test_an_op_whose_every_input_is_stopped_produces_a_leaf(self):
        x = float_var()
        x.stop_grad()
        # Jittor's native policy leaves the output differentiable in its own
        # right (`requires_grad` stays True), so leaf-ness here can only come
        # from connectivity -- there is nothing upstream to send a gradient to.
        self.assertTrue((x * 2).requires_grad)
        self.assert_leaf(x * 2)

    def test_requires_grad_false_is_reversible_and_so_is_the_answer(self):
        x = float_var()
        y = x * 2
        self.assert_not_leaf(y)
        # Unlike stop_grad this does not release the backward graph, so putting
        # it back has to restore the same answer on the same Var.
        y.requires_grad = False
        self.assert_leaf(y)
        y.requires_grad = True
        self.assert_not_leaf(y)

    def test_an_op_downstream_of_a_disabled_input_produces_a_leaf(self):
        x = float_var()
        x.requires_grad = False
        y = x * 2
        self.assertFalse(y.requires_grad)
        self.assert_leaf(y)

    def test_detach_produces_a_leaf_although_the_var_is_not_stopped(self):
        x = float_var()
        detached = (x * 2).detach()
        # detach() stops the clone *op*, not the var (ops/clone_op.cc), so this
        # var still requires grad. Reading only the var's flags calls it a
        # non-leaf; only the edge into the producer says otherwise.
        self.assertTrue(detached.requires_grad)
        self.assertFalse(detached.is_stop_grad())
        self.assert_leaf(detached)

    def test_no_grad_output_is_a_leaf(self):
        x = float_var()
        with jt.no_grad():
            y = x * 2
        self.assertFalse(y.requires_grad)
        self.assert_leaf(y)

    def test_an_integer_var_is_a_leaf(self):
        self.assert_leaf(jt.array(np.arange(4, dtype="int32")))

    def test_in_place_on_an_intermediate_keeps_it_a_non_leaf(self):
        x = float_var()
        y = x * 2
        y[0] = 5.0
        self.assert_not_leaf(y)
        # The producer is now the in-place op, not the multiply.
        self.assertEqual(y.grad_fn_name, "setitem")

    def test_in_place_on_a_leaf_reports_the_graph_as_it_is(self):
        """Torch refuses this; Jittor allows it, so there is a graph to report.

        ``RuntimeError: a leaf Variable that requires grad is being used in an
        in-place operation`` is torch's answer, so this case is deliberately
        outside the parity table. What the query must not do is keep answering
        "leaf" for a Var whose producer is now a real differentiable op.
        """
        param = float_var()
        self.assert_leaf(param)
        param[0] = 5.0
        self.assert_not_leaf(param)


class TestBackwardLeafAfterBackward(LeafAssertions):
    def test_a_held_output_keeps_its_grad_fn_after_backward(self):
        x = float_var()
        y = x * 2
        loss = y.sum()
        jt.grad(loss, [x], retain_graph=False)
        # retain_graph=False freezes the vars nothing holds (grad.cc), and a
        # Python-visible Var is held -- so `y` keeps both its differentiability
        # and its grad_fn, which is what torch does too.
        self.assert_not_leaf(y)
        self.assertEqual(y.grad_fn_name, "binary.multiply")
        self.assert_leaf(x)


class TestBackwardLeafQueryCostAndTraversal(unittest.TestCase):
    """The mechanism, from the C++ side. See src/tests/test_backward_leaf.cc."""

    def test_the_rule_case_by_case(self):
        jt.tests.backward_leaf_query_rule()

    def test_control_dependencies_carry_no_gradient(self):
        jt.tests.backward_leaf_query_ignores_control_dependencies()

    def test_the_query_opens_no_traversal(self):
        # Asserted, not described: every graph walk takes a TraversalEpoch and
        # every epoch advances the runtime stamp, so an unchanged counter across 64
        # queries on a 64-deep chain proves the answer is not a walk. That is
        # the whole cost story -- there is no cache, so nothing to go stale.
        jt.tests.backward_leaf_query_opens_no_traversal()

    def test_the_query_leaves_a_running_traversal_alone(self):
        jt.tests.backward_leaf_query_inside_a_traversal()


class TestBackwardLeafQueryUnderANestedTraversal(unittest.TestCase):
    """The behavioural half: asking while the profiler walks the graph.

    ``MemoryProfiler::check()`` runs a full ``bfs_both`` from inside
    ``Executor::run_sync``'s op loop, and building a backward op re-enters
    ``run_sync``. An attribute read can land anywhere in that, so the answers
    must not depend on whether a traversal is in flight.
    """

    def setUp(self):
        self.previous = jt.flags.profile_memory_enable

    def tearDown(self):
        jt.flags.profile_memory_enable = self.previous

    def snapshot(self):
        x = float_var(64)
        y = x * 2
        loss = (y * y).sum()
        answers = [
            (v.is_backward_leaf, v.grad_fn_name) for v in (x, y, loss)]
        grad = jt.grad(loss, [x], retain_graph=True)[0]
        answers.append((grad.is_backward_leaf, grad.grad_fn_name))
        # Ask again after the backward graph was built on top of the same
        # forward vars.
        answers.extend(
            (v.is_backward_leaf, v.grad_fn_name) for v in (x, y, loss))
        return answers, grad.numpy()

    def test_the_answers_do_not_depend_on_a_traversal_being_in_flight(self):
        baseline, baseline_grad = self.snapshot()
        jt.flags.profile_memory_enable = 1
        try:
            profiled, profiled_grad = self.snapshot()
        finally:
            jt.flags.profile_memory_enable = 0
        self.assertEqual(baseline, profiled)
        np.testing.assert_allclose(baseline_grad, profiled_grad, rtol=1e-5)
        # Not just "the same both times": the answers have to be the right ones,
        # or a query that returned a constant would satisfy the comparison.
        leaves = [leaf for leaf, _ in baseline]
        self.assertEqual(leaves[:3], [True, False, False])
        self.assertEqual(leaves[4:], [True, False, False])


if __name__ == "__main__":
    unittest.main()
