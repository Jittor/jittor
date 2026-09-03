# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The runtime's installers ran, once each, in the declared order.

Task 5.21. Importing jittor rewrites ``Var`` and the root namespace through a
sequence of monkeypatch installers -- 169 ``Var.x = ...`` assignments across
eight files -- and which assignment wins is decided by the order the steps run
in. That order was load-bearing and undeclared; it lived in the physical
arrangement of statements in ``jittor/__init__.py``.

``jittor/_install_order.py`` declares it. These tests check the declaration is
enforced rather than decorative, and that the one place where two spellings of
the same reduction disagreed now agrees.

Run::  python -m pytest tests/core/test_install_order.py
"""

import unittest

import numpy as np

import jittor as jt
from jittor import _install_order


class TestTheSequenceRan(unittest.TestCase):
    def test_import_recorded_every_required_step(self):
        observed = _install_order.observed()
        required = [step.name for step in _install_order.SEQUENCE
                    if step.required]
        for name in required:
            self.assertIn(name, observed,
                          "the runtime imported without running %r" % (name,))

    def test_the_observed_order_follows_the_declaration(self):
        declared = [step.name for step in _install_order.SEQUENCE]
        positions = [declared.index(name)
                     for name in _install_order.observed()]
        self.assertEqual(positions, sorted(positions),
                         "installers ran out of the declared order")

    def test_no_step_ran_twice(self):
        observed = _install_order.observed()
        self.assertEqual(len(observed), len(set(observed)))

    def test_verify_is_idempotent_and_returns_what_ran(self):
        self.assertEqual(_install_order.verify(), _install_order.observed())


class TestTheDeclarationIsEnforced(unittest.TestCase):
    """``record`` must reject the three ways an order goes wrong."""

    def setUp(self):
        self.saved = _install_order.observed()

    def tearDown(self):
        _install_order.reset_for_testing()
        for name in self.saved:
            _install_order.record(name)

    def test_an_unknown_step_is_rejected(self):
        with self.assertRaises(_install_order.InstallOrderError) as caught:
            _install_order.record("no.such.installer")
        self.assertIn("unknown installer", str(caught.exception))

    def test_running_a_step_twice_is_rejected(self):
        with self.assertRaises(_install_order.InstallOrderError) as caught:
            _install_order.record(_install_order.SEQUENCE[0].name)
        self.assertIn("twice", str(caught.exception))

    def test_running_a_step_out_of_order_is_rejected(self):
        _install_order.reset_for_testing()
        last = _install_order.SEQUENCE[-1].name
        first = _install_order.SEQUENCE[0].name
        _install_order.record(last)
        with self.assertRaises(_install_order.InstallOrderError) as caught:
            _install_order.record(first)
        message = str(caught.exception)
        self.assertIn(first, message)
        self.assertIn(last, message)

    def test_a_missing_required_step_fails_verify(self):
        _install_order.reset_for_testing()
        with self.assertRaises(_install_order.InstallOrderError) as caught:
            _install_order.verify()
        self.assertIn("half-patched", str(caught.exception))

    def test_the_nccl_route_is_declared_after_the_hccl_one(self):
        # Its guard is `not hasattr(Var, "mpi_all_reduce")`, so the reverse
        # order would silently disable HCCL on an Ascend box.
        names = [step.name for step in _install_order.SEQUENCE]
        self.assertLess(names.index("collectives.hccl"),
                        names.index("collectives.nccl"))

    def test_compat_composition_is_declared_before_the_optim_refresh(self):
        names = [step.name for step in _install_order.SEQUENCE]
        self.assertLess(names.index("compat.runtime_composition"),
                        names.index("optim.public_exports"))

    def test_the_fast_path_is_declared_before_the_inplace_aliases(self):
        # All Var method providers settle before the explicit in-place
        # allowlist is installed; no later module may invent another alias.
        names = [step.name for step in _install_order.SEQUENCE]
        self.assertLess(names.index("nn.full_reduce_fast_path"),
                        names.index("root.inplace_aliases"))


class TestOneReductionOneNumeric(unittest.TestCase):
    """``jt.sum(x)`` and ``x.sum()`` are the same operation; make them agree."""

    def test_both_spellings_are_routed_through_the_fast_path(self):
        for entry in (jt.Var.sum, jt.Var.mean, jt.sum, jt.mean):
            self.assertTrue(
                hasattr(entry, "_full_reduce_native"),
                "%r is not routed: it keeps the generated atomicAdd kernel "
                "while the other spelling gets the CUB fold" % (entry,))

    def test_the_two_spellings_agree_on_cpu(self):
        value = jt.array(np.random.RandomState(0).randn(1 << 15)
                         .astype("float32"))
        np.testing.assert_array_equal(jt.sum(value).numpy(),
                                      value.sum().numpy())
        np.testing.assert_array_equal(jt.mean(value).numpy(),
                                      value.mean().numpy())

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_the_two_spellings_agree_bit_for_bit_on_cuda(self):
        # Large enough to take the fast path (>= 1<<14 elements). Before this
        # task jt.sum went to a quarter-million atomicAdds and x.sum() to the
        # CUB fold, so the two differed in the last ulps -- and jt.sum differed
        # from itself between runs.
        with jt.flag_scope(use_cuda=1):
            value = jt.array(np.random.RandomState(1).randn(1 << 20)
                             .astype("float32"))
            np.testing.assert_array_equal(jt.sum(value).numpy(),
                                          value.sum().numpy())
            np.testing.assert_array_equal(jt.mean(value).numpy(),
                                          value.mean().numpy())

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_the_root_spelling_is_reproducible_on_cuda(self):
        with jt.flag_scope(use_cuda=1):
            value = jt.array(np.random.RandomState(2).randn(1 << 20)
                             .astype("float32"))
            first = jt.sum(value).numpy()
            for _ in range(4):
                np.testing.assert_array_equal(jt.sum(value).numpy(), first)

    def test_an_axis_argument_still_reaches_the_general_reduce(self):
        value = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
        np.testing.assert_allclose(jt.sum(value, 0).numpy(),
                                   np.arange(12).reshape(3, 4).sum(0))
        np.testing.assert_allclose(value.sum(1).numpy(),
                                   np.arange(12).reshape(3, 4).sum(1))

    def test_a_gradient_still_flows_through_both_spellings(self):
        for reduce in (jt.sum, lambda v: v.sum()):
            value = jt.array(np.ones(1 << 15, dtype="float32"))
            grad = jt.grad(reduce(value), [value])[0]
            np.testing.assert_allclose(grad.numpy(),
                                       np.ones(1 << 15, dtype="float32"))


if __name__ == "__main__":
    unittest.main()
