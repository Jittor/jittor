# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A reduction over zero elements: which ones have an answer and which do not.

``add``, ``multiply`` and ``mean`` have one. 0, 1 and nan are what numpy and
torch return for an empty sum, product and mean, and jittor already agreed --
those cases are pinned here so a fix to the others cannot quietly take them too.

``maximum``, ``minimum``, ``argmax`` and ``argmin`` do not. There is no identity
element to return and no position to point at, so both references raise. jittor
returned the kernel's seed as if it were data::

    >>> jt.zeros((0, 3)).max(0)
    jt.Var([-3.4028235e+38 -3.4028235e+38 -3.4028235e+38], dtype=float32)
    >>> jt.zeros(0).argmax(0)
    (jt.Var(0, dtype=int32), jt.Var(0.0, dtype=float32))

-3.4e38 is ``std::numeric_limits<float>::lowest()``, and index 0 is a position
the input does not have. Both are finite, ordinary-looking values that flow on
into whatever comes next -- the failure mode this file exists to prevent.

Which dim is empty is what decides it, not whether the var is: reducing a
non-empty dim of an empty var is well defined (the result is simply empty), and
that case stays legal.

Reference behaviour, read from numpy 2.5 and a binary PyTorch 2.12 build:

=========================== ========================= ======================
call                        numpy                     torch
=========================== ========================= ======================
``sum`` of empty            ``0.0``                   ``tensor(0.)``
``prod`` of empty           ``1.0``                   ``tensor(1.)``
``mean`` of empty           ``nan`` (with a warning)  ``tensor(nan)``
``max`` of empty            ``ValueError``            ``RuntimeError``
``argmax`` of empty         ``ValueError``            ``IndexError``
=========================== ========================= ======================
"""

import unittest

import numpy as np

import jittor as jt


class TestReductionsWithAnIdentity(unittest.TestCase):
    """add/multiply/mean over zero elements keep their defined answers."""

    def test_sum_of_an_empty_var_is_zero(self):
        np.testing.assert_allclose(jt.zeros(0).sum().data, 0.0)
        np.testing.assert_allclose(jt.zeros((0, 3)).sum().data, 0.0)

    def test_sum_over_an_empty_dim_is_zero_for_each_survivor(self):
        np.testing.assert_allclose(jt.zeros((0, 3)).sum(0).data,
                                   np.zeros((0, 3)).sum(0))
        np.testing.assert_allclose(jt.zeros((3, 0)).sum(1).data,
                                   np.zeros((3, 0)).sum(1))

    def test_a_reduction_that_only_empties_the_output_stays_legal(self):
        # dim 1 has three elements; the result is empty because dim 0 is, which
        # is a shape fact and not a missing answer.
        self.assertEqual(list(jt.zeros((0, 3)).sum(1).shape), [0])
        self.assertEqual(list(jt.zeros((0, 3)).max(1).shape), [0])
        self.assertEqual(list(jt.zeros((0, 3)).argmax(1)[0].shape), [0])

    def test_prod_of_an_empty_var_is_one(self):
        np.testing.assert_allclose(jt.zeros(0).prod().data, 1.0)

    def test_mean_of_an_empty_var_is_nan(self):
        self.assertTrue(np.isnan(jt.zeros(0).mean().data))


class TestReductionsWithoutAnIdentity(unittest.TestCase):
    """max/min/argmax/argmin over zero elements report instead of inventing."""

    def test_max_of_an_empty_var_is_a_catchable_error(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.zeros(0).max().sync()
        text = str(caught.exception)
        self.assertIn("empty", text)
        self.assertIn("identity", text)

    def test_min_of_an_empty_var_is_a_catchable_error(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.zeros(0).min().sync()
        self.assertIn("empty", str(caught.exception))

    def test_max_over_an_empty_dim_names_the_dim(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.zeros((0, 3)).max(0).sync()
        text = str(caught.exception)
        self.assertIn("dim 0", text)
        self.assertIn("empty", text)

    def test_a_max_over_an_empty_dim_does_not_answer_with_lowest(self):
        # The regression itself: the old answer was finite, plausible and wrong.
        try:
            value = jt.zeros((0, 3)).max(0).data
        except RuntimeError:
            return
        self.fail("max over an empty dim returned %r instead of raising"
                  % (value,))

    def test_argmax_of_an_empty_var_is_a_catchable_error(self):
        with self.assertRaises(IndexError) as caught:
            jt.zeros(0).argmax(0)
        text = str(caught.exception)
        self.assertIn("argmax", text)
        self.assertIn("empty", text)

    def test_argmin_over_an_empty_dim_names_the_dim(self):
        with self.assertRaises(IndexError) as caught:
            jt.zeros((0, 3)).argmin(0)
        text = str(caught.exception)
        self.assertIn("argmin", text)
        self.assertIn("dim 0", text)

    def test_amax_reaches_the_same_check(self):
        with self.assertRaises(RuntimeError):
            jt.misc.amax(jt.zeros((0, 3)), 0).sync()


class TestNonEmptyReductionsAreUntouched(unittest.TestCase):

    def test_the_ordinary_answers_still_come_out(self):
        source = np.array([[1.0, 5.0], [3.0, 2.0]], dtype="float32")
        x = jt.array(source)
        np.testing.assert_allclose(x.max().data, source.max())
        np.testing.assert_allclose(x.min().data, source.min())
        np.testing.assert_allclose(x.max(0).data, source.max(0))
        np.testing.assert_allclose(x.min(1).data, source.min(1))
        np.testing.assert_allclose(x.argmax(1)[0].data, source.argmax(1))
        np.testing.assert_allclose(x.argmin(0)[0].data, source.argmin(0))


class TestBoolDivisionIsDefinedNotAnError(unittest.TestCase):
    """``bool / bool`` was on the list of "should this raise?"; it should not.

    numpy gives ``array([1., 0.])`` and torch ``tensor([1., 0.])``: true
    division lifts both operands out of bool, and dividing by ``False`` is the
    ordinary division-by-zero that yields ``inf``. jittor's values agree. The
    result *dtype* is a separate, deliberate native rule (a float wide enough
    for the operand width, so float16 here) which
    ``tests/type/test_type_system`` owns and the Torch shim overrides -- so this
    case is not a missing check.
    """

    def test_bool_true_division_computes_rather_than_raises(self):
        out = jt.array([True, False]) / jt.array([True, True])
        np.testing.assert_allclose(out.data, [1.0, 0.0])

    def test_dividing_by_false_is_an_infinity_not_an_exception(self):
        out = jt.array([True]) / jt.array([False])
        self.assertTrue(np.isinf(out.data).all())


if __name__ == "__main__":
    unittest.main()
