# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``cumsum`` is one implementation with one derivative and one dim contract.

It used to be two, chosen by ``jt.flags.use_cuda``:

* CPU -- ``jt.numpy_code``, a host callback running ``np.cumsum`` per
  execution, with its own backward written as flip/cumsum/flip;
* CUDA -- ``cub_cumsum``, whose backward is ``CubCumsumOp::grad``, a reverse
  scan.

The two rules agree mathematically, which is why nothing caught that they were
two. What did not agree was the ``dim`` guard they shared:
``assert(dim >= -1 and dim < len(x.shape))`` accepts exactly one negative
value, so ``cumsum(x, -2)`` on a 3-D tensor raised an AssertionError while
``cumsum(x, 1)`` -- the same axis -- worked.

numpy is the oracle for the values; the gradient is checked against the closed
form (an inclusive reverse scan of the seed) and, at a few coordinates, against
a finite difference of the numpy forward.
"""

import unittest

import numpy as np

import jittor as jt
from jittor import misc


class _Cumsum:

    use_cuda = 0

    def setUp(self):
        self.rng = np.random.default_rng(20260903)

    def test_matches_numpy_over_shapes_dtypes_and_dims(self):
        cases = [((7,), (0, -1)),
                 ((3, 5), (0, 1, -1, -2)),
                 ((2, 3, 4), (0, 1, 2, -1, -2, -3))]
        for shape, dims in cases:
            for dim in dims:
                for dtype in ("float32", "float64", "int32", "int64"):
                    with self.subTest(shape=shape, dim=dim, dtype=dtype):
                        if dtype.startswith("int"):
                            raw = self.rng.integers(-5, 6, shape).astype(dtype)
                        else:
                            raw = self.rng.standard_normal(shape).astype(dtype)
                        with jt.flag_scope(use_cuda=self.use_cuda):
                            x = jt.array(raw, dtype=dtype)
                            got = jt.cumsum(x, dim).numpy()
                        np.testing.assert_allclose(
                            got, np.cumsum(raw, axis=dim), rtol=1e-5, atol=1e-5)

    def test_a_negative_dim_other_than_minus_one(self):
        """The one the old guard rejected outright."""
        raw = self.rng.standard_normal((2, 3, 4)).astype("float32")
        with jt.flag_scope(use_cuda=self.use_cuda):
            got = jt.cumsum(jt.array(raw), -2).numpy()
            same = jt.cumsum(jt.array(raw), 1).numpy()
        np.testing.assert_allclose(got, np.cumsum(raw, axis=-2),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(got, same)

    def test_an_out_of_range_dim_is_an_indexerror(self):
        with jt.flag_scope(use_cuda=self.use_cuda):
            x = jt.array(self.rng.standard_normal((2, 3)).astype("float32"))
            for dim in (2, -3, 99):
                with self.subTest(dim=dim):
                    with self.assertRaises(IndexError):
                        jt.cumsum(x, dim)

    def test_gradient_is_the_reverse_scan_of_the_seed(self):
        for shape, dim in (((6,), 0), ((3, 5), 1), ((2, 3, 4), -2)):
            with self.subTest(shape=shape, dim=dim):
                raw = self.rng.standard_normal(shape).astype("float32")
                seed = self.rng.standard_normal(shape).astype("float32")
                with jt.flag_scope(use_cuda=self.use_cuda):
                    x = jt.array(raw)
                    loss = (jt.cumsum(x, dim) * jt.array(seed)).sum()
                    got = jt.grad(loss, [x])[0].numpy()
                axis = dim % len(shape)
                expected = np.flip(np.cumsum(np.flip(seed.astype("float64"),
                                                     axis), axis=axis), axis)
                np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_gradient_against_a_finite_difference(self):
        raw = self.rng.standard_normal((4, 5)).astype("float64")
        seed = self.rng.standard_normal((4, 5)).astype("float64")

        def reference_loss(v):
            return float((np.cumsum(v, axis=1) * seed).sum())

        with jt.flag_scope(use_cuda=self.use_cuda):
            x = jt.array(raw, dtype="float64")
            loss = (jt.cumsum(x, 1) * jt.array(seed, dtype="float64")).sum()
            got = jt.grad(loss, [x])[0].numpy()
        eps = 1e-5
        for r, c in ((0, 0), (2, 3), (3, 4)):
            bump = np.zeros_like(raw)
            bump[r, c] = eps
            finite = (reference_loss(raw + bump)
                      - reference_loss(raw - bump)) / (2 * eps)
            self.assertAlmostEqual(got[r, c], finite, places=4)

    def test_empty_and_single_element(self):
        for raw in (np.zeros((0,), dtype="float32"),
                    np.array([2.5], dtype="float32"),
                    np.zeros((3, 0), dtype="float32")):
            with self.subTest(shape=raw.shape):
                with jt.flag_scope(use_cuda=self.use_cuda):
                    got = jt.cumsum(jt.array(raw), -1).numpy()
                    # An empty input used to reach the CUDA block scan with a
                    # zero-row batch, i.e. a zero-block launch. That fails
                    # asynchronously: reading this result back says nothing, and
                    # the error lands on the next test to touch the device.
                    # Drain here so a regression fails in the right place.
                    jt.sync_all()
                np.testing.assert_array_equal(got, np.cumsum(raw, axis=-1))


class TestCumsumCPU(_Cumsum, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCumsumCUDA(_Cumsum, unittest.TestCase):
    use_cuda = 1


class TestCumsumHasOneKernelEntryPoint(unittest.TestCase):

    def test_forward_and_backward_go_through_the_same_scan(self):
        """One entry point, and the derivative defined above it, not in it.

        Both directions reach ``_scan_2d``; the ``reverse`` flag is the only
        difference. If a second kernel or a second gradient rule ever appears,
        one of these counts stops matching.
        """
        calls = []
        real = misc._scan_2d

        def spy(x, reverse):
            calls.append(bool(reverse))
            return real(x, reverse)

        misc._scan_2d = spy
        try:
            with jt.flag_scope(use_cuda=0):
                x = jt.array(np.arange(6, dtype="float32").reshape(2, 3))
                loss = jt.cumsum(x, 1).sum()
                jt.grad(loss, [x])[0].sync()
        finally:
            misc._scan_2d = real
        self.assertEqual(calls, [False, True])

    def test_numpy_cumsum_is_still_available_as_an_independent_reference(self):
        """It is no longer what ``cumsum`` runs, but it is still correct.

        Keeping it costs nothing and gives ``tests/backends/cuda`` an oracle
        for the CUB kernel that shares no code with it.
        """
        raw = np.random.default_rng(7).standard_normal((3, 4)).astype("float32")
        with jt.flag_scope(use_cuda=0):
            np_side = jt.numpy_cumsum(jt.array(raw), -1).numpy()
            unified = jt.cumsum(jt.array(raw), -1).numpy()
        np.testing.assert_allclose(np_side, unified, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
