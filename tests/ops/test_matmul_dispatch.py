# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One capability predicate decides where a matrix product is sent.

There were four guards for the same question, spelled four different ways:

* ``matmul_transpose`` and ``_matmul_2d_cublas`` -- ``a_dtype == b_dtype and
  "float" in a_dtype and "complex" not in a_dtype and "complex" not in
  b_dtype``;
* ``matmul``'s batched branch -- the same, minus the test on ``b``;
* ``bmm_transpose`` -- ``jt.flags.use_cuda and cublas_ops``, and nothing about
  dtypes at all.

The last one is the reachable defect. ``jt.nn.bmm_transpose(a, b)`` and
``jt.nn.matmul(a, b.transpose(-1, -2))`` are the same product; with integer or
complex operands on CUDA the first hit the relay's C++ assertion while the
second computed the answer on the generic path.

``"float" in dtype`` really does match bfloat16 and float64 -- but cuBLAS takes
both, so that part of the audit note is not itself a bug. What the substring
cost is legibility: the two ``"complex" not in`` tests next to it can never
fire, because no complex dtype is spelled with "float" in it either.
"""

import unittest

import numpy as np

import jittor as jt
from jittor.nn.functional import matrix


def _pair(rng, dtype, batched=True):
    shape_a = (2, 3, 4) if batched else (3, 4)
    shape_b = (2, 5, 4) if batched else (5, 4)
    if dtype == "complex64":
        a = (rng.standard_normal(shape_a) + 1j * rng.standard_normal(shape_a))
        b = (rng.standard_normal(shape_b) + 1j * rng.standard_normal(shape_b))
        return a.astype("complex64"), b.astype("complex64")
    if dtype.startswith("int"):
        a = rng.integers(-4, 5, shape_a)
        b = rng.integers(-4, 5, shape_b)
        return a.astype(dtype), b.astype(dtype)
    return (rng.standard_normal(shape_a).astype(dtype),
            rng.standard_normal(shape_b).astype(dtype))


class _Dispatch:

    use_cuda = 0
    # float32 and float64 reach cuBLAS; int32 and complex64 must not, and used
    # to through bmm_transpose
    dtypes = ("float32", "float64", "int32", "complex64")

    def setUp(self):
        self.rng = np.random.default_rng(20260903)

    def test_bmm_transpose_is_matmul_of_the_transpose(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                raw_a, raw_b = _pair(self.rng, dtype)
                with jt.flag_scope(use_cuda=self.use_cuda):
                    a, b = (jt.array(raw_a, dtype=dtype),
                            jt.array(raw_b, dtype=dtype))
                    got = jt.nn.bmm_transpose(a, b).numpy()
                    want = jt.nn.matmul(
                        a, b.transpose((0, 2, 1))).numpy()
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(
                    got, np.matmul(raw_a, np.swapaxes(raw_b, -1, -2)),
                    rtol=1e-4, atol=1e-4)

    def test_matmul_transpose_matches_the_generic_path(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                raw_a, raw_b = _pair(self.rng, dtype, batched=False)
                with jt.flag_scope(use_cuda=self.use_cuda):
                    a, b = (jt.array(raw_a, dtype=dtype),
                            jt.array(raw_b, dtype=dtype))
                    got = jt.nn.matmul_transpose(a, b).numpy()
                np.testing.assert_allclose(
                    got, np.matmul(raw_a, np.swapaxes(raw_b, -1, -2)),
                    rtol=1e-4, atol=1e-4)

    def test_every_relay_is_optional(self):
        """Turn the whole cuBLAS row off; every site must fall back cleanly.

        This is the assertion that would catch a fifth spelling appearing: a
        call site with its own guard would keep taking the relay here.
        """
        saved = matrix._cublas_can_take
        matrix._cublas_can_take = lambda a, b: False
        try:
            for dtype in ("float32", "float64"):
                with self.subTest(dtype=dtype):
                    raw_a, raw_b = _pair(self.rng, dtype)
                    with jt.flag_scope(use_cuda=self.use_cuda):
                        a, b = (jt.array(raw_a, dtype=dtype),
                                jt.array(raw_b, dtype=dtype))
                        generic = (jt.nn.bmm_transpose(a, b).numpy(),
                                   jt.nn.matmul(a, b.transpose((0, 2, 1))).numpy(),
                                   jt.nn.matmul_transpose(a[0], b[0]).numpy())
                    matrix._cublas_can_take = saved
                    with jt.flag_scope(use_cuda=self.use_cuda):
                        a, b = (jt.array(raw_a, dtype=dtype),
                                jt.array(raw_b, dtype=dtype))
                        relayed = (jt.nn.bmm_transpose(a, b).numpy(),
                                   jt.nn.matmul(a, b.transpose((0, 2, 1))).numpy(),
                                   jt.nn.matmul_transpose(a[0], b[0]).numpy())
                    matrix._cublas_can_take = lambda a, b: False
                    for one, two in zip(generic, relayed):
                        np.testing.assert_allclose(one, two, rtol=1e-4,
                                                   atol=1e-4)
        finally:
            matrix._cublas_can_take = saved


class TestDispatchCPU(_Dispatch, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestDispatchCUDA(_Dispatch, unittest.TestCase):
    use_cuda = 1


class TestCapabilityPredicate(unittest.TestCase):
    """The predicate itself, without needing a device to run a product on."""

    def _var(self, dtype):
        # cast rather than jt.array(..., dtype=...): numpy has no bfloat16, so
        # the array constructor cannot make one
        return jt.array(np.zeros((2, 2, 2), dtype="float32")).cast(dtype)

    def test_same_width_is_not_same_dtype(self):
        """float16 and bfloat16 are both two bytes; that is not enough.

        The C++ relay asserts only that the widths agree and instantiates its
        kernel from ``a``'s dtype, so a pair that differs in dtype but agrees in
        width is exactly what the Python guard has to keep out.
        """
        half, bf16 = self._var("float16"), self._var("bfloat16")
        self.assertTrue(matrix._same_floating_dtype(half, half))
        self.assertFalse(matrix._same_floating_dtype(half, bf16))

    def test_non_float_dtypes_are_rejected_without_a_substring_test(self):
        for dtype in ("int32", "int64", "uint8", "bool", "complex64"):
            with self.subTest(dtype=dtype):
                v = self._var(dtype)
                self.assertFalse(matrix._same_floating_dtype(v, v))
                self.assertFalse(matrix._cublas_can_take(v, v))

    def test_float_dtypes_are_accepted(self):
        for dtype in ("float16", "bfloat16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                v = self._var(dtype)
                self.assertTrue(matrix._same_floating_dtype(v, v))


if __name__ == "__main__":
    unittest.main()
