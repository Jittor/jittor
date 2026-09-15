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

from _helpers import capability as _test_capability

import unittest
from unittest import mock

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import override_kernel
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
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                raw_a, raw_b = _pair(self.rng, dtype)
                with jt.flag_scope(use_cuda=self.use_cuda), \
                        override_kernel("matmul", "cuda", matrix._cublas_matmul,
                                        supports=lambda *args: False), \
                        override_kernel("batched_matmul", "cuda", matrix._cublas_batched_matmul,
                                        supports=lambda *args: False):
                    a, b = (jt.array(raw_a, dtype=dtype), jt.array(raw_b, dtype=dtype))
                    self.assertFalse(matrix._cublas_can_take(a, b))
                    generic = (jt.nn.bmm_transpose(a, b).numpy(),
                               jt.nn.matmul(a, b.transpose((0, 2, 1))).numpy(),
                               jt.nn.matmul_transpose(a[0], b[0]).numpy())
                with jt.flag_scope(use_cuda=self.use_cuda):
                    a, b = (jt.array(raw_a, dtype=dtype), jt.array(raw_b, dtype=dtype))
                    relayed = (jt.nn.bmm_transpose(a, b).numpy(),
                               jt.nn.matmul(a, b.transpose((0, 2, 1))).numpy(),
                               jt.nn.matmul_transpose(a[0], b[0]).numpy())
                for one, two in zip(generic, relayed):
                    np.testing.assert_allclose(one, two, rtol=1e-4, atol=1e-4)


class TestDispatchCPU(_Dispatch, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
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


class TestMixedFloatPair(unittest.TestCase):
    """A float32 activation against a float16 weight is a real product.

    cuBLAS takes both operands in one dtype, and the relay rows are selected on
    the operands' own dtypes (``_same_floating_dtype``), so a pair of two
    different floating dtypes matched no kernel. ``matmul``'s 2-D branch and
    ``matmul_transpose`` then fell through to their generic form --

        (a.broadcast(shape) * b.broadcast(shape)).sum(-1)

    -- which *materializes* ``[B, out, in]``. That is what ``torch.autocast``
    produces for every ``nn.Linear`` whose weights were pre-cast to float16:
    vLLM-Omni's MiniMax-H3 video VAE does exactly that to its decoder blocks,
    and the activation stays float32 because the cast is autocast's job.
    Measured at 512x2048x6144 the fallback cost 53.6 ms against cuBLAS's
    0.5 ms, and at the VAE's real 1797 rows the temporary is 86 GiB, which the
    decode died on with an accelerator out-of-memory.

    These cases pin the two halves of the fix: the pair is resolved to one
    dtype and retried on the relay, and the answer is still right.
    """

    def _pair(self, dtype_a, dtype_b, batched=False):
        # One draw per operand: a fresh `standard_normal` for the expectation
        # would advance the same generator and compare against other numbers.
        rng = np.random.default_rng(20260915)
        shape_a = (2, 3, 5) if batched else (3, 5)
        shape_b = (2, 7, 5) if batched else (7, 5)
        raw_a = rng.standard_normal(shape_a).astype("float32")
        raw_b = rng.standard_normal(shape_b).astype("float32")
        a = jt.array(raw_a).cast(dtype_a)
        b = jt.array(raw_b).cast(dtype_b)
        return a, b, raw_a, raw_b

    def test_the_mixed_pair_is_retried_in_a_single_dtype(self):
        """Without the retry the second selection never happens and the pair
        takes the outer-product fallback instead of the relay."""
        a, b, _raw_a, _raw_b = self._pair("float32", "float16")
        seen = []
        real_select = matrix.select_kernel

        def spy(op, x, y, *rest, **kwargs):
            seen.append((op, str(x.dtype).split(".")[-1], str(y.dtype).split(".")[-1]))
            return real_select(op, x, y, *rest, **kwargs)

        with mock.patch.object(matrix, "select_kernel", spy):
            out = jt.nn.matmul_transpose(a, b)
            out.numpy()

        self.assertEqual(seen[0], ("matmul", "float32", "float16"))
        self.assertEqual(seen[-1][0], "matmul")
        # whatever it resolved to, both operands went in together
        self.assertEqual(seen[-1][1], seen[-1][2])
        self.assertEqual(seen[-1][1], str(out.dtype).split(".")[-1])

    def test_the_resolved_dtype_follows_the_amp_region(self):
        a, b, _raw_a, _raw_b = self._pair("float32", "float16")
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
            self.assertEqual(matrix._mixed_float_compute_dtype(a, b), "float16")
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer32):
            self.assertEqual(matrix._mixed_float_compute_dtype(a, b), "float32")
        # No region: promote, widest first.
        with jt.flag_scope(amp_reg=0):
            self.assertEqual(matrix._mixed_float_compute_dtype(a, b), "float32")
            self.assertEqual(matrix._mixed_float_compute_dtype(
                a.cast("float64"), b), "float64")
            self.assertEqual(matrix._mixed_float_compute_dtype(
                a.cast("bfloat16"), b), "float32")  # float16 x bfloat16

    def test_a_matching_pair_is_left_alone(self):
        """The resolution must not touch the routes that already worked."""
        a, _b, _raw_a, _raw_b = self._pair("float32", "float32")
        self.assertIsNone(matrix._mixed_float_compute_dtype(a, a))
        for dtype in ("int32", "bool"):
            v = jt.array(np.zeros((3, 5), dtype="float32")).cast(dtype)
            self.assertIsNone(matrix._mixed_float_compute_dtype(a, v))
            self.assertIsNone(matrix._mixed_float_compute_dtype(v, a))

    def test_mixed_pairs_match_numpy(self):
        """Compute the same product with the operands explicitly cast.

        The expectation is built from the operands *as held* -- casting a
        reduced-precision operand up to float32 is exact, while comparing
        against the original float32 draw would only measure how far the
        operand's own rounding moved. The tolerance therefore tracks the
        narrowest operand, which is what the product loses wherever no relay is
        registered for it (the CPU generic path computes in the operands' own
        precision; on CUDA the pair is resolved and the relay does the work).
        """
        for dtype_a, dtype_b in (("float32", "float16"), ("bfloat16", "float16"),
                                 ("float32", "bfloat16"), ("float64", "float32"),
                                 ("bfloat16", "float32")):
            narrower = [d for d in (dtype_a, dtype_b) if d != "float32"]
            tol = {"bfloat16": 2e-2, "float16": 5e-3}.get(
                narrower[0] if narrower else "", 1e-6)
            for batched in (False, True):
                with self.subTest(dtype_a=dtype_a, dtype_b=dtype_b,
                                  batched=batched):
                    a, b, _raw_a, _raw_b = self._pair(dtype_a, dtype_b,
                                                      batched=batched)
                    want = np.matmul(
                        a.cast("float32").numpy().astype("float64"),
                        np.swapaxes(b.cast("float32").numpy().astype("float64"),
                                    -1, -2))
                    got = jt.nn.matmul_transpose(a, b).numpy()
                    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)
                    if batched:
                        got_mm = jt.nn.matmul(a, b.transpose((0, 2, 1))).numpy()
                        np.testing.assert_allclose(got_mm, want, rtol=tol,
                                                   atol=tol)

    def test_a_linear_with_a_pre_cast_half_weight(self):
        """The shape the H3 decoder actually asks for."""
        rng = np.random.default_rng(7)
        x = jt.array(rng.standard_normal((1, 9, 8)).astype("float32"))
        w = jt.array(rng.standard_normal((12, 8)).astype("float32")).cast("float16")
        out = jt.nn.linear(x, w)
        self.assertEqual(tuple(out.shape), (1, 9, 12))
        np.testing.assert_allclose(
            out.numpy(), np.matmul(x.numpy(), w.numpy().astype("float32").T),
            rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
