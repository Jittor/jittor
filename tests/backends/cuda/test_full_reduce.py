"""Whole-Var ``sum``/``mean`` on CUDA go through the two-stage reduction.

The generated kernel has every thread atomically add into the single output
element, so its cost tracks the thread count rather than the data and its
accuracy suffers from accumulating a million terms into one float. The fast path
folds per block instead. It must produce the same answer, keep gradients, and
decline every case it does not cover.
"""
import unittest

import numpy as np

import jittor as jt

from jittor.nn.backends.full_reduce_cuda import _full_reduce_cuda


class TestFullReduceDispatch(unittest.TestCase):
    """Which calls the fast path accepts. Runs without a GPU."""

    def test_declines_without_cuda(self):
        with jt.flag_scope(use_cuda=0):
            x = jt.array(np.zeros(1 << 16, "float32"))
            self.assertIsNone(_full_reduce_cuda(x))

    @unittest.skipIf(not jt.has_cuda, "CUDA is required")
    def test_declines_non_float32_and_small_inputs(self):
        with jt.flag_scope(use_cuda=1):
            # ``jt.array`` narrows a float64 buffer to float32, so ask for the
            # wide dtype explicitly -- otherwise this asserts nothing.
            wide = jt.array(np.zeros(1 << 16, "float32")).cast("float64")
            self.assertEqual(str(wide.dtype), "float64")
            self.assertIsNone(_full_reduce_cuda(wide))
            self.assertIsNone(
                _full_reduce_cuda(jt.array(np.zeros(1 << 16, "int32"))))
            # Small enough that the generated kernel is not contended.
            self.assertIsNone(
                _full_reduce_cuda(jt.array(np.zeros(64, "float32"))))


@unittest.skipIf(not jt.has_cuda, "CUDA is required")
class TestFullReduce(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(20260828)

    def test_sum_and_mean_match_a_float64_reference(self):
        with jt.flag_scope(use_cuda=1):
            for shape in ((1 << 20,), (8, 197, 768), (2048, 4096)):
                reference = self.rng.randn(*shape).astype("float32")
                x = jt.array(reference)
                wide = reference.astype("float64")
                got_sum = float(x.sum().numpy().reshape(-1)[0])
                got_mean = float(x.mean().numpy().reshape(-1)[0])
                np.testing.assert_allclose(got_sum, wide.sum(), rtol=1e-5,
                                           err_msg="sum %s" % (shape,))
                np.testing.assert_allclose(got_mean, wide.mean(), rtol=1e-5,
                                           err_msg="mean %s" % (shape,))

    def test_result_is_reproducible(self):
        # Folding per block removes the atomics, so repeated runs agree bit for
        # bit rather than depending on the order the atomics landed.
        with jt.flag_scope(use_cuda=1):
            x = jt.array(self.rng.randn(1 << 20).astype("float32"))
            values = {float(x.sum().numpy().reshape(-1)[0]) for _ in range(5)}
            self.assertEqual(len(values), 1, values)

    def test_axis_reductions_and_keepdims_are_untouched(self):
        with jt.flag_scope(use_cuda=1):
            reference = self.rng.randn(64, 128).astype("float32")
            x = jt.array(reference)
            np.testing.assert_allclose(x.sum(dim=0).numpy(), reference.sum(0),
                                       atol=1e-3, rtol=1e-4)
            np.testing.assert_allclose(x.sum(dim=1).numpy(), reference.sum(1),
                                       atol=1e-3, rtol=1e-4)
            np.testing.assert_allclose(x.mean(dim=1).numpy(), reference.mean(1),
                                       atol=1e-5, rtol=1e-5)
            kept = x.sum(keepdims=True)
            self.assertEqual(list(kept.shape), [1, 1])
            np.testing.assert_allclose(float(kept.numpy().reshape(-1)[0]),
                                       reference.astype("float64").sum(),
                                       rtol=1e-4)

    def test_integer_sum_still_works(self):
        with jt.flag_scope(use_cuda=1):
            x = jt.array(np.arange(1 << 16, dtype="int32"))
            expected = int(np.arange(1 << 16, dtype="int64").sum())
            self.assertEqual(int(x.sum().numpy().reshape(-1)[0]), expected)

    def test_gradients(self):
        with jt.flag_scope(use_cuda=1):
            reference = self.rng.randn(512, 512).astype("float32")
            x = jt.array(reference)
            np.testing.assert_allclose(jt.grad(x.sum(), x).numpy(),
                                       np.ones_like(reference), atol=1e-6)
            np.testing.assert_allclose(jt.grad(x.mean(), x).numpy(),
                                       np.full_like(reference, 1.0 / x.numel()),
                                       rtol=1e-5)
            # Through a product, the shape a real loss has.
            np.testing.assert_allclose(jt.grad((x * x).sum(), x).numpy(),
                                       2 * reference, atol=1e-4)

    def test_cpu_results_are_unchanged(self):
        # A float32 sum is order-dependent, and the CPU reduction's order is not
        # fixed: hoisting the accumulator out of the output store let the compiler
        # vectorise the loop, so this now sums in 8 lanes rather than one at a
        # time. Both are valid; which lands closer to the float64 value is down to
        # the data. These 2^18 standard normals cancel down to ~42 out of ~209000
        # of total magnitude, so a fixed relative tolerance here measures that luck
        # rather than the implementation -- the plain serial order happens to beat
        # 8 lanes on this sample. Hold the reduction to the textbook error bound
        # for floating-point summation instead: |err| <= n * eps * sum|x|, using
        # log2(n) for n since a vectorised/blocked sum accumulates in that many
        # sequential steps, not in `size` of them.
        reference = self.rng.randn(1 << 18).astype("float32")
        exact = reference.astype("float64").sum()
        bound = (np.log2(reference.size) * np.finfo("float32").eps
                 * np.abs(reference).astype("float64").sum())
        with jt.flag_scope(use_cuda=0):
            x = jt.array(reference)
            got = float(x.sum().numpy().reshape(-1)[0])
        self.assertLessEqual(
            abs(got - exact), bound,
            "cpu sum %r is off the float64 value %r by more than a float32 "
            "summation can account for (bound %r)" % (got, exact, bound))


if __name__ == "__main__":
    unittest.main()
