
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


@_test_preserve_policy(jt, 'use_cuda')
class _FloorDivideMixin:
    use_cuda = 0

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self.use_cuda))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

    def test_signed_integer_floor_semantics(self):
        dividends = np.array([-7, -6, -5, -1, 0, 1, 5, 6, 7])
        divisors = np.array([3, 3, 3, 3, 3, -3, -3, -3, -3])
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            with self.subTest(dtype=dtype.__name__):
                x = dividends.astype(dtype)
                y = divisors.astype(dtype)
                actual = jt.floor_divide(
                    jt.array(x, dtype=x.dtype.name),
                    jt.array(y, dtype=y.dtype.name),
                ).numpy()
                np.testing.assert_array_equal(actual, np.floor_divide(x, y))

        unsigned_x = np.array([0, 1, 5, 6, 7], dtype=np.uint8)
        unsigned_y = np.array([3, 3, 3, 3, 3], dtype=np.uint8)
        unsigned_actual = jt.floor_divide(
            jt.array(unsigned_x, dtype="uint8"),
            jt.array(unsigned_y, dtype="uint8"),
        ).numpy()
        np.testing.assert_array_equal(
            unsigned_actual, np.floor_divide(unsigned_x, unsigned_y)
        )

    #: Float dividends chosen so the three plausible wrong answers all differ
    #: from the right one. Truncating the *operands* (the KI-OPS-003 defect)
    #: turns -2.7 into -2 and -0.5 into 0; truncating the quotient toward zero
    #: turns -1.35 into -1; flooring an integer quotient that already truncated
    #: is the same mistake one step later. -5.0 and 6.0 are exact multiples of
    #: some divisors below, where every wrong answer happens to agree with the
    #: right one -- they are here as the trap, not as the evidence.
    FLOAT_DIVIDENDS = np.array(
        [-5.0, -2.7, -0.5, 0.5, 1.5, 2.7, 6.0], dtype=np.float32)
    FLOAT_DIVISORS = (2.0, -2.0, 3.0, 0.75)

    def test_float_operands_divide_before_flooring(self):
        """KI-OPS-003: the fraction of the inputs must survive the division.

        Asserts the **dtype** as well as the values. Returning int32 satisfies
        a value-only check for every whole-number case and still breaks callers
        that feed the result into float arithmetic, which is the code this
        exists for -- and it is exactly what the operator used to do.
        """
        for divisor in self.FLOAT_DIVISORS:
            with self.subTest(divisor=divisor):
                got = jt.array(self.FLOAT_DIVIDENDS) // divisor
                np.testing.assert_array_equal(
                    got.numpy(),
                    np.floor_divide(self.FLOAT_DIVIDENDS,
                                    np.float32(divisor)))
                self.assertEqual(
                    str(got.dtype), "float32",
                    "float // float returned {0}; floor division is "
                    "integer-valued, not integer-typed".format(got.dtype))

    def test_float_tensor_divisor_and_mixed_dtypes(self):
        """The tensor//tensor spelling, and float//int, take the same path."""
        divisor = np.full_like(self.FLOAT_DIVIDENDS, -2.0)
        got = jt.array(self.FLOAT_DIVIDENDS) // jt.array(divisor)
        self.assertEqual(str(got.dtype), "float32")
        np.testing.assert_array_equal(
            got.numpy(), np.floor_divide(self.FLOAT_DIVIDENDS, divisor))

        integers = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
        mixed = jt.array(self.FLOAT_DIVIDENDS) // jt.array(integers)
        self.assertEqual(str(mixed.dtype), "float32")
        np.testing.assert_array_equal(
            mixed.numpy(), np.floor_divide(self.FLOAT_DIVIDENDS,
                                           integers.astype(np.float32)))

    def test_float64_operands_divide_at_double_precision(self):
        """A float64 pair must not be evaluated at single precision.

        `1e16 + 2` and `1e16 + 8` are the same float32 number, so a narrowing
        implementation returns the same quotient for both. They are two
        different float64 numbers whose floor-divisions by 3 differ.
        """
        x = np.array([1e16 + 2.0, 1e16 + 8.0], dtype=np.float64)
        y = np.array([3.0, 3.0], dtype=np.float64)
        want = np.floor_divide(x, y)
        self.assertEqual(x[0].astype(np.float32), x[1].astype(np.float32),
                         "the inputs no longer collapse in float32; pick "
                         "different ones or this proves nothing")
        self.assertNotEqual(want[0], want[1],
                            "the float64 answers must differ, otherwise a "
                            "narrowing implementation passes")
        got = jt.array(x, dtype="float64") // jt.array(y, dtype="float64")
        self.assertEqual(str(got.dtype), "float64")
        np.testing.assert_array_equal(got.numpy(), want)

    def test_float16_floors_rather_than_truncating(self):
        """The half table spelled this as a bare `/` and never floored."""
        x = np.array([-5.0, -2.7, -0.5, 2.7], dtype=np.float16)
        y = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float16)
        got = jt.array(x, dtype="float16") // jt.array(y, dtype="float16")
        self.assertEqual(str(got.dtype), "float16")
        np.testing.assert_array_equal(
            got.numpy().astype(np.float64),
            np.floor_divide(x, y).astype(np.float64))

    def test_operator_and_broadcast_semantics(self):
        x = np.array([[-5], [5]], dtype=np.int64)
        y = np.array([[3, -3]], dtype=np.int64)
        actual = (
            jt.array(x, dtype="int64") // jt.array(y, dtype="int64")
        ).numpy()
        np.testing.assert_array_equal(actual, np.floor_divide(x, y))


class TestFloorDivideCPU(_FloorDivideMixin, unittest.TestCase):
    pass


@unittest.skipUnless(
    _test_capability.check_accelerator('cuda', backend=jt).enabled and not _test_capability.check_accelerator('acl', backend=jt).enabled,
    "CUDA is unavailable",
)
class TestFloorDivideCUDA(_FloorDivideMixin, unittest.TestCase):
    use_cuda = 1


@unittest.skipUnless(_test_capability.check_accelerator('acl', backend=jt).enabled, "ACL is unavailable")
class TestFloorDivideNPU(_FloorDivideMixin, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
