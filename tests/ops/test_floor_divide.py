
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
