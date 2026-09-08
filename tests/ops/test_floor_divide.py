import unittest

import numpy as np

import jittor as jt


class _FloorDivideMixin:
    use_cuda = 0

    def setUp(self):
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = self.use_cuda

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._previous_use_cuda

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
    jt.compiler.has_cuda and not getattr(jt.compiler, "has_acl", 0),
    "CUDA is unavailable",
)
class TestFloorDivideCUDA(_FloorDivideMixin, unittest.TestCase):
    use_cuda = 1


@unittest.skipUnless(getattr(jt.compiler, "has_acl", 0), "ACL is unavailable")
class TestFloorDivideNPU(_FloorDivideMixin, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
