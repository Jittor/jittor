"""Direct numerical smoke tests for the public ``jittor.pool`` facade."""

import unittest

import numpy as np

import jittor as jt
from jittor import pool


def _blocks_1d(value, kernel, reduction):
    n, c, length = value.shape
    blocks = value.reshape(n, c, length // kernel, kernel)
    return getattr(blocks, reduction)(axis=3)


def _blocks_2d(value, kernel, reduction):
    n, c, height, width = value.shape
    blocks = value.reshape(
        n, c, height // kernel, kernel, width // kernel, kernel,
    )
    return getattr(blocks, reduction)(axis=(3, 5))


def _blocks_3d(value, kernel, reduction):
    n, c, depth, height, width = value.shape
    blocks = value.reshape(
        n, c, depth // kernel, kernel, height // kernel, kernel,
        width // kernel, kernel,
    )
    return getattr(blocks, reduction)(axis=(3, 5, 7))


class TestLegacyPoolDirect(unittest.TestCase):
    def assert_close(self, actual, expected):
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1e-5, atol=1e-5,
        )

    def test_average_pooling_direct_surface(self):
        one_dimensional = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
        expected_1d = _blocks_1d(one_dimensional, 2, "mean")
        self.assert_close(
            pool.AdaptiveAvgPool1d(4)(jt.array(one_dimensional)), expected_1d,
        )
        self.assert_close(pool.AvgPool1d(2)(jt.array(one_dimensional)), expected_1d)

        two_dimensional = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        expected_2d = _blocks_2d(two_dimensional, 2, "mean")
        self.assert_close(
            pool.AdaptiveAvgPool2d(2)(jt.array(two_dimensional)), expected_2d,
        )
        self.assert_close(pool.AvgPool2d(2)(jt.array(two_dimensional)), expected_2d)
        self.assert_close(
            pool.avg_pool2d(jt.array(two_dimensional), 2), expected_2d,
        )

        three_dimensional = np.arange(64, dtype=np.float32).reshape(
            1, 1, 4, 4, 4,
        )
        expected_3d = _blocks_3d(three_dimensional, 2, "mean")
        self.assert_close(
            pool.AdaptiveAvgPool3d(2)(jt.array(three_dimensional)), expected_3d,
        )
        self.assert_close(
            pool.AvgPool3d(2)(jt.array(three_dimensional)), expected_3d,
        )

    def test_maximum_and_minimum_pooling_direct_surface(self):
        one_dimensional = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
        expected_1d = _blocks_1d(one_dimensional, 2, "max")
        self.assert_close(pool.MaxPool1d(2)(jt.array(one_dimensional)), expected_1d)

        two_dimensional = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        expected_2d_max = _blocks_2d(two_dimensional, 2, "max")
        expected_2d_min = _blocks_2d(two_dimensional, 2, "min")
        self.assert_close(
            pool.AdaptiveMaxPool2d(2)(jt.array(two_dimensional)),
            expected_2d_max,
        )
        self.assert_close(
            pool.max_pool2d(jt.array(two_dimensional), 2), expected_2d_max,
        )
        self.assert_close(
            pool.pool(jt.array(two_dimensional), 2, "minimum"),
            expected_2d_min,
        )

        three_dimensional = np.arange(64, dtype=np.float32).reshape(
            1, 1, 4, 4, 4,
        )
        expected_3d = _blocks_3d(three_dimensional, 2, "max")
        self.assert_close(
            pool.AdaptiveMaxPool3d(2)(jt.array(three_dimensional)), expected_3d,
        )
        self.assert_close(
            pool.pool3d(jt.array(three_dimensional), 2, "maximum"), expected_3d,
        )
        self.assert_close(
            pool.max_pool3d(jt.array(three_dimensional), 2), expected_3d,
        )

    def test_unpooling_direct_surface(self):
        values_2d = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
        indices_2d = np.array([0, 3, 12, 15], dtype=np.int32).reshape(1, 1, 2, 2)
        expected_2d = np.zeros((1, 1, 4, 4), dtype=np.float32)
        expected_2d.reshape(-1)[indices_2d.reshape(-1)] = values_2d.reshape(-1)
        actual_2d = pool.MaxUnpool2d(2)(
            jt.array(values_2d), jt.array(indices_2d), output_size=(1, 1, 4, 4),
        )
        self.assert_close(actual_2d, expected_2d)

        values_3d = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 2, 2, 2)
        indices_3d = np.array(
            [21, 23, 29, 31, 53, 55, 61, 63], dtype=np.int32,
        ).reshape(1, 1, 2, 2, 2)
        expected_3d = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
        expected_3d.reshape(-1)[indices_3d.reshape(-1)] = values_3d.reshape(-1)
        actual_3d = pool.MaxUnpool3d(2)(
            jt.array(values_3d), jt.array(indices_3d),
            output_size=(1, 1, 4, 4, 4),
        )
        self.assert_close(actual_3d, expected_3d)


if __name__ == "__main__":
    unittest.main()
