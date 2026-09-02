# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``MaxUnpool2d`` / ``MaxUnpool3d`` index decoding.

``MaxPool*d(return_indices=True)`` encodes the argmax as a flat offset into the
*original* input volume (``d*H*W + h*W + w``).  Unpooling therefore has to
rebuild a volume with exactly those extents, otherwise the decode uses the
wrong row width and values land in the wrong place -- or, when the decoded
offset falls outside the rebuilt volume, are dropped without any error, because
``reindex_reduce`` treats an out-of-range destination as an overflow.

The expected arrays below were cross-checked once against a binary PyTorch
2.12 build running in a separate process; the tests themselves only need numpy.
"""

import unittest

import numpy as np

import jittor as jt


def _scatter_add(values, indices, out_shape):
    """Jittor's unpool semantics: scatter-add into a zero volume."""
    b, c = values.shape[:2]
    flat = np.zeros((b, c, int(np.prod(out_shape))), dtype=np.float64)
    vi = indices.reshape(b, c, -1)
    vv = values.reshape(b, c, -1)
    for bi in range(b):
        for ci in range(c):
            for index, value in zip(vi[bi, ci], vv[bi, ci]):
                flat[bi, ci, index] += value
    return flat.reshape((b, c) + tuple(out_shape))


class TestMaxUnpool2d(unittest.TestCase):
    def test_default_output_size_with_stride_equal_kernel(self):
        pool = jt.nn.MaxPool2d(2, stride=2, return_indices=True)
        unpool = jt.nn.MaxUnpool2d(2, stride=2)
        x = jt.array(np.arange(1.0, 17.0, dtype="float32").reshape(1, 1, 4, 4))
        value, index = pool(x)
        out = unpool(value, index)
        self.assertEqual(tuple(out.shape), (1, 1, 4, 4))
        expected = np.zeros((4, 4), dtype="float32")
        expected[1, 1] = 6; expected[1, 3] = 8
        expected[3, 1] = 14; expected[3, 3] = 16
        np.testing.assert_allclose(out.numpy().reshape(4, 4), expected)

    def test_default_output_size_with_stride_below_kernel(self):
        """Regression: the default used to be ``pooled * stride``.

        For a 5x5 input pooled with kernel 3 / stride 2 that gives 4x4, so the
        indices (encoded against a width of 5) decoded against a width of 4:
        13 and 15 moved to the last row and 23, 25 fell off the volume
        entirely.  torch's default is ``(pooled - 1) * stride + kernel``.
        """
        x = np.arange(1.0, 26.0, dtype="float32").reshape(1, 1, 5, 5)
        value, index = jt.nn.MaxPool2d(3, stride=2, return_indices=True)(jt.array(x))
        np.testing.assert_array_equal(index.numpy().reshape(-1), [12, 14, 22, 24])
        out = jt.nn.MaxUnpool2d(3, stride=2)(value, index)
        self.assertEqual(tuple(out.shape), (1, 1, 5, 5))
        expected = np.zeros((5, 5), dtype="float32")
        expected[2, 2] = 13; expected[2, 4] = 15
        expected[4, 2] = 23; expected[4, 4] = 25
        np.testing.assert_allclose(out.numpy().reshape(5, 5), expected)
        # Nothing may be lost: every pooled value has a home.
        self.assertAlmostEqual(float(out.numpy().sum()), float(value.numpy().sum()), places=4)

    def test_explicit_output_size_matches_default(self):
        x = np.arange(1.0, 26.0, dtype="float32").reshape(1, 1, 5, 5)
        value, index = jt.nn.MaxPool2d(3, stride=2, return_indices=True)(jt.array(x))
        unpool = jt.nn.MaxUnpool2d(3, stride=2)
        np.testing.assert_allclose(
            unpool(value, index).numpy(),
            unpool(value, index, output_size=(1, 1, 5, 5)).numpy(),
        )

    def test_decodes_with_the_original_row_width(self):
        """A non-square volume catches a width/height mix-up."""
        rng = np.random.default_rng(2)
        x = rng.permutation(1 * 2 * 6 * 9).astype("float32").reshape(1, 2, 6, 9)
        value, index = jt.nn.MaxPool2d(3, stride=2, return_indices=True)(jt.array(x))
        out = jt.nn.MaxUnpool2d(3, stride=2)(value, index, output_size=(6, 9))
        expected = _scatter_add(value.numpy(), index.numpy().astype(np.int64), (6, 9))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_repeated_indices_accumulate(self):
        """Overlapping windows can share an argmax; jittor sums the copies.

        torch overwrites instead.  The behaviour is pinned here so the
        difference stays a deliberate choice rather than a silent regression.
        """
        x = jt.array(np.array([[[[1., 2, 3, 4, 0],
                                 [5, 6, 7, 8, 0],
                                 [9, 10, 11, 12, 0],
                                 [13, 14, 16, 15, 0],
                                 [0, 0, 0, 0, 0]]]], dtype="float32"))
        value, index = jt.nn.MaxPool2d(3, stride=2, return_indices=True)(x)
        np.testing.assert_array_equal(index.numpy().reshape(-1), [12, 13, 17, 17])
        out = jt.nn.MaxUnpool2d(3, stride=2)(value, index, output_size=(1, 1, 5, 5))
        got = out.numpy().reshape(5, 5)
        self.assertEqual(got[3, 2], 32.0)  # 16 + 16, not 16
        self.assertEqual(got[2, 2], 11.0)
        self.assertEqual(got[2, 3], 12.0)


class TestMaxUnpool3d(unittest.TestCase):
    def test_default_output_size_with_stride_equal_kernel(self):
        rng = np.random.default_rng(3)
        x = rng.permutation(1 * 1 * 4 * 4 * 4).astype("float32").reshape(1, 1, 4, 4, 4)
        value, index = jt.nn.MaxPool3d(2, stride=2, return_indices=True)(jt.array(x))
        out = jt.nn.MaxUnpool3d(2, stride=2)(value, index)
        self.assertEqual(tuple(out.shape), (1, 1, 4, 4, 4))
        expected = _scatter_add(value.numpy(), index.numpy().astype(np.int64), (4, 4, 4))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_default_output_size_with_stride_below_kernel(self):
        """Regression: ``pooled * stride`` dropped half of the values here."""
        x = np.arange(1.0, 1.0 + 2 * 5 * 5, dtype="float32").reshape(1, 1, 2, 5, 5)
        value, index = jt.nn.MaxPool3d(
            (1, 3, 3), stride=(1, 2, 2), return_indices=True)(jt.array(x))
        out = jt.nn.MaxUnpool3d((1, 3, 3), stride=(1, 2, 2))(value, index)
        self.assertEqual(tuple(out.shape), (1, 1, 2, 5, 5))
        expected = _scatter_add(value.numpy(), index.numpy().astype(np.int64), (2, 5, 5))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(
            float(out.numpy().sum()), float(value.numpy().sum()), places=3)

    def test_decodes_with_the_original_extents(self):
        """Distinct D/H/W catches a transposed decode."""
        rng = np.random.default_rng(4)
        x = rng.permutation(1 * 2 * 4 * 5 * 6).astype("float32").reshape(1, 2, 4, 5, 6)
        value, index = jt.nn.MaxPool3d(
            (2, 3, 3), stride=(1, 2, 2), return_indices=True)(jt.array(x))
        out = jt.nn.MaxUnpool3d(
            (2, 3, 3), stride=(1, 2, 2))(value, index, output_size=(4, 5, 6))
        expected = _scatter_add(value.numpy(), index.numpy().astype(np.int64), (4, 5, 6))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)


class TestMaxUnpool2dCuda(TestMaxUnpool2d):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


class TestMaxUnpool3dCuda(TestMaxUnpool3d):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
