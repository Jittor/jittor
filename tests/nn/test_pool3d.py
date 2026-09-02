# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reference tests for ``Pool3d`` / ``MaxPool3d``.

The 3D pooling kernels in ``jittor/pool/core_3d.py`` are C++ sources built by
f-string interpolation, with four independent copies of the loop nest (CPU and
CUDA, forward and backward).  They are only exercised through
``jt.nn.MaxPool3d`` / ``jt.nn.AvgPool3d``, so this module pins each copy
against a plain numpy reference.

The index encoding checked here (``d*H*W + h*W + w`` into the *input* volume)
is the one ``torch.nn.functional.max_pool3d(..., return_indices=True)``
produces; it was verified once against a binary PyTorch build in a separate
process and is reproduced here by the numpy reference so the test needs no
torch at runtime.
"""

import unittest

import numpy as np

import jittor as jt


def _triple(v):
    return v if isinstance(v, tuple) else (v, v, v)


def _out_size(size, kernel, stride, padding, ceil_mode):
    if ceil_mode:
        return (size + 2 * padding - kernel + stride - 1) // stride + 1
    return (size + 2 * padding - kernel) // stride + 1


def _reference_maxpool3d(x, kernel, stride, padding, ceil_mode=False):
    """Windowed max plus the flat argmax index, exactly as the kernels define it."""
    kernel, stride, padding = _triple(kernel), _triple(stride), _triple(padding)
    n, c, depth, height, width = x.shape
    od = _out_size(depth, kernel[0], stride[0], padding[0], ceil_mode)
    oh = _out_size(height, kernel[1], stride[1], padding[1], ceil_mode)
    ow = _out_size(width, kernel[2], stride[2], padding[2], ceil_mode)
    values = np.zeros((n, c, od, oh, ow), dtype=x.dtype)
    indices = np.full((n, c, od, oh, ow), -1, dtype=np.int64)
    for bi in range(n):
        for ci in range(c):
            for i in range(od):
                lo0 = i * stride[0] - padding[0]
                hi0 = min(lo0 + kernel[0], depth)
                lo0 = max(lo0, 0)
                for j in range(oh):
                    lo1 = j * stride[1] - padding[1]
                    hi1 = min(lo1 + kernel[1], height)
                    lo1 = max(lo1, 0)
                    for k in range(ow):
                        lo2 = k * stride[2] - padding[2]
                        hi2 = min(lo2 + kernel[2], width)
                        lo2 = max(lo2, 0)
                        best = -np.inf
                        best_index = -1
                        for p in range(lo0, hi0):
                            for q in range(lo1, hi1):
                                for r in range(lo2, hi2):
                                    v = x[bi, ci, p, q, r]
                                    if best < v:
                                        best = v
                                        best_index = (
                                            p * height * width + q * width + r
                                        )
                        values[bi, ci, i, j, k] = best
                        indices[bi, ci, i, j, k] = best_index
    return values, indices


class TestMaxPool3dIndices(unittest.TestCase):
    """``return_indices=True`` used to hang: the innermost loop tested ``q``."""

    def _check(self, shape, kernel, stride, padding=0, ceil_mode=False):
        rng = np.random.default_rng(20240902)
        # Distinct values keep the argmax unambiguous, so the index comparison
        # is exact rather than tie-break dependent.
        x = rng.permutation(int(np.prod(shape))).astype("float32")
        x = (x / x.size).reshape(shape)
        expected, expected_index = _reference_maxpool3d(
            x, kernel, stride, padding, ceil_mode
        )
        layer = jt.nn.MaxPool3d(
            kernel, stride=stride, padding=padding, return_indices=True,
            ceil_mode=ceil_mode,
        )
        value, index = layer(jt.array(x))
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(index.numpy().astype(np.int64), expected_index)
        # An index of -1 means the window was empty, which cannot happen here.
        self.assertTrue((index.numpy() >= 0).all())

    def test_non_overlapping(self):
        self._check((2, 2, 6, 6, 6), 2, 2)

    def test_asymmetric_kernel_and_stride(self):
        self._check((2, 2, 5, 6, 7), (2, 3, 2), (2, 2, 3))

    def test_overlapping_windows(self):
        self._check((1, 3, 7, 7, 7), 3, 2)

    def test_padded(self):
        self._check((1, 2, 6, 6, 7), 3, 2, padding=1)

    def test_ceil_mode(self):
        self._check((1, 2, 7, 7, 5), 2, 2, ceil_mode=True)

    def test_values_match_plain_pool(self):
        """``return_indices`` must not change the pooled values themselves."""
        rng = np.random.default_rng(5)
        x = jt.array(rng.standard_normal((2, 3, 6, 7, 8)).astype("float32"))
        plain = jt.nn.MaxPool3d(3, stride=2)(x).numpy()
        value, _ = jt.nn.MaxPool3d(3, stride=2, return_indices=True)(x)
        np.testing.assert_allclose(value.numpy(), plain, rtol=1e-6, atol=1e-6)


class TestMaxPool3dIndicesCuda(TestMaxPool3dIndices):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
