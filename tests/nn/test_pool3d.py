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


def _reference_maxpool3d_grad(x, dout, kernel, stride, padding, ceil_mode=False):
    """Route each output gradient to the first argmax of its window."""
    _, indices = _reference_maxpool3d(x, kernel, stride, padding, ceil_mode)
    n, c, depth, height, width = x.shape
    grad = np.zeros((n, c, depth * height * width), dtype=np.float64)
    flat_index = indices.reshape(n, c, -1)
    flat_dout = dout.reshape(n, c, -1)
    for bi in range(n):
        for ci in range(c):
            for i, g in zip(flat_index[bi, ci], flat_dout[bi, ci]):
                grad[bi, ci, i] += g
    return grad.reshape(x.shape)


def _pool3d_grad(x, kernel, stride, padding, op, ceil_mode, use_cuda, seed):
    jt.flags.use_cuda = use_cuda
    try:
        xv = jt.array(x)
        y = jt.nn.Pool3d(
            kernel, stride=stride, padding=padding, op=op, ceil_mode=ceil_mode
        )(xv)
        assert tuple(y.shape) == seed.shape, (tuple(y.shape), seed.shape)
        grad, = jt.grad((y * jt.array(seed)).sum(), [xv])
        return grad.numpy().copy(), y.numpy().copy()
    finally:
        jt.flags.use_cuda = 0


class TestPool3dBackward(unittest.TestCase):
    """The CUDA backward kernel looped to ``out_shape`` (the *input* extent).

    ``out`` in a ``cuda_grad_src`` is the gradient w.r.t. the input, so its
    shape is the input's; the loop must be bounded by ``pout_shape`` (the
    forward output), which is what the launch configuration, the CPU backward
    and the 2D kernels all use.  Looping to ``out_shape`` reads ``pout``/``dout``
    out of bounds and accumulates gradient that does not exist.
    """

    #: shapes small enough for the python reference, big enough that the input
    #: extent differs from the pooled extent in every dimension.
    CASES = (
        (2, 2, 0),
        ((2, 3, 2), (2, 2, 3), 0),
        (3, 2, 0),
        (3, 2, 1),
    )

    @staticmethod
    def _inputs(shape, distinct=True):
        rng = np.random.default_rng(20240902)
        if distinct:
            x = rng.permutation(int(np.prod(shape))).astype("float64")
            x = (x / x.size).reshape(shape).astype("float32")
        else:
            # A small alphabet makes ties common, which is what an
            # out-of-bounds ``pout`` read needs in order to be mistaken for a
            # real maximum by the ``@pout == @in0`` test in the kernel.
            x = rng.integers(0, 4, size=shape).astype("float32")
        return x

    def _seed(self, shape, kernel, stride, padding, ceil_mode):
        kernel, stride, padding = _triple(kernel), _triple(stride), _triple(padding)
        sizes = [shape[0], shape[1]] + [
            _out_size(shape[2 + i], kernel[i], stride[i], padding[i], ceil_mode)
            for i in range(3)
        ]
        rng = np.random.default_rng(7)
        return rng.standard_normal(sizes).astype("float32")

    def test_max_backward_cpu_matches_reference(self):
        shape = (2, 2, 6, 7, 8)
        for kernel, stride, padding in self.CASES:
            with self.subTest(kernel=kernel, stride=stride, padding=padding):
                x = self._inputs(shape)
                seed = self._seed(shape, kernel, stride, padding, False)
                got, _ = _pool3d_grad(x, kernel, stride, padding, "maximum", False, 0, seed)
                expected = _reference_maxpool3d_grad(
                    x, seed.astype(np.float64), kernel, stride, padding
                )
                np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_max_backward_cuda_matches_reference(self):
        shape = (2, 2, 6, 7, 8)
        for kernel, stride, padding in self.CASES:
            with self.subTest(kernel=kernel, stride=stride, padding=padding):
                x = self._inputs(shape)
                seed = self._seed(shape, kernel, stride, padding, False)
                got, _ = _pool3d_grad(x, kernel, stride, padding, "maximum", False, 1, seed)
                expected = _reference_maxpool3d_grad(
                    x, seed.astype(np.float64), kernel, stride, padding
                )
                np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_cuda_backward_matches_cpu_backward(self):
        shape = (2, 2, 6, 7, 8)
        for op, ceil_mode in (("maximum", False), ("maximum", True), ("mean", True)):
            for kernel, stride, padding in self.CASES:
                for distinct in (True, False):
                    if op == "mean" and not distinct:
                        continue
                    with self.subTest(op=op, kernel=kernel, stride=stride,
                                      padding=padding, ceil_mode=ceil_mode,
                                      distinct=distinct):
                        x = self._inputs(shape, distinct)
                        seed = self._seed(shape, kernel, stride, padding, ceil_mode)
                        on_cpu, y_cpu = _pool3d_grad(
                            x, kernel, stride, padding, op, ceil_mode, 0, seed)
                        on_cuda, y_cuda = _pool3d_grad(
                            x, kernel, stride, padding, op, ceil_mode, 1, seed)
                        np.testing.assert_allclose(y_cuda, y_cpu, rtol=1e-6, atol=1e-6)
                        np.testing.assert_allclose(on_cuda, on_cpu, rtol=1e-5, atol=1e-5)

    def test_mean_backward_conserves_mass(self):
        """Every output spreads ``dout`` over exactly ``count`` inputs."""
        shape = (2, 2, 6, 7, 8)
        for kernel, stride, padding in ((2, 2, 0), ((2, 3, 2), (2, 2, 3), 0), (3, 2, 0)):
            for use_cuda in ((0, 1) if jt.compiler.has_cuda else (0,)):
                with self.subTest(kernel=kernel, stride=stride, use_cuda=use_cuda):
                    x = self._inputs(shape)
                    seed = self._seed(shape, kernel, stride, padding, True)
                    grad, _ = _pool3d_grad(
                        x, kernel, stride, padding, "mean", True, use_cuda, seed)
                    self.assertAlmostEqual(
                        float(grad.sum()), float(seed.sum()), places=2)


class TestMaxPool3dIndicesCuda(TestMaxPool3dIndices):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
