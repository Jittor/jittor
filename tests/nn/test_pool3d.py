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
        out = (size + 2 * padding - kernel + stride - 1) // stride + 1
        if (out - 1) * stride >= size + padding:
            out -= 1
        return out
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


def _jittor_out_size(size, kernel, stride, padding, ceil_mode):
    if ceil_mode:
        return (size + 2 * padding - kernel + stride - 1) // stride + 1
    return (size + 2 * padding - kernel) // stride + 1


def _torch_out_size(size, kernel, stride, padding, ceil_mode):
    """torch's ``pooling_output_shape``, including the ceil_mode correction."""
    out = _jittor_out_size(size, kernel, stride, padding, ceil_mode)
    if ceil_mode and (out - 1) * stride >= size + padding:
        out -= 1
    return out


def _reference_avgpool3d(x, kernel, stride, padding, out_sizes, count_include_pad):
    """torch's avg_pool3d divisor rules, spelled out.

    ``count_include_pad=True``  -> divide by the window clipped to the *padded*
    volume ``[-p, size + p)`` (the full kernel volume unless ceil_mode pushed the
    window past the padding).
    ``count_include_pad=False`` -> divide by the window clipped to the real input.
    """
    kernel, stride, padding = _triple(kernel), _triple(stride), _triple(padding)
    n, c = x.shape[:2]
    sizes = x.shape[2:]
    out = np.zeros((n, c) + tuple(out_sizes), dtype=np.float64)
    for i in range(out_sizes[0]):
        for j in range(out_sizes[1]):
            for k in range(out_sizes[2]):
                index = (i, j, k)
                starts = [index[a] * stride[a] - padding[a] for a in range(3)]
                padded_ends = [min(starts[a] + kernel[a], sizes[a] + padding[a])
                               for a in range(3)]
                lo = [max(starts[a], 0) for a in range(3)]
                hi = [min(starts[a] + kernel[a], sizes[a]) for a in range(3)]
                window = x[:, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                if count_include_pad:
                    divisor = 1
                    for a in range(3):
                        divisor *= padded_ends[a] - starts[a]
                else:
                    divisor = 1
                    for a in range(3):
                        divisor *= hi[a] - lo[a]
                out[:, :, i, j, k] = window.sum(axis=(2, 3, 4)) / divisor
    return out


class TestAvgPool3dCountIncludePad(unittest.TestCase):
    """``count_include_pad`` selects the averaging divisor -- nothing else.

    ``Pool3d.__init__`` used to compute it as ``count_include_pad and padding != 0``
    on the *raw* argument, so ``padding=(0,0,0)`` (a tuple is never ``== 0``) took a
    different branch than ``padding=0`` and the same pooling produced different
    numbers.  The two divisor implementations were wrong in their own ways too: the
    ``True`` branch of the ceil_mode kernel divided by the whole kernel volume even
    when the window hung past the padded volume, and the non-ceil_mode path went
    through ``reduce('mean')``, which always divides by the kernel volume -- so
    ``count_include_pad=False`` was silently ignored there.

    Both of those are gone: ``Pool3d.execute`` now forwards ``op="mean"`` to
    ``jt.nn.avg_pool3d``, the one implementation of average pooling in the
    package (see ``tests/nn/test_avg_pool_parity.py``).  This class stays as the
    3-D-specific divisor regression; the parity test covers the fact that every
    spelling reaches the same code.

    The reference above is torch's rule; it was checked case-by-case against a
    binary PyTorch 2.12 build in a separate process (48 legal combinations of
    kernel/stride/padding/ceil_mode/count_include_pad).
    """

    SHAPE = (1, 2, 6, 7, 8)
    GEOMETRIES = ((3, 2), ((2, 3, 2), (2, 2, 3)), (2, 2), (3, 3))

    def setUp(self):
        rng = np.random.default_rng(2024)
        self.x = rng.standard_normal(self.SHAPE).astype("float32")

    def _pool(self, kernel, stride, padding, ceil_mode, count_include_pad):
        return jt.pool.Pool3d(
            kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
            count_include_pad=count_include_pad, op="mean")(jt.array(self.x)).numpy()

    def test_scalar_and_tuple_padding_are_equivalent(self):
        """The regression: ``0`` and ``(0,0,0)`` describe the same pooling."""
        for kernel, stride in self.GEOMETRIES:
            for ceil_mode in (False, True):
                for count_include_pad in (True, False):
                    for scalar, tup in ((0, (0, 0, 0)), (1, (1, 1, 1))):
                        with self.subTest(kernel=kernel, ceil_mode=ceil_mode,
                                          count_include_pad=count_include_pad,
                                          padding=scalar):
                            a = self._pool(kernel, stride, scalar, ceil_mode,
                                           count_include_pad)
                            b = self._pool(kernel, stride, tup, ceil_mode,
                                           count_include_pad)
                            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_divisor_matches_torch_rule(self):
        for kernel, stride in self.GEOMETRIES:
            for padding in (0, 1, (0, 0, 0), (1, 1, 1)):
                for ceil_mode in (False, True):
                    for count_include_pad in (True, False):
                        with self.subTest(kernel=kernel, stride=stride,
                                          padding=padding, ceil_mode=ceil_mode,
                                          count_include_pad=count_include_pad):
                            got = self._pool(kernel, stride, padding, ceil_mode,
                                             count_include_pad)
                            k, st, pd = _triple(kernel), _triple(stride), _triple(padding)
                            # Since the mean path moved to jt.nn.avg_pool3d
                            # these sizes agree with torch exactly; the slice
                            # below is a no-op and the assertion right after it
                            # is what says so.
                            out_sizes = [
                                _torch_out_size(self.SHAPE[2 + a], k[a], st[a],
                                                pd[a], ceil_mode)
                                for a in range(3)
                            ]
                            expected = _reference_avgpool3d(
                                self.x.astype(np.float64), kernel, stride, padding,
                                out_sizes, count_include_pad)
                            self.assertEqual(list(got.shape[2:]), out_sizes)
                            np.testing.assert_allclose(
                                got, expected, rtol=1e-4, atol=1e-4)

    def test_count_include_pad_false_is_not_ignored(self):
        """With real padding the two settings must give different numbers."""
        for ceil_mode in (False, True):
            with self.subTest(ceil_mode=ceil_mode):
                on = self._pool(3, 2, 1, ceil_mode, True)
                off = self._pool(3, 2, 1, ceil_mode, False)
                self.assertGreater(float(np.abs(on - off).max()), 1e-3)

    def test_no_padding_makes_the_setting_irrelevant(self):
        for ceil_mode in (False, True):
            for kernel, stride in self.GEOMETRIES:
                with self.subTest(kernel=kernel, ceil_mode=ceil_mode):
                    on = self._pool(kernel, stride, 0, ceil_mode, True)
                    off = self._pool(kernel, stride, 0, ceil_mode, False)
                    np.testing.assert_allclose(on, off, rtol=1e-6, atol=1e-6)

    def test_ceil_mode_output_size_matches_torch_for_the_mean_path(self):
        """The mean path picked up torch's output size along with its divisor."""
        for kernel, stride in self.GEOMETRIES:
            for padding in (0, 1):
                for ceil_mode in (False, True):
                    k, st, pd = _triple(kernel), _triple(stride), _triple(padding)
                    want = tuple(
                        _torch_out_size(self.SHAPE[2 + a], k[a], st[a], pd[a], ceil_mode)
                        for a in range(3))
                    actual = tuple(self._pool(
                        kernel, stride, padding, ceil_mode, True).shape[2:])
                    self.assertEqual(actual, want, (kernel, stride, padding, ceil_mode))

    def test_max_path_ceil_mode_output_size_matches_torch(self):
        for kernel, stride in self.GEOMETRIES:
            for padding in (0, 1):
                for ceil_mode in (False, True):
                    k, st, pd = _triple(kernel), _triple(stride), _triple(padding)
                    want = tuple(
                        _torch_out_size(self.SHAPE[2 + a], k[a], st[a], pd[a], ceil_mode)
                        for a in range(3))
                    actual = tuple(jt.pool.Pool3d(
                        kernel, stride=stride, padding=padding,
                        ceil_mode=ceil_mode, op="maximum",
                    )(jt.array(self.x)).shape[2:])
                    self.assertEqual(actual, want, (kernel, stride, padding, ceil_mode))

    def test_ceil_mode_indices_still_round_trip_through_max_unpool3d(self):
        kernel, stride, padding = (2, 3, 2), (2, 2, 3), 1
        value, index = jt.nn.MaxPool3d(
            kernel, stride=stride, padding=padding, ceil_mode=True,
            return_indices=True,
        )(jt.array(self.x))
        out = jt.nn.MaxUnpool3d(kernel, stride=stride)(
            value, index, output_size=self.x.shape)
        expected = np.zeros(self.x.shape, dtype="float32")
        flat_expected = expected.reshape(self.SHAPE[:2] + (-1,))
        flat_value = value.numpy().reshape(self.SHAPE[:2] + (-1,))
        flat_index = index.numpy().astype(np.int64).reshape(self.SHAPE[:2] + (-1,))
        for bi in range(self.SHAPE[0]):
            for ci in range(self.SHAPE[1]):
                for offset, pooled in zip(flat_index[bi, ci], flat_value[bi, ci]):
                    flat_expected[bi, ci, offset] += pooled
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


class TestAvgPool3dCountIncludePadCuda(TestAvgPool3dCountIncludePad):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        super().setUp()
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


class TestMaxPool3dIndicesCuda(TestMaxPool3dIndices):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
