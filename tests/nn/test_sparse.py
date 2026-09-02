
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Xiangli Li <1905692338@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestSparse(unittest.TestCase):
    def test_sparse_var_dense_transpose_spmm_and_grad(self):
        indices = np.array([[0, 1, 1], [2, 0, 2]], dtype=np.int32)
        values = np.array([3, 4, 5], dtype=np.float32)
        shape = [2,3]
        expected = np.zeros(shape, dtype=np.float32)
        expected[tuple(indices)] = values

        value_var = jt.array(values)
        sparse = jt.sparse.sparse_array(
            jt.array(indices), value_var, jt.NanoVector(shape)
        )
        np.testing.assert_array_equal(sparse.to_dense().numpy(), expected)
        np.testing.assert_array_equal(sparse.t().to_dense().numpy(), expected.T)

        rhs = np.arange(6, dtype=np.float32).reshape(3, 2) / 5
        actual = jt.sparse.spmm(sparse, jt.array(rhs))
        np.testing.assert_allclose(actual.numpy(), expected @ rhs, atol=1e-6, rtol=1e-6)

        grad = jt.grad(actual.sum(), [value_var])[0].numpy()
        reference_grad = rhs[indices[1]].sum(axis=1)
        np.testing.assert_allclose(grad, reference_grad, atol=1e-6, rtol=1e-6)

    def test_sparse_convolution_legacy_module_is_canonical(self):
        import importlib

        canonical = importlib.import_module("jittor.sparse.convolution")
        legacy = importlib.import_module("jittor.nn.sparse")
        self.assertIs(legacy, canonical)
        self.assertIs(jt.nn.submanifold_conv3d, canonical.submanifold_conv3d)
        self.assertIs(jt.sparse.submanifold_conv3d, canonical.submanifold_conv3d)
        
class TestSparseCOODuplicates(unittest.TestCase):
    """COO tensors are uncoalesced: a coordinate may appear several times and
    its value is the *sum* of the duplicates. ``to_dense`` used to assign
    instead of accumulate, so a duplicate silently overwrote its predecessor.
    scipy's ``coo_matrix`` is the oracle -- it defines the same summing rule."""

    def _scipy_dense(self, indices, values, shape):
        from scipy.sparse import coo_matrix
        return coo_matrix((values, (indices[0], indices[1])),
                          shape=tuple(shape)).toarray()

    def _sparse(self, indices, values, shape):
        return jt.sparse.sparse_array(
            jt.array(indices), jt.array(values), jt.NanoVector(list(shape)))

    def test_to_dense_sums_duplicate_coordinates(self):
        indices = np.array([[0, 1, 1, 0, 1], [2, 0, 2, 2, 0]], dtype=np.int32)
        values = np.array([3., 4., 5., 7., 11.], dtype=np.float32)
        shape = [2, 3]
        got = self._sparse(indices, values, shape).to_dense().numpy()
        np.testing.assert_allclose(
            got, self._scipy_dense(indices, values, shape), rtol=1e-6, atol=1e-6)
        # explicitly: the (0,2) entry is 3+7 and (1,0) is 4+11
        self.assertAlmostEqual(float(got[0, 2]), 10.0, places=5)
        self.assertAlmostEqual(float(got[1, 0]), 15.0, places=5)

    def test_to_dense_without_duplicates_is_unchanged(self):
        indices = np.array([[0, 1, 1], [2, 0, 2]], dtype=np.int32)
        values = np.array([3., 4., 5.], dtype=np.float32)
        shape = [2, 3]
        np.testing.assert_allclose(
            self._sparse(indices, values, shape).to_dense().numpy(),
            self._scipy_dense(indices, values, shape), rtol=1e-6, atol=1e-6)

    def test_to_dense_three_dimensional(self):
        indices = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 1]], dtype=np.int32)
        values = np.array([1.5, 2.5, -4.0], dtype=np.float32)
        shape = [2, 2, 3]
        expected = np.zeros(shape, dtype=np.float64)
        np.add.at(expected, tuple(indices), values.astype(np.float64))
        got = self._sparse(indices, values, shape).to_dense().numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_spmm_matches_scipy_with_duplicates(self):
        indices = np.array([[0, 1, 1, 0, 1], [2, 0, 2, 2, 0]], dtype=np.int32)
        values = np.array([3., 4., 5., 7., 11.], dtype=np.float32)
        shape = [2, 3]
        rhs = (np.arange(6, dtype=np.float32).reshape(3, 2) / 5)
        got = jt.sparse.spmm(self._sparse(indices, values, shape),
                             jt.array(rhs)).numpy()
        expected = self._scipy_dense(indices, values, shape) @ rhs
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_spmm_does_not_densify(self):
        # the point of a sparse product: never materialise the sparse operand
        indices = np.array([[0, 1, 1], [2, 0, 2]], dtype=np.int32)
        values = np.array([3., 4., 5.], dtype=np.float32)
        sparse = self._sparse(indices, values, [2, 3])
        calls = []
        original = type(sparse).to_dense

        def spy(self):
            calls.append(1)
            return original(self)

        type(sparse).to_dense = spy
        try:
            jt.sparse.spmm(sparse, jt.array(np.ones((3, 2), dtype=np.float32)))
        finally:
            type(sparse).to_dense = original
        self.assertEqual(calls, [], "spmm must not go through to_dense")

    def test_spmm_gradients_flow_to_values_and_rhs(self):
        indices = np.array([[0, 1, 1, 0], [2, 0, 2, 2]], dtype=np.int32)
        values = np.array([3., 4., 5., 7.], dtype=np.float32)
        rhs = (np.arange(6, dtype=np.float32).reshape(3, 2) / 5)
        value_var = jt.array(values)
        rhs_var = jt.array(rhs)
        sparse = jt.sparse.sparse_array(jt.array(indices), value_var,
                                        jt.NanoVector([2, 3]))
        out = jt.sparse.spmm(sparse, rhs_var)
        gv, gy = jt.grad(out.sum(), [value_var, rhs_var])
        np.testing.assert_allclose(gv.numpy(), rhs[indices[1]].sum(axis=1),
                                   rtol=1e-6, atol=1e-6)
        expected_gy = np.zeros_like(rhs)
        np.add.at(expected_gy, indices[1], np.repeat(values[:, None], 2, axis=1))
        np.testing.assert_allclose(gy.numpy(), expected_gy, rtol=1e-6, atol=1e-6)

    def test_transpose_then_dense_still_matches(self):
        indices = np.array([[0, 1, 1, 0], [2, 0, 2, 2]], dtype=np.int32)
        values = np.array([3., 4., 5., 7.], dtype=np.float32)
        shape = [2, 3]
        sparse = self._sparse(indices, values, shape)
        np.testing.assert_allclose(
            sparse.t().to_dense().numpy(),
            self._scipy_dense(indices, values, shape).T, rtol=1e-6, atol=1e-6)


def _reference_first_occurrence_neighbors(coords, kernel, dilation):
    """Neighbor table with the documented rule: a repeated coordinate resolves
    to its first occurrence."""
    table = {}
    for index, row in enumerate(coords):
        table.setdefault(tuple(int(v) for v in row), index)
    centers = [k // 2 for k in kernel]
    out = np.full((len(coords), kernel[0] * kernel[1] * kernel[2]), -1, "int32")
    for i, row in enumerate(coords):
        tap = 0
        for kz in range(kernel[0]):
            for ky in range(kernel[1]):
                for kx in range(kernel[2]):
                    key = (int(row[0]),
                           int(row[1]) + (kz - centers[0]) * dilation[0],
                           int(row[2]) + (ky - centers[1]) * dilation[1],
                           int(row[3]) + (kx - centers[2]) * dilation[2])
                    out[i, tap] = table.get(key, -1)
                    tap += 1
    return out


class TestSubmanifoldDuplicateCoords(unittest.TestCase):
    """Duplicate coordinates must mean the same thing on both backends, and a
    cached neighbor table must belong to the coordinates it is used with."""

    KERNEL = (1, 1, 3)
    DILATION = (1, 1, 1)

    def _duplicated_coords(self, n_points=512, n_distinct=4):
        # every coordinate appears n_points/n_distinct times; the CPU table
        # keeps the first occurrence and CUDA used to keep whichever thread
        # happened to win an atomicCAS.
        coords = np.zeros((n_points, 4), dtype=np.int32)
        coords[:, 3] = np.arange(n_points, dtype=np.int32) % n_distinct
        return coords

    def _build(self, coords_np, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            return jt.sparse.build_submanifold_conv3d_neighbors(
                jt.array(coords_np), self.KERNEL, dilation=self.DILATION
            ).numpy()

    def test_first_occurrence_on_cpu(self):
        coords_np = self._duplicated_coords()
        got = self._build(coords_np, 0)
        expected = _reference_first_occurrence_neighbors(
            coords_np, self.KERNEL, self.DILATION)
        np.testing.assert_array_equal(got, expected)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_first_occurrence_on_cuda_matches_cpu(self):
        coords_np = self._duplicated_coords()
        expected = _reference_first_occurrence_neighbors(
            coords_np, self.KERNEL, self.DILATION)
        cpu = self._build(coords_np, 0)
        cuda = self._build(coords_np, 1)
        np.testing.assert_array_equal(cpu, expected)
        np.testing.assert_array_equal(cuda, expected)
        np.testing.assert_array_equal(cuda, cpu)

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_convolution_output_agrees_across_backends(self):
        coords_np = self._duplicated_coords(n_points=256, n_distinct=4)
        rng = np.random.RandomState(3)
        feats_np = rng.randn(coords_np.shape[0], 3).astype("float32")
        weight_np = rng.randn(2, 1, 1, 3, 3).astype("float32")

        def run(use_cuda):
            with jt.flag_scope(use_cuda=use_cuda):
                return jt.sparse.submanifold_conv3d(
                    jt.array(feats_np), jt.array(coords_np),
                    jt.array(weight_np), dilation=self.DILATION).numpy()

        np.testing.assert_allclose(run(1), run(0), atol=1e-5, rtol=1e-5)


class TestSubmanifoldNeighborCache(unittest.TestCase):
    KERNEL = (1, 1, 3)

    def _inputs(self, seed=5):
        rng = np.random.RandomState(seed)
        coords = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]], "int32")
        feats = rng.randn(3, 2).astype("float32")
        weight = rng.randn(2, 1, 1, 3, 2).astype("float32")
        return coords, feats, weight

    def test_matching_cache_is_accepted(self):
        coords_np, feats_np, weight_np = self._inputs()
        coords = jt.array(coords_np)
        cache = jt.sparse.build_submanifold_conv3d_neighbors(coords, self.KERNEL)
        cached = jt.sparse.submanifold_conv3d(
            jt.array(feats_np), coords, jt.array(weight_np), neighbors=cache)
        fresh = jt.sparse.submanifold_conv3d(
            jt.array(feats_np), coords, jt.array(weight_np))
        np.testing.assert_allclose(cached.numpy(), fresh.numpy(),
                                   atol=1e-6, rtol=1e-6)

    def test_equal_but_distinct_coords_object_is_accepted(self):
        coords_np, feats_np, weight_np = self._inputs()
        cache = jt.sparse.build_submanifold_conv3d_neighbors(
            jt.array(coords_np), self.KERNEL)
        out = jt.sparse.submanifold_conv3d(
            jt.array(feats_np), jt.array(coords_np), jt.array(weight_np),
            neighbors=cache)
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_cache_for_other_coordinates_is_rejected(self):
        coords_np, feats_np, weight_np = self._inputs()
        other = coords_np.copy()
        other[2, 3] = 9        # same shape, different topology
        cache = jt.sparse.build_submanifold_conv3d_neighbors(
            jt.array(other), self.KERNEL)
        with self.assertRaises(ValueError) as ctx:
            jt.sparse.submanifold_conv3d(
                jt.array(feats_np), jt.array(coords_np), jt.array(weight_np),
                neighbors=cache)
        assert "different coordinates" in str(ctx.exception), ctx.exception

    def test_cache_for_other_dilation_is_rejected(self):
        coords_np, feats_np, weight_np = self._inputs()
        coords = jt.array(coords_np)
        cache = jt.sparse.build_submanifold_conv3d_neighbors(
            coords, self.KERNEL, dilation=2)
        with self.assertRaises(ValueError) as ctx:
            jt.sparse.submanifold_conv3d(
                jt.array(feats_np), coords, jt.array(weight_np),
                dilation=1, neighbors=cache)
        assert "dilation" in str(ctx.exception), ctx.exception

    def test_cache_for_other_kernel_is_rejected(self):
        coords_np, feats_np, weight_np = self._inputs()
        coords = jt.array(coords_np)
        cache = jt.sparse.build_submanifold_conv3d_neighbors(coords, (3, 1, 1))
        with self.assertRaises(ValueError) as ctx:
            jt.sparse.submanifold_conv3d(
                jt.array(feats_np), coords, jt.array(weight_np),
                neighbors=cache)
        assert "kernel" in str(ctx.exception), ctx.exception

    def test_untracked_neighbors_tensor_is_rejected(self):
        coords_np, feats_np, weight_np = self._inputs()
        raw = jt.array(np.zeros((3, 3), dtype=np.int32))
        with self.assertRaises(ValueError) as ctx:
            jt.sparse.submanifold_conv3d(
                jt.array(feats_np), jt.array(coords_np), jt.array(weight_np),
                neighbors=raw)
        assert "build_submanifold_conv3d_neighbors" in str(ctx.exception)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestSparseCOODuplicatesCuda(TestSparseCOODuplicates):
    """Same contract on CUDA: the scatter-add there goes through atomics, and
    duplicate coordinates are exactly the case that exercises them."""

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=1)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
