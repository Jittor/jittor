
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
