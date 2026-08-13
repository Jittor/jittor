
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
        
if __name__ == "__main__":
    unittest.main()
