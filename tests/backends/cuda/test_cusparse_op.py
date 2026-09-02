# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Shizhan Lu <578752274@qq.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

cusparse_ops = None


def setUpModule():
    global cusparse_ops
    cusparse_ops = getattr(jt.compile_extern, "cusparse_ops", None)
    if cusparse_ops is None:
        raise unittest.SkipTest("cuSPARSE support is unavailable")


class TestSpmmCsrOp(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, lazy_execution=0)
    def test_spmm_csr_forward_float32_int32(self):
        x = jt.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype="float32")
        col_indices = jt.array([0, 1, 1, 2], dtype="int32")
        row_offset = jt.array([0, 2, 3, 4], dtype="int32")
        csr_weight = jt.array([3.0, 1.0, 4.0, 2.0], dtype="float32")
        output = jt.zeros((3, 3), dtype="float32")
        cusparse_ops.cusparse_spmmcsr(
            output, x, col_indices, csr_weight, row_offset,
            3, 3 ,False, False
        ).fetch_sync()
        expected_output = np.array([
            [12.0, 8.0, 4.0],
            [12.0, 8.0, 4.0],
            [6.0, 4.0, 2.0]
        ])
        np.testing.assert_allclose(output.data, expected_output, atol=1e-5)

    @jt.flag_scope(use_cuda=1, lazy_execution=0)
    def test_spmm_csr_forward_float16_int32(self):
        x = jt.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype="float16")
        col_indices = jt.array([0, 1, 1, 2], dtype="int32")
        row_offset = jt.array([0, 2, 3, 4], dtype="int32")
        csr_weight = jt.array([3.0, 1.0, 4.0, 2.0], dtype="float16")
        output = jt.zeros((3, 3), dtype="float16")
        cusparse_ops.cusparse_spmmcsr(
            output, x, col_indices, csr_weight, row_offset,
            3, 3,False, False
        ).fetch_sync()
        expected_output = np.array([
            [12.0, 8.0, 4.0],
            [12.0, 8.0, 4.0],
            [6.0, 4.0, 2.0]
        ], dtype="float16")
        np.testing.assert_allclose(output.data, expected_output, atol=1e-5)

    @jt.flag_scope(use_cuda=1, lazy_execution=0)
    def test_spmm_csr_forward_float64_int32(self):
        # Commented out rather than fixed until 6.B08: the op passed
        # CUDA_R_32F and float alpha/beta whatever the dtype was, so this case
        # was computed in single precision and scaled by whatever eight bytes
        # started at &alpha. atol is tight on purpose -- 1e-5 would have
        # passed for an fp32 accumulation too.
        x = jt.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype="float64")
        col_indices = jt.array([0, 1, 1, 2], dtype="int32")
        row_offset = jt.array([0, 2, 3, 4], dtype="int32")
        csr_weight = jt.array([3.0, 1.0, 4.0, 2.0], dtype="float64")
        output = jt.zeros((3, 3), dtype="float64")
        cusparse_ops.cusparse_spmmcsr(
            output, x, col_indices, csr_weight, row_offset,
            3, 3,False, False
        ).fetch_sync()
        assert str(output.dtype) == "float64", output.dtype
        expected_output = np.array([
            [12.0, 8.0, 4.0],
            [12.0, 8.0, 4.0],
            [6.0, 4.0, 2.0]
        ], dtype="float64")
        np.testing.assert_allclose(output.data, expected_output, atol=1e-12)

    
    @jt.flag_scope(use_cuda=1, lazy_execution=0)
    def test_spmm_csr_forward_float32_int64(self):
        x = jt.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype="float32")
        col_indices = jt.array([0, 1, 1, 2], dtype="int64")
        row_offset = jt.array([0, 2, 3, 4], dtype="int64")
        csr_weight = jt.array([3.0, 1.0, 4.0, 2.0], dtype="float32")
        output = jt.zeros((3, 3), dtype="float32")
        cusparse_ops.cusparse_spmmcsr(
            output, x, col_indices, csr_weight, row_offset,
            3, 3,False, False
        ).fetch_sync()
        expected_output = np.array([
            [12.0, 8.0, 4.0],
            [12.0, 8.0, 4.0],
            [6.0, 4.0, 2.0]
        ], dtype="float32")
        np.testing.assert_allclose(output.data, expected_output, atol=1e-5)

    @jt.flag_scope(use_cuda=1, lazy_execution=0)
    def test_spmm_coo(self):
        x=jt.array([[3.0, 2.0, 1.0],[4.0, 2.0, 2.0],[1.0, 2.0, 3.0]], dtype="float32")
        edge_index=jt.array([[0,0,1,2],[1,2,2,1]],dtype="int32")
        row_indices=edge_index[0,:]
        col_indices=edge_index[1,:]
        edge_weight = jt.array([1.0, 1.0, 1.0, 1.0], dtype="float32")
        feature_dim=jt.size(x,1) 
        output=jt.zeros(3,feature_dim)
        cusparse_ops.cusparse_spmmcoo(output,x,row_indices,col_indices,edge_weight,3,3,False, False).fetch_sync()
        print("Output:", output)
        expected_output = np.array([
            [5.0, 4.0, 5.0],
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 2.0]
        ], dtype="float32")
        np.testing.assert_allclose(output.data, expected_output, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
