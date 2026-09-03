import unittest

import numpy as np

import jittor as jt
from jittor import nn
from _helpers.assertions import expect_error


@unittest.skipIf(not jt.has_cuda, "CUDA is required")
class TestCublasMatmulGrad(unittest.TestCase):
    def test_batched_non_float_inputs_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[[1, 2]]], dtype="int32")
            b = jt.array([[[1], [2]]], dtype="int32")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, False, False),
                exc_type=RuntimeError,
                match="floating-point inputs",
            )

    def test_batched_mixed_input_dtypes_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[[1.0, 2.0]]], dtype="float32")
            b = jt.array([[[1.0], [2.0]]], dtype="float64")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, False, False),
                exc_type=RuntimeError,
                match="same dtype",
            )

    def test_non_float_inputs_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[1, 2]], dtype="int32")
            b = jt.array([[1], [2]], dtype="int32")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_matmul(a, b, False, False),
                exc_type=RuntimeError,
                match="floating-point inputs",
            )

    def test_mixed_input_dtypes_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[1.0, 2.0]], dtype="float32")
            b = jt.array([[1.0], [2.0]], dtype="float64")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_matmul(a, b, False, False),
                exc_type=RuntimeError,
                match="same dtype",
            )

    def setUp(self):
        self.old_tf32 = int(getattr(jt.flags, "cuda_allow_tf32", 0))
        if hasattr(jt.flags, "cuda_allow_tf32"):
            jt.flags.cuda_allow_tf32 = 0

    def tearDown(self):
        if hasattr(jt.flags, "cuda_allow_tf32"):
            jt.flags.cuda_allow_tf32 = self.old_tf32

    def test_all_transpose_combinations(self):
        rng = np.random.RandomState(20260710)
        m, k, n = 3, 4, 5
        with jt.flag_scope(use_cuda=1):
            for trans_a in (False, True):
                for trans_b in (False, True):
                    a_np = rng.randn(*( (k, m) if trans_a else (m, k) )).astype("float32")
                    b_np = rng.randn(*( (n, k) if trans_b else (k, n) )).astype("float32")
                    go_np = rng.randn(m, n).astype("float32")
                    a = jt.array(a_np)
                    b = jt.array(b_np)
                    go = jt.array(go_np)
                    out = jt.compile_extern.cublas_ops.cublas_matmul(
                        a, b, trans_a, trans_b)
                    da, db = jt.grad((out * go).sum(), [a, b])
                    got_out, got_da, got_db = jt.fetch_sync([out, da, db])

                    op_a = a_np.T if trans_a else a_np
                    op_b = b_np.T if trans_b else b_np
                    ref_out = op_a @ op_b
                    ref_da_op = go_np @ op_b.T
                    ref_db_op = op_a.T @ go_np
                    ref_da = ref_da_op.T if trans_a else ref_da_op
                    ref_db = ref_db_op.T if trans_b else ref_db_op
                    label = f"trans_a={trans_a}, trans_b={trans_b}"
                    np.testing.assert_allclose(got_out, ref_out, atol=2e-5, rtol=2e-5,
                                               err_msg=label)
                    np.testing.assert_allclose(got_da, ref_da, atol=2e-5, rtol=2e-5,
                                               err_msg=label)
                    np.testing.assert_allclose(got_db, ref_db, atol=2e-5, rtol=2e-5,
                                               err_msg=label)

    def test_linear_3d_random_projection_grad(self):
        rng = np.random.RandomState(20260711)
        x_np = rng.randn(2, 3, 4).astype("float32")
        w_np = rng.randn(5, 4).astype("float32")
        b_np = rng.randn(5).astype("float32")
        go_np = rng.randn(2, 3, 5).astype("float32")
        with jt.flag_scope(use_cuda=1):
            x = jt.array(x_np)
            w = jt.array(w_np)
            b = jt.array(b_np)
            out = nn.linear(x, w, b)
            dx, dw, db = jt.grad((out * jt.array(go_np)).sum(), [x, w, b])
            got_out, got_dx, got_dw, got_db = jt.fetch_sync([out, dx, dw, db])

        flat_x = x_np.reshape((-1, x_np.shape[-1]))
        flat_go = go_np.reshape((-1, go_np.shape[-1]))
        np.testing.assert_allclose(got_out, x_np @ w_np.T + b_np,
                                   atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(got_dx, go_np @ w_np,
                                   atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(got_dw, flat_go.T @ flat_x,
                                   atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(got_db, flat_go.sum(axis=0),
                                   atol=2e-5, rtol=2e-5)

    def test_float64_2d_and_batched_precision(self):
        expected = np.array([[100000001.0]], dtype=np.float64)
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[1e8, 1.0]]).float64()
            b = jt.array([[1.0], [1.0]]).float64()
            out_2d = nn.matmul(a, b)
            out_batched = nn.matmul(a.reshape((1, 1, 2)), b.reshape((1, 2, 1)))
            out_acc = jt.compile_extern.cublas_ops.cublas_acc_matmul(
                a, b, 0, 0, -1, -1, 0, 0
            )
            got_2d, got_batched, got_acc = jt.fetch_sync(
                [out_2d, out_batched, out_acc]
            )

        self.assertEqual(got_2d.dtype, np.float64)
        self.assertEqual(got_batched.dtype, np.float64)
        self.assertEqual(got_acc.dtype, np.float64)
        np.testing.assert_array_equal(got_2d, expected)
        np.testing.assert_array_equal(got_batched, expected.reshape((1, 1, 1)))
        np.testing.assert_array_equal(got_acc, expected)


if __name__ == "__main__":
    unittest.main()
