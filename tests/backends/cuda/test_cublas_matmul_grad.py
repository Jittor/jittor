
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from _helpers.assertions import expect_error


@_test_preserve_policy(jt, 'cuda_allow_tf32')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA is required")
class TestCublasMatmulGrad(unittest.TestCase):
    def test_acc_non_float_inputs_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[1, 2]], dtype="int32")
            b = jt.array([[1], [2]], dtype="int32")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_acc_matmul(
                    a, b, 0, 0, -1, -1, 0, 0),
                exc_type=RuntimeError,
                match="floating-point inputs",
            )

    def test_acc_mixed_input_dtypes_are_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.array([[1.0, 2.0]], dtype="float32")
            b = jt.array([[1.0], [2.0]], dtype="float64")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_acc_matmul(
                    a, b, 0, 0, -1, -1, 0, 0),
                exc_type=RuntimeError,
                match="same dtype",
            )

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

    def test_batched_and_acc_error_then_compute(self):
        """A caught cuBLAS user error must not poison the next CUDA launch."""
        with jt.flag_scope(use_cuda=1):
            batched_a = jt.array([[[1, 2]]], dtype="int32")
            batched_b = jt.array([[[1], [2]]], dtype="int32")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_batched_matmul(
                    batched_a, batched_b, False, False),
                exc_type=RuntimeError,
                match="floating-point inputs",
            )

            acc_a = jt.array([[1, 2]], dtype="int32")
            acc_b = jt.array([[1], [2]], dtype="int32")
            expect_error(
                lambda: jt.compile_extern.cublas_ops.cublas_acc_matmul(
                    acc_a, acc_b, 0, 0, -1, -1, 0, 0),
                exc_type=RuntimeError,
                match="floating-point inputs",
            )

            out = jt.compile_extern.cublas_ops.cublas_batched_matmul(
                jt.array([[[1.0, 2.0]]], dtype="float32"),
                jt.array([[[3.0], [4.0]]], dtype="float32"),
                False, False,
            )
            got, = jt.fetch_sync([out])
            np.testing.assert_array_equal(
                got, np.array([[[11.0]]], dtype=np.float32))

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
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self.old_tf32 = int(getattr(jt.introspection.policy.runtime, "cuda_allow_tf32", 0))
        if hasattr(jt.introspection.policy.runtime, "cuda_allow_tf32"):
            _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=0))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            if hasattr(jt.introspection.policy.runtime, "cuda_allow_tf32"):
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=self.old_tf32))

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

    def test_half_gradients_from_a_float32_consumer(self):
        """A float32 cotangent for a half/bf16 product gives half/bf16 gradients.

        `y + w_fp32` promotes the product, so the gradient reaching the GEMM is
        float32 while its saved operands are half; the gradient GEMM used to
        fail the same-dtype check. Every transpose combination of the 2-D op
        and the batched op must answer gradients in the operands' dtype.
        """
        cublas = jt.compile_extern.cublas_ops
        rng = np.random.RandomState(20261006)
        m, k, n = 3, 4, 5
        with jt.flag_scope(use_cuda=1):
            for dtype, tol in (("float16", 2e-2), ("bfloat16", 1e-1)):
                for trans_a in (False, True):
                    for trans_b in (False, True):
                        label = f"{dtype} trans_a={trans_a} trans_b={trans_b}"
                        a_np = rng.randn(*((k, m) if trans_a else (m, k))).astype("float32")
                        b_np = rng.randn(*((n, k) if trans_b else (k, n))).astype("float32")
                        go_np = rng.randn(m, n).astype("float32")
                        a = jt.array(a_np).cast(dtype)
                        b = jt.array(b_np).cast(dtype)
                        out = cublas.cublas_matmul(a, b, trans_a, trans_b)
                        shift = jt.array(rng.randn(m, n).astype("float32"))
                        loss = ((out + shift) * jt.array(go_np)).sum()
                        da, db = jt.grad(loss, [a, b])
                        self.assertEqual(str(da.dtype), dtype, label)
                        self.assertEqual(str(db.dtype), dtype, label)
                        a_ref = a.float32().numpy()
                        b_ref = b.float32().numpy()
                        op_a = a_ref.T if trans_a else a_ref
                        op_b = b_ref.T if trans_b else b_ref
                        ref_da = go_np @ op_b.T
                        ref_db = op_a.T @ go_np
                        np.testing.assert_allclose(
                            da.float32().numpy(), ref_da.T if trans_a else ref_da,
                            atol=tol, rtol=tol, err_msg=label)
                        np.testing.assert_allclose(
                            db.float32().numpy(), ref_db.T if trans_b else ref_db,
                            atol=tol, rtol=tol, err_msg=label)

                a = jt.array(rng.randn(2, m, k).astype("float32")).cast(dtype)
                b = jt.array(rng.randn(2, k, n).astype("float32")).cast(dtype)
                go_np = rng.randn(2, m, n).astype("float32")
                out = cublas.cublas_batched_matmul(a, b, False, False)
                shift = jt.array(rng.randn(2, m, n).astype("float32"))
                da, db = jt.grad(((out + shift) * jt.array(go_np)).sum(), [a, b])
                self.assertEqual(str(da.dtype), dtype, f"{dtype} batched")
                self.assertEqual(str(db.dtype), dtype, f"{dtype} batched")
                a_ref, b_ref = a.float32().numpy(), b.float32().numpy()
                np.testing.assert_allclose(da.float32().numpy(), go_np @ b_ref.transpose(0, 2, 1),
                                           atol=tol, rtol=tol, err_msg=f"{dtype} batched da")
                np.testing.assert_allclose(db.float32().numpy(), a_ref.transpose(0, 2, 1) @ go_np,
                                           atol=tol, rtol=tol, err_msg=f"{dtype} batched db")

                # The public spelling reaches the same ops.
                x = jt.array(rng.randn(2, m, k).astype("float32")).cast(dtype)
                w = jt.array(rng.randn(n, k).astype("float32")).cast(dtype)
                bias = jt.array(rng.randn(n).astype("float32"))
                dx, dw = jt.grad((nn.linear(x, w) + bias).sum(), [x, w])
                self.assertEqual((str(dx.dtype), str(dw.dtype)), (dtype, dtype))

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
