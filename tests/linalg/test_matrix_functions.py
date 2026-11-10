"""``jittor.linalg.matrix_exp`` and ``eigvals``.

matrix_exp is scaling-and-squaring over a degree-18 Taylor series; its values
and gradient are compared with closed forms here (and were compared once with
``torch.linalg.matrix_exp`` from PyTorch 2.9.0 to ~1e-15). eigvals returns
native complex eigenvalues in the input's precision and differentiates with
torch's eigenvalue backward ``V^{-H} diag(gw) V^H``.
"""
import unittest

import numpy as np

import jittor as jt
from jittor.linalg import eigvals, matrix_exp


class TestMatrixExp(unittest.TestCase):
    def test_rotation_generator(self):
        a = jt.array64(np.array([[0.0, 2.5], [-2.5, 0.0]]))
        expected = np.array([[np.cos(2.5), np.sin(2.5)], [-np.sin(2.5), np.cos(2.5)]])
        np.testing.assert_allclose(matrix_exp(a).numpy(), expected, atol=1e-13)

    def test_batched_diagonal_and_large_norm(self):
        d = np.array([[[1.0, 0.0], [0.0, -2.0]], [[10.0, 0.0], [0.0, 0.5]]])
        out = matrix_exp(jt.array64(d)).numpy()
        np.testing.assert_allclose(out[:, [0, 1], [0, 1]], np.exp(d[:, [0, 1], [0, 1]]), rtol=1e-13)

    def test_gradient_of_trace(self):
        # d tr(exp(A)) / dA = exp(A)^T
        a = jt.array64(np.array([[0.1, 0.4], [-0.3, 0.2]]))
        grad = jt.grad(jt.diagonal(matrix_exp(a), 0, 0, 1).sum(), a)
        np.testing.assert_allclose(grad.numpy(), matrix_exp(a).numpy().T, rtol=1e-12)

    def test_rejects_non_square(self):
        with self.assertRaises(RuntimeError):
            matrix_exp(jt.array64(np.zeros((2, 3))))


class TestEigvals(unittest.TestCase):
    def test_precision_and_values(self):
        a = np.array([[2.0, 1.0], [0.0, 3.0]])
        w = eigvals(jt.array64(a))
        self.assertEqual(jt.core._checked_dtype_name(w.dtype), "complex128")
        np.testing.assert_allclose(sorted(w.numpy().real), [2.0, 3.0], rtol=1e-12)

    def test_gradient_of_eigenvalue_sum_is_identity(self):
        # sum(eigvals(A)) = tr(A), so the gradient is the identity.
        a = jt.array64(np.array([[0.3, -1.2], [2.0, 0.1]]))
        grad = jt.grad(eigvals(a).real.sum(), a)
        np.testing.assert_allclose(grad.numpy(), np.eye(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
