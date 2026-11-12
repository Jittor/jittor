"""``jittor.autograd.functional.jacobian`` / ``hessian`` follow torch's layout.

torch returns ``Jacobian[i][j]`` with shape ``outputs[i].shape + inputs[j].shape``
and unpacks single Tensors on either side; ``hessian`` is the jacobian of the
jacobian of a single-element function. The expected values are plain calculus;
the layouts and values were also compared once against
``torch.autograd.functional`` from PyTorch 2.9.0 in a separate process.
"""

import unittest

import numpy as np

import jittor as jt
from jittor.autograd.functional import hessian, jacobian

X = np.array([0.3, -0.7, 1.1])


class TestJacobian(unittest.TestCase):
    def test_vector_function(self):
        x = jt.array64(X)
        jac = jacobian(lambda a: jt.sin(a) * a.sum(), x)
        expected = np.diag(np.cos(X) * X.sum()) + np.sin(X)[:, None]
        self.assertEqual(tuple(jac.shape), (3, 3))
        np.testing.assert_allclose(jac.numpy(), expected, rtol=1e-12)

    def test_tuple_layout(self):
        x, y = jt.array64(X), jt.array64(np.array(0.4))
        jac = jacobian(lambda a, b: (a * b, (a ** 2).sum() * b), (x, y))
        self.assertEqual([[tuple(t.shape) for t in row] for row in jac], [[(3, 3), (3,)], [(3,), ()]])
        np.testing.assert_allclose(jac[0][0].numpy(), np.eye(3) * 0.4, rtol=1e-12)
        np.testing.assert_allclose(jac[0][1].numpy(), X, rtol=1e-12)
        np.testing.assert_allclose(jac[1][0].numpy(), 2 * X * 0.4, rtol=1e-12)
        np.testing.assert_allclose(jac[1][1].numpy(), (X ** 2).sum(), rtol=1e-12)

    def test_forward_mode_matches_reverse_mode(self):
        x = jt.array64(X)
        fn = lambda a: jt.exp(a) * a[0]
        forward = jacobian(fn, x, vectorize=True, strategy="forward-mode")
        np.testing.assert_allclose(forward.numpy(), jacobian(fn, x).numpy(), rtol=1e-12)
        with self.assertRaises(NotImplementedError):
            jacobian(fn, x, strategy="forward-mode")

    def test_create_graph_keeps_history(self):
        x = jt.array64(X)
        jac = jacobian(lambda a: (a ** 3).sum(), x, create_graph=True)
        np.testing.assert_allclose(jac.numpy(), 3 * X ** 2, rtol=1e-12)
        second = jt.grad(jac.sum(), x)
        self.assertIsNotNone(second)

    def test_complex_output_requires_explicit_seed(self):
        x = jt.array64(X)
        with self.assertRaisesRegex(RuntimeError, "real scalar outputs"):
            jacobian(lambda a: a * jt.array(np.array(1j)), x)



class TestHessian(unittest.TestCase):
    def test_single_input(self):
        x = jt.array64(X)
        hess = hessian(lambda a: (jt.sin(a) * a).sum(), x)
        np.testing.assert_allclose(hess.numpy(), np.diag(2 * np.cos(X) - X * np.sin(X)), rtol=1e-12)

    def test_tuple_inputs(self):
        x, y = jt.array64(X), jt.array64(np.array(0.4))
        hess = hessian(lambda a, b: (a ** 2).sum() * b ** 3, (x, y))
        np.testing.assert_allclose(hess[0][0].numpy(), 2 * 0.4 ** 3 * np.eye(3), rtol=1e-12)
        np.testing.assert_allclose(hess[0][1].numpy(), 6 * X * 0.4 ** 2, rtol=1e-12)
        np.testing.assert_allclose(hess[1][1].numpy(), 6 * (X ** 2).sum() * 0.4, rtol=1e-12)

    def test_rejects_vector_output(self):
        with self.assertRaisesRegex(RuntimeError, "single element"):
            hessian(lambda a: a * 2, jt.array64(X))


if __name__ == "__main__":
    unittest.main()
