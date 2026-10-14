"""A singular operand raises on every device, as with NumPy and PyTorch.

The forward of solve/inv/cholesky runs in a ``numpy_code`` callback, which on
CUDA receives CuPy. CuPy returns NaN for a singular matrix where NumPy raises,
so the same call raised on CPU and went silently wrong on CUDA -- including the
QR backward, which solves against R.
"""

import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, get_all_device_types, use_cuda_for


SINGULAR = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)


class TestSingularInputRaises(JittorTestCase):
    def _devices(self, body):
        for device in get_all_device_types():
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda_for(device)):
                    body()

    def test_solve_inv_cholesky(self):
        def body():
            a = jt.array(SINGULAR)
            with self.assertRaises(np.linalg.LinAlgError):
                jt.linalg.solve(a, jt.ones((2, 1))).sync()
            with self.assertRaises(np.linalg.LinAlgError):
                jt.linalg.inv(a).sync()
            with self.assertRaises(np.linalg.LinAlgError):
                jt.linalg.cholesky(jt.array(-np.eye(2, dtype=np.float32))).sync()
        self._devices(body)

    def test_regular_input_still_solves(self):
        def body():
            a = np.array([[2.0, 1.0], [1.0, 3.0]], np.float32)
            b = np.array([[1.0], [2.0]], np.float32)
            np.testing.assert_allclose(jt.linalg.solve(jt.array(a), jt.array(b)).numpy(),
                                       np.linalg.solve(a, b), rtol=1e-5)
            np.testing.assert_allclose(jt.linalg.inv(jt.array(a)).numpy(),
                                       np.linalg.inv(a), rtol=1e-5)
            np.testing.assert_allclose(jt.linalg.cholesky(jt.array(a)).numpy(),
                                       np.linalg.cholesky(a), rtol=1e-5)
        self._devices(body)

    def test_qr_backward_through_singular_r(self):
        def body():
            x = jt.zeros((2, 2))
            q, r = jt.linalg.qr(jt.array(SINGULAR) + x)
            with self.assertRaises(np.linalg.LinAlgError):
                jt.grad(jt.matmul(q, r).sum(), x).sync()
        self._devices(body)


if __name__ == "__main__":
    unittest.main()
