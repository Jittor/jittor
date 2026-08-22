import pickle
import unittest

import numpy as np

import jittor as jt


class Test0DParity(unittest.TestCase):
    def test_scalar_creation_and_numpy(self):
        x = jt.array(1.0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.numpy().shape, ())
        self.assertEqual(x.item(), 1.0)

        y = jt.array(np.float32(2.0))
        self.assertEqual(y.shape, [])
        self.assertEqual(y.numpy().shape, ())

    def test_reduce_index_reshape_transpose(self):
        x = jt.array([1.0, 2.0, 3.0])
        self.assertEqual(x.sum().shape, [])
        self.assertEqual(x[0].shape, [])
        self.assertEqual(jt.tensordot(x, x, dims=1).shape, [])
        self.assertEqual(jt.array(1.0).reshape([]).shape, [])
        self.assertEqual(jt.array(1.0).transpose().shape, [])

    def test_scalar_protocols(self):
        x = jt.array(1.0)
        with self.assertRaises(TypeError):
            len(x)
        with self.assertRaises(TypeError):
            iter(x)

    def test_complex_scalar_backward(self):
        x = jt.array(np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64))
        loss = jt.real(x.sum())
        gradient = jt.grad(loss, x)
        self.assertEqual(loss.shape, [])
        np.testing.assert_allclose(
            gradient.numpy(), np.ones(2, dtype=np.complex64)
        )

    def test_scalar_power_backward(self):
        x = jt.array(np.float32(2.0))
        loss = x**3
        gradient = jt.grad(loss, x)
        self.assertEqual(loss.shape, [])
        self.assertEqual(gradient.shape, [])
        self.assertEqual(gradient.item(), 12.0)

    def test_pickle_roundtrip(self):
        x = jt.array(np.float32(3.0))
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(y.shape, [])
        self.assertEqual(y.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
