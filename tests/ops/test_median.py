import unittest

import numpy as np

import jittor as jt


def _median_reference(x, dim, keepdim):
    if dim is None:
        flat = x.reshape(-1)
        k = (flat.size - 1) // 2
        values = np.sort(flat)[k:k + 1]
        grad_indices = np.array([np.argsort(flat)[k]], dtype=np.int64)
        return values, grad_indices

    dim %= x.ndim
    k = (x.shape[dim] - 1) // 2
    order = np.argsort(x, axis=dim)
    selected = np.take(order, k, axis=dim)
    values = np.take_along_axis(x, np.expand_dims(selected, dim), axis=dim)
    if not keepdim:
        values = np.squeeze(values, axis=dim)
    return values, selected


class _MedianMixin:
    use_cuda = 0

    def setUp(self):
        self._previous_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = self.use_cuda

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._previous_use_cuda

    def test_values_and_gradients_across_axes(self):
        source = np.array([
            [[9., 1., 5., 3.], [8., 4., 2., 6.], [7., 0., 10., 11.]],
            [[-1., -9., -5., -3.], [-8., -4., -2., -6.], [-7., .5, -10., -11.]],
        ], dtype=np.float32)

        for dim in (None, 0, 1, 2, -1, -2):
            for keepdim in (False, True):
                with self.subTest(dim=dim, keepdim=keepdim):
                    expected, selected = _median_reference(source, dim, keepdim)
                    x = jt.array(source)
                    actual = jt.median(x, dim=dim, keepdim=keepdim)
                    np.testing.assert_array_equal(actual.numpy(), expected)

                    weights = np.arange(
                        1, actual.numel() + 1, dtype=np.float32
                    ).reshape(actual.shape)
                    grad = jt.grad((actual * jt.array(weights)).sum(), x).numpy()
                    expected_grad = np.zeros_like(source)
                    if dim is None:
                        expected_grad.reshape(-1)[selected[0]] = weights.reshape(-1)[0]
                    else:
                        axis = dim % source.ndim
                        expanded_indices = np.expand_dims(selected, axis)
                        expanded_weights = weights
                        if not keepdim:
                            expanded_weights = np.expand_dims(weights, axis)
                        np.put_along_axis(
                            expected_grad, expanded_indices, expanded_weights, axis=axis
                        )
                    np.testing.assert_array_equal(grad, expected_grad)

    def test_invalid_dimension_raises(self):
        x = jt.array(np.ones((2, 3), dtype=np.float32))
        for dim in (-3, 2):
            with self.subTest(dim=dim):
                with self.assertRaises(IndexError):
                    jt.median(x, dim=dim)


class TestMedianCPU(_MedianMixin, unittest.TestCase):
    pass


@unittest.skipUnless(jt.compiler.has_cuda, "CUDA is unavailable")
class TestMedianCUDA(_MedianMixin, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
