"""The saved vocabulary normalizer must be repeatable, not only rounded loss."""
import unittest

import numpy as np
import jittor as jt
from _helpers import capability


_CUDA = capability.check_accelerator('cuda', backend=jt).enabled
_DEVICES = [('cpu', 0)] + ([('cuda', 1)] if _CUDA else [])


class TestCrossEntropyRepeatability(unittest.TestCase):
    def test_complete_vocabulary_forward_and_gradient_repeat(self):
        values = np.random.RandomState(41).normal(0., 3., (16, 151936)).astype('float32')
        labels = np.random.RandomState(42).randint(0, 151936, size=16).astype('int64')
        for device, enabled in _DEVICES:
            with self.subTest(device=device), jt.runtime.scope(use_cuda=enabled):
                baseline = None
                for _ in range(12):
                    x = jt.array(values)
                    loss = jt.nn.cross_entropy_loss(x, jt.array(labels))
                    # Match Tensor.backward's settled-forward condition from
                    # the real Trainer, without changing the loss's math.
                    loss.sync()
                    gradient = jt.grad(loss, [x])[0]
                    current = loss.numpy().copy(), gradient.numpy().copy()
                    if baseline is None:
                        baseline = current
                    else:
                        for actual, expected in zip(current, baseline):
                            np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(_CUDA, 'Requires actual CUDA row-LSE kernel')
    def test_row_lse_is_small_and_uses_actual_input_backend(self):
        from jittor.backends.cuda.kernels.nn import softmax_cuda
        with jt.runtime.scope(use_cuda=0):
            token = jt.core._set_construction_placement(0, 0)
            try:
                host = jt.array(np.zeros((2, 151936), dtype='float32'))
            finally:
                jt.core._restore_construction_placement(token)
            self.assertEqual(host.placement_backend, 0)
            host.sync()
        with jt.runtime.scope(use_cuda=1):
            self.assertIsNone(softmax_cuda._cross_entropy_lse(host))
            double = jt.array(np.zeros((2, 7), dtype='float64'), dtype='float64')
            self.assertEqual(str(double.dtype), 'float64')
            self.assertIsNone(softmax_cuda._cross_entropy_lse(double))
            x = jt.array(np.zeros((2, 151936), dtype='float32'))
            with jt.flag_scope(transpose_storage_view=1):
                view = x.transpose()
            self.assertFalse(view._storage_is_contiguous())
            self.assertIsNone(softmax_cuda._cross_entropy_lse(view))
            result = softmax_cuda._cross_entropy_lse(x)
            self.assertEqual(tuple(result.shape), (2,))
            result.sync()
            self.assertEqual(result.location(), 'device')
            np.testing.assert_allclose(result.numpy(), np.log(151936.), rtol=1e-6, atol=1e-6)

    def test_nonfinite_rows_and_fp64_fallback(self):
        for device, enabled in _DEVICES:
            with self.subTest(device=device), jt.runtime.scope(use_cuda=enabled):
                values = np.array([[0., np.nan], [0., np.inf], [-np.inf, -np.inf], [1., 2.]], dtype='float32')
                x = jt.array(values)
                loss = jt.nn.cross_entropy_loss(x, jt.array([0, 0, 0, 0]), reduction='none')
                gradient = jt.grad(loss.sum(), [x])[0].numpy()
                self.assertTrue(np.isnan(loss.numpy()[:3]).all())
                self.assertTrue(np.isnan(gradient[:3]).all())
                np.testing.assert_allclose(loss.numpy()[3], np.log1p(np.exp(1.)), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(gradient[3], [-1/(1+np.exp(-1.)), 1/(1+np.exp(-1.))], rtol=1e-6, atol=1e-6)
                double = jt.array(np.array([[1., 2., 3.]], dtype='float64'), dtype='float64')
                self.assertEqual(str(double.dtype), 'float64')
                loss = jt.nn.cross_entropy_loss(double, jt.array([1]))
                self.assertEqual(str(loss.dtype), 'float64')
                np.testing.assert_allclose(loss.numpy(), np.log(np.exp(np.array([-2., -1., 0.])).sum())+1., rtol=1e-12, atol=1e-12)
