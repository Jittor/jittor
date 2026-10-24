"""CUDA Softplus rounding, independent analytic references and higher gradients.

The exact CUDA arithmetic-order check fails the old decomposed VJP. Independent
Native Torch CUDA confirmation is a separate interpreter control; NumPy here
only constructs public fixtures and float64 analytic expectations.
"""
import unittest
import numpy as np
import jittor as jt
from _helpers import capability
from jittor._runtime.dispatch import select_kernel


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA hardware is required; a skip is not acceptance')
class TestSoftplusCUDA(unittest.TestCase):
    def _snapshot(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        before = (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                  value.device_id, value.placement_backend, value.location())
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            host.sync()
            self.assertIsNot(host, value)
            self.assertEqual(host.location(), 'cpu')
            array = np.array(host.numpy(), copy=True, order='C')
        self.assertEqual(before, (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                                 value.device_id, value.placement_backend, value.location()))
        return array

    def test_rounded_forward_and_vjp(self):
        raw = np.linspace(-24, 24, 1216, dtype=np.float32).reshape(1, 76, 16)
        raw.reshape(-1)[:7] = [0., 1e-8, -1e-8, 20., np.nextafter(np.float32(20), np.float32(np.inf)),
                              np.nextafter(np.float32(20), np.float32(-np.inf)), -20.]
        index = np.arange(raw.size, dtype=np.int64).reshape(raw.shape)
        weights = (((index * 53 + 11) % 257) - 128).astype(np.float32) / np.float32(1021)
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for beta in (1., 2.):
                for scale in (1, 128, 1024):
                    with self.subTest(beta=beta, scale=scale):
                        x = jt.array(raw).start_grad()
                        dy = jt.array(weights * np.float32(scale)).stop_grad()
                        y = jt.nn.softplus(x, beta=beta)
                        gradient = jt.grad((y * dy).sum(), x)
                        with jt.no_grad():
                            scaled = beta * x
                            z = scaled.exp()
                            expected = jt.ternary(scaled > 20., dy, (dy * z) / (z + 1.))
                        actual, wanted = self._snapshot(gradient), self._snapshot(expected)
                        self.assertTrue(np.array_equal(actual.view(np.uint32), wanted.view(np.uint32)))
                        wide = raw.astype(np.float64)
                        ref = np.where(wide * beta > 20., wide, np.logaddexp(0., wide * beta) / beta)
                        np.testing.assert_allclose(self._snapshot(y), ref, atol=3e-6, rtol=3e-6)
                        self.assertTrue(np.array_equal(self._snapshot(x).view(np.uint32), raw.view(np.uint32)))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_input_and_cotangent_second_derivatives(self):
        rng = np.random.RandomState(117)
        raw = rng.uniform(-4, 4, (3, 7)).astype(np.float32)
        weights = rng.uniform(-2, 2, raw.shape).astype(np.float32)
        vector = rng.uniform(-1, 1, raw.shape).astype(np.float32)
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for beta in (.5, 1., 2., 3.5):
                with self.subTest(beta=beta):
                    x, dy, direction = (jt.array(v).start_grad() for v in (raw, weights, vector))
                    self.assertIsNotNone(select_kernel('nn.softplus', x, beta=beta, threshold=20.))
                    y = jt.nn.softplus(x, beta=beta)
                    first = jt.grad((y * dy).sum(), x)
                    gx, gdy = jt.grad((first * direction).sum(), (x, dy))
                    wide = raw.astype(np.float64)
                    sigmoid = 1. / (1. + np.exp(-wide * beta))
                    for actual, expected in ((first, weights * sigmoid),
                            (gx, vector * weights * beta * sigmoid * (1. - sigmoid)),
                            (gdy, vector * sigmoid)):
                        np.testing.assert_allclose(self._snapshot(actual), expected, atol=3e-6, rtol=3e-6)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_layout_empty_and_stopped_inputs_and_declines(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            matrix = np.arange(15, dtype=np.float32).reshape(3, 5) / 4 - 2
            with jt.flag_scope(transpose_storage_view=1):
                transposed = jt.array(matrix).transpose((1, 0))
            self.assertFalse(transposed._storage_is_contiguous())
            cases = [(jt.array(np.array(-.5, dtype=np.float32)), np.array(-.5, dtype=np.float32)),
                     (jt.array(np.empty((0, 3), dtype=np.float32)), np.empty((0, 3), dtype=np.float32)),
                     (transposed, matrix.T)]
            for x, original in cases:
                with self.subTest(shape=tuple(x.shape)):
                    x.start_grad()
                    y = jt.nn.softplus(x)
                    dx = jt.grad(y.sum(), x)
                    wide = original.astype(np.float64)
                    for actual, expected in ((y, np.logaddexp(0., wide)), (dx, 1. / (1. + np.exp(-wide)))):
                        np.testing.assert_allclose(self._snapshot(actual), expected, atol=3e-6, rtol=3e-6)
            self.assertFalse(transposed._storage_is_contiguous())
            frozen = jt.array([-.5, .25, 1.]).stop_grad()
            self.assertTrue(jt.nn.softplus(frozen).is_stop_grad())
            active = jt.array([-.5, .25, 1.]).start_grad()
            with jt.no_grad():
                self.assertTrue(jt.nn.softplus(active).is_stop_grad())
            self.assertFalse(active.is_stop_grad())
            for dtype in ('float16', 'float64', 'int32'):
                self.assertIsNone(select_kernel('nn.softplus', jt.ones((3,), dtype=dtype)))
            for beta in (0., -1., float('nan'), float('inf'), True):
                self.assertIsNone(select_kernel('nn.softplus', active, beta=beta, threshold=20.))
            self.assertEqual(jt.core.backend_fallback_count(), 0)


if __name__ == '__main__':
    unittest.main()
