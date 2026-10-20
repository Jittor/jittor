"""Native reciprocal-square-root opmath and its public complex composition."""
import unittest

import numpy as np

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.fallback import forbid_backend_fallbacks


class _RsqrtNativeChecks:
    cuda_enabled = 0

    def evaluate(self, raw, name):
        with jt.runtime.scope(use_cuda=self.cuda_enabled, auto_convert_64_to_32=0,
                              amp_reg=0, backend_fallback='error'):
            with forbid_backend_fallbacks():
                before = jt.core.backend_fallback_count()
                value = jt.array(raw).cast(name)
                result = jt.rsqrt(value)
                result.sync()
                self.assertEqual(dtype_name(result.dtype), name)
                self.assertEqual(jt.core.dispatch_context([result])[0],
                                 'cuda' if self.cuda_enabled else 'cpu')
                actual = result.float32().numpy() if name in ('float16', 'bfloat16') else result.numpy()
                self.assertEqual(jt.core.backend_fallback_count(), before)
                return actual

    def test_positive_forward_keeps_dtype(self):
        for name in ('float16', 'bfloat16', 'float32', 'float64'):
            with self.subTest(dtype=name):
                raw = np.array([.0625, .25, 1., 4., 16., 64.], dtype='float64')
                expected = 1 / np.sqrt(raw)
                np.testing.assert_array_equal(self.evaluate(raw, name), expected)

    def test_special_value_signs_and_classification(self):
        for name in ('float16', 'bfloat16', 'float32', 'float64'):
            with self.subTest(dtype=name):
                actual = self.evaluate(np.array([-np.inf, -1., -0., 0., np.inf, np.nan]), name)
                self.assertTrue(np.isnan(actual[[0, 1, 5]]).all())
                self.assertTrue(np.isneginf(actual[2]))
                self.assertTrue(np.isposinf(actual[3]))
                self.assertEqual(actual[4], 0.)
                self.assertFalse(np.signbit(actual[4]))

    def test_forward_and_gradient_for_reduced_opmath(self):
        for name in ('float16', 'bfloat16', 'float32', 'float64'):
            with self.subTest(dtype=name):
                with jt.runtime.scope(use_cuda=self.cuda_enabled, auto_convert_64_to_32=0,
                                      amp_reg=0, backend_fallback='error'), forbid_backend_fallbacks():
                    raw = np.array([.25, 1., 4., 16.], dtype='float64')
                    x = jt.array(raw).cast(name)
                    y = x.rsqrt()
                    grad = jt.grad(y.sum(), x)
                    self.assertEqual(dtype_name(grad.dtype), name)
                    actual = grad.float32().numpy() if name in ('float16', 'bfloat16') else grad.numpy()
                    np.testing.assert_array_equal(actual, -.5 * np.power(raw, -1.5))
                    self.assertEqual(jt.core.dispatch_context([grad])[0],
                                     'cuda' if self.cuda_enabled else 'cpu')

    def test_float64_does_not_narrow(self):
        raw = np.array([1.0 + 2.0**-51, 1.e-200, 1.e200], dtype='float64')
        actual = self.evaluate(raw, 'float64')
        np.testing.assert_allclose(actual, 1 / np.sqrt(raw), rtol=2e-15, atol=0)
        self.assertNotEqual(actual[0], 1.)

    def test_complex_public_composition_is_preserved(self):
        raw = np.array([1+2j, 3-4j, -2+.25j], dtype='complex64')
        np.testing.assert_allclose(self.evaluate(raw, 'complex64'),
                                   1 / np.sqrt(raw), rtol=2e-6, atol=2e-7)


class TestRsqrtNativeCPU(_RsqrtNativeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(jt.has_cuda, 'requires real CUDA')
class TestRsqrtNativeCUDA(_RsqrtNativeChecks, unittest.TestCase):
    cuda_enabled = 1


if __name__ == '__main__':
    unittest.main()
