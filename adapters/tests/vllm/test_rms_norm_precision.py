"""The private compiled RMS contract multiplies in FP32 before the final cast."""
import os
import unittest

import numpy as np
import jittor as jt
import torch

from jittor_adapters.vllm.custom_ops import _rms_norm


class TestPrivateRMSNormPrecision(unittest.TestCase):
    def _check(self, dtype):
        device = os.environ.get('JITTOR_TEST_DEVICE', 'cpu')
        with jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error'):
            before = jt.core.backend_fallback_count()
            rng = np.random.RandomState(3)
            value = torch.tensor(rng.normal(0, 1, (7, 32)).astype('float32'),
                                 dtype=getattr(torch, dtype), device=device, requires_grad=True)
            weight = torch.tensor(rng.uniform(.6, 1.4, (32,)).astype('float32'),
                                  dtype=getattr(torch, dtype), device=device, requires_grad=True)
            x = value.detach().float().cpu().numpy().copy()
            w = weight.detach().float().cpu().numpy().copy()
            expected = (x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))) * w
            # Only conversion to the declared output format uses the backend;
            # the normalization and affine arithmetic above are NumPy's.
            expected = torch.tensor(expected, device=device, dtype=getattr(torch, dtype)).float().cpu().numpy()
            output = torch.zeros_like(value)
            alias = output.view(-1)
            pointer = output.data_ptr()
            identity = id(output)
            self.assertIsNone(_rms_norm(output, value, weight, 1e-6))
            self.assertEqual(id(output), identity)
            self.assertEqual(output.data_ptr(), pointer)
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)
            output.sync()
            self.assertEqual(output.location(), 'device' if device == 'cuda' else 'cpu')
            observed = output.float().cpu().numpy().copy()
            np.testing.assert_array_equal(alias.float().cpu().numpy(), observed.reshape(-1))
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(value.detach().float().cpu().numpy(), x)
            np.testing.assert_array_equal(weight.detach().float().cpu().numpy(), w)
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_float32(self):
        self._check('float32')

    def test_float16(self):
        self._check('float16')

    def test_bfloat16(self):
        self._check('bfloat16')


if __name__ == '__main__':
    unittest.main()
