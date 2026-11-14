"""Real multi-axis norm uses original axes and has Torch's zero subgradient."""
import unittest
import numpy as np
import torch

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestMultiAxisNorm(cu.JittorTestCase):
    def test_frozen_input_stays_frozen_with_grad_enabled(self, device):
        with torch.enable_grad():
            value = torch.ones((2, 3, 4), device=device, requires_grad=False)
            result = value.norm(p=2, dim=(1, 2))
            self.assertFalse(result.requires_grad)
            np.testing.assert_allclose(result.cpu().numpy(), np.sqrt(12.))

    def test_large_spatial_norm_matches_float64(self, device):
        array = np.random.default_rng(72).normal(size=(2, 56, 56, 128)).astype('float32')
        expected = np.sqrt(np.square(array.astype('float64')).sum((1, 2), keepdims=True))
        with torch.no_grad():
            value = torch.tensor(array, device=device).norm(p=2, dim=(1, 2), keepdim=True)
            if value.device.type == 'cuda':
                value.sync()
                self.assertEqual(value.location(), 'device')
            np.testing.assert_allclose(value.cpu().numpy(), expected, rtol=2e-6, atol=1e-6)

    def test_axes_refer_to_the_original_tensor_rank(self, device):
        array = np.arange(1, 361, dtype='float32').reshape(3, 4, 5, 6) / 100
        for dtype in (torch.float32, torch.float64):
            for dims in ((1, 2), (-3, 2), (1, -2)):
                for keepdim in (False, True):
                    with self.subTest(dtype=dtype, dims=dims, keepdim=keepdim):
                        expected = np.sqrt(np.square(array.astype('float64')).sum(dims, keepdims=keepdim))
                        x = torch.tensor(array, dtype=dtype, device=device, requires_grad=True)
                        value = x.norm(p=2, dim=dims, keepdim=keepdim)
                        self.assertEqual(value.device.type, device.split(':')[0])
                        if value.device.type == 'cuda':
                            value.sync()
                            self.assertEqual(value.location(), 'device')
                        np.testing.assert_allclose(value.detach().cpu().numpy(), expected, rtol=2e-6, atol=1e-6)
                        value.sum().backward()
                        divisor = np.sqrt(np.square(array.astype('float64')).sum(dims, keepdims=True))
                        np.testing.assert_allclose(x.grad.cpu().numpy(), array / divisor, rtol=2e-5, atol=1e-6)

    def test_zero_norm_has_zero_gradient_without_hiding_nan(self, device):
        x = torch.zeros((2, 3, 4), device=device, requires_grad=True)
        value = torch.norm(x, p=2, dim=(1, 2))
        value.sum().backward()
        np.testing.assert_array_equal(value.detach().cpu().numpy(), np.zeros(2))
        np.testing.assert_array_equal(x.grad.cpu().numpy(), np.zeros((2, 3, 4)))
        with torch.no_grad():
            values = torch.tensor([[[float('nan')]], [[float('inf')]]], device=device)
            observed = values.norm(p=2, dim=(1, 2)).cpu().numpy()
        self.assertTrue(np.isnan(observed[0]))
        self.assertTrue(np.isposinf(observed[1]))

    def test_duplicate_axes_are_rejected(self, device):
        with self.assertRaises(RuntimeError):
            torch.ones((2, 3, 4), device=device).norm(p=2, dim=(1, -2))


instantiate_device_type_tests(TestMultiAxisNorm, globals())
