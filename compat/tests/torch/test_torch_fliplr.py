"""Torch fliplr uses native flip with correct values and gradients."""
import unittest

from _helpers.device_types import instantiate_device_type_tests

import numpy as np
import torch


class TestTorchFliplr(unittest.TestCase):
    def test_values_and_gradient(self, device):
        for shape in ((3, 4), (2, 3, 4)):
            with self.subTest(device=device, shape=shape):
                count = int(np.prod(shape))
                x = torch.arange(count, device=device,
                                 dtype=torch.float32).reshape(shape).requires_grad_()
                y = torch.fliplr(x)
                self.assertEqual(y.device.type, device)
                expected = np.flip(np.arange(count).reshape(shape), 1)
                np.testing.assert_array_equal(y.detach().cpu().numpy(), expected)
                weights = torch.arange(count, device=device,
                                       dtype=torch.float32).reshape(shape)
                (y * weights).sum().backward()
                self.assertEqual(x.grad.device.type, device)
                np.testing.assert_array_equal(x.grad.cpu().numpy(), expected)

    def test_minimum_rank(self, device):
        with self.assertRaises(RuntimeError):
            torch.fliplr(torch.ones(3, device=device))

    def test_noncontiguous_and_copy(self, device):
        x = torch.arange(12, device=device).reshape(3, 4).transpose(0, 1)
        original = x.cpu().numpy().copy()
        y = torch.fliplr(x)
        np.testing.assert_array_equal(y.cpu().numpy(), np.flip(original, 1))
        y[0, 0] = -1
        np.testing.assert_array_equal(x.cpu().numpy(), original)


instantiate_device_type_tests(TestTorchFliplr, globals(), only_for=("cpu", "cuda"))
