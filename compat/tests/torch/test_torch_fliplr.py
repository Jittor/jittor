"""Torch fliplr uses native flip with correct values and gradients."""
import unittest

import numpy as np
import torch


class TestTorchFliplr(unittest.TestCase):
    def test_values_and_gradient(self):
        for device in ("cpu", "cuda"):
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

    def test_minimum_rank(self):
        with self.assertRaises(RuntimeError):
            torch.fliplr(torch.ones(3, device="cuda"))

    def test_noncontiguous_and_copy(self):
        x = torch.arange(12, device="cuda").reshape(3, 4).transpose(0, 1)
        original = x.cpu().numpy().copy()
        y = torch.fliplr(x)
        np.testing.assert_array_equal(y.cpu().numpy(), np.flip(original, 1))
        y[0, 0] = -1
        np.testing.assert_array_equal(x.cpu().numpy(), original)
