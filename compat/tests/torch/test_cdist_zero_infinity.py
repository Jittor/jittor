"""Regression tests for cdist's zero and infinite orders."""

import unittest

import numpy as np
import torch


class TestCdistZeroInfinity(unittest.TestCase):
    def setUp(self):
        self.x1_values = np.array([[0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
        self.x2_values = np.array([[1.0, 0.0], [2.0, 4.0]], dtype=np.float32)

    def test_zero_order_counts_differing_coordinates(self):
        result = torch.cdist(torch.tensor(self.x1_values),
                             torch.tensor(self.x2_values), p=0)
        expected = np.count_nonzero(
            self.x1_values[:, None, :] - self.x2_values[None, :, :], axis=-1)
        np.testing.assert_array_equal(result.detach().numpy(), expected.astype(np.float32))

    def test_infinite_order_uses_largest_absolute_difference(self):
        result = torch.cdist(torch.tensor(self.x1_values),
                             torch.tensor(self.x2_values), p=float("inf"))
        expected = np.max(
            np.abs(self.x1_values[:, None, :] - self.x2_values[None, :, :]),
            axis=-1)
        np.testing.assert_array_equal(result.detach().numpy(), expected)

    def test_infinite_order_gradient_follows_unique_maximum(self):
        x1 = torch.tensor(self.x1_values, requires_grad=True)
        x2 = torch.tensor(self.x2_values, requires_grad=True)
        torch.cdist(x1, x2, p=float("inf")).sum().backward()

        np.testing.assert_array_equal(
            x1.grad.numpy(), np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=np.float32))
        np.testing.assert_array_equal(
            x2.grad.numpy(), np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    def test_zero_order_retains_zero_gradients_for_each_differentiable_input(self):
        for first_grad, second_grad in ((True, True), (True, False), (False, True)):
            x1 = torch.tensor(self.x1_values, requires_grad=first_grad)
            x2 = torch.tensor(self.x2_values, requires_grad=second_grad)
            result = torch.cdist(x1, x2, p=0)
            self.assertTrue(result.requires_grad)
            result.sum().backward()
            for value, enabled in ((x1, first_grad), (x2, second_grad)):
                if enabled:
                    np.testing.assert_array_equal(value.grad.numpy(),
                                                  np.zeros(tuple(value.shape)))
                else:
                    self.assertIsNone(value.grad)

    def test_infinite_order_gradient_reaches_every_tied_coordinate(self):
        x1 = torch.tensor([[1.0, -1.0], [0.0, 0.0]], requires_grad=True)
        x2 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        torch.cdist(x1, x2, p=float("inf")).sum().backward()
        # torch.cdist gives each maximum coordinate its full derivative.
        np.testing.assert_array_equal(x1.grad.numpy(), [[1.0, -2.0], [-1.0, -1.0]])
        np.testing.assert_array_equal(x2.grad.numpy(), [[-1.0, 1.0], [1.0, 2.0]])

    def test_zero_features_return_zeros_and_empty_gradients(self):
        for p in (float("inf"), 0):
            x1 = torch.tensor(np.empty((2, 0), dtype=np.float32), requires_grad=True)
            x2 = torch.tensor(np.empty((3, 0), dtype=np.float32), requires_grad=True)
            result = torch.cdist(x1, x2, p=p)
            self.assertTrue(result.requires_grad)
            np.testing.assert_array_equal(result.detach().numpy(), np.zeros((2, 3)))
            result.sum().backward()
            self.assertEqual(tuple(x1.grad.shape), (2, 0))
            self.assertEqual(tuple(x2.grad.shape), (3, 0))

    def test_empty_rows_and_batches_retain_zero_gradients(self):
        for first_shape, second_shape in (((0, 3), (2, 3)),
                                          ((2, 3), (0, 3)),
                                          ((0, 2, 3), (1, 4, 3))):
            for p in (0, float("inf")):
                x1 = torch.tensor(np.zeros(first_shape, dtype=np.float32),
                                  requires_grad=True)
                x2 = torch.tensor(np.zeros(second_shape, dtype=np.float32),
                                  requires_grad=True)
                result = torch.cdist(x1, x2, p=p)
                self.assertEqual(result.numel(), 0)
                self.assertTrue(result.requires_grad)
                result.sum().backward()
                np.testing.assert_array_equal(x1.grad.numpy(), np.zeros(first_shape))
                np.testing.assert_array_equal(x2.grad.numpy(), np.zeros(second_shape))

    def test_broadcast_batches_sum_every_pair_gradient(self):
        first = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3)
        second = np.arange(36, dtype=np.float32).reshape(1, 3, 4, 3)
        delta = first[..., :, None, :] - second[..., None, :, :]
        for p in (0, float("inf")):
            x1 = torch.tensor(first, requires_grad=True)
            x2 = torch.tensor(second, requires_grad=True)
            result = torch.cdist(x1, x2, p=p)
            if p == 0:
                expected = np.count_nonzero(delta, axis=-1)
                pair_gradient = np.zeros_like(delta)
            else:
                expected = np.abs(delta).max(axis=-1)
                pair_gradient = (np.abs(delta) == expected[..., None]) * np.sign(delta)
            np.testing.assert_array_equal(result.detach().numpy(), expected)
            result.sum().backward()
            np.testing.assert_array_equal(
                x1.grad.numpy(), pair_gradient.sum(axis=-2).sum(axis=1, keepdims=True))
            np.testing.assert_array_equal(
                x2.grad.numpy(), -pair_gradient.sum(axis=-3).sum(axis=0, keepdims=True))

    def test_float64_preserves_small_coordinate_differences(self):
        for p in (0, float("inf")):
            x1 = torch.tensor([[1.0000000001, 2.5]], dtype=torch.float64,
                              requires_grad=True)
            x2 = torch.tensor([[1.0, 2.7]], dtype=torch.float64, requires_grad=True)
            result = torch.cdist(x1, x2, p=p)
            self.assertEqual(result.dtype, torch.float64)
            expected = [[2.0]] if p == 0 else [[2.7 - 2.5]]
            np.testing.assert_array_equal(result.detach().numpy(), expected)
            result.sum().backward()
            np.testing.assert_array_equal(x1.grad.numpy(), [[0.0, 0.0 if p == 0 else -1.0]])
            np.testing.assert_array_equal(x2.grad.numpy(), [[0.0, 0.0 if p == 0 else 1.0]])


if __name__ == "__main__":
    unittest.main()
