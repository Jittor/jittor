"""Packed jagged reductions preserve dense values, gradients and placement."""
import unittest
import numpy as np
import torch
from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestJaggedSum(cu.JittorTestCase):
    def test_ragged_axis_values_dtype_and_gradient(self, device):
        with forbid_backend_fallbacks():
            values = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device))
            offsets = torch.tensor([0, 1, 3], dtype=torch.int64, device=device)
            nested = torch.nested.nested_tensor_from_jagged(values, offsets)
            result = nested.sum(dim=1)
            result.sum().backward()
            result.sync()
            values.grad.sync()
            self.assertEqual(result.location(), 'cpu' if device == 'cpu' else 'device')
            np.testing.assert_array_equal(result.detach().cpu().numpy(), [[1., 2.], [8., 10.]])
            np.testing.assert_array_equal(values.grad.detach().cpu().numpy(), np.ones((3, 2)))
            kept = nested.sum(dim=-2, keepdim=True, dtype=torch.float32)
            self.assertEqual(tuple(kept.shape), (2, 1, 2))
            self.assertTrue(kept.dtype == torch.float32)

    def test_full_sum_includes_holes_but_row_sum_refuses_them(self, device):
        with forbid_backend_fallbacks():
            values = torch.tensor([1., 2., 30., 4.], device=device)
            offsets = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
            lengths = torch.tensor([1, 1], dtype=torch.int64, device=device)
            nested = torch.nested.nested_tensor_from_jagged(values, offsets, lengths=lengths)
            self.assertEqual(nested.sum().item(), 37.)
            with self.assertRaisesRegex(RuntimeError, 'holes'):
                nested.sum(dim=1)
            with self.assertRaises(NotImplementedError):
                nested.sum(dim=0)
            with self.assertRaises(NotImplementedError):
                nested.sum(keepdim=True)
            with self.assertRaises(IndexError):
                nested.sum(dim=3)


instantiate_device_type_tests(TestJaggedSum, globals())
if __name__ == '__main__':
    unittest.main()
