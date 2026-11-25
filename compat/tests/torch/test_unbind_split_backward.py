"""Tensor iteration shares native multi-output split backward ownership."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestUnbindSplitBackward(cu.JittorTestCase):
    def test_iteration_uses_one_multioutput_split_and_accumulates(self, device):
        with forbid_backend_fallbacks():
            x = torch.nn.Parameter(torch.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), device=device))
            # One split owns all row cotangents, rather than a full-input
            # setitem-gradient graph per independently indexed row.
            with patch.object(torch, 'split', wraps=torch.split) as split:
                rows = tuple(iter(x))
            self.assertEqual(split.call_count, 1)
            loss = sum((i + 1) * row.sum() for i, row in enumerate(rows))
            loss.backward()
            x.grad.sync()
            self.assertEqual(x.grad.location(), 'cpu' if device == 'cpu' else 'device')
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), np.repeat([[1.], [2.], [3.]], 4, axis=1))
            for i, row in enumerate(rows):
                self.assertEqual(tuple(row.shape), (4,))
                np.testing.assert_array_equal(row.detach().cpu().numpy(), np.arange(4, dtype=np.float32) + i * 4)

    def test_negative_axis_and_transposed_input_gradient(self, device):
        with forbid_backend_fallbacks():
            x = torch.nn.Parameter(torch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=device))
            outputs = x.transpose(0, 1).unbind(dim=-1)
            self.assertIsInstance(outputs, tuple)
            self.assertEqual(len(outputs), 2)
            (outputs[0].sum() + outputs[1].sum() * 2).backward()
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [[1., 1., 1.], [2., 2., 2.]])

    def test_empty_and_invalid_dimensions(self, device):
        with forbid_backend_fallbacks():
            empty = torch.empty((0, 3), device=device)
            self.assertEqual(empty.unbind(0), ())
            self.assertEqual(tuple(iter(empty)), ())
            self.assertEqual(len(empty.unbind(-1)), 3)
            for dim in (2, -3):
                with self.assertRaises(IndexError):
                    empty.unbind(dim)
            scalar = torch.tensor(1., device=device)
            with self.assertRaises(TypeError):
                iter(scalar)
            with self.assertRaises(IndexError):
                scalar.unbind(0)


instantiate_device_type_tests(TestUnbindSplitBackward, globals())
if __name__ == '__main__':
    unittest.main()
