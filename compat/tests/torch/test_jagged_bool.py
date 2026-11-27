"""Jagged bool conversion changes values without rewriting packed metadata."""
import unittest
import numpy as np
import torch
from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestJaggedBool(cu.JittorTestCase):
    def test_holes_offsets_lengths_dtype_and_placement(self, device):
        with forbid_backend_fallbacks():
            values = torch.tensor([0., -2., 0., 4., 5.], device=device)
            offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=device)
            lengths = torch.tensor([1, 2], dtype=torch.int64, device=device)
            nested = torch.nested.nested_tensor_from_jagged(values, offsets, lengths=lengths)
            result = nested.bool()
            self.assertTrue(result.dtype == torch.bool)
            self.assertTrue(nested.dtype == values.dtype)
            self.assertTrue(result.offsets().dtype == torch.int64)
            self.assertTrue(result.lengths().dtype == torch.int64)
            for value in (result.values(), result.offsets(), result.lengths()):
                value.sync()
                self.assertEqual(value.location(), 'cpu' if device == 'cpu' else 'device')
            np.testing.assert_array_equal(result.values().cpu().numpy(), [False, True, False, True, True])
            np.testing.assert_array_equal(result.offsets().cpu().numpy(), [0, 2, 5])
            np.testing.assert_array_equal(result.lengths().cpu().numpy(), [1, 2])
            rows = result.unbind()
            np.testing.assert_array_equal(rows[0].cpu().numpy(), [False])
            np.testing.assert_array_equal(rows[1].cpu().numpy(), [False, True])
            self.assertIs(result.bool(), result)

    def test_without_lengths_preserves_offsets(self, device):
        with forbid_backend_fallbacks():
            values = torch.tensor([0, 1, -1], device=device)
            offsets = torch.tensor([0, 1, 3], dtype=torch.int64, device=device)
            result = torch.nested.nested_tensor_from_jagged(values, offsets).bool()
            self.assertIsNone(result.lengths())
            np.testing.assert_array_equal(result.offsets().cpu().numpy(), [0, 1, 3])
            np.testing.assert_array_equal(result.values().cpu().numpy(), [False, True, True])


instantiate_device_type_tests(TestJaggedBool, globals())
if __name__ == '__main__':
    unittest.main()
