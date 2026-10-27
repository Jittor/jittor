"""Upper-triangular coordinates match row-major NumPy reference on CPU/CUDA."""
import unittest

from _helpers.device_types import instantiate_device_type_tests

import numpy as np
import torch


class TestTriuIndices(unittest.TestCase):
    def test_coordinates(self, device):
        shapes = ((3, 5, 0), (5, 3, 1), (4, 4, -2),
                  (3, 4, 7), (0, 3, 0), (3, 0, 0))
        for dtype in (torch.int32, torch.int64):
            for row, col, offset in shapes:
                with self.subTest(device=device, dtype=dtype,
                                  shape=(row, col), offset=offset):
                    result = torch.triu_indices(row, col, offset,
                                                dtype=dtype, device=device)
                    expected = np.asarray(
                        [(i, j) for i in range(row) for j in range(col)
                         if j - i >= offset], dtype=np.int64,
                    ).reshape(-1, 2).T
                    self.assertEqual(result.device.type, device)
                    self.assertIs(result.dtype, dtype)
                    np.testing.assert_array_equal(result.cpu().numpy(), expected)

    def test_default_and_invalid(self, device):
        self.assertIs(torch.triu_indices(2, 2, device=device).dtype, torch.int64)
        for row, col in ((-1, 2), (2, -1)):
            with self.assertRaises(RuntimeError):
                torch.triu_indices(row, col, device=device)
        with self.assertRaises(RuntimeError):
            torch.triu_indices(2, 2, dtype=torch.float32, device=device)

    def test_index_assignment(self, device):
        rows, cols = torch.triu_indices(4, 4, 1, device=device)
        matrix = torch.zeros((4, 4), device=device)
        matrix[rows, cols] = 1
        np.testing.assert_array_equal(matrix.cpu().numpy(),
                                      np.triu(np.ones((4, 4)), 1))


instantiate_device_type_tests(TestTriuIndices, globals(), only_for=("cpu", "cuda"))
