"""Upper-triangular coordinates match row-major NumPy reference on CPU/CUDA."""
import unittest

import numpy as np
import torch


class TestTriuIndices(unittest.TestCase):
    def test_coordinates(self):
        shapes = ((3, 5, 0), (5, 3, 1), (4, 4, -2),
                  (3, 4, 7), (0, 3, 0), (3, 0, 0))
        for device in ("cpu", "cuda"):
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
                        self.assertEqual(result.dtype, dtype)
                        np.testing.assert_array_equal(result.cpu().numpy(), expected)

    def test_default_and_invalid(self):
        self.assertEqual(torch.triu_indices(2, 2).dtype, torch.int64)
        for row, col in ((-1, 2), (2, -1)):
            with self.assertRaises(RuntimeError):
                torch.triu_indices(row, col)
        with self.assertRaises(RuntimeError):
            torch.triu_indices(2, 2, dtype=torch.float32)

    def test_cuda_index_assignment(self):
        rows, cols = torch.triu_indices(4, 4, 1, device="cuda")
        matrix = torch.zeros((4, 4), device="cuda")
        matrix[rows, cols] = 1
        np.testing.assert_array_equal(matrix.cpu().numpy(),
                                      np.triu(np.ones((4, 4)), 1))
