import unittest

import numpy as np
import torch
import jittor as jt


class TestTorchNumPyIndex(unittest.TestCase):
    """PyTorch accepts a NumPy integer/bool array as an advanced index; the
    Torch frontend normalizes it to a Jittor array before dispatch so the
    values and the gather gradient match. Backend-agnostic (CPU/CUDA)."""

    def test_numpy_integer_array_index_values_and_gradient(self):
        self.assertIsNot(torch, jt)
        self.assertIs(torch.Tensor._frontend_backend, jt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for dtype in (np.int32, np.int64):
            with self.subTest(dtype=dtype):
                source = torch.tensor([10.0, 20.0, 30.0], device=device,
                                      requires_grad=True)
                indices = np.array([2, 0], dtype=dtype)
                selected = source[indices]
                np.testing.assert_array_equal(
                    selected.detach().cpu().numpy(), [30.0, 10.0])
                selected.sum().backward()
                np.testing.assert_array_equal(
                    source.grad.detach().cpu().numpy(), [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
