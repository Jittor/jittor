"""CUDA parity for the in-place Tensor diagonal fill used by ms-swift InfoNCE."""
import unittest

import numpy as np
import torch


class TestFillDiagonal(unittest.TestCase):
    def test_values_identity_and_wrap(self):
        cases = (((3, 3), False), ((2, 4), False), ((7, 3), False),
                 ((7, 3), True), ((2, 2, 2), False), ((0, 3), False),
                 ((3, 0), True))
        for device in ("cpu", "cuda"):
            for shape, wrap in cases:
                with self.subTest(device=device, shape=shape, wrap=wrap):
                    value = torch.zeros(shape, device=device)
                    returned = value.fill_diagonal_(5, wrap=wrap)
                    self.assertIs(returned, value)
                    self.assertEqual(value.device.type, device)
                    expected = np.zeros(shape, dtype=np.float32)
                    if len(shape) == 2 and wrap and shape[0] > shape[1] and shape[1]:
                        linear = np.arange(0, expected.size, shape[1] + 1)
                        expected[linear // shape[1], linear % shape[1]] = 5
                    else:
                        diagonal = np.arange(min(shape))
                        expected[(diagonal,) * len(shape)] = 5
                    np.testing.assert_array_equal(value.cpu().numpy(), expected)

    def test_invalid_shapes(self):
        for shape in ((3,), (2, 3, 2)):
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError):
                    torch.zeros(shape, device="cuda").fill_diagonal_(1)

    def test_nonleaf_gradient(self):
        leaf = torch.arange(9, device="cuda", dtype=torch.float32).requires_grad_()
        value = leaf.reshape(3, 3) * 2
        self.assertIs(value.fill_diagonal_(0), value)
        value.sum().backward()
        np.testing.assert_array_equal(
            leaf.grad.cpu().numpy(),
            np.array([0, 2, 2, 2, 0, 2, 2, 2, 0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
