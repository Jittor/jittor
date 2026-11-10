"""Tensor.addcmul_ and Tensor.lerp_ in-place semantics (schedulefree step)."""

from __future__ import print_function

import unittest

import torch


class TestTensorAddcmulLerp(unittest.TestCase):
    def test_addcmul_in_place(self):
        tensor = torch.tensor([1.0, 2.0])
        result = tensor.addcmul_(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]), value=0.5)
        self.assertEqual(result.tolist(), [5.0, 9.5])

    def test_lerp_in_place(self):
        tensor = torch.tensor([0.0, 10.0])
        result = tensor.lerp_(torch.tensor([10.0, 0.0]), 0.25)
        self.assertEqual(result.tolist(), [2.5, 7.5])


if __name__ == '__main__':
    unittest.main()
