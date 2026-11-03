"""A public Linear subclass may initialize only Module and own its parameters."""

import unittest

import numpy as np
import torch
from torch import nn


class TestLinearPublicInheritance(unittest.TestCase):
    def test_skip_linear_constructor_owns_only_custom_parameters(self):
        class Custom(nn.Linear):
            def __init__(self):
                super(nn.Linear, self).__init__()
                self.projection = nn.Parameter(torch.tensor(
                    [[1., 2., -1.], [-2., 1., 3.]]))

            def forward(self, value):
                return value @ self.projection.t()

        model = Custom()
        self.assertFalse(hasattr(model, "weight"))
        self.assertFalse(hasattr(model, "bias"))
        self.assertEqual(list(model.state_dict()), ["projection"])
        self.assertEqual([name for name, _ in model.named_parameters()], ["projection"])
        value = torch.tensor([[2., -1., 3.], [1., 4., -2.]], requires_grad=True)
        actual = model(value)
        np.testing.assert_array_equal(actual.detach().numpy(), [[-3., 4.], [11., -4.]])
        actual.sum().backward()
        np.testing.assert_array_equal(value.grad.numpy(), [[-1., 3., 2.], [-1., 3., 2.]])
        np.testing.assert_array_equal(model.projection.grad.numpy(), [[3., 3., 1.], [3., 3., 1.]])

    def test_normal_constructor_and_super_forward(self):
        class Custom(nn.Linear):
            def __init__(self):
                super().__init__(3, 2, bias=False)

            def forward(self, value):
                return super().forward(value) + 1

        model = Custom()
        self.assertEqual(list(model.state_dict()), ["weight"])
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 2., -1.], [-2., 1., 3.]]))
        value = torch.tensor([[2., -1., 3.]], requires_grad=True)
        actual = model(value)
        np.testing.assert_array_equal(actual.detach().numpy(), [[-2., 5.]])
        actual.sum().backward()
        np.testing.assert_array_equal(value.grad.numpy(), [[-1., 3., 2.]])
        np.testing.assert_array_equal(model.weight.grad.numpy(), [[2., -1., 3.], [2., -1., 3.]])

    def test_frontend_linear_keeps_native_subtype(self):
        import jittor as jt

        model = nn.Linear(3, 2)
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(model, jt.nn.Linear)
        self.assertEqual(tuple(model.weight.shape), (2, 3))
