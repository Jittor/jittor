"""A custom Function keeps its backward edge when forward returns a detached tensor."""
import unittest

import numpy as np
import torch


class TestDetachedCustomFunctionOutput(unittest.TestCase):
    def test_detached_forward_output_uses_custom_backward(self):
        class BackwardBoundary(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.detach()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        source = torch.tensor([2.0, 3.0], requires_grad=True)
        result = BackwardBoundary.apply(source)
        self.assertTrue(result.requires_grad)
        self.assertIsNotNone(result.grad_fn)
        result.retain_grad()
        (result * 3).sum().backward()
        np.testing.assert_allclose(source.grad.numpy(), [3.0, 3.0])
        np.testing.assert_allclose(result.grad.numpy(), [3.0, 3.0])
    def test_no_grad_scope_keeps_custom_output_detached(self):
        class BackwardBoundary(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.detach()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        source = torch.tensor([2.0], requires_grad=True)
        with torch.no_grad():
            result = BackwardBoundary.apply(source)
        self.assertFalse(result.requires_grad)

    def test_integer_custom_output_remains_non_differentiable(self):
        class IntegerBoundary(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.to(dtype=torch.int32)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        source = torch.tensor([2.0], requires_grad=True)
        result = IntegerBoundary.apply(source)
        self.assertFalse(result.requires_grad)
