"""torch.autograd.gradcheck checks analytic gradients against finite differences."""
import unittest

import torch


class TestGradcheck(unittest.TestCase):
    def test_quadratic_scalar_output_passes(self):
        def func(x):
            return (x * x).sum()

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        x.requires_grad = True
        self.assertTrue(torch.autograd.gradcheck(func, (x,)))

    def test_constant_input_is_not_differentiated(self):
        def func(x, target):
            return ((x - target.double()) ** 2).mean()

        x = torch.tensor([0.5, 1.5], dtype=torch.float64)
        x.requires_grad = True
        target = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.assertTrue(torch.autograd.gradcheck(func, (x, target)))

    def test_tuple_output_passes(self):
        def func(x):
            return x.sum(), (x * x).sum()

        x = torch.tensor([0.5, -1.5, 2.0], dtype=torch.float64)
        x.requires_grad = True
        self.assertTrue(torch.autograd.gradcheck(func, (x,)))

    def test_mismatch_returns_false_without_raising(self):
        def func(x):
            return (x ** 3).sum()

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x.requires_grad = True
        # A central difference with eps=1 is not the derivative of x**3.
        self.assertFalse(torch.autograd.gradcheck(func, (x,), eps=1.0, raise_exception=False))

    def test_mismatch_raises_by_default(self):
        def func(x):
            return (x ** 3).sum()

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x.requires_grad = True
        with self.assertRaises(RuntimeError):
            torch.autograd.gradcheck(func, (x,), eps=1.0)

    def test_unsupported_options_raise(self):
        def func(x):
            return (x * x).sum()

        x = torch.tensor([1.0], dtype=torch.float64)
        x.requires_grad = True
        with self.assertRaises(NotImplementedError):
            torch.autograd.gradcheck(func, (x,), check_forward_ad=True)
        with self.assertRaises(NotImplementedError):
            torch.autograd.gradcheck(func, (x,), masked=True)


class TestNonfiniteGradcheck(unittest.TestCase):
    def test_nan_backward_cannot_pass_a_finite_forward(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * float("nan")

        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        self.assertFalse(torch.autograd.gradcheck(
            BadGradient.apply, (x,), raise_exception=False))
        with self.assertRaises(RuntimeError):
            torch.autograd.gradcheck(BadGradient.apply, (x,))

    def test_matching_nan_jacobians_do_not_pass(self):
        x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        self.assertFalse(torch.autograd.gradcheck(
            lambda value: value * float("nan"), (x,), raise_exception=False))
