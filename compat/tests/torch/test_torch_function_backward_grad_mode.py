"""Custom backward callbacks inherit the caller's create_graph request."""
import unittest
import numpy as np
import torch
import jittor as jt

@unittest.skipUnless(jt.has_cuda, "requires a real CUDA device")
class TestFunctionBackwardGradMode(unittest.TestCase):
    def setUp(self):
        scope = jt.runtime.scope(use_cuda=1, backend_fallback="error")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def check_entry(self, entry, create_graph):
        seen = []
        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x
            @staticmethod
            def backward(ctx, g):
                seen.append(torch.is_grad_enabled())
                x, = ctx.saved_tensors
                return 2 * x * g
        x = torch.tensor([.5, 1.5], device="cuda", requires_grad=True)
        y = Square.apply(x).sum()
        if entry == "grad":
            value, = torch.autograd.grad(y, (x,), create_graph=create_graph)
        elif entry == "public":
            torch.autograd.backward(y, create_graph=create_graph)
            value = x.grad
        else:
            y.backward(create_graph=create_graph)
            value = x.grad
        self.assertEqual(seen, [create_graph])
        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(value.device.type, "cuda")
        np.testing.assert_allclose(value.detach().cpu().numpy(), [1., 3.], rtol=1e-5, atol=1e-6)
        x.grad = None

    def test_first_order_callbacks_disable_grad(self):
        for entry in ("grad", "public", "tensor"):
            with self.subTest(entry=entry):
                self.check_entry(entry, False)

    def test_higher_order_callbacks_enable_grad(self):
        for entry in ("grad", "public", "tensor"):
            with self.subTest(entry=entry):
                self.check_entry(entry, True)

    def test_exception_restores_ambient_mode_and_request(self):
        seen = []
        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * x
            @staticmethod
            def backward(ctx, g):
                seen.append(torch.is_grad_enabled())
                raise RuntimeError("intentional custom backward failure")
        x = torch.tensor(2., device="cuda", requires_grad=True)
        y = Failing.apply(x)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "intentional custom backward failure"):
                torch.autograd.grad(y, (x,), create_graph=True)
            self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(seen, [True])
        self.check_entry("grad", False)

    def test_nested_grad_request_restores_outer_callback(self):
        seen = []
        class Inner(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * x
            @staticmethod
            def backward(ctx, g):
                seen.append(("inner", torch.is_grad_enabled()))
                return g
        class Outer(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x
            @staticmethod
            def backward(ctx, g):
                seen.append(("before", torch.is_grad_enabled()))
                z = torch.tensor(2., device="cuda", requires_grad=True)
                torch.autograd.grad(Inner.apply(z), (z,), create_graph=False)
                seen.append(("after", torch.is_grad_enabled()))
                x, = ctx.saved_tensors
                return 2 * x * g
        x = torch.tensor(3., device="cuda", requires_grad=True)
        value, = torch.autograd.grad(Outer.apply(x), (x,), create_graph=True)
        self.assertEqual(seen, [("before", True), ("inner", False), ("after", True)])
        np.testing.assert_allclose(value.detach().cpu().numpy(), 6., rtol=1e-5)
