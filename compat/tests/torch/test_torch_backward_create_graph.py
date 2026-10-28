"""Public backward must preserve requested higher-order differentiation."""
import unittest
import numpy as np
import torch
import jittor as jt

@unittest.skipUnless(jt.has_cuda, "requires a real CUDA device")
class TestBackwardCreateGraph(unittest.TestCase):
    def setUp(self):
        scope = jt.runtime.scope(use_cuda=1, backend_fallback="error")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_public_backward_retains_graph_for_second_derivative(self):
        x = torch.tensor([.5, 1.5, 2.], device="cuda", requires_grad=True)
        torch.autograd.backward((x ** 3).sum(), create_graph=True)
        first = x.grad
        self.assertIsNotNone(first)
        second, = torch.autograd.grad(first.sum(), (x,))
        np.testing.assert_allclose(first.detach().cpu().numpy(), [0.75, 6.75, 12.], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(second.detach().cpu().numpy(), [3., 9., 12.], rtol=1e-5, atol=1e-6)
        x.grad = None

    def test_weighted_public_backward_second_derivative(self):
        x = torch.tensor([.5, 1.5], device="cuda", requires_grad=True)
        weight = torch.tensor([2., 3.], device="cuda")
        torch.autograd.backward(x ** 3, grad_tensors=weight, create_graph=True)
        second, = torch.autograd.grad(x.grad.sum(), (x,))
        np.testing.assert_allclose(second.detach().cpu().numpy(), [6., 27.], rtol=1e-5, atol=1e-6)
        x.grad = None
