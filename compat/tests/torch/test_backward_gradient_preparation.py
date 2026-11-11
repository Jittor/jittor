"""Ordinary backward prepares fresh gradients without changing AD semantics."""
from unittest.mock import patch

import numpy as np
import torch
import jittor as jt

from _helpers.device_types import instantiate_device_type_tests


class HalfGradient(jt.Function):
    def execute(self, value):
        return value * 2

    def grad(self, gradient):
        return (gradient * 2).cast('float16')


class TestBackwardGradientPreparation:
    def test_missing_target_and_stopped_gradient(self, device):
        first = torch.tensor([2., 3.], device=device, requires_grad=True)
        unused = torch.tensor([7.], device=device, requires_grad=True)
        gradients, prepared = jt.core._grad_optional_prepare(
            first.square().sum(), [first, unused], False, False)
        assert prepared and gradients[1] is None
        assert not gradients[0].requires_grad
        np.testing.assert_array_equal(gradients[0].cpu().numpy(), [4., 6.])

    def test_public_dtype_fallback(self, device):
        value = torch.tensor([1., 2.], device=device, requires_grad=True)
        results = []
        native = jt.core._grad_optional_prepare

        def observe(*args):
            result = native(*args)
            results.append(result[1])
            return result

        with patch.object(jt.core, '_grad_optional_prepare', observe):
            HalfGradient.apply(value).sum().backward()
        assert results == [False]
        assert value.grad.dtype == value.dtype
        assert not value.grad.requires_grad
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [2., 2.])

    def test_scalar_gradient_and_fresh_calls(self, device):
        value = torch.tensor(3., device=device, requires_grad=True)
        value.square().backward()
        retained = value.grad
        assert tuple(retained.shape) == tuple(value.shape) == ()
        assert not retained.requires_grad and float(retained) == 6.
        value.grad = None
        with torch.no_grad():
            value.fill_(4.)
        value.square().backward()
        assert value.grad is not retained and float(value.grad) == 8.
        assert float(retained) == 6.

    def test_second_order_backward(self, device):
        value = torch.tensor([2., 3.], device=device, requires_grad=True)
        value.pow(3).sum().backward(create_graph=True)
        first = value.grad
        assert first.requires_grad
        second, = torch.autograd.grad(first.sum(), value)
        np.testing.assert_array_equal(second.cpu().numpy(), [12., 18.])

    def test_hooks_and_retained_graph_accumulate(self, device):
        value = torch.tensor([2., 3.], device=device, requires_grad=True)
        observed = []
        handle = value.register_hook(lambda gradient: gradient * 3)
        value.register_post_accumulate_grad_hook(
            lambda parameter: observed.append(parameter.grad.cpu().numpy().copy()))
        loss = value.square().sum()
        loss.backward(retain_graph=True)
        retained = value.grad
        loss.backward()
        assert value.grad is retained
        np.testing.assert_array_equal(observed[0], [12., 18.])
        np.testing.assert_array_equal(observed[1], [24., 36.])
        handle.remove()

    def test_tracing_uses_original_gradient_route(self, device):
        from jittor._runtime import step_capture
        value = torch.tensor([3.], device=device, requires_grad=True)
        with patch.object(step_capture, 'tracing', return_value=True), \
                patch.object(jt.core, '_grad_optional_prepare') as native:
            value.square().sum().backward()
        native.assert_not_called()
        assert not value.grad.requires_grad
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [6.])


instantiate_device_type_tests(TestBackwardGradientPreparation, globals(), only_for=('cpu', 'cuda'))
