"""CUDA custom Function forward owns no internal gradient recording."""
import numpy as np
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot


def test_forward_no_grad_preserves_inputs_and_custom_backward_edge():
    observed = {}

    class Accumulator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            observed['enabled'] = torch.is_grad_enabled()
            observed['input_requires_grad'] = x.requires_grad
            observed['needs'] = ctx.needs_input_grad
            value = torch.zeros((8, 8), device='cuda')
            for _ in range(16):
                value.add_(x @ x.t())
            observed['accumulator_requires_grad'] = value.requires_grad
            ctx.save_for_backward(x)
            return value

        @staticmethod
        def backward(ctx, dy):
            x, = ctx.saved_tensors
            return ((dy + dy.t()) @ x) * 16

    x = torch.ones((8, 4), device='cuda', requires_grad=True)
    y = Accumulator.apply(x)
    assert observed == dict(enabled=False, input_requires_grad=True,
                            needs=(True,), accumulator_requires_grad=False)
    assert y.requires_grad and torch.is_grad_enabled()
    dx, = torch.autograd.grad(y.sum(), [x])
    np.testing.assert_array_equal(snapshot(y), np.full((8, 8), 64.))
    np.testing.assert_array_equal(snapshot(dx), np.full((8, 4), 256.))


def test_explicit_forward_enable_grad_restores_mode_and_custom_derivative():
    observed = []

    class Explicit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            observed.append(torch.is_grad_enabled())
            with torch.enable_grad():
                observed.append(torch.is_grad_enabled())
                y = x * x
            observed.append(torch.is_grad_enabled())
            return y

        @staticmethod
        def backward(ctx, dy):
            return 3 * dy

    x = torch.tensor([2., 4.], device='cuda', requires_grad=True)
    y = Explicit.apply(x)
    dx, = torch.autograd.grad(y.sum(), [x])
    assert observed == [False, True, False]
    np.testing.assert_array_equal(snapshot(y), [4., 16.])
    np.testing.assert_array_equal(snapshot(dx), [3., 3.])
    assert torch.is_grad_enabled()


def test_outer_no_grad_does_not_attach_custom_backward():
    observed = []

    class NoGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            observed.append((torch.is_grad_enabled(), x.requires_grad))
            return x * x

        @staticmethod
        def backward(ctx, dy):
            raise AssertionError('backward must not run')

    x = torch.ones(2, device='cuda', requires_grad=True)
    with torch.no_grad():
        y = NoGrad.apply(x)
        assert not y.requires_grad
        assert not torch.is_grad_enabled()
    assert observed == [(False, True)]
    np.testing.assert_array_equal(snapshot(y), [1., 1.])
    assert torch.is_grad_enabled()


def test_forward_exception_restores_enclosing_grad_mode():
    class Failing(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            assert not torch.is_grad_enabled()
            raise RuntimeError('intentional forward failure')

        @staticmethod
        def backward(ctx, dy):
            return dy

    x = torch.ones(2, device='cuda', requires_grad=True)
    with pytest.raises(RuntimeError, match='intentional forward failure'):
        Failing.apply(x)
    assert torch.is_grad_enabled()
    with torch.no_grad():
        with pytest.raises(RuntimeError, match='intentional forward failure'):
            Failing.apply(x)
        assert not torch.is_grad_enabled()
