"""CUDA saved input edges and custom call-context lifetime."""
import gc
import weakref
import numpy as np
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return 2 * x * dy


@pytest.mark.parametrize('mode,first_expected,second_expected', [
    ('single', [4., 6.], [2., 2.]),
    ('nonleaf', [16., 24.], [8., 8.]),
    ('chain', [32., 108.], [48., 108.]),
])
def test_saved_input_higher_derivatives(mode, first_expected, second_expected):
    x = torch.tensor([2., 3.], device='cuda', requires_grad=True)
    value = 2 * x if mode == 'nonleaf' else x
    if mode == 'chain':
        value = Square.apply(value)
    y = Square.apply(value)
    first, = torch.autograd.grad(y.sum(), [x], create_graph=True)
    second, = torch.autograd.grad(first.sum(), [x])
    np.testing.assert_array_equal(snapshot(first), first_expected)
    np.testing.assert_array_equal(snapshot(second), second_expected)


def test_saved_parameter_and_input_cross_derivatives():
    class Product(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)
            return (x * weight).sum()

        @staticmethod
        def backward(ctx, dy):
            x, weight = ctx.saved_tensors
            return weight * dy, x * dy

    x = torch.tensor([2., 3.], device='cuda', requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor([4., 5.], device='cuda'))
    gx, gw = torch.autograd.grad(Product.apply(x, weight), [x, weight], create_graph=True)
    cross_w, = torch.autograd.grad(gx.sum(), [weight], retain_graph=True)
    cross_x, = torch.autograd.grad(gw.sum(), [x])
    np.testing.assert_array_equal(snapshot(gx), [4., 5.])
    np.testing.assert_array_equal(snapshot(gw), [2., 3.])
    np.testing.assert_array_equal(snapshot(cross_w), [1., 1.])
    np.testing.assert_array_equal(snapshot(cross_x), [1., 1.])


def test_reused_factory_keeps_each_call_input_origin():
    factory = Square()
    x = torch.tensor([2.], device='cuda', requires_grad=True)
    z = torch.tensor([5.], device='cuda', requires_grad=True)
    y, other = factory(x), factory(z)
    gx, gz = torch.autograd.grad(y.sum() + other.sum(), [x, z], create_graph=True)
    xx, zz = torch.autograd.grad(gx.sum() + gz.sum(), [x, z])
    np.testing.assert_array_equal(snapshot(gx), [4.])
    np.testing.assert_array_equal(snapshot(gz), [10.])
    np.testing.assert_array_equal(snapshot(xx), [2.])
    np.testing.assert_array_equal(snapshot(zz), [2.])
    assert '_torch_caller_inputs' not in factory.__dict__


def test_unsaved_arguments_and_context_release_without_gc():
    refs = []

    class NoSave(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            refs.append(weakref.ref(ctx))
            return x + 1

        @staticmethod
        def backward(ctx, dy):
            return dy

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        x = torch.ones(2, device='cuda', requires_grad=True)
        input_ref = weakref.ref(x)
        y = NoSave.apply(x)
        del x
        assert input_ref() is None
        y.sum().backward()
        np.testing.assert_array_equal(snapshot(y), [2., 2.])
        del y
        assert refs[0]() is None
    finally:
        if was_enabled:
            gc.enable()


def test_saved_original_input_released_when_output_dies_without_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        x = torch.ones(2, device='cuda', requires_grad=True)
        input_ref = weakref.ref(x)
        y = Square.apply(x)
        del x
        assert input_ref() is not None
        np.testing.assert_array_equal(snapshot(y), [1., 1.])
        del y
        assert input_ref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_forward_exception_cleans_temporary_context_references():
    refs = []

    class Failing(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            refs.append(weakref.ref(ctx))
            ctx.save_for_backward(x)
            raise RuntimeError('intentional saved-input failure')

        @staticmethod
        def backward(ctx, dy):
            return dy

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        x = torch.ones(2, device='cuda', requires_grad=True)
        input_ref = weakref.ref(x)
        try:
            Failing.apply(x)
        except RuntimeError as error:
            assert 'intentional saved-input failure' in str(error)
            # A user-held traceback legitimately retains its frames.
            error.__traceback__ = None
        else:
            pytest.fail('forward should raise')
        del x
        assert input_ref() is None
        assert refs[0]() is None
        assert torch.is_grad_enabled()
    finally:
        if was_enabled:
            gc.enable()
