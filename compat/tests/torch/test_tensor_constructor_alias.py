"""Tensor(source) preserves storage and the grad-mode-dependent graph alias."""

import contextlib
import ctypes

import numpy as np
import pytest
import torch

from _rng_init_support import _array, require_rng_device


@pytest.fixture(autouse=True)
def forbid_fallbacks():
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with forbid_backend_fallbacks():
            yield
            jt.sync_all(True)
    else:
        yield


def assert_physical_device(value, device):
    assert value.device.type == device
    if hasattr(torch, '_torch_compat_install_context'):
        value.sync()
        assert value.location() == ('device' if device == 'cuda' else 'cpu')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('intermediate', [False, True])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('disabled', [False, True])
def test_tensor_source_alias(device, dtype, requires_grad, intermediate, transposed, disabled):
    require_rng_device(device)
    guard = contextlib.nullcontext()
    if hasattr(torch, '_torch_compat_install_context'):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        guard = forbid_backend_fallbacks()
    with guard:
        leaf = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype,
                            device=device, requires_grad=requires_grad)
        source = leaf * 2 if intermediate else leaf
        if transposed:
            source = source.transpose(0, 1)
        with torch.set_grad_enabled(not disabled):
            alias = torch.Tensor(source)
        assert type(alias) is torch.Tensor and alias is not source
        assert alias.dtype == dtype and alias.device == source.device
        assert alias.requires_grad == requires_grad
        assert alias.is_leaf == (disabled or not requires_grad)
        assert (alias.grad_fn is None) == alias.is_leaf
        assert alias.shape == source.shape and alias.stride() == source.stride()
        assert alias.data_ptr() == source.data_ptr()
        if hasattr(torch, '_torch_compat_install_context'):
            alias.sync()
            assert alias.location() == ('device' if device == 'cuda' else 'cpu')
        if device == 'cuda':
            ordinal = ctypes.c_int(-1)
            driver = ctypes.CDLL('libcuda.so.1')
            assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                               ctypes.c_void_p(alias.data_ptr())) == 0
            assert ordinal.value == alias.device.index
        if requires_grad:
            (alias * 3).sum().backward()
            if disabled:
                assert leaf.grad is None
                assert alias.grad is None
            else:
                np.testing.assert_array_equal(_array(leaf.grad), np.full((2, 3), 6. if intermediate else 3.))
        with torch.no_grad():
            alias.fill_(9.)
        np.testing.assert_array_equal(_array(source), np.full(tuple(source.shape), 9.))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('variant', ['inherited', 'enable', 'toggle', 'detach', 'clone', 'alias', 'retain'])
def test_unrecorded_alias_gradient_edge(device, variant):
    require_rng_device(device)
    source = torch.tensor([1., 2.], requires_grad=True, device=device)
    with torch.no_grad():
        alias = torch.Tensor(source)
    target = alias
    if variant == 'enable':
        target.requires_grad_(True)
    elif variant == 'toggle':
        target.requires_grad_(False).requires_grad_(True)
    elif variant == 'detach':
        target = alias.detach().requires_grad_(True)
    elif variant == 'clone':
        target = alias.clone()
    elif variant == 'alias':
        target = torch.Tensor(alias)
    elif variant == 'retain':
        target.retain_grad()
    assert_physical_device(source, device)
    assert_physical_device(target, device)
    loss = (target * 3).sum()
    source_gradient, target_gradient = torch.autograd.grad(
        loss, (source, target), allow_unused=True, retain_graph=True)
    assert source_gradient is None
    if variant in ('inherited', 'retain'):
        assert target_gradient is None
    else:
        assert_physical_device(target_gradient, device)
        np.testing.assert_array_equal(_array(target_gradient), [3., 3.])
    loss.backward()
    assert source.grad is None
    if variant in ('enable', 'toggle'):
        np.testing.assert_array_equal(_array(alias.grad), [3., 3.])
    else:
        assert alias.grad is None
    if variant == 'detach':
        np.testing.assert_array_equal(_array(target.grad), [3., 3.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_alias_activation_does_not_connect_existing_graph(device):
    require_rng_device(device)
    source = torch.tensor([1., 2.], requires_grad=True, device=device)
    with torch.no_grad():
        alias = torch.Tensor(source)
    old_loss = (alias * 3).sum()
    assert old_loss.requires_grad and not old_loss.is_leaf and old_loss.grad_fn is not None
    alias.requires_grad_(True)
    new_loss = (alias * 5).sum()
    assert torch.autograd.grad(old_loss, (source, alias), allow_unused=True) == (None, None)
    source_gradient, alias_gradient = torch.autograd.grad(new_loss, (source, alias), allow_unused=True)
    assert source_gradient is None
    assert_physical_device(alias_gradient, device)
    np.testing.assert_array_equal(_array(alias_gradient), [5., 5.])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_unrecorded_alias_cannot_be_a_backward_root(device):
    require_rng_device(device)
    source = torch.tensor([1., 2.], requires_grad=True, device=device)
    with torch.no_grad():
        alias = torch.Tensor(source)
    gradient = torch.ones_like(alias)
    with pytest.raises(RuntimeError, match='does not require grad.*does not have a grad_fn'):
        torch.autograd.grad(alias, source, gradient)
    with pytest.raises(RuntimeError, match='does not require grad.*does not have a grad_fn'):
        alias.backward(gradient)

