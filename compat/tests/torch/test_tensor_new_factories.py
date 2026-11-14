"""Tensor.new_* factories create leaves with an explicit gradient policy."""
import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('name', ['new_empty', 'new_ones', 'new_zeros', 'new_full', 'new_tensor'])
@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('disabled', [False, True])
@jt.flag_scope(backend_fallback='error')
def test_new_factories_detach_and_honor_requires_grad(device, name, requires_grad, disabled):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    source = torch.ones((4, 1), dtype=torch.float64, device=device, requires_grad=True)
    args = (source,) if name == 'new_tensor' else ((4, 1), 2.) if name == 'new_full' else ((4, 1),)
    with torch.set_grad_enabled(not disabled):
        value = getattr(source, name)(*args, requires_grad=requires_grad)
    assert value.is_leaf and value.grad_fn is None
    assert value.requires_grad == requires_grad
    assert value.dtype == torch.float64 and value.device.type == device
    if not requires_grad:
        value.bernoulli_(.75)
        assert np.isin(value.cpu().numpy(), [0., 1.]).all()
    else:
        with pytest.raises(RuntimeError, match='leaf Variable'):
            value.bernoulli_(.75)
        value.sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), np.ones((4, 1)))
    assert source.grad is None


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@jt.flag_scope(backend_fallback='error')
def test_new_empty_defaults_allow_stochastic_depth_in_place(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    source = torch.ones((32, 3), device=device, requires_grad=True)
    mask = source.new_empty((32, 1))
    assert not mask.requires_grad and mask.is_leaf
    mask.bernoulli_(.8).div_(.8)
    output = source*mask
    output.sum().backward()
    np.testing.assert_array_equal(source.grad.cpu().numpy(),
                                  np.repeat(mask.cpu().numpy(), 3, axis=1))
