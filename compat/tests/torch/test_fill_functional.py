"""Device and autograd regression for the public torch.fill APIs."""
import numpy as np
import pytest
import torch

from _device_assertions import require_residency


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
def test_fill_does_not_mutate_and_keeps_shape_dtype(device, dtype):
    x = torch.tensor([0, 1, 0], dtype=dtype, device=device)
    before = x.detach().cpu().numpy().copy()
    actual = torch.fill(x, 1)
    require_residency(actual, device)
    assert actual.shape == x.shape and actual.dtype == dtype
    np.testing.assert_array_equal(x.detach().cpu().numpy(), before)
    np.testing.assert_array_equal(actual.detach().cpu().numpy(), np.ones(3))


def test_fill_gradients_and_ignores_nan_inputs(device):
    x = torch.tensor([float('nan'), float('inf')], dtype=torch.float32, device=device, requires_grad=True)
    value = torch.tensor(2.5, dtype=torch.float32, device=device, requires_grad=True)
    result = torch.fill(x, value)
    require_residency(result, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), [2.5, 2.5])
    result.sum().backward()
    require_residency(x.grad, device)
    require_residency(value.grad, device)
    np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [0., 0.])
    assert value.grad.item() == 2.


def test_fill_scalar_and_empty_shapes(device):
    scalar = torch.fill(torch.tensor(3., device=device), 4.)
    empty = torch.fill(torch.empty((0, 2), device=device), 4.)
    require_residency(scalar, device)
    require_residency(empty, device)
    assert tuple(scalar.shape) == () and scalar.item() == 4.
    assert tuple(empty.shape) == (0, 2) and empty.numel() == 0


def test_fill_inplace_and_invalid_value(device):
    x = torch.zeros(2, device=device)
    result = torch.fill_(x, 3.)
    assert result is x
    np.testing.assert_array_equal(x.cpu().numpy(), [3., 3.])
    with pytest.raises(RuntimeError, match='0-dimension'):
        torch.fill(x, torch.ones(2, device=device))
    with pytest.raises(TypeError):
        torch.fill(x, 3, out=x)
