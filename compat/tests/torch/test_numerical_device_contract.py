"""Activation and distance regressions execute on the requested real device."""
import numpy as np
import pytest
import torch
from _device_assertions import require_residency


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.mark.parametrize('name', ['elu', 'selu', 'celu'])
def test_activation_values_and_gradients_on_device(name, device):
    values = np.array([-np.inf, -1., -1e-7, 0., 1e-7, np.inf], dtype=np.float32)
    x = torch.tensor(values, device=device, requires_grad=True)
    y = getattr(torch.nn.functional, name)(x)
    y.sum().backward()
    with np.errstate(over='ignore', invalid='ignore'):
        expected = np.where(values > 0, values, np.expm1(values))
        grad = np.where(values > 0, 1., np.exp(values))
    if name == 'selu':
        scale, alpha = 1.0507009873554805, 1.6732632423543772
        expected = scale * np.where(values > 0, values, alpha * expected)
        grad = scale * np.where(values > 0, 1., alpha * grad)
    for tensor in (x, y, x.grad):
        require_residency(tensor, device)
    np.testing.assert_allclose(y.detach().cpu().numpy(), expected, rtol=3e-6, atol=1e-12)
    np.testing.assert_allclose(x.grad.cpu().numpy(), grad, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize('p', [0., float('inf')])
def test_cdist_zero_and_infinity_backward_on_device(p, device):
    left = torch.tensor([[1., -1.]], device=device, requires_grad=True)
    right = torch.tensor([[0., 0.]], device=device, requires_grad=True)
    result = torch.cdist(left, right, p=p)
    result.sum().backward()
    expected = np.array([[0., 0.] if p == 0 else [1., -1.]])
    for tensor in (result, left.grad, right.grad):
        require_residency(tensor, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), [[2. if p == 0 else 1.]])
    np.testing.assert_array_equal(left.grad.cpu().numpy(), expected)
    np.testing.assert_array_equal(right.grad.cpu().numpy(), -expected)


def test_cdist_shared_embedding_gradient_at_coincident_pairs(device):
    x = torch.tensor([[0., 0.], [3., 4.], [3., 4.]], device=device, requires_grad=True)
    distance = torch.cdist(x, x, p=2.)
    distance.sum().backward()

    for tensor in (distance, x.grad):
        require_residency(tensor, device)
    np.testing.assert_array_equal(distance.detach().cpu().numpy(), [[0., 5., 5.], [5., 0., 0.], [5., 0., 0.]])
    np.testing.assert_allclose(x.grad.cpu().numpy(), [[-2.4, -3.2], [1.2, 1.6], [1.2, 1.6]], rtol=1e-6)


@pytest.mark.parametrize("dim", [None, 0, (0,)])
def test_clamped_l2_norm_has_finite_gradient_at_zero(dim, device):
    value = torch.zeros((2, 3), device=device, dtype=torch.float32, requires_grad=True)
    norm = value.norm(dim=dim)
    (value / norm.clamp(min=1e-4)).sum().backward()

    for tensor in (value, norm, value.grad):
        require_residency(tensor, device)
    np.testing.assert_array_equal(norm.detach().cpu().numpy(), np.zeros(tuple(norm.shape)))
    np.testing.assert_array_equal(value.grad.cpu().numpy(), np.full((2, 3), 1e4))
