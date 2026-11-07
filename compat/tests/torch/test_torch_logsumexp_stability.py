"""Fixed-order Torch CUDA logsumexp retains value and autograd semantics."""
import numpy as np
import pytest
import sys
import torch


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')


@pytest.fixture(autouse=True)
def _no_jittor_backend_fallbacks():
    jt = sys.modules.get('jittor')
    before = jt.core.backend_fallback_count() if jt is not None else None
    yield
    if jt is not None:
        assert jt.core.backend_fallback_count() == before


@pytest.mark.cuda
def test_large_logsumexp_repeats_and_has_correct_first_and_second_gradients():
    _cuda()
    data = np.linspace(-4., 4., 65536, dtype=np.float32).reshape(2, 32768)
    direction = np.cos(np.arange(data.size, dtype=np.float32) * np.float32(.003)).reshape(data.shape)
    x = torch.tensor(data, device='cuda', requires_grad=True)
    observations = [torch.logsumexp(x, -1).detach().cpu().numpy().copy() for _ in range(12)]
    for value in observations[1:]:
        np.testing.assert_array_equal(value, observations[0])
    y = torch.logsumexp(x, -1)
    assert tuple(y.shape) == (2,)
    assert tuple(torch.logsumexp(x, 1, keepdim=True).shape) == (2, 1)
    stacked = torch.stack([x.detach(), x.detach()])
    assert tuple(torch.logsumexp(stacked, -1).shape) == (2, 2)
    first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    v = torch.tensor(direction, device='cuda')
    second = torch.autograd.grad((first * v).sum(), x)[0]
    wide = data.astype(np.float64)
    maximum = np.max(wide, axis=-1, keepdims=True)
    exponent = np.exp(wide - maximum)
    probability = exponent / exponent.sum(axis=-1, keepdims=True)
    expected_y = (maximum + np.log(exponent.sum(axis=-1, keepdims=True))).reshape(-1)
    expected_second = probability * (direction - (probability * direction).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(y.detach().cpu().numpy(), expected_y, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(first.detach().cpu().numpy(), probability, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(second.detach().cpu().numpy(), expected_second, rtol=1e-4, atol=1e-7)


@pytest.mark.cuda
def test_large_logsumexp_preserves_nan_and_infinity_semantics():
    _cuda()
    data = np.zeros((5, 32768), dtype=np.float32)
    data[0] = -np.inf
    data[1, 0] = np.inf
    data[2, 0] = np.nan
    data[3, 0] = np.inf
    data[3, 1] = np.nan
    x = torch.tensor(data, device='cuda', requires_grad=True)
    y = torch.logsumexp(x, dim=-1)
    result = y.detach().cpu().numpy()
    assert np.isneginf(result[0])
    assert np.isposinf(result[1])
    assert np.isnan(result[2]) and np.isnan(result[3])
    np.testing.assert_allclose(result[4], np.log(32768.), rtol=1e-6, atol=2e-6)
    y.sum().backward()
    gradient = x.grad.detach().cpu().numpy()
    assert np.isnan(gradient[0]).all()
    assert np.isnan(gradient[1, 0]) and np.all(gradient[1, 1:] == 0)
    assert np.isnan(gradient[2]).all() and np.isnan(gradient[3]).all()


@pytest.mark.cuda
def test_large_logsumexp_autocast_and_noncontiguous_input():
    _cuda()
    data = np.linspace(-2., 3., 65536, dtype=np.float32).reshape(32768, 2)
    x = torch.tensor(data, device='cuda').t()
    assert not x.is_contiguous()
    wide = data.T.astype(np.float64)
    maximum = wide.max(axis=-1, keepdims=True)
    expected = (maximum + np.log(np.exp(wide - maximum).sum(axis=-1, keepdims=True))).reshape(-1)
    contiguous = torch.tensor(np.ascontiguousarray(data.T), device='cuda')
    for amp_dtype in (torch.float16, torch.bfloat16):
        with torch.autocast('cuda', dtype=amp_dtype):
            result = torch.logsumexp(x, dim=-1)
            contiguous_result = torch.logsumexp(contiguous, dim=-1)
        assert result.dtype == contiguous_result.dtype == torch.float32
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(contiguous_result.cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
