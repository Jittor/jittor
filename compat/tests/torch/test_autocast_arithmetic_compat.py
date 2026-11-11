"""Torch autocast operator policy, exercised on independent CPU/CUDA backends."""
import numpy as np
import pytest
import torch


from _device_assertions import require_residency


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    return request.param


@pytest.mark.parametrize('precision', ['float16', 'bfloat16'])
def test_autocast_fp32_arithmetic_and_rsqrt(device, precision):
    if device == 'cpu' and precision == 'float16':
        pytest.skip('harness-limited: CPU FP16 autocast is outside this check')
    values = np.array([1.00123, 2.34567, 4.12345], dtype=np.float32)
    x = torch.tensor(values, dtype=torch.float32, device=device).requires_grad_(True)
    with torch.autocast(device_type=device, dtype=getattr(torch, precision)):
        outputs = (x + x, x * x, x.mul(x), torch.mul(x, x), x.pow(2), torch.pow(x, 2),
                   x.square(), torch.rsqrt(x), x.rsqrt())
        for output in outputs:
            assert output.dtype == torch.float32
            require_residency(output, device)
        loss = (x * x).sum()
    for output in outputs[:1]:
        np.testing.assert_allclose(output.detach().cpu().numpy(), values + values, rtol=1e-6, atol=1e-7)
    for output in outputs[1:7]:
        np.testing.assert_allclose(output.detach().cpu().numpy(), values * values, rtol=1e-6, atol=1e-7)
    for output in outputs[7:]:
        np.testing.assert_allclose(output.detach().cpu().numpy(), 1 / np.sqrt(values), rtol=1e-6, atol=1e-7)
    loss.backward()
    require_residency(x.grad, device)
    np.testing.assert_allclose(x.grad.detach().cpu().numpy(), values * 2, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('precision', ['float16', 'bfloat16'])
def test_autocast_rsqrt_half_input_and_matmul_policy(device, precision):
    if device == 'cpu' and precision == 'float16':
        pytest.skip('harness-limited: CPU FP16 autocast is outside this check')
    dtype = getattr(torch, precision)
    half = torch.tensor([1.25, 2.5], dtype=dtype, device=device)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
    with torch.autocast(device_type=device, dtype=dtype):
        inverse = torch.rsqrt(half)
        product = torch.mm(matrix, matrix)
    assert inverse.dtype == (torch.float32 if device == 'cuda' else dtype)
    assert product.dtype == dtype
    require_residency(inverse, device)
    require_residency(product, device)


def test_disabled_nested_autocast_restores_fp32(device):
    x = torch.tensor([1.00123, 2.34567], dtype=torch.float32, device=device)
    outer = torch.float16 if device == 'cuda' else torch.bfloat16
    with torch.autocast(device_type=device, dtype=outer):
        with torch.autocast(device_type=device, enabled=False):
            result = x * x
            assert result.dtype == torch.float32
    np.testing.assert_allclose(result.detach().cpu().numpy(), np.array([1.00123, 2.34567], dtype=np.float32) ** 2,
                               rtol=1e-6, atol=1e-7)
