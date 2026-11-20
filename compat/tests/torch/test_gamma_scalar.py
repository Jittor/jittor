"""Scalar gamma functions preserve placement, shape, and autograd."""
import numpy as np
import pytest
from scipy import special
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


@pytest.mark.parametrize(
    ("name", "value", "gradient"),
    [
        ("lgamma", special.gammaln(2.5), special.digamma(2.5)),
        ("digamma", special.digamma(2.5), special.polygamma(1, 2.5)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gamma_after_full_reduction(device, name, value, gradient, dtype):
    source = torch.tensor([1.0, 1.5], device=device, dtype=dtype, requires_grad=True)
    result = getattr(torch, name)(source.sum())
    assert result.shape == ()
    assert result.dtype == dtype
    assert result.device.type == device
    np.testing.assert_allclose(result.detach().cpu().numpy(), value, rtol=3e-6)

    result.backward()

    np.testing.assert_allclose(source.grad.cpu().numpy(), [gradient, gradient], rtol=3e-6)


@pytest.mark.parametrize("name", ["lgamma", "digamma"])
def test_gamma_accepts_empty_input(device, name):
    result = getattr(torch, name)(torch.empty((0, 3), device=device))
    assert result.shape == (0, 3)
    assert result.device.type == device
    assert result.cpu().numpy().size == 0
