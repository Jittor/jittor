"""Public remainder spellings use the native floor-modulo operation."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32, torch.float64])
def test_remainder_signed_values_and_aliases(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    data = np.array([-7, -3, 0, 3, 7])
    if dtype == torch.int64:
        data = np.concatenate([data, [2 ** 45 + 3, -(2 ** 45 + 3)]])
    x = torch.tensor(data, dtype=dtype, device=device)
    for divisor in (4, -4):
        wanted = np.remainder(data, divisor)
        for got in (torch.remainder(x, divisor),
                    torch.remainder(input=x, other=divisor), x.remainder(divisor)):
            assert got.dtype == dtype and got.device == x.device
            np.testing.assert_array_equal(got.cpu().numpy(), wanted)
    got = torch.remainder(7, torch.tensor([-4, 4], dtype=dtype, device=device))
    np.testing.assert_array_equal(got.cpu().numpy(), [-1, 3])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_remainder_gradients_preserve_both_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.tensor([-7.5, 3.5], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([4., -4.], dtype=torch.float64, device=device, requires_grad=True)
    z = torch.remainder(x, y)
    np.testing.assert_array_equal(z.detach().cpu().numpy(), [.5, -.5])
    z.sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), [1., 1.])
    np.testing.assert_array_equal(y.grad.cpu().numpy(), [2., 1.])
