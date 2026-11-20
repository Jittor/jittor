"""Local response normalization should support channel boundaries and backward."""

import jittor as jt
import numpy as np
import pytest
import torch
from _helpers import capability


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(2, 4, 2), (2, 4, 2, 3), (2, 4, 2, 2, 2)])
@pytest.mark.parametrize("size", [3, 4])
def test_local_response_norm_values_and_gradients(device, shape, size):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires a real CUDA device")
    values = np.linspace(-2, 3, np.prod(shape)).reshape(shape)
    flat = values.reshape(shape[0], shape[1], -1)
    alpha, beta, k = 0.4, 0.5, 1.25
    channels = np.arange(shape[1])
    offsets = channels[None, :] - channels[:, None]
    window = ((offsets >= -(size // 2)) & (offsets <= (size - 1) // 2)) / size
    scale = k + alpha * (window @ (flat ** 2))
    expected = flat * scale ** (-beta)
    gradient = scale ** (-beta) - (
        (2 * alpha * beta) * flat * (window.T @ (flat * scale ** (-beta - 1)))
    )
    with jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
        x = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
        norm = torch.nn.LocalResponseNorm(size, alpha, beta, k)
        assert torch.nn.modules.normalization.LocalResponseNorm is type(norm)
        result = norm(x)
        direct = torch.nn.functional.local_response_norm(x, size, alpha, beta, k)
        result.sum().backward()
        assert result.device.type == x.grad.device.type == device
        for output in (result, direct):
            np.testing.assert_allclose(output.detach().cpu().numpy(), expected.reshape(shape), rtol=1e-10)
        np.testing.assert_allclose(x.grad.cpu().numpy(), gradient.reshape(shape), rtol=1e-10)


def test_local_response_norm_empty_and_short_inputs():
    norm = torch.nn.LocalResponseNorm(3)
    empty = torch.empty(0, 4, 2)
    assert norm(empty) is empty
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        norm(torch.ones(2, 4))
