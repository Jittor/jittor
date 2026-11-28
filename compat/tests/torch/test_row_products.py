import numpy as np
import pytest
import torch

import jittor as jt


def dyadic_samples(rng, shape):
    return rng.integers(-7, 8, size=shape).astype(np.float32) / 32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
@torch.no_grad()
def test_single_row_linear(device, bias, offset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    rng = np.random.default_rng(27)
    inputs = dyadic_samples(rng, (1, 512 + offset))
    weights = dyadic_samples(rng, (768, 512))
    biases = dyadic_samples(rng, (768,)) if bias else None
    expected = inputs[:, offset:] @ weights.T
    if biases is not None:
        expected += biases
    with jt.flag_scope(backend_fallback="error"):
        x = torch.tensor(inputs, device=device)[:, offset:]
        w = torch.tensor(weights, device=device)
        b = torch.tensor(biases, device=device) if biases is not None else None
        actual = torch.nn.functional.linear(x, w, b)
        assert actual.device.type == device
        assert actual.dtype == torch.float32
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_batched_single_row_product(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    rng = np.random.default_rng(31)
    inputs = dyadic_samples(rng, (4, 1, 96))
    weights = dyadic_samples(rng, (4, 83, 96))
    expected = inputs @ weights.swapaxes(-1, -2)

    with jt.flag_scope(backend_fallback="error"):
        x = torch.tensor(inputs, device=device)
        w = torch.tensor(weights, device=device)
        actual = x @ w.transpose(-1, -2)
        assert actual.device.type == device
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
