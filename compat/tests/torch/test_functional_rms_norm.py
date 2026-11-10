"""Functional RMSNorm preserves values, gradients and normalized axes."""
import os

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("shape,axes,weighted", [
    ((2, 3, 64), (64,), True),
    ((2, 3, 64), (64,), False),
    ((2, 3, 64), (3, 64), True),
    ((2, 3, 17), (17,), True),
])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("device", os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def test_functional_rms_values_and_derivatives(shape, axes, weighted, training, device):
    rng = np.random.default_rng(123)
    source = rng.standard_normal(shape).astype("float32")
    weights = rng.uniform(.5, 1.5, axes).astype("float32")
    upstream = rng.standard_normal(shape).astype("float32")
    value = torch.tensor(source, device=device, requires_grad=training)
    weight = torch.tensor(weights, device=device, requires_grad=training) if weighted else None
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        actual = torch.nn.functional.rms_norm(value, axes, weight, eps=1e-6)
        assert actual.device.type == device
        assert actual.dtype == torch.float32
        if training:
            (actual * torch.tensor(upstream, device=device)).sum().backward()
    dims = tuple(range(len(shape) - len(axes), len(shape)))
    x = source.astype("float64")
    inv = 1 / np.sqrt(np.mean(x*x, axis=dims, keepdims=True) + 1e-6)
    w = weights.astype("float64") if weighted else 1.
    np.testing.assert_allclose(actual.detach().cpu().numpy(), x*inv*w, rtol=2e-5, atol=2e-6)
    if training:
        g = upstream*w
        expected = g*inv - x*inv**3*np.mean(g*x, axis=dims, keepdims=True)
        np.testing.assert_allclose(value.grad.cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        if weighted:
            dw = (upstream*x*inv).sum(axis=tuple(range(len(shape)-len(axes))))
            np.testing.assert_allclose(weight.grad.cpu().numpy(), dw, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def test_rms_norm_strided_inputs_use_live_weights_and_epsilon(device):
    rng = np.random.default_rng(914)
    source = rng.standard_normal((17, 3)).astype("float32")
    weights = rng.uniform(.5, 1.5, 34).astype("float32")
    with torch.no_grad():
        value = torch.tensor(source, device=device).transpose(0, 1)
        weight = torch.tensor(weights, device=device)[::2]
        assert not value.is_contiguous()
        assert not weight.is_contiguous()
        first = torch.nn.functional.rms_norm(value, (17,), weight, eps=1e-6)
        saved = first.cpu().numpy().copy()
        weight.copy_(torch.full_like(weight, 2.))
        second = torch.nn.functional.rms_norm(value, (17,), weight, eps=.1)
        assert first is not second
        assert first.device.type == second.device.type == device
        assert not first.requires_grad and not second.requires_grad
        raw = source.T.astype("float64")
        expected_first = raw / np.sqrt(np.mean(raw*raw, -1, keepdims=True) + 1e-6) * weights[::2]
        expected_second = raw / np.sqrt(np.mean(raw*raw, -1, keepdims=True) + .1) * 2.
        np.testing.assert_allclose(saved, expected_first, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(second.cpu().numpy(), expected_second, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(first.cpu().numpy(), saved)
