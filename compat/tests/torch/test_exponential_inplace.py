"""Exponential draws on one explicitly selected device per test process.

Run with JITTOR_TEST_DEVICE=cpu or cuda in separate backend processes.
Seeded generators are an explicit unsupported boundary in the shim for now.
"""

import os

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    return os.environ.get("JITTOR_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_exponential_preserves_object_shape_dtype_and_device(device, dtype):
    tensor = torch.zeros((16, 32), dtype=dtype, device=device)
    result = tensor.exponential_()
    assert result is tensor
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == (16, 32)
    assert tensor.device.type == device.split(":")[0]
    values = tensor.cpu().numpy()
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)


@pytest.mark.parametrize("rate", [0.5, 2.0])
def test_exponential_matches_analytic_distribution(device, rate):
    tensor = torch.empty((65536,), device=device).exponential_(rate)
    values = tensor.cpu().numpy()
    np.testing.assert_allclose(values.mean(), 1 / rate, rtol=0.04)
    np.testing.assert_allclose(values.var(), 1 / rate ** 2, rtol=0.08)
    for cutoff in (0.5, 1.0, 2.0):
        empirical = np.mean(values <= cutoff / rate)
        assert abs(empirical - (1 - np.exp(-cutoff))) < 0.015


def test_exponential_retained_view_updates_only_selected_elements(device):
    base = torch.zeros((4, 8), device=device)
    selected = base[:, ::2]
    assert selected.exponential_() is selected
    values = base.cpu().numpy()
    assert np.all(values[:, ::2] > 0)
    np.testing.assert_array_equal(values[:, 1::2], 0)


def test_exponential_global_seed_reproduces_and_stream_advances(device):
    torch.manual_seed(712)
    first = torch.empty((128,), device=device).exponential_().cpu().numpy()
    next_draw = torch.empty((128,), device=device).exponential_().cpu().numpy()
    torch.manual_seed(712)
    repeated = torch.empty((128,), device=device).exponential_().cpu().numpy()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, next_draw)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_exponential_rejects_nonpositive_rate_without_mutation(device, rate):
    tensor = torch.ones((8,), device=device)
    with pytest.raises((RuntimeError, ValueError)):
        tensor.exponential_(rate)
    np.testing.assert_array_equal(tensor.cpu().numpy(), 1)


def test_exponential_rejects_integral_target(device):
    tensor = torch.zeros((8,), dtype=torch.int32, device=device)
    with pytest.raises((RuntimeError, TypeError)):
        tensor.exponential_()


def test_exponential_explicit_generator_is_not_silently_ignored(device):
    generator = torch.Generator(device=device).manual_seed(812)
    tensor = torch.ones((8,), device=device)
    if hasattr(torch, "_torch_compat_install_context"):
        with pytest.raises(NotImplementedError, match="generator"):
            tensor.exponential_(generator=generator)
        np.testing.assert_array_equal(tensor.cpu().numpy(), 1)
    else:
        first = tensor.exponential_(generator=generator).cpu().numpy().copy()
        generator.manual_seed(812)
        second = tensor.exponential_(generator=generator).cpu().numpy()
        np.testing.assert_array_equal(first, second)
