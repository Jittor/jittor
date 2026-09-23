"""Exponential draws on one explicitly selected device per test process.

Run with JITTOR_TEST_DEVICE=cpu or cuda in separate backend processes.
Explicit generators must own a reproducible, isolated device stream.
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
    tensor = torch.ones((128,), device=device)
    first = tensor.exponential_(generator=generator).cpu().numpy().copy()
    advanced = tensor.exponential_(generator=generator).cpu().numpy().copy()
    generator.manual_seed(812)
    repeated = tensor.exponential_(generator=generator).cpu().numpy()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, advanced)


def test_explicit_generators_are_isolated_from_each_other_and_global_rng(device):
    a = torch.Generator(device=device).manual_seed(99)
    b = torch.Generator(device=device).manual_seed(99)
    first = torch.empty((129,), device=device).exponential_(generator=a).cpu().numpy()
    torch.manual_seed(771)
    expected_global = torch.empty((32,), device=device).exponential_().cpu().numpy()
    torch.manual_seed(771)
    same = torch.empty((129,), device=device).exponential_(generator=b).cpu().numpy()
    observed_global = torch.empty((32,), device=device).exponential_().cpu().numpy()
    np.testing.assert_array_equal(first, same)
    np.testing.assert_array_equal(expected_global, observed_global)
    # Advancing the global stream cannot affect either explicit stream.
    next_a = torch.empty((129,), device=device).exponential_(generator=a).cpu().numpy()
    torch.empty((999,), device=device).exponential_()
    next_b = torch.empty((129,), device=device).exponential_(generator=b).cpu().numpy()
    np.testing.assert_array_equal(next_a, next_b)


def test_generator_snapshot_restores_stream_after_draws(device):
    generator = torch.Generator(device=device).manual_seed(321)
    torch.empty((17,), device=device).exponential_(generator=generator)
    state = generator.get_state()
    assert state.dtype == torch.uint8 and state.device.type == "cpu"
    expected = torch.empty((257,), device=device).exponential_(generator=generator).cpu().numpy()
    torch.empty((257,), device=device).exponential_(generator=generator)
    generator.set_state(state)
    actual = torch.empty((257,), device=device).exponential_(generator=generator).cpu().numpy()
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_explicit_generator_distribution_and_device(device, dtype):
    generator = torch.Generator(device=device).manual_seed(1234)
    values = torch.empty((65536,), device=device, dtype=dtype).exponential_(2, generator=generator)
    assert values.device.type == device.split(":")[0]
    assert values.dtype == dtype
    arr = values.cpu().numpy().astype(np.float64)
    assert np.all(np.isfinite(arr)) and np.all(arr > 0)
    np.testing.assert_allclose(arr.mean(), .5, rtol=.04)
    np.testing.assert_allclose(arr.var(), .25, rtol=.08)
    assert abs(np.mean(arr <= .5) - (1-np.exp(-1))) < .015


def test_explicit_generator_retained_view(device):
    base = torch.zeros((4, 8), device=device)
    view = base[:, ::2]
    assert view.exponential_(generator=torch.Generator(device=device).manual_seed(42)) is view
    actual = base.cpu().numpy()
    assert np.all(actual[:, ::2] > 0)
    np.testing.assert_array_equal(actual[:, 1::2], 0)


def test_explicit_generator_device_mismatch_does_not_mutate(device):
    if device.split(":")[0] != "cuda":
        pytest.skip("requires an accelerator tensor and a CPU generator")
    generator = torch.Generator(device="cpu").manual_seed(42)
    tensor = torch.ones((8,), device=device)
    with pytest.raises(RuntimeError, match="[Gg]enerator|device"):
        tensor.exponential_(generator=generator)
    np.testing.assert_array_equal(tensor.cpu().numpy(), 1)


def test_cuda_generator_offset_can_rewind_one_sampler_draw(device):
    if device.split(":")[0] != "cuda":
        pytest.skip("CPU PyTorch generators have no offset API")
    generator = torch.Generator(device=device).manual_seed(42)
    before = generator.get_offset()
    expected = torch.empty((50272,), device=device).exponential_(generator=generator).cpu().numpy()
    assert generator.get_offset() - before == 4
    generator.set_offset(generator.get_offset() - 4)
    actual = torch.empty((50272,), device=device).exponential_(generator=generator).cpu().numpy()
    np.testing.assert_array_equal(expected, actual)


def test_generator_snapshot_preserves_factory_stream_too(device):
    generator = torch.Generator(device=device).manual_seed(821)
    torch.randn((9,), device=device, generator=generator)
    state = generator.get_state()
    exp_expected = torch.empty((35,), device=device).exponential_(generator=generator).cpu().numpy()
    normal_expected = torch.randn((9,), device=device, generator=generator).cpu().numpy()
    generator.set_state(state)
    exp_actual = torch.empty((35,), device=device).exponential_(generator=generator).cpu().numpy()
    normal_actual = torch.randn((9,), device=device, generator=generator).cpu().numpy()
    np.testing.assert_array_equal(exp_expected, exp_actual)
    np.testing.assert_array_equal(normal_expected, normal_actual)


def test_generator_draws_capture_state_before_lazy_execution(device):
    generator = torch.Generator(device=device).manual_seed(71)
    first = torch.empty((33,), device=device).exponential_(generator=generator)
    second = torch.empty((33,), device=device).exponential_(generator=generator)
    # Resetting the Python generator before deferred evaluation must not change
    # either already-created draw; evaluate in reverse order to exercise this.
    generator.manual_seed(71)
    reference = torch.empty((33,), device=device).exponential_(generator=generator)
    b, a, expected = second.cpu().numpy(), first.cpu().numpy(), reference.cpu().numpy()
    np.testing.assert_array_equal(a, expected)
    assert not np.array_equal(a, b)


def test_explicit_generator_empty_draw_does_not_advance_state(device):
    generator = torch.Generator(device=device).manual_seed(731)
    before = generator.get_state().cpu().numpy().copy()
    tensor = torch.empty((0, 3), device=device)
    assert tensor.exponential_(generator=generator) is tensor
    assert tuple(tensor.shape) == (0, 3)
    np.testing.assert_array_equal(before, generator.get_state().cpu().numpy())


@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_generator_clone_preserves_advanced_independent_streams(device, method):
    import copy
    import pickle
    generator = torch.Generator(device=device).manual_seed(277)
    torch.empty((33,), device=device).exponential_(generator=generator).cpu().numpy()
    torch.randn((11,), device=device, generator=generator).cpu().numpy()
    cloned = (pickle.loads(pickle.dumps(generator)) if method == "pickle"
              else copy.deepcopy(generator))
    assert cloned is not generator
    expected = torch.empty((35,), device=device).exponential_(generator=generator).cpu().numpy()
    actual = torch.empty((35,), device=device).exponential_(generator=cloned).cpu().numpy()
    np.testing.assert_array_equal(expected, actual)
    expected_normal = torch.randn((11,), device=device, generator=generator).cpu().numpy()
    actual_normal = torch.randn((11,), device=device, generator=cloned).cpu().numpy()
    np.testing.assert_array_equal(expected_normal, actual_normal)
