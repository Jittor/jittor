"""Real vLLM GPU sampler semantics, usable in shim and native Torch processes.

``apply_temperature`` controls scaling, not whether sampling is stochastic.
The installed vLLM sampler is deliberately used instead of an adapter helper.
"""

import numpy as np
import pytest


def _sample(*, apply_temperature, temperatures=None, mapping=None, positions=None,
            processed=None):
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    count, vocab = 64, 17
    values = np.zeros((count, vocab), dtype=np.float32)
    values[:, 0] = 0.01  # Unique greedy winner; stochastic draws stay near uniform.
    if temperatures is None:
        temperatures = np.ones(count, dtype=np.float32)
    if mapping is None:
        mapping = np.arange(count, dtype=np.int32)
    if positions is None:
        positions = np.full(count, 11, dtype=np.int64)
    logits = torch.tensor(values, dtype=torch.float32, device="cuda")
    indices = gumbel_sample(
        logits,
        torch.tensor(mapping, dtype=torch.int32, device="cuda"),
        torch.tensor(temperatures, dtype=torch.float32, device="cuda"),
        torch.tensor(np.arange(count) + 123, dtype=torch.int64, device="cuda"),
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
        apply_temperature,
        output_processed_logits=processed,
    )
    assert indices.device.type == "cuda"
    assert indices.dtype == torch.int64
    actual = indices.cpu().numpy()
    assert actual.shape == (count,)
    assert np.all((actual >= 0) & (actual < vocab))
    return actual


@pytest.mark.parametrize("apply_temperature", [False, True])
def test_stochastic_fixed_seed_and_position_are_reproducible(apply_temperature):
    first = _sample(apply_temperature=apply_temperature)
    second = _sample(apply_temperature=apply_temperature)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("apply_temperature", [False, True])
def test_stochastic_multiple_seeds_do_not_collapse_to_greedy(apply_temperature):
    sampled = _sample(apply_temperature=apply_temperature)
    # For 64 near-uniform independent draws from 17 categories, collapsing to
    # fewer than 3 categories is overwhelmingly unlikely, but argmax always does.
    assert np.unique(sampled).size >= 3, sampled
    assert np.any(sampled != 0), "nonzero temperature silently became greedy"


def test_unit_temperature_scaling_flag_does_not_change_randomness():
    np.testing.assert_array_equal(
        _sample(apply_temperature=False),
        _sample(apply_temperature=True),
    )


def test_stochastic_position_changes_draws():
    first = _sample(apply_temperature=False)
    next_position = _sample(
        apply_temperature=False, positions=np.full(64, 12, dtype=np.int64)
    )
    assert np.any(first != next_position), "sampler ignored token position"


@pytest.mark.parametrize("apply_temperature", [False, True])
def test_mixed_greedy_stochastic_requests_respect_state_mapping(apply_temperature):
    temperatures = np.ones(64, dtype=np.float32)
    temperatures[::2] = 0
    mapping = np.arange(63, -1, -1, dtype=np.int32)
    sampled = _sample(
        apply_temperature=apply_temperature,
        temperatures=temperatures,
        mapping=mapping,
    )
    greedy_rows = temperatures[mapping] == 0
    np.testing.assert_array_equal(sampled[greedy_rows], 0)
    assert np.unique(sampled[~greedy_rows]).size >= 3


def test_sampler_writes_processed_logits_for_greedy_requests():
    import torch

    output = torch.full((64, 17), -99.0, dtype=torch.float32, device="cuda")
    _sample(
        apply_temperature=False,
        temperatures=np.zeros(64, dtype=np.float32),
        processed=output,
    )
    expected = np.zeros((64, 17), dtype=np.float32)
    expected[:, 0] = 0.01
    np.testing.assert_allclose(output.cpu().numpy(), expected, atol=1e-7, rtol=0)
