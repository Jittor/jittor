"""Empty CUDA RNG collections are valid in a CPU-only process."""

import random

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("api,states,expected", [
    ("get_rng_state_all", None, []),
    ("set_rng_state_all", [], None),
    ("set_rng_state_all", (), None),
])
def test_empty_cuda_rng_collection_preserves_other_streams(api, states, expected):
    if torch.cuda.device_count():
        pytest.skip("requires a process with no visible CUDA devices")
    cpu_before = torch.get_rng_state().numpy().copy()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    initialized_before = torch.cuda.is_initialized()
    result = getattr(torch.cuda, api)(*(() if states is None else (states,)))
    assert result == expected
    assert torch.cuda.is_initialized() == initialized_before
    np.testing.assert_array_equal(torch.get_rng_state().numpy(), cpu_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
