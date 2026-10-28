"""Shared state assertions for random-initialization tests; no sampling at import."""

import random
import sys
from contextlib import contextmanager

import numpy as np
import pytest
import torch


def _array(value):
    return value.detach().cpu().to(dtype=torch.float64).numpy().copy()


def _states():
    return [_array(torch.get_rng_state())] + [
        _array(state) for state in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    ]


def _same_states(left, right):
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        np.testing.assert_array_equal(actual, expected)


@contextmanager
def preserve_random_streams():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    python, numpy = random.getstate(), np.random.get_state()
    if "jittor" in sys.modules:
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with forbid_backend_fallbacks():
            yield
            sys.modules["jittor"].sync_all(True)
    else:
        yield
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(python)
    np.random.set_state(numpy)


def require_rng_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires real CUDA")
    return device
