"""Saving and restoring the CUDA RNG state, for real.

The native CUDA backend owns a versioned Philox seed/counter state. Torch
compatibility only encodes that state as the CPU ``uint8`` value expected by
``torch.cuda.get_rng_state``; generated values and state bytes are intentionally
Jittor-specific.
"""
from functools import wraps

import jittor as jt
import numpy as np
import pytest
import torch


def _cuda_rng(func):
    """Run on a CUDA build with native RNG state support, `use_cuda` scoped.

    The flag is set through `jt.flag_scope` rather than by assigning
    `jt.flags.use_cuda`: it is process-global, and a test that leaves it on makes
    every later file in the session run on the accelerator. The flag-scope gate
    (`tests/structure/runtime/test_flag_scope_contract.py`) flagged the previous
    assignments for exactly that, and it is right to -- nothing restored them.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if not jt.has_cuda:
            pytest.skip("no CUDA device")
        if not hasattr(jt, "get_cuda_rng_state") or not hasattr(jt, "set_cuda_rng_state"):
            pytest.skip("this build has no native CUDA RNG state API")
        with jt.flag_scope(use_cuda=1):
            return func(*args, **kwargs)
    return inner


def _cuda_rng_seed_only(func):
    """The same, for the two tests with no saved position to round-trip.

    Spelled out rather than derived from ``_cuda_rng``: a module-level
    ``_alias = factory(...)`` is an assignment whose value calls a local helper,
    which the collection-side-effect gate reads as work a bare import would do.
    The decorator form is not an assignment and needs no allowlist entry.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if not jt.has_cuda:
            pytest.skip("no CUDA device")
        with jt.flag_scope(use_cuda=1):
            return func(*args, **kwargs)
    return inner


@_cuda_rng
def test_state_round_trips_for_uniform():
    torch.cuda.manual_seed(1234)
    jt.random((10,)).sync()

    state = torch.cuda.get_rng_state()
    expected = jt.random((8,)).numpy().copy()

    torch.cuda.set_rng_state(state)
    np.testing.assert_array_equal(jt.random((8,)).numpy(), expected)


@_cuda_rng
def test_state_round_trips_after_a_mixed_history():
    # The history the doc says cannot be expressed: uniform, then normal, then
    # an odd-length normal, then float64.
    torch.cuda.manual_seed(99)
    jt.random((5,)).sync()
    jt.random((6,), type="normal").sync()
    jt.random((7,), type="normal").sync()
    jt.random((3,), dtype="float64").sync()

    state = torch.cuda.get_rng_state()
    expected_u = jt.random((8,)).numpy().copy()
    expected_n = jt.random((4,), type="normal").numpy().copy()

    torch.cuda.set_rng_state(state)
    np.testing.assert_array_equal(jt.random((8,)).numpy(), expected_u)
    np.testing.assert_array_equal(jt.random((4,), type="normal").numpy(), expected_n)


@_cuda_rng
def test_a_restored_state_is_not_just_a_reseed():
    # Seeding rewinds; restoring continues. If `set_rng_state` were a reseed in
    # disguise, the draw after it would be the sequence's *first* values.
    torch.cuda.manual_seed(7)
    first = jt.random((8,)).numpy().copy()
    state = torch.cuda.get_rng_state()
    after = jt.random((8,)).numpy().copy()
    assert not (first == after).all(), "the draw did not advance at all"

    torch.cuda.set_rng_state(state)
    np.testing.assert_array_equal(jt.random((8,)).numpy(), after)


@_cuda_rng
def test_a_foreign_state_is_refused():
    with pytest.raises(ValueError) as caught:
        torch.cuda.set_rng_state(torch.zeros(24, dtype=torch.uint8))
    assert "format" in str(caught.value) or "jittor" in str(caught.value)


@_cuda_rng
def test_a_refused_state_leaves_the_generator_alone():
    torch.cuda.manual_seed(31)
    state = torch.cuda.get_rng_state()
    expected = jt.random((8,)).numpy().copy()
    torch.cuda.set_rng_state(state)

    with pytest.raises(ValueError):
        torch.cuda.set_rng_state(torch.zeros(24, dtype=torch.uint8))
    np.testing.assert_array_equal(jt.random((8,)).numpy(), expected)


@_cuda_rng
def test_get_rng_state_all_answers_per_device():
    states = torch.cuda.get_rng_state_all()
    assert len(states) == int(jt.get_device_count())
    torch.cuda.set_rng_state_all(states)


@_cuda_rng_seed_only
def test_initial_seed_reports_the_seed_in_use():
    # No native counting needed: the seed is expressible either way.
    torch.cuda.manual_seed(4321)
    assert torch.cuda.initial_seed() == 4321


@_cuda_rng_seed_only
def test_seed_actually_reseeds():
    torch.cuda.manual_seed(11)
    torch.cuda.seed()
    assert torch.cuda.initial_seed() != 11
