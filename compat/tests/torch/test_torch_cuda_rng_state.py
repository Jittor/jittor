"""Saving and restoring the CUDA RNG state, for real.

`get_rng_state` returned the constant `[0]` and `set_rng_state` did nothing, so
`accelerator.save_state()` wrote a byte that meant nothing, `load_state()`
restored nothing, and the resumed run drew a different sequence than the one it
was continuing -- silently, which is the shape every bug in this file's
neighbourhood takes.

The state is a seed and a position. cuRAND will set an offset but not report
one, so jittor counts: measured against CURAND_RNG_PSEUDO_DEFAULT, a uniform
draw of n advances the generator by n and a normal draw of n advances it by
n/2, whatever the precision; a mixed history advances by the sum; and after a
restore every one of the four kinds continues exactly. The C5 issue doc's
"问题一" says a single offset cannot describe a mixed history -- on this cuRAND
it can; the measurement is described in the commit that added this.
"""
from functools import wraps

import jittor as jt
import numpy as np
import pytest
import torch


def _cuda_rng(func):
    """Run on a CUDA build with cuRAND offset accounting, `use_cuda` scoped.

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
        backend = getattr(jt.compile_extern, "curand", None)
        if backend is None or not hasattr(backend, "curand_restore_state"):
            # The counting lives in the native cuRAND wrapper. Without it there
            # is no position to save, and everything below asserts that the
            # position round-trips.
            pytest.skip("this build has no native cuRAND offset accounting")
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
