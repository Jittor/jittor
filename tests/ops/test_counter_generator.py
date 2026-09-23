"""Native independent device RNG: known Philox vector and lazy state ownership."""
import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def selected_backend():
    import jittor as jt
    with jt.flag_scope(use_cuda=int(os.environ.get("JITTOR_TEST_DEVICE", "cpu") == "cuda")):
        yield


def test_philox_known_vector_and_state():
    import jittor as jt
    from jittor.ops.random import CounterGenerator
    like = jt.zeros((4,), dtype="float64")
    generator = CounterGenerator(0)
    # Philox4x32-10, zero key and zero counter (Random123 known-answer vector).
    words = np.array([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8], dtype=np.uint64)
    expected = ((words.astype(np.float64) + .5) * 2.0**-32).astype(np.float32)
    first = generator.uniform_like(like, dtype="float32")
    snapshot = generator.get_state()
    second = generator.uniform_like(like, dtype="float32")
    generator.set_state(snapshot)
    repeated = generator.uniform_like(like, dtype="float32")
    np.testing.assert_array_equal(first.numpy(), expected)
    np.testing.assert_array_equal(second.numpy(), repeated.numpy())
    assert not np.array_equal(first.numpy(), second.numpy())


def test_counter_seed_high_bits_and_lazy_capture():
    import jittor as jt
    from jittor.ops.random import CounterGenerator
    like = jt.zeros((31,))
    low = CounterGenerator(7)
    high = CounterGenerator(2**63 + 7)
    expected = low.uniform_like(like)
    different = high.uniform_like(like)
    low.manual_seed(9)
    np.testing.assert_array_equal(expected.numpy(), CounterGenerator(7).uniform_like(like).numpy())
    assert not np.array_equal(expected.numpy(), different.numpy())


@pytest.mark.parametrize("state", [(0, -4), (0, 1), (2**64, 0), (-1, 0)])
def test_bad_counter_state_does_not_mutate(state):
    from jittor.ops.random import CounterGenerator
    generator = CounterGenerator(51)
    before = generator.get_state()
    with pytest.raises(ValueError):
        generator.set_state(state)
    assert generator.get_state() == before
