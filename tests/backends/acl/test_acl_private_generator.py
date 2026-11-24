"""Native ACL private Philox contract on a real, externally admitted device.

The Python integer oracle supplies expectations only: actual outputs must be
ACL-resident Vars and fallback accounting must remain zero. These tests do
not claim compatibility with CANN offsets or vLLM's offset-minus-four rule.
"""

import gc
from contextlib import contextmanager

import numpy as np
import pytest

import jittor as jt
from _helpers.capability import require_accelerator


_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_MAX_OFFSET = (1 << 63) - 1


def _philox_block(seed, block):
    """Independent arbitrary-precision integer Philox4x32-10 reference."""
    counter = [block & _MASK32, block >> 32, 0, 0]
    key = [seed & _MASK32, (seed & _MASK64) >> 32]
    for round_index in range(10):
        high0, low0 = divmod(0xD2511F53 * counter[0], 1 << 32)
        high1, low1 = divmod(0xCD9E8D57 * counter[2], 1 << 32)
        counter = [high1 ^ counter[1] ^ key[0], low1,
                   high0 ^ counter[3] ^ key[1], low0]
        if round_index != 9:
            key = [(key[0] + 0x9E3779B9) & _MASK32,
                   (key[1] + 0xBB67AE85) & _MASK32]
    return tuple(counter)


def _reference(seed, offset, count):
    words = []
    for block in range(offset // 4, (offset + count + 3) // 4):
        words.extend(_philox_block(seed, block))
    lane = offset % 4
    # Both conversion and power-of-two scaling are exact in float32.
    return np.asarray([word >> 8 for word in words[lane:lane + count]],
                      dtype=np.float32) * np.float32(2.0 ** -24)


@contextmanager
def _placement(backend, ordinal):
    # Native construction and frontend context are distinct scopes. Set both
    # metadata requests without changing the selected execution device.
    token = jt.core._set_tensor_placement(backend, ordinal)
    previous = jt.core._set_construction_placement(backend, ordinal)
    try:
        yield
    finally:
        jt.core._restore_construction_placement(previous)
        jt.core._reset_tensor_placement(token)


@pytest.fixture
def acl():
    require_accelerator("acl")
    assert jt.compiler.build_config.backend == "acl"
    assert jt.compiler.core_build_is_current()
    ordinal = int(jt.core.current_device())
    assert ordinal >= 0
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), jt.flag_scope(
            lazy_execution=1, async_execution=0, auto_flush_ops=0,
            amp_reg=0, auto_mixed_precision_level=0):
        before = int(jt.core.backend_fallback_count())
        yield ordinal
        jt.sync_all(True)
        assert int(jt.core.backend_fallback_count()) == before


def _owner(ordinal, seed=0, offset=0):
    result = jt.Generator("acl:%d" % ordinal).manual_seed(seed)
    result.set_offset(offset)
    return result


def _draw(owner, ordinal, count, dtype="float32", distribution="uniform"):
    # Bypass jt.random's explicit low-precision cast lowering when testing the
    # native RandomOp dtype boundary.
    with _placement(2, ordinal):
        return jt.core.ops.random((count,), dtype, distribution, owner)


def _resident(value, ordinal):
    value.sync()
    assert int(value.placement_backend) == 2
    assert int(value.device_id) == ordinal
    assert value.location() == "device"
    assert str(value.dtype) == "float32"
    return value.numpy().copy()


def test_official_philox_known_answer_vector(acl):
    # Random123's published zero-counter/zero-key Philox4x32-10 KAT:
    # https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    official = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    assert _philox_block(0, 0) == official
    expected = np.asarray([word >> 8 for word in official], np.float32) * np.float32(2.0 ** -24)
    owner = _owner(acl)
    np.testing.assert_array_equal(_resident(_draw(owner, acl, 4), acl), expected)
    assert owner.get_offset() == 4


@pytest.mark.parametrize("count", [1, 3, 4, 5, 255, 256, 257, 1025])
def test_lane_and_ub_tile_boundaries_match_integer_reference(acl, count):
    seed, offset = 0x0123456789ABCDEF, 3
    owner = _owner(acl, seed, offset)
    np.testing.assert_array_equal(_resident(_draw(owner, acl, count), acl),
                                  _reference(seed, offset, count))
    assert owner.get_offset() == offset + count


@pytest.mark.parametrize("seed", [0, 19, -1, -(1 << 63), 0x123456789ABCDEF])
@pytest.mark.parametrize("offset", [1, (1 << 34) + 3, (1 << 61) + 5, _MAX_OFFSET - 18])
def test_signed_seed_bits_and_high_raw32_positions(acl, seed, offset):
    owner = _owner(acl, seed, offset)
    assert owner.initial_seed() == seed
    np.testing.assert_array_equal(_resident(_draw(owner, acl, 17), acl),
                                  _reference(seed, offset, 17))
    assert owner.get_offset() == offset + 17


def test_public_random_uses_native_private_owner(acl):
    owner = _owner(acl, 43, 7)
    with _placement(2, acl):
        result = jt.random((257,), dtype="float32", generator=owner)
    np.testing.assert_array_equal(_resident(result, acl), _reference(43, 7, 257))


def test_issue_order_reserves_spans_before_reverse_execution(acl):
    owner = _owner(acl, 71, 3)
    first, second = _draw(owner, acl, 5), _draw(owner, acl, 7)
    assert not first.is_finished and not second.is_finished
    assert owner.get_offset() == 15
    checkpoint = owner.get_state()
    np.testing.assert_array_equal(_resident(second, acl), _reference(71, 8, 7))
    np.testing.assert_array_equal(_resident(first, acl), _reference(71, 3, 5))
    assert owner.get_state() == checkpoint


@pytest.mark.parametrize("action", ["reseed", "restore", "delete", "restore_and_delete"])
def test_pending_draw_captures_seed_offset_and_owner_lifetime(acl, action):
    owner = _owner(acl, 131, 3)
    replacement = _owner(acl, -1, 101).get_state()
    pending = _draw(owner, acl, 257)
    assert not pending.is_finished
    assert owner.get_offset() == 260
    if action == "reseed":
        owner.manual_seed(9)
        assert owner.get_offset() == 0
    if "restore" in action:
        owner.set_state(replacement)
        assert owner.get_offset() == 101 and owner.initial_seed() == -1
    if "delete" in action:
        del owner
        gc.collect()
    assert not pending.is_finished
    np.testing.assert_array_equal(_resident(pending, acl), _reference(131, 3, 257))


def test_private_interleaving_does_not_change_default_or_another_owner(acl):
    default_before = jt.core.rng_state("acl", acl)
    first, other = _owner(acl, 47), _owner(acl, -103)
    one = _resident(_draw(first, acl, 5), acl)
    _resident(_draw(other, acl, 257), acl)
    two = _resident(_draw(first, acl, 11), acl)
    np.testing.assert_array_equal(np.concatenate([one, two]), _reference(47, 0, 16))
    assert first.get_offset() == 16 and other.get_offset() == 257
    assert jt.core.rng_state("acl", acl) == default_before


def test_empty_and_limit_reservations_are_exact_and_overflow_is_atomic(acl):
    owner = _owner(acl, 31, _MAX_OFFSET - 1)
    np.testing.assert_array_equal(_resident(_draw(owner, acl, 1), acl),
                                  _reference(31, _MAX_OFFSET - 1, 1))
    before = owner.get_state()
    empty = _draw(owner, acl, 0)
    empty.sync()
    assert tuple(empty.shape) == (0,) and int(empty.placement_backend) == 2
    assert owner.get_state() == before and owner.get_offset() == _MAX_OFFSET
    with pytest.raises(RuntimeError, match="overflows int64"):
        _draw(owner, acl, 1)
    assert owner.get_state() == before
    for bad in [-1, 1 << 63]:
        with pytest.raises((RuntimeError, OverflowError, ValueError)):
            owner.set_offset(bad)
        assert owner.get_state() == before


@pytest.mark.parametrize("dtype,distribution", [
    ("float32", "normal"), ("float16", "uniform"),
    ("bfloat16", "uniform"), ("float64", "uniform")])
def test_native_unsupported_draw_rejected_before_reservation(acl, dtype, distribution):
    owner = _owner(acl, 29, 11)
    before = owner.get_state()
    with pytest.raises(RuntimeError, match="float32 uniform"):
        _draw(owner, acl, 7, dtype, distribution)
    assert owner.get_state() == before


def test_backend_and_ordinal_mismatch_rejected_before_reservation(acl):
    owner = _owner(acl, 41, 17)
    before = owner.get_state()
    for target_backend in [0, 1]:
        with _placement(target_backend, 0), pytest.raises(RuntimeError, match="backend"):
            jt.core.ops.random((7,), "float32", "uniform", owner)
        assert owner.get_state() == before
    for device in ["cpu", "cuda:0"]:
        foreign = jt.Generator(device).manual_seed(3)
        foreign_before = foreign.get_state()
        with pytest.raises(RuntimeError, match="backend"):
            _draw(foreign, acl, 7)
        assert foreign.get_state() == foreign_before
    # The rejection uses construction metadata only: neither case launches
    # work or allocates memory on the second ordinal.
    other = _owner(acl + 1, 7, 13)
    other_before = other.get_state()
    with pytest.raises(RuntimeError, match="device must match"):
        _draw(other, acl, 7)
    assert other.get_state() == other_before
    with pytest.raises(RuntimeError, match="device must match"):
        _draw(owner, acl + 1, 7)
    assert owner.get_state() == before


@pytest.mark.parametrize("device", ["acl:", "acl:-1", "acl:1x", "acl:2147483648"])
def test_invalid_acl_device_spelling_is_rejected(acl, device):
    with pytest.raises(RuntimeError, match="ordinal"):
        jt.Generator(device)


def test_version_algorithm_backend_device_and_numeric_state_validation_is_atomic(acl):
    owner = _owner(acl, -17, 11)
    before = owner.get_state()
    fields = before.split()
    assert fields == ["JITTOR_ACL_RNG_1", "philox4x32-10-raw32-v1",
                      "acl", str(acl), "-17", "11"]
    invalid = ["", " ".join(fields[:-1]), before + " trailing",
               jt.Generator("cpu").get_state(), jt.Generator("cuda:0").get_state()]
    for index, replacement in [(0, "JITTOR_ACL_RNG_2"), (1, "unknown"),
                               (2, "cuda"), (3, str(acl + 1)),
                               (4, str(1 << 63)), (4, "nan"),
                               (5, "-1"), (5, str(1 << 63))]:
        changed = fields.copy()
        changed[index] = replacement
        invalid.append(" ".join(changed))
    for bad in invalid:
        with pytest.raises(RuntimeError):
            owner.set_state(bad)
        assert owner.get_state() == before
        assert owner.initial_seed() == -17 and owner.get_offset() == 11
    for device in ["cpu", "cuda:0"]:
        foreign = jt.Generator(device).manual_seed(5)
        foreign_before = foreign.get_state()
        with pytest.raises(RuntimeError):
            foreign.set_state(before)
        assert foreign.get_state() == foreign_before
    expected = _resident(_draw(owner, acl, 257), acl)
    assert owner.set_state(before) is owner
    np.testing.assert_array_equal(_resident(_draw(owner, acl, 257), acl), expected)
    np.testing.assert_array_equal(expected, _reference(-17, 11, 257))
