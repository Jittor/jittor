"""Full-width seed APIs and device-native explicit generator streams."""
import ctypes
import pickle
import random
import struct

import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def require_native_sampling():
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


@pytest.fixture(params=["cpu", "cuda"])
def rng_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def _array(value):
    return value.detach().cpu().numpy().copy()


def _numpy_state_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


@pytest.mark.parametrize("seed", [-(1 << 63), -1, 0, (1 << 31) - 1,
                                  1 << 31, (1 << 32) + 17, (1 << 63) - 1,
                                  1 << 63, (1 << 64) - 1])
def test_global_full_width_seed_and_state(seed):
    expected = seed & ((1 << 64) - 1)
    assert torch.manual_seed(seed) is torch.default_generator
    assert torch.initial_seed() == expected
    state = torch.get_rng_state()
    assert state.dtype == torch.uint8 and state.device.type == "cpu"
    wanted = _array(torch.rand(19, device="cpu"))
    torch.manual_seed(71)
    torch.set_rng_state(state)
    assert torch.initial_seed() == expected
    np.testing.assert_array_equal(_array(torch.rand(19, device="cpu")), wanted)


@pytest.mark.parametrize("seed", [-(1 << 63) - 1, 1 << 64])
def test_seed_overflow_does_not_change_global_or_explicit_state(seed, rng_device):
    torch.manual_seed(19)
    global_before = _array(torch.get_rng_state())
    cuda_before = _array(torch.cuda.get_rng_state()) if rng_device == "cuda" else None
    with pytest.raises(ValueError, match="Overflow"):
        torch.manual_seed(seed)
    assert torch.initial_seed() == 19
    np.testing.assert_array_equal(_array(torch.get_rng_state()), global_before)
    if cuda_before is not None:
        np.testing.assert_array_equal(_array(torch.cuda.get_rng_state()), cuda_before)
    generator = torch.Generator(device=rng_device).manual_seed(31)
    before = _array(generator.get_state())
    with pytest.raises(ValueError, match="Overflow"):
        generator.manual_seed(seed)
    assert generator.initial_seed() == 31
    np.testing.assert_array_equal(_array(generator.get_state()), before)


@pytest.mark.parametrize("seed", [-(1 << 63), -1, 17, 1 << 63, (1 << 64) - 1])
def test_explicit_seed_state_and_pending_draw(seed, rng_device):
    generator = torch.Generator(device=rng_device)
    assert generator.initial_seed() == 67280421310721
    assert generator.manual_seed(seed) is generator
    assert generator.initial_seed() == seed & ((1 << 64) - 1)
    pending = torch.rand(17, generator=generator, device=rng_device)
    state = generator.get_state()
    assert state.dtype == torch.uint8 and state.device.type == "cpu"
    wanted = _array(torch.randn(13, generator=generator, device=rng_device))
    assert generator.set_state(state) is generator
    np.testing.assert_array_equal(_array(torch.randn(13, generator=generator, device=rng_device)), wanted)
    assert np.isfinite(_array(pending)).all()
    restored = torch.Generator(device=rng_device).manual_seed(99)
    restored.set_state(state)
    assert restored.initial_seed() == seed & ((1 << 64) - 1)
    np.testing.assert_array_equal(_array(torch.randn(13, generator=restored, device=rng_device)), wanted)


def test_generators_do_not_reseed_global_or_host_streams(rng_device):
    torch.manual_seed(709)
    expected = _array(torch.rand(11, device=rng_device))
    torch.manual_seed(709)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    generator = torch.Generator(device=rng_device).manual_seed(-1)
    _array(torch.randn(37, generator=generator, device=rng_device))
    np.testing.assert_array_equal(_array(torch.rand(11, device=rng_device)), expected)
    assert python_before == random.getstate()
    assert _numpy_state_equal(numpy_before, np.random.get_state())


def test_explicit_factories_use_native_owner_without_numpy_draws(monkeypatch, rng_device):
    def host_prng_forbidden(*args, **kwargs):
        raise AssertionError("a native generator must not use NumPy draws")
    monkeypatch.setattr(np.random, "default_rng", host_prng_forbidden)
    generator = torch.Generator(device=rng_device).manual_seed(2147483648)
    assert isinstance(generator._native, jt.Generator)
    original = jt.core.ops.random
    seen = []
    def observe(*args, **kwargs):
        handle = args[3] if len(args) > 3 else kwargs.get("generator")
        seen.append(handle)
        return original(*args, **kwargs)
    monkeypatch.setattr(jt.core.ops, "random", observe)
    with forbid_backend_fallbacks():
        source = torch.full((19,), 0.5, device=rng_device)
        draws = [torch.rand(19, generator=generator, device=rng_device),
                 torch.randn(19, generator=generator, device=rng_device, dtype=torch.float64),
                 torch.normal(0., 1., (19,), generator=generator, device=rng_device),
                 torch.normal(source, 0.2, generator=generator),
                 torch.randint(0, 100, (19,), generator=generator, device=rng_device),
                 torch.randperm(19, generator=generator, device=rng_device),
                 torch.rand_like(source, generator=generator),
                 torch.randn_like(source, generator=generator),
                 torch.bernoulli(source, generator=generator),
                 torch.multinomial(source, 7, replacement=True, generator=generator),
                 torch.multinomial(source, 7, replacement=False, generator=generator),
                 torch.normal(mean=0., std=source, generator=generator)]
        for output in draws:
            output.sync()
            assert output.device.type == rng_device
            if rng_device == "cuda":
                # Inspect the actual allocation before the explicit host read.
                ordinal = ctypes.c_int(-1)
                driver = ctypes.CDLL("libcuda.so.1")
                rc = driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                                 ctypes.c_void_p(output.data_ptr()))
                assert rc == 0 and ordinal.value == output.device.index
            else:
                assert output.placement_backend == 0
            assert np.isfinite(_array(output)).all()
        assert len(seen) == len(draws)
        assert all(handle is generator._native for handle in seen)
        assert sorted(_array(draws[5]).tolist()) == list(range(19))
        assert np.all((_array(draws[4]) >= 0) & (_array(draws[4]) < 100))
    seen.clear()


def test_generator_state_rejects_corruption_without_changing_stream(rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(53)
    state = generator.get_state()
    corrupted = torch.tensor(np.concatenate([_array(state), np.array([88], dtype=np.uint8)]),
                             dtype=torch.uint8, device="cpu")
    for bad in (torch.tensor([53], dtype=torch.int64, device="cpu"), corrupted):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            generator.set_state(bad)
        np.testing.assert_array_equal(_array(generator.get_state()), _array(state))
        assert generator.initial_seed() == 53


def test_pickled_generator_replays_native_stream(rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(-1)
    _array(torch.rand(13, generator=generator, device=rng_device))
    restored = pickle.loads(pickle.dumps(generator))
    assert restored.initial_seed() == generator.initial_seed()
    np.testing.assert_array_equal(_array(torch.rand(17, generator=restored, device=rng_device)),
                                  _array(torch.rand(17, generator=generator, device=rng_device)))


def test_generator_device_mismatch_is_rejected_before_advancing():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    for source, target in (("cpu", "cuda"), ("cuda", "cpu")):
        generator = torch.Generator(device=source).manual_seed(43)
        before = _array(generator.get_state())
        with pytest.raises(RuntimeError, match="device type"):
            torch.rand(5, generator=generator, device=target)
        np.testing.assert_array_equal(_array(generator.get_state()), before)


@pytest.mark.parametrize("seed", [-1, 1 << 31, (1 << 32) + 17, 1 << 63, (1 << 64) - 1])
def test_cuda_seed_preserves_other_streams_and_full_state(seed):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(17)
    cpu_before = _array(torch.get_rng_state())
    python_before, numpy_before = random.getstate(), np.random.get_state()
    with forbid_backend_fallbacks():
        torch.cuda.manual_seed(seed)
        assert torch.cuda.initial_seed() == seed & ((1 << 64) - 1)
        state = torch.cuda.get_rng_state()
        assert state.dtype == torch.uint8 and state.device.type == "cpu"
        expected = _array(torch.randn(19, device="cuda", dtype=torch.float64))
        torch.cuda.manual_seed(3)
        torch.cuda.set_rng_state(state)
        assert torch.cuda.initial_seed() == seed & ((1 << 64) - 1)
        np.testing.assert_array_equal(_array(torch.randn(19, device="cuda", dtype=torch.float64)), expected)
    np.testing.assert_array_equal(_array(torch.get_rng_state()), cpu_before)
    assert python_before == random.getstate()
    assert _numpy_state_equal(numpy_before, np.random.get_state())


def test_cuda_legacy_v1_state_and_invalid_offset_are_atomic():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.manual_seed(-1)
    wanted = _array(torch.rand(17, device="cuda"))
    legacy = torch.tensor(np.frombuffer(b"JTCURAND" + struct.pack("<Iqq", 1, -1, 0), dtype=np.uint8).copy(),
                          dtype=torch.uint8, device="cpu")
    torch.cuda.set_rng_state(legacy)
    assert torch.cuda.initial_seed() == (1 << 64) - 1
    np.testing.assert_array_equal(_array(torch.rand(17, device="cuda")), wanted)
    before = torch.cuda.get_rng_state()
    invalid = torch.tensor(np.frombuffer(b"JTCURAND" + struct.pack("<IQq", 2, 7, -1), dtype=np.uint8).copy(),
                           dtype=torch.uint8, device="cpu")
    with pytest.raises(ValueError, match="nonnegative"):
        torch.cuda.set_rng_state(invalid)
    np.testing.assert_array_equal(_array(torch.cuda.get_rng_state()), _array(before))


@pytest.mark.parametrize("seed", [-1, 1 << 31, 1 << 63, (1 << 64) - 1])
def test_global_seed_reaches_cuda_without_reseeding_python_or_numpy(seed):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    python_before, numpy_before = random.getstate(), np.random.get_state()
    assert torch.manual_seed(seed) is torch.default_generator
    assert torch.initial_seed() == torch.cuda.initial_seed() == seed & ((1 << 64) - 1)
    expected = _array(torch.rand(23, device="cuda"))
    torch.manual_seed(seed)
    np.testing.assert_array_equal(_array(torch.rand(23, device="cuda")), expected)
    assert python_before == random.getstate()
    assert _numpy_state_equal(numpy_before, np.random.get_state())


def test_cuda_device_seeds_and_restores_are_independent():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    seeds = [(1 << 64) - 1, (1 << 63) + 17]
    torch.manual_seed(127)
    cpu_before = _array(torch.get_rng_state())
    python_before, numpy_before = random.getstate(), np.random.get_state()
    original_device = torch.cuda.current_device()
    with forbid_backend_fallbacks():
        try:
            for index, seed in enumerate(seeds):
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(seed)
                    assert torch.cuda.initial_seed() == seed
            from jittor.distributed import checkpoint
            with jt.flag_scope(use_cuda=1):
                native_state = checkpoint._rng_snapshot()
            assert [item[0] for item in native_state["cuda"][:2]] == seeds
            states = torch.cuda.get_rng_state_all()
            assert len(states) == torch.cuda.device_count()
            assert all(state.dtype == torch.uint8 and state.device.type == "cpu" for state in states)
            expected = []
            for index in range(2):
                output = torch.rand(19, device="cuda:%d" % index)
                output.sync()
                ordinal = ctypes.c_int(-1)
                rc = ctypes.CDLL("libcuda.so.1").cuPointerGetAttribute(
                    ctypes.byref(ordinal), 9, ctypes.c_void_p(output.data_ptr()))
                assert rc == 0 and ordinal.value == index
                expected.append(_array(output))
            legacy_state = dict(native_state, cuda=list(native_state["cuda"]))
            legacy_state["cuda"][0] = (-1, native_state["cuda"][0][1])
            for snapshot in (native_state, legacy_state):
                with jt.flag_scope(use_cuda=1):
                    checkpoint._restore_rng(snapshot)
                for index in range(2):
                    with torch.cuda.device(index):
                        assert torch.cuda.initial_seed() == seeds[index]
                    np.testing.assert_array_equal(_array(torch.rand(19, device="cuda:%d" % index)), expected[index])
            for invalid_seed in (-(1 << 63) - 1, 1 << 64):
                invalid_snapshot = dict(native_state, cuda=list(native_state["cuda"]))
                invalid_snapshot["cuda"][0] = (invalid_seed, 0)
                before = [_array(state) for state in torch.cuda.get_rng_state_all()]
                with jt.flag_scope(use_cuda=1), pytest.raises(ValueError, match="out of range"):
                    checkpoint._restore_rng(invalid_snapshot)
                for actual, wanted in zip(torch.cuda.get_rng_state_all(), before):
                    np.testing.assert_array_equal(_array(actual), wanted)
            untouched = _array(torch.cuda.get_rng_state(1))
            torch.cuda.set_rng_state(states[0], 0)
            np.testing.assert_array_equal(_array(torch.cuda.get_rng_state(1)), untouched)
            with torch.cuda.device(1):
                assert torch.cuda.initial_seed() == seeds[1]
            np.testing.assert_array_equal(_array(torch.rand(19, device="cuda:0")), expected[0])
            torch.cuda.set_rng_state_all(states)
            for index in range(2):
                with torch.cuda.device(index):
                    assert torch.cuda.initial_seed() == seeds[index]
                np.testing.assert_array_equal(_array(torch.rand(19, device="cuda:%d" % index)), expected[index])
            before = [_array(state) for state in torch.cuda.get_rng_state_all()]
            invalid = list(states)
            invalid[1] = torch.tensor([0], dtype=torch.uint8, device="cpu")
            with pytest.raises(ValueError, match="byte length"):
                torch.cuda.set_rng_state_all(invalid)
            for actual, wanted in zip(torch.cuda.get_rng_state_all(), before):
                np.testing.assert_array_equal(_array(actual), wanted)
            generators = [torch.Generator(device="cuda:%d" % index).manual_seed(seed)
                          for index, seed in enumerate(seeds)]
            untouched = _array(generators[1].get_state())
            state = generators[0].get_state()
            wanted = _array(torch.randn(19, generator=generators[0], device="cuda:0"))
            generators[0].set_state(state)
            np.testing.assert_array_equal(_array(generators[1].get_state()), untouched)
            np.testing.assert_array_equal(
                _array(torch.randn(19, generator=generators[0], device="cuda:0")), wanted)
            with torch.cuda.device(1):
                assert torch.cuda.initial_seed() == seeds[1]
            torch.cuda.manual_seed_all((1 << 32) + 29)
            for index in range(2):
                with torch.cuda.device(index):
                    assert torch.cuda.initial_seed() == (1 << 32) + 29
        finally:
            torch.cuda.set_device(original_device)
    np.testing.assert_array_equal(_array(torch.get_rng_state()), cpu_before)
    assert python_before == random.getstate()
    assert _numpy_state_equal(numpy_before, np.random.get_state())


@pytest.mark.parametrize("value, expected", [(True, 1), (1.5, 1), ("7", 7),
                                             (np.int64(-1), (1 << 64) - 1)])
def test_global_seed_coercion_matches_torch(value, expected):
    assert torch.manual_seed(value) is torch.default_generator
    assert torch.initial_seed() == expected


@pytest.mark.parametrize("value, error", [(True, RuntimeError), (1.5, RuntimeError),
                                          ("7", RuntimeError), (np.int64(-1), TypeError)])
def test_generator_seed_rejects_non_python_integers_atomically(value, error, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(131)
    state = _array(generator.get_state())
    with pytest.raises(error):
        generator.manual_seed(value)
    assert generator.initial_seed() == 131
    np.testing.assert_array_equal(_array(generator.get_state()), state)
