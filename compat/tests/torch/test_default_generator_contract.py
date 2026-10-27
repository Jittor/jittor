"""Global CPU generator views, private copies and device-local mutation."""

import copy
import pickle
import random

import jittor as jt
import numpy as np
import pytest
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


def _array(value):
    return value.detach().cpu().numpy().copy()


@pytest.fixture(autouse=True)
def restore_rng():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with forbid_backend_fallbacks():
        yield
        jt.sync_all(True)
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


@pytest.mark.parametrize("seed", [17, -1, (1 << 63) + 29])
@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))])
def test_default_generator_copies_are_private_streams(seed, clone):
    default = torch.default_generator
    assert type(default) is torch.Generator
    assert torch.manual_seed(seed) is default
    _array(torch.rand(23, device="cpu"))
    state = default.get_state()
    expected = _array(torch.randn(19, device="cpu"))
    default.set_state(state)
    private = clone(default)
    assert type(private) is torch.Generator and private is not default
    assert private.initial_seed() == seed & ((1 << 64) - 1)
    np.testing.assert_array_equal(_array(torch.randn(19, generator=private, device="cpu")), expected)
    np.testing.assert_array_equal(_array(default.get_state()), _array(state))
    private.manual_seed(51)
    np.testing.assert_array_equal(_array(default.get_state()), _array(state))
    np.testing.assert_array_equal(_array(torch.randn(19, device="cpu")), expected)


@pytest.mark.parametrize("source_is_default", [False, True])
@pytest.mark.parametrize("seed", [37, -1, (1 << 63) + 29])
def test_cpu_generator_state_interchange_preserves_pending_position(source_is_default, seed):
    private = torch.Generator().manual_seed(seed)
    torch.default_generator.manual_seed(seed)
    source, target = ((torch.default_generator, private) if source_is_default
                      else (private, torch.default_generator))
    pending = torch.rand(29, generator=source, device="cpu")
    state = source.get_state()
    expected = _array(torch.rand(17, generator=source, device="cpu"))
    assert target.set_state(state) is target
    np.testing.assert_array_equal(_array(torch.rand(17, generator=target, device="cpu")), expected)
    assert np.isfinite(_array(pending)).all()
    assert target.initial_seed() == seed & ((1 << 64) - 1)


@pytest.mark.parametrize("method", ["manual_seed", "seed", "set_state"])
def test_default_generator_methods_preserve_real_cuda_stream(method):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(8100)
    devices = ["cuda:%d" % index for index in range(torch.cuda.device_count())]
    for device in devices:
        first = torch.rand(31, device=device)
        assert first.device.type == "cuda"
        _array(first)
    cuda = torch.cuda.get_rng_state_all()
    expected = [_array(torch.rand(27, device=device)) for device in devices]
    torch.cuda.set_rng_state_all(cuda)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    default = torch.default_generator
    if method == "manual_seed":
        assert default.manual_seed(-1) is default
        assert default.initial_seed() == (1 << 64) - 1
    elif method == "seed":
        seed = default.seed()
        assert 0 <= seed < (1 << 64) and seed == torch.initial_seed()
    else:
        state = torch.Generator().manual_seed(31).get_state()
        assert default.set_state(state) is default
        assert default.initial_seed() == 31
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    actual = torch.cuda.get_rng_state_all()
    assert len(actual) == len(cuda)
    for before, after in zip(cuda, actual):
        np.testing.assert_array_equal(_array(after), _array(before))
    for device, reference in zip(devices, expected):
        np.testing.assert_array_equal(_array(torch.rand(27, device=device)), reference)


@pytest.mark.parametrize("bad", [True, 1.5, "17", -(1 << 63) - 1, 1 << 64])
def test_default_manual_seed_rejects_invalid_input_before_mutation(bad):
    before = torch.get_rng_state()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        torch.default_generator.manual_seed(bad)
    np.testing.assert_array_equal(_array(torch.get_rng_state()), _array(before))
