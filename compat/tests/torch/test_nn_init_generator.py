"""In-place initialization consumes the selected device-native RNG stream."""

import ctypes
import random
import sys

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


@pytest.fixture(autouse=True)
def preserve_streams():
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


@pytest.fixture(params=["cpu", "cuda"])
def rng_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires real CUDA")
    return request.param


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_private_initialization_preserves_global_streams_and_replays(name, dtype, rng_device):
    torch.manual_seed(127)
    generator = torch.Generator(device=rng_device).manual_seed((1 << 63) + 17)
    tensor = torch.tensor(np.zeros((5, 7)), dtype=dtype, device=rng_device, requires_grad=True)
    assert tensor.is_leaf and tensor.requires_grad
    global_before, private_before = _states(), generator.get_state()
    host_before = random.getstate(), np.random.get_state()
    initialize = getattr(torch.nn.init, name)
    result = initialize(tensor, generator=generator)
    assert result is tensor and result.dtype == dtype and result.device.type == rng_device
    values = _array(result)
    assert not np.array_equal(_array(private_before), _array(generator.get_state()))
    _same_states(_states(), global_before)
    assert random.getstate() == host_before[0]
    numpy_after = np.random.get_state()
    assert numpy_after[0] == host_before[1][0] and numpy_after[2:] == host_before[1][2:]
    np.testing.assert_array_equal(numpy_after[1], host_before[1][1])
    generator.set_state(private_before)
    np.testing.assert_array_equal(_array(initialize(torch.zeros_like(tensor), generator=generator)), values)
    assert tensor.is_leaf and tensor.requires_grad
    (tensor * 2).sum().backward()
    np.testing.assert_array_equal(_array(tensor.grad), np.full((5, 7), 2.))


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_pending_initializations_advance_one_private_stream(name, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(-1)
    state = generator.get_state()
    initialize = getattr(torch.nn.init, name)
    first = initialize(torch.zeros(37, dtype=torch.float64, device=rng_device), generator=generator)
    second = initialize(torch.zeros_like(first), generator=generator)
    following = torch.rand(23, device=rng_device, generator=generator)
    wanted = [_array(value) for value in (first, second, following)]
    assert not np.array_equal(wanted[0], wanted[1])
    generator.set_state(state)
    restored = [initialize(torch.zeros_like(first), generator=generator) for _ in range(2)]
    restored.append(torch.rand(23, device=rng_device, generator=generator))
    for actual, expected in zip(restored, wanted):
        np.testing.assert_array_equal(_array(actual), expected)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_default_initialization_replays_selected_global_device(name, rng_device):
    torch.manual_seed(51)
    state = torch.cuda.get_rng_state() if rng_device == "cuda" else torch.get_rng_state()
    initialize = getattr(torch.nn.init, name)
    first = initialize(torch.zeros(31, dtype=torch.float64, device=rng_device))
    wanted = _array(first)
    current = torch.cuda.get_rng_state() if rng_device == "cuda" else torch.get_rng_state()
    assert not np.array_equal(_array(state), _array(current))
    if rng_device == "cuda":
        torch.cuda.set_rng_state(state)
    else:
        torch.set_rng_state(state)
    np.testing.assert_array_equal(_array(initialize(torch.zeros_like(first))), wanted)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_explicit_default_cpu_generator_is_same_global_stream(name):
    torch.manual_seed(73)
    state = torch.get_rng_state()
    initialize = getattr(torch.nn.init, name)
    wanted = _array(initialize(torch.zeros(19, device="cpu"), generator=torch.default_generator))
    torch.set_rng_state(state)
    np.testing.assert_array_equal(_array(initialize(torch.zeros(19, device="cpu"))), wanted)


@pytest.mark.parametrize("name,args", [("uniform_", (2., 1.)), ("uniform_", (float("nan"), 1.)),
                                       ("uniform_", (-float("inf"), 1.)),
                                       ("uniform_", (0., float("inf"))),
                                       ("normal_", (0., -1.)), ("normal_", (0., float("nan")))])
def test_invalid_distribution_is_atomic(name, args, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(61)
    tensor = torch.full((17,), 3., dtype=torch.float64, device=rng_device)
    states, before = _states(), generator.get_state()
    with pytest.raises((RuntimeError, ValueError)):
        getattr(torch.nn.init, name)(tensor, *args, generator=generator)
    np.testing.assert_array_equal(_array(tensor), np.full(17, 3.))
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), states)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_invalid_generator_and_integral_target_are_atomic(name, rng_device):
    initialize = getattr(torch.nn.init, name)
    generator = torch.Generator(device=rng_device).manual_seed(91)
    states, before = _states(), generator.get_state()
    with pytest.raises(TypeError):
        initialize(torch.zeros(17, device=rng_device), generator=object())
    integer = torch.zeros(17, dtype=torch.int64, device=rng_device)
    with pytest.raises((RuntimeError, ValueError)):
        initialize(integer, generator=generator)
    np.testing.assert_array_equal(_array(integer), np.zeros(17))
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), states)


def test_uniform_bounds_and_normal_parameters(rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(41)
    uniform = _array(torch.nn.init.uniform_(torch.zeros(4096, dtype=torch.float64, device=rng_device),
                                          -3.5, -1.25, generator=generator))
    assert np.all(uniform >= -3.5) and np.all(uniform < -1.25)
    normal = _array(torch.nn.init.normal_(torch.zeros(4096, dtype=torch.float64, device=rng_device),
                                        7., 2., generator=generator))
    assert abs(normal.mean() - 7.) < .15 and abs(normal.std() - 2.) < .15
    for name, args, expected in (("uniform_", (2.5, 2.5), 2.5), ("normal_", (-7., 0.), -7.)):
        before = generator.get_state()
        tensor = torch.zeros(19, dtype=torch.float64, device=rng_device)
        np.testing.assert_array_equal(_array(getattr(torch.nn.init, name)(tensor, *args, generator=generator)),
                                      np.full(19, expected))
        assert not np.array_equal(_array(before), _array(generator.get_state()))


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_initialization_rejects_generator_device_mismatch(name):
    if not torch.cuda.is_available():
        pytest.skip("requires real CUDA")
    for source, target in (("cpu", "cuda"), ("cuda", "cpu")):
        generator = torch.Generator(device=source).manual_seed(67)
        before, states = generator.get_state(), _states()
        tensor = torch.full((19,), 4., device=target)
        with pytest.raises(RuntimeError):
            getattr(torch.nn.init, name)(tensor, generator=generator)
        np.testing.assert_array_equal(_array(tensor), np.full(19, 4.))
        np.testing.assert_array_equal(_array(before), _array(generator.get_state()))
        _same_states(_states(), states)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_native_initialization_draw_owner_and_actual_allocation(name, rng_device, monkeypatch):
    import jittor as jt

    def forbidden(*args, **kwargs):
        raise AssertionError("native initialization must not draw through NumPy")

    for entry in ("normal", "uniform", "default_rng"):
        monkeypatch.setattr(np.random, entry, forbidden)
    original, seen = jt.core.ops.random, []

    def observe(*args, **kwargs):
        handle = args[3] if len(args) > 3 else kwargs.get("generator")
        output = original(*args, **kwargs)
        seen.append((handle, output))
        return output

    monkeypatch.setattr(jt.core.ops, "random", observe)
    generator = torch.Generator(device=rng_device).manual_seed(29)
    tensor = torch.zeros(23, dtype=torch.float64, device=rng_device)
    getattr(torch.nn.init, name)(tensor, generator=generator)
    tensor.sync()
    assert len(seen) == 1 and seen[0][0] is generator._native
    output = seen[0][1]
    output.sync()
    assert output.dtype == torch.float64 and output.device.type == rng_device
    if rng_device == "cuda":
        driver, ordinal = ctypes.CDLL("libcuda.so.1"), ctypes.c_int(-1)
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(output.data_ptr())) == 0
        assert ordinal.value == output.device.index
    else:
        assert output.placement_backend == 0
    assert np.isfinite(_array(tensor)).all()
    seen.clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_uniform_dtype_range_validation_is_atomic(dtype, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(59)
    tensor = torch.zeros(17, dtype=dtype, device=rng_device)
    limit = torch.finfo(dtype).max
    for low, high in ((-2. * limit, 0.), (0., 2. * limit), (-limit, limit)):
        before, states = generator.get_state(), _states()
        with pytest.raises(RuntimeError):
            torch.nn.init.uniform_(tensor, low, high, generator=generator)
        np.testing.assert_array_equal(_array(tensor), np.zeros(17))
        np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
        _same_states(_states(), states)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_initialization_writes_retained_views_without_freezing(name, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(43)
    state = generator.get_state()
    parent = torch.tensor(np.zeros((4, 9)), dtype=torch.float64, device=rng_device, requires_grad=True)
    view = parent[:, 1:8:2]
    assert getattr(torch.nn.init, name)(view, generator=generator) is view
    values = _array(parent)
    generator.set_state(state)
    replay_parent = torch.zeros((4, 9), dtype=torch.float64, device=rng_device)
    expected = _array(getattr(torch.nn.init, name)(replay_parent[:, 1:8:2], generator=generator))
    np.testing.assert_array_equal(values[:, 1:8:2], expected)
    np.testing.assert_array_equal(values[:, ::2], np.zeros((4, 5)))
    assert parent.is_leaf and parent.requires_grad and view.requires_grad
    (parent * 3).sum().backward()
    np.testing.assert_array_equal(_array(parent.grad), np.full((4, 9), 3.))
