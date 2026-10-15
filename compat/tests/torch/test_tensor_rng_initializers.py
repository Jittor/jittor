"""Tensor random fills share samplers, preserving in-place autograd semantics."""

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

def _fill(tensor, name, *args, **kwargs):
    with torch.no_grad():
        return getattr(tensor, name)(*args, **kwargs)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_tensor_private_stream_replays_without_global_mutation(name, dtype, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed((1 << 63) + 17)
    tensor = torch.tensor(np.zeros((5, 7)), dtype=dtype, device=rng_device, requires_grad=True)
    assert tensor.is_leaf
    before, global_before = generator.get_state(), _states()
    host_before = random.getstate(), np.random.get_state()
    assert _fill(tensor, name, generator=generator) is tensor
    assert tensor.dtype == dtype and tensor.device.type == rng_device
    values = _array(tensor)
    assert not np.array_equal(_array(before), _array(generator.get_state()))
    _same_states(_states(), global_before)
    assert random.getstate() == host_before[0]
    numpy_after = np.random.get_state()
    assert numpy_after[0] == host_before[1][0] and numpy_after[2:] == host_before[1][2:]
    np.testing.assert_array_equal(numpy_after[1], host_before[1][1])
    generator.set_state(before)
    frozen = torch.zeros_like(tensor)
    assert not frozen.requires_grad
    np.testing.assert_array_equal(_array(_fill(frozen, name, generator=generator)), values)
    assert not frozen.requires_grad
    assert tensor.is_leaf and tensor.requires_grad
    (tensor * 2).sum().backward()
    np.testing.assert_array_equal(_array(tensor.grad), np.full((5, 7), 2.))


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_pending_fills_and_factory_share_one_private_stream(name, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(-1)
    before = generator.get_state()
    first = _fill(torch.zeros(37, dtype=torch.float64, device=rng_device), name, generator=generator)
    second = _fill(torch.zeros_like(first), name, generator=generator)
    following = torch.rand(23, device=rng_device, generator=generator)
    expected = [_array(value) for value in (first, second, following)]
    assert not np.array_equal(expected[0], expected[1])
    generator.set_state(before)
    replay = [_fill(torch.zeros_like(first), name, generator=generator) for _ in range(2)]
    replay.append(torch.rand(23, device=rng_device, generator=generator))
    for actual, wanted in zip(replay, expected):
        np.testing.assert_array_equal(_array(actual), wanted)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_default_stream_replays_and_matches_init_consumer(name, rng_device):
    torch.manual_seed(51)
    state = torch.cuda.get_rng_state() if rng_device == "cuda" else torch.get_rng_state()
    tensor = torch.zeros(31, dtype=torch.float64, device=rng_device)
    expected = _array(_fill(tensor, name))
    advanced = torch.cuda.get_rng_state() if rng_device == "cuda" else torch.get_rng_state()
    assert not np.array_equal(_array(state), _array(advanced))
    if rng_device == "cuda":
        torch.cuda.set_rng_state(state)
    else:
        torch.set_rng_state(state)
    np.testing.assert_array_equal(_array(getattr(torch.nn.init, name)(torch.zeros_like(tensor))), expected)


@pytest.mark.parametrize("name,args", [("uniform_", (2., 1.)), ("uniform_", (float("nan"), 1.)),
                                       ("uniform_", (-float("inf"), 1.)),
                                       ("uniform_", (0., float("inf"))),
                                       ("normal_", (0., -1.)), ("normal_", (0., float("nan")))])
def test_tensor_invalid_distribution_is_atomic(name, args, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(61)
    tensor = torch.full((17,), 3., dtype=torch.float64, device=rng_device)
    before, global_before = generator.get_state(), _states()
    with pytest.raises((RuntimeError, ValueError)):
        _fill(tensor, name, *args, generator=generator)
    np.testing.assert_array_equal(_array(tensor), np.full(17, 3.))
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), global_before)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_invalid_generator_and_integer_dtype_are_atomic(name, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(91)
    before, global_before = generator.get_state(), _states()
    with pytest.raises(TypeError):
        _fill(torch.zeros(17, device=rng_device), name, generator=object())
    tensor = torch.zeros(17, dtype=torch.int64, device=rng_device)
    with pytest.raises((RuntimeError, ValueError)):
        _fill(tensor, name, generator=generator)
    np.testing.assert_array_equal(_array(tensor), np.zeros(17))
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), global_before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_tensor_uniform_dtype_bounds_are_atomic(dtype, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(59)
    tensor = torch.zeros(17, dtype=dtype, device=rng_device)
    limit = torch.finfo(dtype).max
    for low, high in ((-2. * limit, 0.), (0., 2. * limit), (-limit, limit)):
        before, global_before = generator.get_state(), _states()
        with pytest.raises(RuntimeError):
            _fill(tensor, "uniform_", low, high, generator=generator)
        np.testing.assert_array_equal(_array(tensor), np.zeros(17))
        np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
        _same_states(_states(), global_before)


def test_tensor_distribution_parameters(rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(41)
    uniform = _array(_fill(torch.zeros(4096, dtype=torch.float64, device=rng_device), "uniform_",
                           -3.5, -1.25, generator=generator))
    assert np.all(uniform >= -3.5) and np.all(uniform < -1.25)
    normal = _array(_fill(torch.zeros(4096, dtype=torch.float64, device=rng_device), "normal_",
                          7., 2., generator=generator))
    assert abs(normal.mean() - 7.) < .15 and abs(normal.std() - 2.) < .15
    for name, args, expected in (("uniform_", (2.5, 2.5), 2.5), ("normal_", (-7., 0.), -7.)):
        before = generator.get_state()
        actual = _fill(torch.zeros(19, dtype=torch.float64, device=rng_device), name, *args, generator=generator)
        np.testing.assert_array_equal(_array(actual), np.full(19, expected))
        assert not np.array_equal(_array(before), _array(generator.get_state()))


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
@pytest.mark.parametrize("use_view", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_tensor_leaf_inplace_rejects_before_drawing(name, use_view, explicit, rng_device):
    parent = torch.tensor(np.ones((3, 7)), dtype=torch.float64, device=rng_device, requires_grad=True)
    target = parent[:, 1:5] if use_view else parent
    generator = torch.Generator(device=rng_device).manual_seed(19)
    before, global_before = generator.get_state(), _states()
    with pytest.raises(RuntimeError, match="leaf"):
        getattr(target, name)(generator=generator if explicit else None)
    np.testing.assert_array_equal(_array(parent), np.ones((3, 7)))
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), global_before)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_explicit_default_generator_is_same_cpu_stream(name):
    torch.manual_seed(73)
    before = torch.get_rng_state()
    tensor = torch.zeros(19, device="cpu")
    expected = _array(_fill(tensor, name, generator=torch.default_generator))
    torch.set_rng_state(before)
    np.testing.assert_array_equal(_array(_fill(torch.zeros_like(tensor), name)), expected)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_nonleaf_overwrite_has_zero_derivative(name, rng_device):
    root = torch.tensor([float("nan"), float("inf"), 3., -2.], dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    target = root * 2
    generator = torch.Generator(device=rng_device).manual_seed(31)
    before = generator.get_state()
    assert getattr(target, name)(generator=generator) is target
    assert not target.is_leaf and target.requires_grad
    values = _array(target)
    assert np.isfinite(values).all()
    target.sum().backward()
    np.testing.assert_array_equal(_array(root.grad), np.zeros(4))
    generator.set_state(before)
    np.testing.assert_array_equal(_array(_fill(torch.zeros_like(target), name, generator=generator)), values)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_retained_view_writeback_under_no_grad(name, rng_device):
    generator = torch.Generator(device=rng_device).manual_seed(43)
    state = generator.get_state()
    parent = torch.tensor(np.zeros((4, 9)), dtype=torch.float64, device=rng_device, requires_grad=True)
    view = parent[:, 1:8:2]
    assert _fill(view, name, generator=generator) is view
    values = _array(parent)
    generator.set_state(state)
    replay_parent = torch.zeros((4, 9), dtype=torch.float64, device=rng_device)
    expected = _array(_fill(replay_parent[:, 1:8:2], name, generator=generator))
    np.testing.assert_array_equal(values[:, 1:8:2], expected)
    np.testing.assert_array_equal(values[:, ::2], np.zeros((4, 5)))
    assert parent.is_leaf and parent.requires_grad and view.requires_grad
    (parent * 3).sum().backward()
    np.testing.assert_array_equal(_array(parent.grad), np.full((4, 9), 3.))


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_generator_device_mismatch_is_atomic(name):
    if not torch.cuda.is_available():
        pytest.skip("requires real CUDA")
    for source, target in (("cpu", "cuda"), ("cuda", "cpu")):
        generator = torch.Generator(device=source).manual_seed(67)
        tensor = torch.full((19,), 4., device=target)
        before, global_before = generator.get_state(), _states()
        with pytest.raises(RuntimeError):
            _fill(tensor, name, generator=generator)
        np.testing.assert_array_equal(_array(tensor), np.full(19, 4.))
        np.testing.assert_array_equal(_array(before), _array(generator.get_state()))
        _same_states(_states(), global_before)


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_native_tensor_fill_uses_selected_generator_and_device(name, rng_device, monkeypatch):
    import jittor as jt

    def forbidden(*args, **kwargs):
        raise AssertionError("a native fill must not draw through NumPy")

    for entry in ("normal", "uniform", "default_rng"):
        monkeypatch.setattr(np.random, entry, forbidden)
    original, seen = jt.core.ops.random, []

    def observe(*args, **kwargs):
        output = original(*args, **kwargs)
        seen.append((args[3] if len(args) > 3 else kwargs.get("generator"), output))
        return output

    monkeypatch.setattr(jt.core.ops, "random", observe)
    generator = torch.Generator(device=rng_device).manual_seed(29)
    tensor = torch.zeros(23, dtype=torch.float64, device=rng_device)
    _fill(tensor, name, generator=generator)
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


@pytest.mark.parametrize("name", ["uniform_", "normal_"])
def test_tensor_no_grad_nonleaf_preserves_old_graph_and_pending_reader(name, rng_device):
    source = torch.tensor([1., 2., 3., 4.], dtype=torch.float64, device=rng_device, requires_grad=True)
    target = source * 2
    pointer = target.data_ptr()
    pending_reader = target * 3
    generator = torch.Generator(device=rng_device).manual_seed(47)
    assert _fill(target, name, generator=generator) is target
    assert target.data_ptr() == pointer and not target.is_leaf and target.requires_grad
    np.testing.assert_array_equal(_array(pending_reader), np.arange(1., 5.) * 6)
    assert np.isfinite(_array(target)).all()
    target.sum().backward()
    np.testing.assert_array_equal(_array(source.grad), np.full(4, 2.))
