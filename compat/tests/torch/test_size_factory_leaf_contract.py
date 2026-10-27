"""Shape factories publish fresh leaves, including scalar normal samplers."""

import contextlib
import ctypes

import numpy as np
import pytest
import torch

from _rng_init_support import (
    _array,
    _same_states,
    _states,
    preserve_random_streams,
    require_rng_device,
)


FACTORIES = (
    ("empty", ((2, 3),)), ("zeros", ((2, 3),)),
    ("ones", ((2, 3),)), ("full", ((2, 3), .5)),
    ("arange", (6,)), ("linspace", (0., 1., 6)),
    ("rand", ((2, 3),)), ("randn", ((2, 3),)),
    ("normal", (2., 3., (2, 3))),
    ("randint", (0, 7, (2, 3))), ("randperm", (6,)),
)
SAMPLERS = tuple(case for case in FACTORIES if case[0] in
                 ("rand", "randn", "normal", "randint", "randperm"))


@pytest.fixture(autouse=True)
def preserve_streams():
    with preserve_random_streams():
        yield


@pytest.fixture(params=["cpu", "cuda"])
def factory_device(request):
    return require_rng_device(request.param)


def _physical_device(value, device):
    assert value.device.type == device
    if hasattr(torch, "_torch_compat_install_context"):
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
    if device == "cuda":
        ordinal = ctypes.c_int(-1)
        driver = ctypes.CDLL("libcuda.so.1")
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                           ctypes.c_void_p(value.data_ptr())) == 0
        assert ordinal.value == value.device.index


def _assert_leaf_and_backward(value, dtype, device, requires_grad):
    assert value.is_leaf and value.grad_fn is None
    assert value.requires_grad == requires_grad and value.dtype == dtype
    _physical_device(value, device)
    with torch.no_grad():
        value.fill_(.5)
    assert value.is_leaf and value.requires_grad == requires_grad
    value.requires_grad_(True)
    (value * 2).sum().backward()
    assert value.grad is not None and value.grad.dtype == dtype
    np.testing.assert_array_equal(_array(value.grad), np.full(tuple(value.shape), 2.))
    (value * 3).sum().backward()
    np.testing.assert_array_equal(_array(value.grad), np.full(tuple(value.shape), 5.))


@pytest.mark.parametrize("name,args", FACTORIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("inside_no_grad", [False, True])
def test_size_factory_leaf_receives_complete_accumulated_gradients(
        name, args, dtype, requires_grad, inside_no_grad, factory_device):
    options = dict(dtype=dtype, device=factory_device, requires_grad=requires_grad)
    with torch.no_grad() if inside_no_grad else contextlib.nullcontext():
        value = getattr(torch, name)(*args, **options)
        peer = getattr(torch, name)(*args, **options)
    assert value.data_ptr() != peer.data_ptr()
    _assert_leaf_and_backward(value, dtype, factory_device, requires_grad)


@pytest.mark.parametrize("name,args", SAMPLERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("inside_no_grad", [False, True])
def test_private_generator_factory_is_a_leaf_without_changing_its_rng_contract(
        name, args, dtype, requires_grad, inside_no_grad, factory_device):
    generator = torch.Generator(device=factory_device).manual_seed(37)
    before, global_before = generator.get_state(), _states()
    options = dict(dtype=dtype, device=factory_device,
                   requires_grad=requires_grad, generator=generator)
    with torch.no_grad() if inside_no_grad else contextlib.nullcontext():
        value = getattr(torch, name)(*args, **options)
    _physical_device(value, factory_device)
    sampled = _array(value)
    assert not np.array_equal(_array(before), _array(generator.get_state()))
    _same_states(_states(), global_before)
    generator.set_state(before)
    with torch.no_grad() if inside_no_grad else contextlib.nullcontext():
        replay = getattr(torch, name)(*args, **options)
    np.testing.assert_array_equal(_array(replay), sampled)
    _assert_leaf_and_backward(value, dtype, factory_device, requires_grad)


@pytest.mark.parametrize("name,args", FACTORIES[1:4])
@pytest.mark.parametrize("fill", ["uniform_", "normal_"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
                                    torch.float32, torch.float64])
def test_size_factory_leaf_random_write_rejects_before_advancing_rng(
        name, args, fill, dtype, factory_device):
    value = getattr(torch, name)(*args, dtype=dtype, device=factory_device, requires_grad=True)
    assert value.is_leaf and value.requires_grad
    original = _array(value)
    generator = torch.Generator(device=factory_device).manual_seed(47)
    before, global_before = generator.get_state(), _states()
    with pytest.raises(RuntimeError, match="leaf"):
        getattr(value, fill)(generator=generator)
    np.testing.assert_array_equal(_array(value), original)
    np.testing.assert_array_equal(_array(generator.get_state()), _array(before))
    _same_states(_states(), global_before)
