"""Integer ranges remain fresh leaves on their requested physical device."""

import ctypes
import functools

import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def _assert_device(value, device):
    assert value.device.type == device
    if hasattr(value, "location"):
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
    if device == "cuda" and value.numel():
        ordinal = ctypes.c_int(-1)
        driver = ctypes.CDLL("libcuda.so.1")
        assert driver.cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(value.data_ptr())) == 0
        assert ordinal.value == value.device.index


@pytest.mark.parametrize("bounds", [(6,), (2, 8), (2, 10, 3), (0,),
                                    (2**40, 2**40 + 9, 3)])
@pytest.mark.parametrize("options", [{}, {"dtype": None},
                                    {"dtype": torch.int64, "requires_grad": False}])
def test_integer_arange_fresh_leaf_and_independent_mutation(device, bounds, options):
    value = torch.arange(*bounds, device=device, **options)
    peer = torch.arange(*bounds, device=device, **options)
    expected = np.arange(*bounds, dtype=np.int64)
    assert value.dtype == torch.int64
    assert value.is_leaf and value.grad_fn is None and not value.requires_grad
    _assert_device(value, device)
    np.testing.assert_array_equal(value.cpu().numpy(), expected)
    value.add_(7)
    np.testing.assert_array_equal(value.cpu().numpy(), expected + 7)
    np.testing.assert_array_equal(peer.cpu().numpy(), expected)


def test_arange_floating_leaf_keeps_gradients(device):
    value = torch.arange(2, 8, dtype=torch.float64, device=device, requires_grad=True)
    assert value.is_leaf and value.requires_grad
    _assert_device(value, device)
    (value * 3).sum().backward()
    np.testing.assert_array_equal(value.grad.cpu().numpy(), np.full(6, 3.))


@pytest.mark.parametrize("bounds", [(np.int64(2), np.int64(8)), (0., 4., 1.)])
def test_arange_non_python_integer_bounds(device, bounds):
    value = torch.arange(*bounds, device=device, dtype=torch.int64)
    _assert_device(value, device)
    np.testing.assert_array_equal(value.cpu().numpy(), np.arange(*bounds, dtype=np.int64))


def test_overridden_arange_factory_still_detaches_its_result(device):
    jt = pytest.importorskip("jittor")
    from jittor.compat.torch.api_delegates import bind_delegates
    from jittor.compat.torch.context import get_install_context
    from jittor.compat.torch.installers import factories

    context = get_install_context(jt)
    registry = context.state["factory_implementations"]
    held = torch.arange(4, device=device)

    def custom_range(*args, dtype=None):
        return held

    replacement = functools.partial(factories._constructor_adapter, "arange",
                                    custom_range, True)
    bind_delegates(context, "factory_implementations", {**registry, "arange": replacement})
    try:
        value = torch.arange(4, device=device)
        assert value is not held
        assert value.is_leaf and not value.requires_grad
        _assert_device(value, device)
        np.testing.assert_array_equal(value.cpu().numpy(), np.arange(4))
    finally:
        bind_delegates(context, "factory_implementations", registry)
