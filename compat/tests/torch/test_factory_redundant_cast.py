"""Factory casts must convert a dtype, rather than re-enter an unchanged cast.

Observe real Tensor.cast calls while preserving their implementation. Native
factories may themselves cast intermediate values; only the compatibility
constructor's final cast is subject to the no-redundant-call contract.
"""

import os
import sys

import numpy as np
import pytest


@pytest.fixture
def runtime():
    import jittor as jt
    import torch
    device = os.environ.get("JITTOR_FACTORY_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    if device == "cuda":
        from _helpers import capability
        if not capability.check_accelerator("cuda", backend=jt).enabled:
            pytest.skip("accelerator prerequisite: real CUDA required")
    previous = torch.get_default_dtype()
    with jt.flag_scope(use_cuda=int(device == "cuda"), auto_flush_ops=0):
        torch.set_default_dtype(torch.float32)
        try:
            yield jt, torch, device
        finally:
            torch.set_default_dtype(previous)


@pytest.fixture
def cast_calls(runtime, monkeypatch):
    _, torch, _ = runtime
    from jittor._core.dtypes import dtype_name
    original = torch.Tensor.cast
    calls = []

    def observed(value, target, *args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code.co_name == "_constructor_adapter":
            calls.append((dtype_name(value.dtype), dtype_name(target)))
        return original(value, target, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cast", observed)
    return calls


_FACTORIES = (
    "zeros", "ones", "empty", "full", "rand", "randn", "zeros_like",
    "ones_like", "full_like", "rand_like", "randn_like",
    "arange", "randint", "randperm", "normal",
)


def _construct(torch, name, device, dtype, source=None, **kwargs):
    kwargs.update(dtype=dtype, device=device)
    if name.endswith("_like"):
        args = (source, 3) if name == "full_like" else (source,)
    elif name == "full":
        args = ((2, 3), 3)
    elif name == "arange":
        args = (1, 7)
    elif name == "randint":
        args = (1, 7, (2, 3))
    elif name == "randperm":
        args = (6,)
    elif name == "normal":
        args = (3., 0., (2, 3))
    else:
        args = ((2, 3),)
    return getattr(torch, name)(*args, **kwargs)


def _assert_values(value, name):
    if name in ("empty", "empty_like"):
        value.fill_(7)
        np.testing.assert_array_equal(value.numpy(), np.full((2, 3), 7))
        return
    actual = value.numpy()
    assert np.isfinite(actual).all()
    if name.startswith("zeros"):
        np.testing.assert_array_equal(actual, np.zeros((2, 3)))
    elif name.startswith("ones"):
        np.testing.assert_array_equal(actual, np.ones((2, 3)))
    elif name.startswith("full") or name == "normal":
        np.testing.assert_array_equal(actual, np.full((2, 3), 3))
    elif name == "arange":
        np.testing.assert_array_equal(actual, np.arange(1, 7))
    elif name == "randperm":
        np.testing.assert_array_equal(np.sort(actual), np.arange(6))
    elif name == "randint":
        assert ((actual >= 1) & (actual < 7)).all()
    elif name in ("rand", "rand_like"):
        assert ((actual >= 0) & (actual < 1)).all()


@pytest.mark.parametrize("name", _FACTORIES)
@pytest.mark.parametrize("wide", [False, True])
def test_explicit_dtype_skips_redundant_constructor_cast(runtime, cast_calls, name, wide):
    _, torch, device = runtime
    integer = name in ("arange", "randint", "randperm")
    dtype = (torch.int64 if wide else torch.int32) if integer else (
        torch.float64 if wide else torch.float32)
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, device=device)
    cast_calls.clear()
    result = _construct(torch, name, device, dtype, source)
    observed = list(cast_calls)
    assert result.dtype is dtype
    assert result.device.type == device
    assert not result.requires_grad
    _assert_values(result, name)
    assert not [(before, after) for before, after in observed if before == after], (
        name, observed)


@pytest.mark.parametrize("name", ["zeros", "ones", "empty", "rand", "randn"])
def test_default_dtype_is_read_each_call(runtime, cast_calls, name):
    _, torch, device = runtime
    retained = []
    all_calls = []
    for dtype in (torch.float64, torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        cast_calls.clear()
        value = getattr(torch, name)((2, 3), device=device)
        observed = list(cast_calls)
        assert value.dtype is dtype and value.device.type == device
        _assert_values(value, name)
        retained.append(value)
        all_calls.extend(observed)
    assert [value.dtype for value in retained] == [torch.float64, torch.float32, torch.float64]
    assert not [(before, after) for before, after in all_calls if before == after]


@pytest.mark.parametrize("name", ["zeros_like", "ones_like", "full_like"])
def test_like_inheritance_and_requested_conversion(runtime, cast_calls, name):
    _, torch, device = runtime
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64,
                          device=device, requires_grad=True)
    torch.set_default_dtype(torch.float32)
    args = (source, 3) if name == "full_like" else (source,)
    inherited = getattr(torch, name)(*args)
    assert inherited.dtype is torch.float64
    assert inherited.device.type == device and not inherited.requires_grad
    cast_calls.clear()
    result = _construct(torch, name, device, torch.float32, source)
    observed = list(cast_calls)
    assert result.dtype is torch.float32 and result.device.type == device
    _assert_values(result, name)
    if name == "ones_like":
        assert ("float64", "float32") in observed, "native ones_like cannot receive dtype"
    assert source.dtype is torch.float64 and source.requires_grad
    np.testing.assert_array_equal(source.numpy(), [[1., 2., 3.], [4., 5., 6.]])


@pytest.mark.parametrize("name", ["ones", "full", "rand", "randn"])
def test_factory_leaf_and_gradient_survive_cast_elision(runtime, name):
    _, torch, device = runtime
    with torch.no_grad():
        value = _construct(torch, name, device, torch.float64, requires_grad=True)
    assert value.requires_grad and value.is_leaf
    assert value.dtype is torch.float64 and value.device.type == device
    (value * 3).sum().backward()
    assert value.grad.dtype is torch.float64 and value.grad.device.type == device
    np.testing.assert_array_equal(value.grad.numpy(), np.full((2, 3), 3))


def test_accepts_dtype_does_not_guarantee_native_result_dtype(runtime, cast_calls):
    _, torch, device = runtime
    # normal uses dtype for its random input, but a tensor mean can promote the
    # native result again. Removing all final casts loses the requested dtype.
    mean = torch.tensor([1.25, 2.5], dtype=torch.float64, device=device)
    cast_calls.clear()
    result = torch.normal(mean, 0., dtype=torch.float32, device=device)
    assert result.dtype is torch.float32 and result.device.type == device
    np.testing.assert_array_equal(result.numpy(), [1.25, 2.5])
    assert ("float64", "float32") in cast_calls


def test_fractional_arange_keeps_native_conversion(runtime):
    _, torch, device = runtime
    value = torch.arange(0.25, 1.25, 0.25, dtype=torch.float64, device=device)
    assert value.dtype is torch.float64 and value.device.type == device
    np.testing.assert_array_equal(value.numpy(), [0.25, 0.5, 0.75, 1.])


@pytest.mark.parametrize("name", ["zeros", "ones", "full"])
@pytest.mark.parametrize("dtype_name", ["bool", "float16"])
def test_typed_constants_avoid_noop_cast(runtime, cast_calls, name, dtype_name):
    _, torch, device = runtime
    dtype = getattr(torch, dtype_name)
    result = _construct(torch, name, device, dtype)
    observed = list(cast_calls)
    assert result.dtype is dtype and result.device.type == device
    expected = np.full((2, 3), {"zeros": 0, "ones": 1, "full": 3}[name], dtype=dtype_name)
    np.testing.assert_array_equal(result.numpy(), expected)
    assert not [(before, after) for before, after in observed if before == after]


@pytest.mark.parametrize("name", ["tril", "triu"])
def test_tensor_factory_retains_required_cast(runtime, cast_calls, name):
    _, torch, device = runtime
    source = torch.tensor([[1.25, 2.5], [3.75, 4.]], dtype=torch.float64, device=device)
    result = getattr(torch, name)(source, dtype=torch.float32, device=device)
    assert result.dtype is torch.float32 and result.device.type == device
    expected = getattr(np, name)(np.array([[1.25, 2.5], [3.75, 4.]], dtype="float32"))
    np.testing.assert_array_equal(result.numpy(), expected)
    assert ("float64", "float32") in cast_calls


def test_like_dtype_tracks_holder_rebinding(runtime):
    jt, torch, device = runtime
    source = torch.tensor([1., 2.], dtype=torch.float32, device=device)
    first = torch.ones_like(source)
    replacement = torch.tensor([3, 4, 5], dtype=torch.int64, device=device)
    jt.Var.assign(source, replacement)
    second = torch.ones_like(source)
    assert first.dtype is torch.float32 and tuple(first.shape) == (2,)
    assert second.dtype is torch.int64 and tuple(second.shape) == (3,)
    assert first.device.type == second.device.type == device
    np.testing.assert_array_equal(first.numpy(), [1., 1.])
    np.testing.assert_array_equal(second.numpy(), [1, 1, 1])
