"""Pending matrix updates keep aliases, snapshots and copy destinations distinct."""
import numpy as np
import pytest
import jittor as jt
import torch
from _helpers.common import device_flags_for, get_all_device_types


@pytest.fixture(params=get_all_device_types())
def device(request):
    options = device_flags_for(request.param)
    options['auto_flush_ops'] = 0
    with jt.flag_scope(**options):
        yield request.param


def _matrix(device, dtype=torch.float32):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) % 4 - 1
    b = np.arange(24, dtype=np.float32).reshape(2, 4, 3) % 3 - 1
    x = torch.tensor(a, device=device, dtype=dtype)
    y = torch.tensor(b, device=device, dtype=dtype)
    return torch.matmul(x, y), a @ b


def _physical(value, device):
    assert value.device.type == device
    assert value.location() == ("cpu" if device == "cpu" else "device")
    assert value.placement_backend == (0 if device == "cpu" else 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unshared_pending_matmul_update(device, dtype):
    with torch.no_grad():
        value, expected = _matrix(device, dtype)
        assert not value._mem_ptr_now
        result = value.add_(torch.ones((3, 3), device=device, dtype=dtype))
        assert result is value
        if device == "cuda":
            assert not value._mem_ptr_now, "an unobserved destination needs no eager store"
        np.testing.assert_array_equal(value.numpy(), expected + 1)
        _physical(value, device)


@pytest.mark.parametrize("alias_kind", ["detach", "data", "view", "snapshot", "holder"])
def test_pending_matmul_with_an_existing_reader(device, alias_kind):
    with torch.no_grad():
        value, expected = _matrix(device)
        if alias_kind == "detach":
            other = value.detach()
        elif alias_kind == "data":
            other = value.data
        elif alias_kind == "view":
            other = value.view(2, 9)
        elif alias_kind == "holder":
            other = value._alias()
        else:
            other = value.clone()
        value.add_(2)
        np.testing.assert_array_equal(value.numpy(), expected + 2)
        target = expected if alias_kind == "snapshot" else expected + 2
        np.testing.assert_array_equal(other.numpy(), target.reshape(tuple(other.shape)))
        _physical(value, device)
        _physical(other, device)


def test_deferred_copy_does_not_alias_the_rhs(device):
    with torch.no_grad():
        value, expected = _matrix(device)
        rhs = value + 1
        assert value.copy_(rhs) is value
        if device == "cuda":
            assert not value._mem_ptr_now
        value.add_(2)
        np.testing.assert_array_equal(rhs.numpy(), expected + 1)
        np.testing.assert_array_equal(value.numpy(), expected + 3)
        rhs.mul_(3)
        np.testing.assert_array_equal(rhs.numpy(), (expected + 1) * 3)
        np.testing.assert_array_equal(value.numpy(), expected + 3)
        assert value.data_ptr() != rhs.data_ptr()
        _physical(value, device)


def test_materialized_matmul_keeps_its_storage(device):
    with torch.no_grad():
        value, expected = _matrix(device)
        pointer = value.data_ptr()
        value.add_(3)
        assert value.data_ptr() == pointer
        np.testing.assert_array_equal(value.numpy(), expected + 3)
        _physical(value, device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deferred_matmul_update_matches_materialized_rounding(device, dtype):
    rng = np.random.RandomState(72941)
    a = torch.tensor(rng.randn(2, 3, 4).astype(np.float32), device=device, dtype=dtype)
    b = torch.tensor(rng.randn(2, 4, 3).astype(np.float32), device=device, dtype=dtype)
    bias = torch.tensor(rng.randn(2, 3, 3).astype(np.float32), device=device, dtype=dtype)
    with torch.no_grad():
        reference = torch.matmul(a, b)
        reference.numpy()
        reference.add_(bias)
        expected = reference.numpy().copy()
        value = torch.matmul(a, b)
        value.add_(bias)
        if device == "cuda":
            assert not value._mem_ptr_now
        np.testing.assert_array_equal(value.numpy(), expected)
        _physical(value, device)


def test_pending_updates_while_auto_flush_submits_work(device):
    values = []
    with torch.no_grad(), jt.flag_scope(auto_flush_ops=4):
        for index in range(24):
            value, expected = _matrix(device)
            value.add_(index)
            values.append((value, expected + index))
    jt.sync_all(True)
    for value, expected in values:
        np.testing.assert_array_equal(value.numpy(), expected)
        _physical(value, device)
