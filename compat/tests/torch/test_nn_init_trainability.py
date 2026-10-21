"""Initialization writes values without replacing the destination's graph."""

import numpy as np
import pytest
import torch

from _rng_init_support import _array, preserve_random_streams, require_rng_device


_INITIALIZERS = (
    "uniform_", "normal_", "zeros_", "ones_", "constant_",
    "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
)


@pytest.fixture(autouse=True)
def preserve_streams():
    with preserve_random_streams():
        yield


@pytest.fixture(params=["cpu", "cuda"])
def rng_device(request):
    return require_rng_device(request.param)


def _initialize(name, tensor):
    return getattr(torch.nn.init, name)(tensor, *((1.25,) if name == "constant_" else ()))


def _assert_initialized(name, tensor):
    values = _array(tensor)
    assert np.isfinite(values).all()
    assert not np.array_equal(values, np.full(values.shape, 99.))
    if name in ("zeros_", "ones_", "constant_"):
        expected = {"zeros_": 0., "ones_": 1., "constant_": 1.25}[name]
        np.testing.assert_array_equal(values, np.full(values.shape, expected))


@pytest.mark.parametrize("name", _INITIALIZERS)
@pytest.mark.parametrize("parameter", [False, True])
@pytest.mark.parametrize("state", ["enabled", "constructed_frozen", "reversible_frozen"])
def test_initialization_preserves_actual_leaf_trainability(name, parameter, state, rng_device):
    enabled = state != "constructed_frozen"
    tensor = torch.tensor(np.full((3, 4), 99.), dtype=torch.float64,
                          device=rng_device, requires_grad=enabled)
    if parameter:
        tensor = torch.nn.Parameter(tensor, requires_grad=enabled)
    if state == "reversible_frozen":
        tensor.requires_grad_(False)
    trainable = tensor.requires_grad
    pointer, snapshot = tensor.data_ptr(), tensor.clone()
    result = _initialize(name, tensor)
    assert result is tensor and tensor.data_ptr() == pointer
    assert tensor.requires_grad == trainable and tensor.is_leaf
    assert tensor.dtype == torch.float64 and tensor.device.type == rng_device
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 4), 99.))
    _assert_initialized(name, tensor)
    # Both kinds of frozen leaf must remain usable after explicit re-enabling.
    tensor.requires_grad_(True)
    (tensor * 2).sum().backward()
    np.testing.assert_array_equal(_array(tensor.grad), np.full((3, 4), 2.))


@pytest.mark.parametrize("name", _INITIALIZERS)
def test_initialization_preserves_nonleaf_graph(name, rng_device):
    root = torch.tensor(np.full((3, 4), 33.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    tensor = root * 3
    pointer, snapshot = tensor.data_ptr(), tensor.clone()
    assert not tensor.is_leaf
    assert _initialize(name, tensor) is tensor
    assert tensor.data_ptr() == pointer and not tensor.is_leaf and tensor.requires_grad
    assert tensor.dtype == torch.float64 and tensor.device.type == rng_device
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 4), 99.))
    _assert_initialized(name, tensor)
    tensor.sum().backward()
    np.testing.assert_array_equal(_array(root.grad), np.full((3, 4), 3.))

@pytest.mark.parametrize("name", ["uniform_", "normal_", "kaiming_normal_"])
def test_initialization_preserves_leaf_view_owner(name, rng_device):
    root = torch.tensor(np.full((3, 8), 99.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    view = root[:, 1:5]
    snapshot = root.clone()
    assert _initialize(name, view) is view
    assert root.is_leaf and root.requires_grad and view.requires_grad and not view.is_leaf
    assert view.dtype == torch.float64 and view.device.type == rng_device
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 8), 99.))
    _assert_initialized(name, view)
    np.testing.assert_array_equal(_array(root)[:, :1], np.full((3, 1), 99.))
    np.testing.assert_array_equal(_array(root)[:, 5:], np.full((3, 3), 99.))
    view.sum().backward()
    expected = np.zeros((3, 8))
    expected[:, 1:5] = 1.
    np.testing.assert_array_equal(_array(root.grad), expected)
