"""No-grad storage copies retain basic views and their pre-existing gradients."""

import ctypes
import sys

import numpy as np
import pytest
import torch

from _rng_init_support import _array, preserve_random_streams, require_rng_device


@pytest.fixture(autouse=True)
def preserve_streams():
    with preserve_random_streams():
        yield


@pytest.fixture(params=["cpu", "cuda"])
def rng_device(request):
    return require_rng_device(request.param)


def _view(value, layout):
    if layout == "slice":
        return value[:, 1:5]
    if layout == "step":
        return value[:, ::2]
    if layout == "transpose":
        return value.T
    return value[:, 1:7:2].T


@pytest.mark.parametrize("layout", ["slice", "step", "transpose", "nested"])
@pytest.mark.parametrize("nonleaf_base", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_no_grad_copy_preserves_strided_graph_and_pending_readers(layout, nonleaf_base, dtype, rng_device):
    root = torch.tensor(np.full((3, 8), 11.), dtype=dtype, device=rng_device, requires_grad=True)
    scale = 3 if nonleaf_base else 1
    base = root * scale if nonleaf_base else root
    view = _view(base, layout)
    pointer, base_pointer, snapshot = view.data_ptr(), base.data_ptr(), base.clone()
    values = np.arange(view.numel(), dtype=np.float32).reshape(tuple(view.shape))
    # Also exercise dtype conversion and an actual CPU -> CUDA copy.
    source = torch.tensor(values, dtype=torch.float32, device="cpu", requires_grad=True)
    with torch.no_grad():
        assert view.copy_(source) is view
    assert view.data_ptr() == pointer and base.data_ptr() == base_pointer
    assert view.requires_grad and not view.is_leaf
    assert base.is_leaf == (not nonleaf_base) and base.requires_grad
    assert view.dtype == dtype and view.device.type == rng_device
    assert source.requires_grad and source.is_leaf and source.device.type == "cpu"
    expected = np.full((3, 8), 11. * scale)
    _view(expected, layout)[...] = values
    np.testing.assert_array_equal(_array(base), expected)
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 8), 11. * scale))
    view.sum().backward()
    gradient = np.zeros((3, 8))
    _view(gradient, layout)[...] = scale
    np.testing.assert_array_equal(_array(root.grad), gradient)
    assert source.grad is None


@pytest.mark.parametrize("alias", ["identical", "interleaved", "disjoint_rows"])
def test_no_grad_copy_handles_source_aliases(alias, rng_device):
    root = torch.tensor(np.arange(16.).reshape(4, 4), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    if alias == "identical":
        destination = source = root[:, ::2]
        expected = np.arange(16.).reshape(4, 4)
    elif alias == "interleaved":
        destination, source = root[:, ::2], root[:, 1::2]
        expected = np.arange(16.).reshape(4, 4)
        expected[:, ::2] = expected[:, 1::2].copy()
    else:
        destination, source = root[2:], root[:2]
        expected = np.arange(16.).reshape(4, 4)
        expected[2:] = expected[:2].copy()
    pointer, snapshot = root.data_ptr(), root.clone()
    with torch.no_grad():
        assert destination.copy_(source) is destination
    assert root.data_ptr() == pointer and root.is_leaf and root.requires_grad
    assert not destination.is_leaf and destination.requires_grad
    np.testing.assert_array_equal(_array(root), expected)
    np.testing.assert_array_equal(_array(snapshot), np.arange(16.).reshape(4, 4))
    root.sum().backward()
    np.testing.assert_array_equal(_array(root.grad), np.ones((4, 4)))


@pytest.mark.parametrize("overlap", ["shift", "transpose", "expand"])
def test_no_grad_copy_rejects_known_overlaps_before_writing(overlap, rng_device):
    root = torch.tensor(np.arange(16.).reshape(4, 4), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    if overlap == "shift":
        flat = root.view(-1)
        destination, source = flat[1:], flat[:-1]
    elif overlap == "transpose":
        destination, source = root, root.T
    else:
        destination = root[:, :1].expand(4, 4)
        source = torch.tensor(np.ones((4, 4)), dtype=torch.float64, device=rng_device)
    pointer = root.data_ptr()
    with torch.no_grad(), pytest.raises(RuntimeError, match="overlap|single memory"):
        destination.copy_(source)
    assert root.data_ptr() == pointer and root.is_leaf and root.requires_grad
    np.testing.assert_array_equal(_array(root), np.arange(16.).reshape(4, 4))


@pytest.mark.parametrize("destination_kind", ["root", "slice", "transpose"])
@pytest.mark.parametrize("nonleaf_base", [False, True])
def test_no_grad_copy_refreshes_materialized_siblings(destination_kind, nonleaf_base, rng_device):
    root = torch.tensor(np.full((3, 8), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    scale = 3 if nonleaf_base else 1
    base = root * scale if nonleaf_base else root
    sibling, nested = base.T, base[:, 1:7:2].T
    destination = base if destination_kind == "root" else _view(base, destination_kind)
    tensors = (base, destination, sibling, nested)
    pointers, snapshot = [value.data_ptr() for value in tensors], base.clone()
    values = np.arange(destination.numel(), dtype=np.float64).reshape(tuple(destination.shape))
    source = torch.tensor(values, dtype=torch.float64, device=rng_device, requires_grad=True)
    with torch.no_grad():
        assert destination.copy_(source) is destination
    assert [value.data_ptr() for value in tensors] == pointers
    assert base.is_leaf == (not nonleaf_base) and base.requires_grad
    assert not sibling.is_leaf and not nested.is_leaf
    expected = np.full((3, 8), 11. * scale)
    if destination_kind == "root":
        expected[...] = values
    else:
        _view(expected, destination_kind)[...] = values
    np.testing.assert_array_equal(_array(base), expected)
    np.testing.assert_array_equal(_array(sibling), expected.T)
    np.testing.assert_array_equal(_array(nested), expected[:, 1:7:2].T)
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 8), 11. * scale))
    (destination.sum() + sibling.sum() * 2 + nested.sum() * 3).backward()
    gradient = np.full((3, 8), 2.)
    gradient[:, 1:7:2] += 3
    if destination_kind == "root":
        gradient += 1
    else:
        _view(gradient, destination_kind)[...] += 1
    np.testing.assert_array_equal(_array(root.grad), gradient * scale)
    assert source.grad is None


@pytest.mark.parametrize("name", ["uniform_", "normal_", "kaiming_normal_"])
def test_initialization_preserves_materialized_view_owner(name, rng_device):
    root = torch.tensor(np.full((3, 4), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    view, sibling = root.T, root.T
    tensors = (root, view, sibling)
    pointers, snapshot = [value.data_ptr() for value in tensors], root.clone()
    assert getattr(torch.nn.init, name)(view) is view
    assert [value.data_ptr() for value in tensors] == pointers
    assert root.is_leaf and root.requires_grad and not view.is_leaf and not sibling.is_leaf
    values = _array(root)
    assert np.isfinite(values).all() and not np.array_equal(values, np.full((3, 4), 11.))
    np.testing.assert_array_equal(_array(view), values.T)
    np.testing.assert_array_equal(_array(sibling), values.T)
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 4), 11.))
    (view.sum() + sibling.sum() * 2).backward()
    np.testing.assert_array_equal(_array(root.grad), np.full((3, 4), 3.))


def test_copy_refreshes_readonly_expanded_materialized_sibling(rng_device):
    root = torch.tensor(np.full((3, 4), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    sibling = root.T.unsqueeze(0).expand(2, 4, 3)
    pointers, snapshot = (root.data_ptr(), sibling.data_ptr()), root.clone()
    source = torch.tensor(np.arange(12.).reshape(3, 4), dtype=torch.float64, device=rng_device)
    with torch.no_grad():
        assert root.copy_(source) is root
    assert (root.data_ptr(), sibling.data_ptr()) == pointers
    assert root.is_leaf and not sibling.is_leaf and sibling.requires_grad
    np.testing.assert_array_equal(_array(sibling),
                                  np.broadcast_to(np.arange(12.).reshape(3, 4).T, (2, 4, 3)))
    np.testing.assert_array_equal(_array(snapshot), np.full((3, 4), 11.))
    (root.sum() + sibling.sum()).backward()
    np.testing.assert_array_equal(_array(root.grad), np.full((3, 4), 3.))


def test_native_copy_observer_retains_actual_device_buffers(rng_device):
    if "jittor" not in sys.modules:
        pytest.skip("native binding observer")
    root = torch.tensor(np.full((3, 8), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    view = root[:, 1:5]
    source = torch.tensor(np.arange(12.).reshape(3, 4), dtype=torch.float64, device="cpu")
    pointer = view.data_ptr()
    with torch.no_grad():
        view._copy_into(source)
    view.sync()
    assert view.data_ptr() == pointer and not view.is_leaf and view.requires_grad
    assert source.location() == "cpu"
    if rng_device == "cuda":
        assert view.location() == "device" and root.location() == "device"
        driver, ordinal = ctypes.CDLL("libcuda.so.1"), ctypes.c_int(-1)
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) == 0
        assert ordinal.value == view.device.index
    else:
        assert view.location() == "cpu" and root.location() == "cpu"
    np.testing.assert_array_equal(_array(view), np.arange(12.).reshape(3, 4))
    view.sum().backward()
    expected = np.zeros((3, 8))
    expected[:, 1:5] = 1.
    np.testing.assert_array_equal(_array(root.grad), expected)


def test_native_copy_dtype_validation_is_atomic(rng_device):
    if "jittor" not in sys.modules:
        pytest.skip("native binding observer")
    root = torch.tensor(np.full((3, 8), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    source = torch.tensor(np.ones((3, 4)), dtype=torch.float32, device=rng_device)
    view = root[:, 1:5]
    pointer = root.data_ptr()
    with torch.no_grad(), pytest.raises(RuntimeError, match="dtype mismatch"):
        view._copy_into(source)
    assert root.data_ptr() == pointer and root.is_leaf and root.requires_grad
    assert view.requires_grad and not view.is_leaf
    np.testing.assert_array_equal(_array(root), np.full((3, 8), 11.))


def test_native_copy_refuses_unproven_gapped_alias_before_writing(rng_device):
    if "jittor" not in sys.modules:
        pytest.skip("native binding observer")
    root = torch.tensor(np.arange(16.).reshape(4, 4), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    destination, source = root[:, 1:], root[:, :-1]
    with torch.no_grad(), pytest.raises(RuntimeError, match="overlap"):
        destination._copy_into(source)
    assert root.is_leaf and root.requires_grad and not destination.is_leaf
    np.testing.assert_array_equal(_array(root), np.arange(16.).reshape(4, 4))


def test_native_copy_keeps_original_graph_nodes(rng_device):
    if "jittor" not in sys.modules:
        pytest.skip("native binding observer")
    root = torch.tensor(np.full((3, 8), 11.), dtype=torch.float64,
                        device=rng_device, requires_grad=True)
    base = root * 3
    view, sibling = base.T, base[:, 1:7:2].T
    tensors = (root, base, view, sibling)
    nodes = [value.grad_fn_node_id for value in tensors]
    assert nodes[0] == -1 and all(node >= 0 for node in nodes[1:])
    source = torch.tensor(np.arange(24.).reshape(8, 3), dtype=torch.float64, device=rng_device)
    with torch.no_grad():
        assert view.copy_(source) is view
    assert [value.grad_fn_node_id for value in tensors] == nodes
    (view.sum() + sibling.sum() * 2).backward()
    expected = np.full((3, 8), 3.)
    expected[:, 1:7:2] += 6
    np.testing.assert_array_equal(_array(root.grad), expected)
