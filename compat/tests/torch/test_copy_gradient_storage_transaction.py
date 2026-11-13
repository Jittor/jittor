"""Differentiable copies commit real bytes and retain the existing view graph."""

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


def _values(tensor):
    return tensor.detach().cpu().numpy().copy()


def _assert_values(tensor, expected):
    assert tensor is not None
    np.testing.assert_array_equal(_values(tensor), expected)


class TestCopyGradientStorageTransaction:
    def test_copy_retains_destination_storage_and_both_gradient_edges(self, device):
        old = torch.tensor([2., 4., 6.], device=device, requires_grad=True)
        source = torch.tensor([7., 8., 9.], device=device, requires_grad=True)
        destination = old * 2
        pointer = destination.data_ptr()
        version = destination._version
        assert destination.copy_(source) is destination
        assert destination.data_ptr() == pointer
        assert destination._version == version + 1
        _assert_values(destination, [7., 8., 9.])
        (destination * torch.tensor([1., 2., 3.], device=device)).sum().backward()
        _assert_values(old.grad, [0., 0., 0.])
        _assert_values(source.grad, [1., 2., 3.])

    def test_constant_source_preserves_explicit_zero_destination_gradient(self, device):
        old = torch.tensor([2., 4., 6.], device=device, requires_grad=True)
        destination = old * 3
        destination.copy_(torch.tensor([7., 8., 9.], device=device))
        destination.sum().backward()
        _assert_values(old.grad, [0., 0., 0.])

    def test_broadcast_copy_reduces_real_source_gradient(self, device):
        old = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device, requires_grad=True)
        source = torch.tensor([7., 8., 9.], device=device, requires_grad=True)
        destination = old * 2
        destination.copy_(source)
        destination.sum().backward()
        _assert_values(old.grad, np.zeros((2, 3)))
        _assert_values(source.grad, [2., 2., 2.])

    def test_partial_view_commit_retains_root_sibling_and_gradients(self, device):
        old = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], device=device, requires_grad=True)
        root = old * 2
        destination, sibling = root[:, 1:3], root.T
        pointers = root.data_ptr(), destination.data_ptr(), sibling.data_ptr()
        source = torch.tensor([[11., 12.], [13., 14.]], device=device, requires_grad=True)
        version = root._version
        assert destination.copy_(source) is destination
        assert (root.data_ptr(), destination.data_ptr(), sibling.data_ptr()) == pointers
        assert root._version == version + 1
        assert destination._version == root._version == sibling._version
        expected = np.array([[2., 11., 12., 8.], [10., 13., 14., 16.]])
        _assert_values(root, expected)
        _assert_values(sibling, expected.T)
        root.sum().backward()
        _assert_values(old.grad, [[2., 0., 0., 2.], [2., 0., 0., 2.]])
        _assert_values(source.grad, np.ones((2, 2)))

    def test_unobserved_first_write_has_one_native_version(self, device):
        destination = torch.tensor([1., 2.], device=device)
        destination.copy_(torch.tensor([3., 4.], device=device))
        assert destination._version == 1
        detached = destination.detach()
        with torch.no_grad():
            detached.copy_(torch.tensor([5., 6.], device=device))
        assert destination._version == detached._version == 2
        _assert_values(destination, [5., 6.])

    def test_invalid_shape_does_not_change_bytes_or_version(self, device):
        old = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
        destination = old * 2
        pointer, version = destination.data_ptr(), destination._version
        with pytest.raises(RuntimeError):
            destination.copy_(torch.tensor([4., 5.], device=device, requires_grad=True))
        assert destination.data_ptr() == pointer and destination._version == version
        _assert_values(destination, [2., 4., 6.])
        destination.sum().backward()
        _assert_values(old.grad, [2., 2., 2.])

    def test_leaf_guard_is_atomic(self, device):
        destination = torch.tensor([1., 2.], device=device, requires_grad=True)
        pointer, version = destination.data_ptr(), destination._version
        with pytest.raises(RuntimeError, match="leaf"):
            destination.copy_(torch.tensor([3., 4.], device=device))
        assert destination.data_ptr() == pointer and destination._version == version
        _assert_values(destination, [1., 2.])

    def test_index_copy_uses_same_storage_and_gradient_transaction(self, device):
        old = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device, requires_grad=True)
        source = torch.tensor([[7., 8.]], device=device, requires_grad=True)
        destination = old * 2
        pointer, version = destination.data_ptr(), destination._version
        index = torch.tensor([1], device=device, dtype=torch.int64)
        assert destination.index_copy_(0, index, source) is destination
        assert destination.data_ptr() == pointer and destination._version == version + 1
        _assert_values(destination, [[2., 4.], [7., 8.], [10., 12.]])
        destination.sum().backward()
        _assert_values(old.grad, [[2., 2.], [0., 0.], [2., 2.]])
        _assert_values(source.grad, [[1., 1.]])


instantiate_device_type_tests(TestCopyGradientStorageTransaction, globals(), only_for=("cpu", "cuda"))
