"""Tensor-list clearing preserves real storage and gradient-state policies."""
import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestForeachZero:
    def test_pending_snapshot_observes_values_before_clear(self, device):
        value = torch.tensor([1., 2.], device=device)
        snapshot = value.clone()
        torch._foreach_zero_([value])
        np.testing.assert_array_equal(snapshot.cpu().numpy(), [1., 2.])
        np.testing.assert_array_equal(value.cpu().numpy(), [0., 0.])

    def test_mixed_dtype_and_detached_alias_preserve_storage(self, device):
        source = torch.tensor([1., 2.], device=device)
        integer = torch.tensor([3, 4], dtype=torch.int64, device=device)
        tensors = [source.detach(), integer, torch.tensor([True, False], device=device)]
        if device == "cuda":
            tensors.append(torch.tensor([5., 6.], dtype=torch.bfloat16, device=device))
        pointers = [tensor.data_ptr() for tensor in tensors]
        dtypes = [tensor.dtype for tensor in tensors]
        assert torch._foreach_zero_(tensors) is tensors
        assert pointers == [tensor.data_ptr() for tensor in tensors]
        assert dtypes == [tensor.dtype for tensor in tensors]
        for tensor in tensors:
            assert tensor.device.type == device and not tensor.requires_grad
            np.testing.assert_array_equal(tensor.float().cpu().numpy(), np.zeros(tensor.shape))
        np.testing.assert_array_equal(source.cpu().numpy(), [0., 0.])

    def test_tuple_duplicate_empty_and_scalar_tensor(self, device):
        value = torch.tensor([1., 2.], device=device)
        empty = torch.empty((0,), device=device)
        scalar = torch.tensor(3., device=device)
        tensors = (value, value, empty, scalar)
        assert torch._foreach_zero_(tensors) is tensors
        np.testing.assert_array_equal(value.cpu().numpy(), [0., 0.])
        assert empty.numel() == 0
        assert scalar.item() == 0

    def test_invalid_lists_are_rejected_before_mutation(self, device):
        value = torch.tensor([1., 2.], device=device)
        with pytest.raises(TypeError, match="expected Tensor"):
            torch._foreach_zero_([value, 3])
        np.testing.assert_array_equal(value.cpu().numpy(), [1., 2.])
        with pytest.raises(TypeError):
            torch._foreach_zero_(iter([value]))
        with pytest.raises(RuntimeError, match="at least one tensor"):
            torch._foreach_zero_([])

    def test_leaf_rejection_is_atomic_and_no_grad_can_clear(self, device):
        ordinary = torch.tensor([1., 2.], device=device)
        leaf = torch.tensor([3., 4.], device=device, requires_grad=True)
        with pytest.raises(RuntimeError, match="leaf Variable"):
            torch._foreach_zero_([ordinary, leaf])
        np.testing.assert_array_equal(ordinary.cpu().numpy(), [1., 2.])
        pointer = leaf.data_ptr()
        with torch.no_grad():
            tensors = [leaf]
            assert torch._foreach_zero_(tensors) is tensors
        assert leaf.requires_grad and leaf.is_leaf and leaf.data_ptr() == pointer
        leaf.square().sum().backward()
        np.testing.assert_array_equal(leaf.grad.cpu().numpy(), [0., 0.])


instantiate_device_type_tests(TestForeachZero, globals())


def test_differentiable_nonleaf_lists_fail_explicitly():
    source = torch.tensor([1., 2.], requires_grad=True)
    nonleaf = source * 3
    with pytest.raises(NotImplementedError, match="differentiable tensor lists"):
        torch._foreach_zero_([nonleaf])
    np.testing.assert_array_equal(nonleaf.detach().cpu().numpy(), [3., 6.])
    nonleaf.sum().backward()
    np.testing.assert_array_equal(source.grad.cpu().numpy(), [3., 3.])


def test_views_fail_before_other_inputs_are_mutated():
    ordinary = torch.tensor([1., 2.])
    source = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    view = source[:, ::2]
    assert not view.is_contiguous()
    with pytest.raises(NotImplementedError, match="non-contiguous tensor lists"):
        torch._foreach_zero_([ordinary, view])
    np.testing.assert_array_equal(ordinary.cpu().numpy(), [1., 2.])
    np.testing.assert_array_equal(source.cpu().numpy(), [[1., 2., 3., 4.], [5., 6., 7., 8.]])
