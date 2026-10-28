"""In-place detach preserves object/storage and earlier reverse-mode graphs."""
import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorDetachInplace:
    def test_leaf_keeps_identity_storage_and_existing_gradient(self, device):
        dtypes = (torch.float32, torch.bfloat16) if device == "cuda" else (torch.float32,)
        for dtype in dtypes:
            value = torch.tensor([1., 2.], device=device, dtype=dtype, requires_grad=True)
            assert value.is_leaf and value.grad_fn is None and value.requires_grad
            value.square().sum().backward()
            gradient = value.grad
            pointer = value.data_ptr()
            assert value.detach_() is value
            assert value.data_ptr() == pointer and value.dtype == dtype and value.device.type == device
            assert not value.requires_grad and value.is_leaf and value.grad_fn is None
            assert value.grad is gradient
            np.testing.assert_array_equal(value.float().cpu().numpy(), [1., 2.])
            np.testing.assert_array_equal(gradient.float().cpu().numpy(), [2., 4.])

    def test_nonleaf_preserves_earlier_consumers_but_cuts_future_history(self, device):
        source = torch.tensor([1., 2.], device=device, requires_grad=True)
        value = source * 3
        earlier = (value * 2).sum()
        pointer = value.data_ptr()
        assert value.detach_() is value
        assert value.data_ptr() == pointer and value.grad_fn is None and value.is_leaf
        earlier.backward()
        np.testing.assert_array_equal(source.grad.cpu().numpy(), [6., 6.])
        source.grad = None
        value.requires_grad_()
        value.square().sum().backward()
        assert source.grad is None
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [6., 12.])

    def test_leaf_reenabled_before_earlier_backward_keeps_its_edge(self, device):
        value = torch.tensor([1., 2.], device=device, requires_grad=True)
        earlier = (value * 2).sum()
        value.detach_().requires_grad_()
        earlier.backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [2., 2.])

    def test_views_are_rejected_without_mutation(self, device):
        for detached in (False, True):
            source = torch.tensor([1., 2.], device=device, requires_grad=True)
            view = (source.detach() if detached else source)[:1]
            pointer, requires_grad = view.data_ptr(), view.requires_grad
            with pytest.raises(RuntimeError, match="detach views in-place"):
                view.detach_()
            assert view.data_ptr() == pointer and view.requires_grad == requires_grad
            np.testing.assert_array_equal(source.detach().cpu().numpy(), [1., 2.])

    def test_default_backward_publishes_detached_gradients_and_accumulates_in_place(self, device):
        value = torch.tensor([1., 2.], device=device, requires_grad=True)
        value.square().sum().backward()
        gradient = value.grad
        assert not gradient.requires_grad and gradient.is_leaf and gradient.grad_fn is None
        value.square().sum().backward()
        assert value.grad is gradient
        assert not gradient.requires_grad and gradient.is_leaf and gradient.grad_fn is None
        np.testing.assert_array_equal(gradient.cpu().numpy(), [4., 8.])

    def test_advanced_index_copy_can_detach_in_place(self, device):
        source = torch.tensor([1., 2.], device=device, requires_grad=True)
        copied = source[[0]]
        assert copied.detach_() is copied and not copied.requires_grad
        np.testing.assert_array_equal(copied.cpu().numpy(), [1.])

    def test_create_graph_retains_gradient_history(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        value.square().sum().backward(create_graph=True)
        assert value.grad.requires_grad and value.grad.grad_fn is not None
        second, = torch.autograd.grad(value.grad.sum(), value)
        np.testing.assert_array_equal(second.cpu().numpy(), [2.])


instantiate_device_type_tests(TestTensorDetachInplace, globals())
