"""Inplace gradient-domain rules distinguish leaf views from legal writes."""
import gc
import weakref

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


def _mutations(device):
    return (
        (lambda value: value.copy_(torch.tensor([8., 10.], device=device)), [8., 10.]),
        (lambda value: value.div_(2.0), [1.5, 2.]),
        (lambda value: value.mul_(2.0), [6., 8.]),
        (lambda value: value.zero_(), [0., 0.]),
    )


class TestInplaceGradViews:
    def test_leaf_views_reject_before_mutation(self, device):
        for mutation, _expected in _mutations(device):
            root = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
            view = root.narrow(0, 2, 2)
            with pytest.raises(RuntimeError, match="view.*leaf"):
                mutation(view)
            np.testing.assert_array_equal(root.detach().cpu().numpy(), [1., 2., 3., 4.])

    def test_leaf_views_retain_actual_root_until_the_view_is_released(self, device):
        for mutation, _expected in _mutations(device):
            root = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
            view = root.view(2, 2).narrow(0, 1, 1).view(2)
            reference = weakref.ref(root)
            del root
            gc.collect()
            assert reference() is not None and view._base is reference()
            with pytest.raises(RuntimeError, match="view.*leaf"):
                mutation(view)
            np.testing.assert_array_equal(view.detach().cpu().numpy(), [3., 4.])
            del view
            gc.collect()
            assert reference() is None

    def test_no_grad_leaf_view_writes_keep_storage_and_later_gradient(self, device):
        for mutation, expected in _mutations(device):
            root = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
            view = root.narrow(0, 2, 2)
            alias = view.detach()
            pointers = [value.data_ptr() for value in (root, view, alias)]
            with torch.no_grad():
                assert mutation(view) is view
            assert pointers == [value.data_ptr() for value in (root, view, alias)]
            assert root.requires_grad and root.is_leaf and view.requires_grad
            np.testing.assert_array_equal(alias.cpu().numpy(), expected)
            root.square().sum().backward()
            np.testing.assert_array_equal(root.grad.cpu().numpy(), [2., 4., *(2 * x for x in expected)])

    def test_nonleaf_views_keep_real_input_and_rhs_gradients(self, device):
        for name, expected, source_gradient, rhs_gradient in (
                ("copy_", [2., 2.], [0., 0., 0., 0.], [1., 1.]),
                ("mul_", [12., 18.], [0., 6., 6., 0.], [6., 9.]),
                ("div_", [3., 4.5], [0., 1.5, 1.5, 0.], [-1.5, -2.25]),
                ("zero_", [0., 0.], [0., 0., 0., 0.], None)):
            source = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
            base = source * 3
            view = base.narrow(0, 1, 2)
            rhs = torch.tensor([2., 2.], device=device, requires_grad=True)
            assert (view.zero_() if name == "zero_" else getattr(view, name)(rhs)) is view
            np.testing.assert_array_equal(base.detach().cpu().numpy(), [3., *expected, 12.])
            view.sum().backward()
            assert source.grad is not None
            np.testing.assert_array_equal(source.grad.cpu().numpy(), source_gradient)
            if rhs_gradient is not None:
                assert rhs.grad is not None
                np.testing.assert_array_equal(rhs.grad.cpu().numpy(), rhs_gradient)

    def test_freezing_a_leaf_keeps_detached_storage_aliases(self, device):
        for mutation, expected in _mutations(device):
            root = torch.tensor([3., 4.], device=device, requires_grad=True)
            root.requires_grad_(False)
            alias = root.detach()
            pointers = [value.data_ptr() for value in (root, alias)]
            assert mutation(root) is root
            assert pointers == [value.data_ptr() for value in (root, alias)]
            assert not root.requires_grad and root.is_leaf
            np.testing.assert_array_equal(alias.cpu().numpy(), expected)

    def test_frozen_views_can_acquire_a_real_rhs_gradient_edge(self, device):
        for temporarily_frozen in (False, True):
            for name, expected, gradient in (
                    ("copy_", [2., 2.], [1., 1.]),
                    ("mul_", [4., 6.], [2., 3.]),
                    ("div_", [1., 1.5], [-.5, -.75])):
                base = torch.tensor([1., 2., 3., 4.], device=device,
                                    requires_grad=temporarily_frozen)
                if temporarily_frozen:
                    base.requires_grad_(False)
                view = base.narrow(0, 1, 2)
                rhs = torch.tensor([2., 2.], device=device, requires_grad=True)
                assert getattr(view, name)(rhs) is view
                assert base.requires_grad and view.requires_grad
                assert not base.is_leaf and not view.is_leaf
                np.testing.assert_array_equal(base.detach().cpu().numpy(), [1., *expected, 4.])
                view.sum().backward()
                assert rhs.grad is not None
                np.testing.assert_array_equal(rhs.grad.cpu().numpy(), gradient)


instantiate_device_type_tests(TestInplaceGradViews, globals())
