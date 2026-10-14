"""Tensor hooks observe complete node gradients without changing the graph."""
import gc
import weakref

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorGradientHooks:
    def test_leaf_late_registration_and_order(self, device):
        value = torch.tensor([2., 3.], device=device, requires_grad=True)
        loss = (value.square() + value * 3).sum()
        events = []
        first = value.register_hook(lambda gradient: (events.append("first"), gradient * 2)[1])
        second = value.register_hook(lambda gradient: (events.append("second"), gradient + 1)[1])
        assert value.is_leaf and value.grad_fn is None
        loss.backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [15., 19.])
        assert events == ["first", "second"]
        first.remove()
        second.remove()
        value.grad = None
        value.square().sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [4., 6.])

    def test_nonleaf_hook_changes_upstream_and_requested_target(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        middle = value * 3
        events = []
        middle.register_hook(lambda gradient: (events.append(float(gradient.item())), gradient * 2)[1])
        gx, gm = torch.autograd.grad(middle.square().sum(), (value, middle))
        np.testing.assert_array_equal(gm.cpu().numpy(), [12.])
        np.testing.assert_array_equal(gx.cpu().numpy(), [72.])
        assert events == [12.] and value.grad is None
        middle = value * 3
        middle.register_hook(lambda gradient: gradient * 2)
        gm, = torch.autograd.grad(middle.square().sum(), (middle,))
        np.testing.assert_array_equal(gm.cpu().numpy(), [24.])

    def test_loss_hook_and_leaf_callback_run_once_per_backward(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        loss = value.square().sum()
        events = []
        value.register_hook(lambda gradient: events.append(("leaf", float(gradient.item()))))
        loss.register_hook(lambda gradient: (events.append(("loss", float(gradient.item()))), gradient * 3)[1])
        loss.backward(retain_graph=True)
        loss.backward()
        assert events == [("loss", 1.), ("leaf", 12.)] * 2
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [24.])

    def test_disconnected_hook_does_not_fire(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        absent = torch.tensor([3.], device=device, requires_grad=True)
        absent.register_hook(lambda gradient: pytest.fail("disconnected hook ran"))
        gx, ga = torch.autograd.grad(value.square().sum(), (value, absent), allow_unused=True)
        assert ga is None
        np.testing.assert_array_equal(gx.cpu().numpy(), [4.])

    def test_hook_survives_deleted_nonleaf_wrapper(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        middle = value * 3
        events = []
        middle.register_hook(lambda gradient: (events.append("middle"), gradient * 2)[1])
        loss = middle.square().sum()
        del middle
        gc.collect()
        loss.backward()
        assert events == ["middle"]
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [72.])

    def test_removable_handle_does_not_retain_tensor(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        reference = weakref.ref(value)
        handle = value.register_hook(lambda gradient: gradient)
        del value
        gc.collect()
        assert reference() is None
        handle.remove()
        handle.remove()

    def test_closed_hook_cycle_is_collected(self, device):
        def create():
            value = torch.tensor([2.], device=device, requires_grad=True)
            value.register_hook(lambda gradient: gradient + value.detach() * 0)
            return weakref.ref(value)
        reference = create()
        gc.collect()
        assert reference() is None

    def test_leaf_hook_survives_data_rebinding(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        events = []
        value.register_hook(lambda gradient: (events.append(float(gradient.item())), gradient * 2)[1])
        value.data = torch.tensor([3.], device=device)
        value.square().sum().backward()
        assert events == [6.]
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [12.])

    def test_hook_failure_and_bad_replacements_are_visible(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        handle = value.register_hook(lambda gradient: "invalid")
        with pytest.raises((TypeError, RuntimeError), match="(Tensor|tensor|Variable|variable)"):
            value.square().sum().backward()
        assert value.grad is None
        handle.remove()
        handle = value.register_hook(lambda gradient: gradient.expand(2))
        with pytest.raises(RuntimeError, match="(shape|size)"):
            value.square().sum().backward()
        handle.remove()
        value.square().sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [4.])

    def test_requires_grad_validation(self, device):
        value = torch.tensor([2.], device=device)
        with pytest.raises(RuntimeError, match="require"):
            value.register_hook(lambda gradient: gradient)
        value.requires_grad_(True)

    def test_post_accumulate_observes_accumulation_and_final_order(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        events = []
        engine = torch.autograd.Variable._execution_engine
        def before(gradient):
            events.append(("gradient", float(gradient.item())))
            engine.queue_callback(lambda: events.append(("final", float(value.grad.item()))))
        def after(parameter):
            assert parameter is value and not torch.is_grad_enabled()
            events.append(("post", float(parameter.grad.item())))
            parameter.grad.mul_(2)
        value.register_hook(before)
        handle = value.register_post_accumulate_grad_hook(after)
        value.square().sum().backward()
        assert events == [("gradient", 4.), ("post", 4.), ("final", 8.)]
        events.clear()
        value.square().sum().backward()
        assert events == [("gradient", 4.), ("post", 12.), ("final", 24.)]
        handle.remove()

    def test_post_accumulate_optimizer_write_through_and_clear(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        optimizer = torch.optim.SGD([value], lr=.1)
        def double(parameter):
            parameter.grad.mul_(2)
        handle = value.register_post_accumulate_grad_hook(double)
        value.square().sum().backward()
        optimizer.step()
        np.testing.assert_allclose(value.detach().cpu().numpy(), [1.2], rtol=1e-6)
        handle.remove()
        optimizer.zero_grad(set_to_none=True)
        value.register_post_accumulate_grad_hook(lambda parameter: setattr(parameter, "grad", None))
        value.square().sum().backward()
        assert value.grad is None
        optimizer.step()
        np.testing.assert_allclose(value.detach().cpu().numpy(), [1.2], rtol=1e-6)

    def test_post_accumulate_rejects_nonleaf_and_grad_does_not_call_it(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        with pytest.raises(RuntimeError, match="non-leaf"):
            (value * 2).register_post_accumulate_grad_hook(lambda parameter: None)
        value.register_post_accumulate_grad_hook(lambda parameter: pytest.fail("autograd.grad accumulated"))
        result, = torch.autograd.grad(value.square().sum(), (value,))
        np.testing.assert_array_equal(result.cpu().numpy(), [4.])
        assert value.grad is None


instantiate_device_type_tests(TestTensorGradientHooks, globals(), only_for=("cpu", "cuda"))
