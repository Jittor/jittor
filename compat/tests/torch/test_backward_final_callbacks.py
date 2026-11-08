"""Graph-task callbacks run after accumulation and fail visibly on errors."""
import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestBackwardFinalCallbacks:
    def test_callback_order_and_accumulated_gradient(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        events = []
        engine = torch.autograd.Variable._execution_engine
        def hook(gradient):
            events.append("hook")
            def first():
                events.append(("first", float(value.grad.item())))
                engine.queue_callback(lambda: events.append("last"))
            engine.queue_callback(first)
            engine.queue_callback(lambda: events.append("second"))
            return gradient
        loss = value.square().sum()
        handle = value.register_hook(hook)
        loss.backward()
        assert events == ["hook", ("first", 4.), "second", "last"]
        events.clear()
        handle.remove()
        loss = value.square().sum()
        handle = value.register_hook(hook)
        loss.backward()
        assert events == ["hook", ("first", 8.), "second", "last"]
        handle.remove()

    def test_callbacks_outside_graph_and_failure_cleanup(self, device):
        engine = torch.autograd.Variable._execution_engine
        with pytest.raises(RuntimeError, match="during backward"):
            engine.queue_callback(lambda: None)
        value = torch.tensor([2.], device=device, requires_grad=True)
        events = []
        def hook(gradient):
            engine.queue_callback(lambda: events.append("should-not-run"))
            raise RuntimeError("callback hook failure")
        loss = value.square().sum()
        handle = value.register_hook(hook)
        with pytest.raises(RuntimeError, match="callback hook failure"):
            loss.backward()
        assert not events
        handle.remove()
        value.square().sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [4.])
        assert not events

    def test_callback_failure_propagates_without_polluting_next_graph(self, device):
        engine = torch.autograd.Variable._execution_engine
        value = torch.tensor([2.], device=device, requires_grad=True)
        def failure():
            raise RuntimeError("final callback failed")
        loss = value.square().sum()
        handle = value.register_hook(lambda gradient: (engine.queue_callback(failure), gradient)[1])
        with pytest.raises(RuntimeError, match="final callback failed"):
            loss.backward()
        handle.remove()
        value.grad = None
        value.square().sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [4.])

    def test_grad_returns_run_final_callback_without_accumulating(self, device):
        engine = torch.autograd.Variable._execution_engine
        value = torch.tensor([3.], device=device, requires_grad=True)
        events = []
        def hook(gradient):
            engine.queue_callback(lambda: events.append(value.grad))
            return gradient
        loss = value.square().sum()
        value.register_hook(hook)
        result, = torch.autograd.grad(loss, (value,))
        np.testing.assert_array_equal(result.cpu().numpy(), [6.])
        assert events == [None] and value.grad is None


instantiate_device_type_tests(TestBackwardFinalCallbacks, globals(), only_for=("cpu", "cuda"))
