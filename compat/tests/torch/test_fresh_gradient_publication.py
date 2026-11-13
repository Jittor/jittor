"""Fresh backward slots and custom Python gradient accessors stay coherent."""
import os
from unittest import mock

import numpy as np
import pytest
import torch
import jittor as jt

from jittor.compat.torch import tensor_object_state as state


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def _array(value):
    return value.detach().cpu().numpy().copy()


def _prepared(device):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    optimizer = torch.optim.SGD([parameter], lr=.125)
    parameter.grad = None
    return parameter, optimizer


def _observe():
    native = jt.core._try_publish_fresh_gradients
    decisions = []

    def observed(*args):
        result = native(*args)
        decisions.append(result)
        return result

    return decisions, mock.patch.object(jt.core, "_try_publish_fresh_gradients", observed)


def test_fresh_backward_and_accumulation_keep_slots_and_gradient_identity(device):
    parameter, optimizer = _prepared(device)
    decisions, observer = _observe()
    with observer:
        parameter.square().sum().backward()
        first = parameter.grad
        assert optimizer.param_groups[0]["grads"][0] is first
        np.testing.assert_array_equal(_array(first), [2., 4.])
        parameter.square().sum().backward()
        assert parameter.grad is first
        np.testing.assert_array_equal(_array(first), [4., 8.])
        parameter.grad = None
        parameter.square().sum().backward()
        assert parameter.grad is not first
        np.testing.assert_array_equal(_array(parameter.grad), [2., 4.])
    assert decisions == [False, False, True]
    optimizer.step()
    np.testing.assert_array_equal(_array(parameter), [.75, 1.5])
    assert parameter.grad.device.type == device


@pytest.mark.parametrize("change", ["property", "closure", "helper", "state-slot"])
def test_custom_gradient_accessors_keep_their_python_protocol(device, monkeypatch, change):
    parameter, optimizer = _prepared(device)
    events = []
    if change == "property":
        original = torch.Tensor._torch_grad

        def read(value):
            events.append("read")
            return original.fget(value)

        def write(value, gradient):
            events.append("write")
            original.fset(value, gradient)

        monkeypatch.setattr(torch.nn.Parameter, "_torch_grad", property(read, write))
    elif change == "closure":
        # Identical accessor bytecode with a different backing slot must not
        # accidentally publish to the ordinary grad slot.
        descriptor = state._state_property("_torch_grad", "data_owner", None)
        monkeypatch.setattr(torch.nn.Parameter, "_torch_grad", descriptor)
    elif change == "helper":
        original = state.get_tensor_object_state

        def observed(value, create=False):
            events.append("helper")
            return original(value, create=create)

        monkeypatch.setattr(state, "get_tensor_object_state", observed)
    else:
        original = state.TensorObjectState.grad

        def read(value):
            events.append("slot-read")
            return original.__get__(value)

        def write(value, gradient):
            events.append("slot-write")
            original.__set__(value, gradient)

        monkeypatch.setattr(state.TensorObjectState, "grad", property(read, write))
    decisions, observer = _observe()
    with observer:
        parameter.square().sum().backward()
    assert decisions and not any(decisions)
    gradient = parameter.grad
    np.testing.assert_array_equal(_array(gradient), [2., 4.])
    assert optimizer.param_groups[0]["grads"][0] is gradient
    if change != "closure":
        assert events
    optimizer.step()
    np.testing.assert_array_equal(_array(parameter), [.75, 1.5])
