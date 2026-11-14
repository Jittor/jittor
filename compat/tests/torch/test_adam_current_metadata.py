"""Adam observes current eligibility and step protocols around kernel selection."""
import os
import types
from contextlib import ExitStack
from unittest import mock

import numpy as np
import pytest
import torch

from jittor.compat.torch import optimizer_adam as adam
from jittor._runtime import dispatch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def _initialized(device):
    parameter = torch.nn.Parameter(torch.full((2,), .125, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, fused=True)
    parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    # Do not expose optimizer.state here: observing its Tensor counter must
    # itself switch subsequent updates to the ordinary live Tensor protocol.
    return parameter, optimizer, optimizer.param_groups[0]


def _provider(monkeypatch, events, select=None):
    original = dispatch.select_kernel

    def update(entries, *args):
        events.append("update")
        return [(p, m, v) for p, m, v, _, _ in entries]

    def choose(name, entries, *args, **kwargs):
        if name != "optim.adamw_fused":
            return original(name, entries, *args, **kwargs)
        events.append("select")
        if select is not None:
            select(entries)
        return update

    monkeypatch.setattr(dispatch, "select_kernel", choose)


@pytest.mark.parametrize("change", ["freeze", "gradient-none", "integer-step", "exposed-step"])
def test_provider_selection_rechecks_live_metadata(device, monkeypatch, change):
    parameter, optimizer, group = _initialized(device)
    before = parameter.detach().cpu().numpy().copy()
    selected, exposed, events = [], [], []

    def select(entries):
        selected.append(entries[0][4])
        if change == "freeze":
            parameter.requires_grad_(False)
        elif change == "gradient-none":
            group["grads"][0] = None
        elif change == "integer-step":
            group["_torch_steps"][0].value = 7
        else:
            counter = group["_torch_steps"][0].tensor()
            counter.data.fill_(7)
            exposed.append(counter)

    _provider(monkeypatch, events, select)
    optimizer.step()
    assert selected == [1]
    assert events == ["select", "update"]
    counter = optimizer.state[parameter]["step"]
    assert counter.item() == (1 if change in ("freeze", "gradient-none") else 8)
    if exposed:
        assert counter is exposed[0]
    np.testing.assert_array_equal(parameter.detach().cpu().numpy(), before)


@pytest.mark.parametrize("owner,name,event", [
    (adam._AdamStep, "__int__", "int"),
    (adam._AdamStep, "set", "set"),
    (adam._AdamSteps, "__setitem__", "setitem"),
])
def test_in_place_counter_method_code_changes_are_observed(device, monkeypatch, owner, name, event):
    parameter, optimizer, _ = _initialized(device)
    events = []
    _provider(monkeypatch, events)
    function = getattr(owner, name)
    original = types.FunctionType(function.__code__, function.__globals__,
                                  function.__name__, function.__defaults__, function.__closure__)
    namespace = {"_metadata_probe_events": events, "_metadata_probe_original": original,
                 "_metadata_probe_event": event}

    def replacement(self, *args):
        _metadata_probe_events.append(_metadata_probe_event)
        return _metadata_probe_original(self, *args)

    closure_marker = None

    def replacement_with_closure(self, *args):
        _ = closure_marker
        _metadata_probe_events.append(_metadata_probe_event)
        return _metadata_probe_original(self, *args)

    # __setitem__ uses zero-argument super() and therefore has a __class__
    # closure cell. Code replacement must retain its number of free variables.
    code = replacement_with_closure.__code__ if function.__closure__ else replacement.__code__
    original_code = function.__code__
    with mock.patch.dict(function.__globals__, namespace):
        try:
            function.__code__ = code
            optimizer.step()
        finally:
            function.__code__ = original_code
    selected, updated = events.index("select"), events.index("update")
    assert event in events[selected + 1:updated]
    if event == "int":
        assert event in events[:selected]
    assert optimizer.state[parameter]["step"].item() == 2


def test_instance_counter_set_override_is_observed(device, monkeypatch):
    parameter, optimizer, group = _initialized(device)
    events = []
    _provider(monkeypatch, events)
    step = group["_torch_steps"][0]
    original = step.set

    def set_step(value):
        events.append("set")
        original(value)

    step.set = set_step
    optimizer.step()
    assert "set" in events[events.index("select") + 1:events.index("update")]
    assert optimizer.state[parameter]["step"].item() == 2


def test_changed_requires_grad_descriptor_is_observed(device):
    parameter, optimizer, _ = _initialized(device)

    def requires_grad(self):
        raise RuntimeError("live requires_grad observer")

    with mock.patch.object(torch.nn.Parameter, "requires_grad", property(requires_grad)), \
            pytest.raises(RuntimeError, match="live requires_grad observer"):
        optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 1


def test_counter_container_iteration_protocol_is_preserved(device, monkeypatch):
    parameter, optimizer, group = _initialized(device)
    events = []
    _provider(monkeypatch, events)

    class Counters(adam._AdamSteps):
        def __getitem__(self, index):
            events.append("counter-read")
            return super().__getitem__(index)

    group["_torch_steps"] = Counters(group, group["_torch_steps"])
    optimizer.step()
    assert "counter-read" in events[:events.index("select")]
    assert "counter-read" in events[events.index("select") + 1:events.index("update")]
    assert optimizer.state[parameter]["step"].item() == 2


@pytest.mark.parametrize("kind", ["requires-grad", "int", "set"])
def test_provider_can_change_method_code_between_the_two_passes(device, monkeypatch, kind):
    parameter, optimizer, _ = _initialized(device)
    events, selected = [], []
    if kind == "requires-grad":
        from jittor.compat.torch.installers.tensor.autograd_api import _rg_get
        function = _rg_get
    else:
        function = getattr(adam._AdamStep, "__int__" if kind == "int" else "set")
    original_code = function.__code__
    original = types.FunctionType(original_code, function.__globals__,
                                  function.__name__, function.__defaults__, function.__closure__)

    def replacement(self, *args):
        _metadata_probe_events.append("protocol")
        return _metadata_probe_original(self, *args)

    with ExitStack() as scopes:
        def select(entries):
            selected.append(entries[0][4])
            scopes.enter_context(mock.patch.dict(function.__globals__, {
                "_metadata_probe_events": events, "_metadata_probe_original": original,
            }))
            function.__code__ = replacement.__code__

        _provider(monkeypatch, events, select)
        try:
            optimizer.step()
        finally:
            function.__code__ = original_code
    assert selected == [1]
    assert "protocol" in events[events.index("select") + 1:events.index("update")]
    assert optimizer.state[parameter]["step"].item() == 2


@pytest.mark.parametrize("unsupported", ["exposed-counter", "instance-set"])
def test_native_counter_decline_does_not_partially_advance_earlier_entries(device, unsupported):
    if device != "cuda":
        pytest.skip("native Adam metadata batching requires CUDA")
    parameters = [torch.nn.Parameter(torch.full((2,), value, device=device))
                  for value in (.125, .25)]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, fused=True)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    group = optimizer.param_groups[0]
    steps = group["_torch_steps"]
    if unsupported == "exposed-counter":
        steps[1].tensor()
    else:
        steps[1].set = lambda value: pytest.fail("native decline invoked a custom setter")
    assert adam._adam_metadata_batch(group, group["grads"], steps, advance=True) is None
    assert steps[0].value == 1 and steps[1].value == 1


def test_native_batch_is_reachable_and_reads_current_counters(device):
    if device != "cuda":
        pytest.skip("native Adam metadata batching requires CUDA")
    parameter, _, group = _initialized(device)
    steps, gradients = group["_torch_steps"], group["grads"]
    first = adam._adam_metadata_batch(group, gradients, steps)
    assert first is not None and len(first) == 1 and first[0][0] is parameter
    assert first[0][4] == 1
    steps[0].value = 7
    assert adam._adam_metadata_batch(group, gradients, steps)[0][4] == 7
    assert adam._adam_metadata_batch(group, gradients, steps, advance=True) == [0]
    assert steps[0].value == 8
    # Once observed, a counter's actual storage and all its aliases remain
    # authoritative. A native decline must not change that storage or count.
    tensor = steps[0].tensor()
    tensor.data.fill_(12)
    assert adam._adam_metadata_batch(group, gradients, steps) is None
    assert adam._adam_metadata_batch(group, gradients, steps, advance=True) is None
    assert tensor.item() == 12


def test_native_batch_declines_a_protocol_with_a_missing_counter_method(device):
    if device != "cuda":
        pytest.skip("native Adam metadata batching requires CUDA")
    import jittor as jt
    _, _, group = _initialized(device)
    protocol = list(adam._ADAM_METADATA_PROTOCOL)
    protocol[0], protocol[2] = object, ()
    protocol[7] = dict(protocol[7], _AdamStep=object)
    frontend = adam.get_install_context(jt).target_namespace
    assert jt.core._try_optimizer_adam_entries(
        group, group["grads"], group["_torch_steps"],
        (jt.core.Var, frontend.Tensor, frontend.nn.Parameter), tuple(protocol), False) is None
    assert group["_torch_steps"][0].value == 1


@pytest.mark.parametrize("change", ["epoch", "property", "getter-code", "detach-method"])
def test_native_batch_preserves_capture_epochs_and_custom_counter_access(device, monkeypatch, change):
    if device != "cuda":
        pytest.skip("native Adam metadata batching requires CUDA")
    _, _, group = _initialized(device)
    steps, gradients = group["_torch_steps"], group["grads"]
    assert adam._adam_metadata_batch(group, gradients, steps) is not None
    if change == "epoch":
        steps._advance_capture_steps([0])
        steps._advance_capture_steps([0])
        expected = 3
    elif change == "property":
        monkeypatch.setattr(adam._AdamStep, "value", property(lambda self: 17))
        expected = 17
    elif change == "getter-code":
        def replacement(self):
            return 19
        monkeypatch.setattr(adam._AdamStep.value.fget, "__code__", replacement.__code__)
        expected = 19
    else:
        steps[0]._detach_capture_epoch = lambda: pytest.fail("custom detachment was bypassed")
        expected = 1
    assert adam._adam_metadata_batch(group, gradients, steps) is None
    assert adam._adam_metadata_batch(group, gradients, steps, advance=True) is None
    assert int(steps[0]) == expected
