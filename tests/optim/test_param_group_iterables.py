"""Appended parameter iterables remain usable across optimizer steps."""

import numpy as np
import pytest
import jittor as jt

from _helpers.common import device_flags_for, get_all_device_types
from jittor._runtime.fallback import forbid_backend_fallbacks


ALGORITHMS = ["SGD", "Adam", "AdamW", "RMSprop", "Adan"]
INPUTS = [[2.0, -3.0], [0.5, 4.0, -2.0]]


def trajectory(name, form, device):
    first = jt.array([1.0], dtype="float32")
    parameters = [jt.array(value, dtype="float32") for value in INPUTS]
    optimizer = getattr(jt.optim, name)([first], lr=0.1)
    optimizer.step((first * first).sum())
    yielded = []

    def generate():
        for parameter in parameters:
            yielded.append(parameter)
            yield parameter

    supplied = {"list": parameters, "tuple": tuple(parameters),
                "iterator": iter(parameters), "generator": generate()}[form]
    group = {"params": supplied, "lr": 0.03}
    optimizer.add_param_group(group)
    assert optimizer.param_groups[-1] is group
    assert group["lr"] == 0.03
    if form in ("list", "tuple"):
        assert group["params"] is supplied
    assert len(group["params"]) == len(parameters)
    assert all(actual is expected for actual, expected in zip(group["params"], parameters))
    if form == "generator":
        assert len(yielded) == len(parameters)
    snapshots = []
    for index in range(3):
        optimizer.step(sum((p * p).sum() for p in [first] + parameters))
        step = []
        for parameter in [first] + parameters:
            parameter.sync()
            assert parameter.location() == ("cpu" if device == "cpu" else "device")
            step.append(parameter.numpy().copy())
        snapshots.append(step)
        if name == "SGD":
            for value, initial in zip(step[1:], INPUTS):
                np.testing.assert_allclose(
                    value, np.asarray(initial) * 0.94 ** (index + 1), rtol=1e-6)
    for final, initial in zip(snapshots[-1][1:], INPUTS):
        assert not np.array_equal(final, initial), "appended parameter did not update"
    return snapshots


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("form", ["list", "tuple", "iterator", "generator"])
@pytest.mark.parametrize("device", get_all_device_types())
def test_appended_iterable_matches_list_training(name, form, device):
    with jt.flag_scope(**device_flags_for(device)), forbid_backend_fallbacks():
        actual = trajectory(name, form, device)
        expected = trajectory(name, "list", device)
        for actual_step, expected_step in zip(actual, expected):
            for value, reference in zip(actual_step, expected_step):
                np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", ["Optimizer"] + ALGORITHMS)
def test_empty_iterable_keeps_empty_group_behavior(name):
    optimizer = getattr(jt.optim, name)([jt.ones(1)], lr=0.1)
    empty = {"params": iter(())}
    optimizer.add_param_group(empty)
    assert optimizer.param_groups[-1] is empty
    assert empty["params"] == []


@pytest.mark.parametrize("name", ["Optimizer"] + ALGORITHMS)
def test_failing_iterable_does_not_register_partial_group(name):
    optimizer = getattr(jt.optim, name)([jt.ones(1)], lr=0.1)
    before = len(optimizer.param_groups)

    def failing():
        yield jt.ones(1)
        raise RuntimeError("parameter source failed")

    with pytest.raises(RuntimeError, match="parameter source failed"):
        optimizer.add_param_group({"params": failing()})
    assert len(optimizer.param_groups) == before
