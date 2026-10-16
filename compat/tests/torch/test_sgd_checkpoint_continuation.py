"""Portable SGD checkpoint continuation contract for binary Torch and the shim.

SGD_CHECKPOINT_DEVICE selects a real CPU/CUDA worker. The external fresh-process
harness imports the same small helpers; optimizer mathematics stays in SGD.
"""
import io
import os

import numpy as np
import pytest
import torch

from _optimizer_contract_support import require_optimizer_backend  # noqa: F401


PARAMETERS = [0.0172119140625, -0.0023345947265625,
              -0.004791259765625, -0.0089111328125]
GRADIENTS = [-0.3359375, -0.0220947265625, -0.019775390625, -0.052001953125]
FACTORS = (1., -.5, 1.5, .25, -1.)
CONFIGURATIONS = {
    "plain": {},
    "momentum": {"momentum": .9, "weight_decay": .03, "dampening": .2},
    "nesterov": {"momentum": .9, "weight_decay": .03, "nesterov": True},
}


def make_optimizer(dtype, configuration, device, *, restored=False):
    dtype = getattr(torch, dtype)
    parameters = [
        torch.tensor(PARAMETERS, dtype=dtype, device=device, requires_grad=True),
        torch.tensor(np.asarray(PARAMETERS).reshape(2, 2), dtype=dtype,
                     device=device, requires_grad=True),
    ]
    if restored:
        with torch.no_grad():
            for value in parameters:
                value.fill_(5.)
    groups = [{"params": [parameters[0]], "lr": .4 if restored else .01},
              {"params": [parameters[1]], "lr": .6 if restored else .02}]
    # Deliberately different constructor defaults prove checkpoint group restore.
    options = {} if restored else CONFIGURATIONS[configuration]
    return parameters, torch.optim.SGD(groups, lr=.5 if restored else .01, **options)


def values(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def public_groups(optimizer):
    return [{key: value for key, value in group.items() if key != "params"}
            for group in optimizer.state_dict()["param_groups"]]


def capture(parameters, optimizer):
    result = {"groups": public_groups(optimizer), "parameters": [], "momentum": []}
    for parameter in parameters:
        result["parameters"].append(values(parameter))
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        result["momentum"].append(None if buffer is None else values(buffer))
    return result


def assert_capture_exact(actual, expected):
    assert actual["groups"] == expected["groups"]
    for name in ("parameters", "momentum"):
        for left, right in zip(actual[name], expected[name]):
            if right is None:
                assert left is None
            else:
                np.testing.assert_array_equal(left, right)


def advance(parameters, optimizer, start, end, observe=None):
    rows = []
    for step in range(start, end):
        optimizer.zero_grad()
        gradients = [torch.tensor(np.asarray(GRADIENTS).reshape(tuple(p.shape)) * FACTORS[step],
                                  dtype=p.dtype, device=p.device) for p in parameters]
        loss = sum((p * gradient).sum() for p, gradient in zip(parameters, gradients))
        loss.backward()
        published = [parameter.grad for parameter in parameters]
        for gradient, reference in zip(published, gradients):
            assert gradient is not None
            np.testing.assert_array_equal(values(gradient), values(reference))
        if observe is not None:
            observe("before_step", step, parameters, optimizer, published)
        optimizer.step()
        for parameter, gradient in zip(parameters, published):
            assert parameter.grad is gradient
            assert str(parameter.dtype) == str(gradient.dtype)
        row = capture(parameters, optimizer)
        row["gradients"] = [values(gradient) for gradient in published]
        rows.append(row)
        if observe is not None:
            observe("after_step", step, parameters, optimizer, published)
    return rows


def checkpoint_bytes(parameters, optimizer):
    stream = io.BytesIO()
    torch.save({"parameters": parameters, "optimizer": optimizer.state_dict()}, stream)
    return stream.getvalue()


def restore(payload, dtype, configuration, device):
    saved = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    parameters, optimizer = make_optimizer(dtype, configuration, device, restored=True)
    identities = [id(parameter) for parameter in parameters]
    with torch.no_grad():
        for parameter, value in zip(parameters, saved["parameters"]):
            assert parameter.dtype == value.dtype
            assert parameter.device == value.device
            parameter.copy_(value)
    optimizer.load_state_dict(saved["optimizer"])
    assert identities == [id(parameter) for parameter in parameters]
    return parameters, optimizer


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
@pytest.mark.parametrize("configuration", tuple(CONFIGURATIONS))
def test_sgd_checkpoint_continuation(dtype, configuration):
    device = os.environ.get("SGD_CHECKPOINT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    parameters, optimizer = make_optimizer(dtype, configuration, device)
    advance(parameters, optimizer, 0, 3)
    before = capture(parameters, optimizer)
    payload = checkpoint_bytes(parameters, optimizer)
    assert_capture_exact(capture(parameters, optimizer), before)
    baseline = advance(parameters, optimizer, 3, 5)
    restored, new_optimizer = restore(payload, dtype, configuration, device)
    assert_capture_exact(capture(restored, new_optimizer), before)
    resumed = advance(restored, new_optimizer, 3, 5)
    for actual, expected in zip(resumed, baseline):
        assert_capture_exact(actual, expected)
        for gradient, reference in zip(actual["gradients"], expected["gradients"]):
            np.testing.assert_array_equal(gradient, reference)
