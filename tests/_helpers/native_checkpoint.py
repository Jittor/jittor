"""Shared native checkpoint fixtures and independent NumPy training references."""

import jittor as jt
from jittor import nn
from jittor.distributed import checkpoint
import numpy as np


def model_optimizer():
    model = nn.Linear(3, 2)
    model.weight.update(jt.array(np.arange(6, dtype=np.float32).reshape(2, 3) / 20,
                                dtype="float32"))
    model.bias.update(jt.array([0.1, -0.2], dtype="float32"))
    optimizer = jt.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.95))
    return model, optimizer


def inputs(rank=0):
    x = np.arange(12, dtype=np.float32).reshape(4, 3) / 10 + rank / 7
    target = np.arange(8, dtype=np.float32).reshape(4, 2) / 11 - rank / 9
    return x, target


def training_step(model, optimizer, rank=0, observer=None):
    x, target = inputs(rank)
    prediction = model(jt.array(x, dtype="float32"))
    loss = ((prediction - jt.array(target, dtype="float32")) ** 2).mean()
    if observer is not None:
        observer(prediction)
        observer(loss)
    loss_value = float(loss.item())
    optimizer.backward(loss)
    if observer is not None:
        for group in optimizer.param_groups:
            for gradient in group["grads"]:
                observer(gradient)
    gradients = [np.array(value.numpy(), copy=True)
                 for group in optimizer.param_groups for value in group["grads"]]
    optimizer.step()
    jt.sync_all(True)
    if observer is not None:
        for value in model.state_dict().values():
            observer(value)
        for group in optimizer.param_groups:
            for field in ("m", "values"):
                for value in group.get(field, ()):
                    observer(value)
    trace = {
        "model": {name: np.array(value.numpy(), copy=True) for name, value in model.state_dict().items()},
        "m": [np.array(value.numpy(), copy=True) for group in optimizer.param_groups for value in group["m"]],
        "v": [np.array(value.numpy(), copy=True) for group in optimizer.param_groups for value in group["values"]],
        "steps": [group["n_step"] for group in optimizer.param_groups],
        "gradients": gradients,
        "loss": loss_value,
    }
    return trace


def numpy_trajectory(steps, world_size=1, frozen_bias=False):
    # Independent FP64 reference for native Adam's epsilon-before-bias-scaling
    # convention. This explicitly differs from torch.optim.Adam's convention.
    w = (np.arange(6, dtype=np.float32).reshape(2, 3) / 20).astype(np.float64)
    b = np.array([0.1, -0.2], dtype=np.float32).astype(np.float64)
    moments = [np.zeros_like(w), np.zeros_like(b)]
    variances = [np.zeros_like(w), np.zeros_like(b)]
    traces = []
    for step in range(1, steps + 1):
        gradients = [np.zeros_like(w), np.zeros_like(b)]
        losses = []
        for rank in range(world_size):
            x, target = inputs(rank)
            error = x.astype(np.float64) @ w.T + b - target.astype(np.float64)
            losses.append(float(np.mean(error ** 2)))
            dout = 2 * error / error.size
            gradients[0] += dout.T @ x / world_size
            if not frozen_bias:
                gradients[1] += dout.sum(axis=0) / world_size
        parameters = [w, b]
        for index, gradient in enumerate(gradients):
            moments[index] = 0.85 * moments[index] + 0.15 * gradient
            variances[index] = 0.95 * variances[index] + 0.05 * gradient ** 2
            factor = 0.01 * np.sqrt(1 - 0.95 ** step) / (1 - 0.85 ** step)
            parameters[index] -= moments[index] * factor / (np.sqrt(variances[index]) + 1e-8)
        traces.append({"model": {"weight": w.copy(), "bias": b.copy()},
                       "m": [value.copy() for value in moments],
                       "v": [value.copy() for value in variances],
                       "steps": [step], "gradients": [value.copy() for value in gradients],
                       "losses": losses})
    return traces


def assert_trace(actual, expected, *, exact=False, rank=0, model_prefix=""):
    for name, value in expected["model"].items():
        got = actual["model"][model_prefix + name]
        if exact:
            np.testing.assert_array_equal(got, value)
        else:
            np.testing.assert_allclose(got, value, atol=2e-6, rtol=2e-5)
    for field in ("m", "v", "gradients"):
        for got, want in zip(actual[field], expected[field]):
            if exact:
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)
    assert actual["steps"] == expected["steps"]
    loss = expected["loss"] if exact else expected["losses"][rank]
    assert actual["loss"] == loss if exact else abs(actual["loss"] - loss) < 2e-6


class Bfloat16TrainingModel(jt.Module):
    def __init__(self):
        self.weight = jt.array([0.25, -0.5], dtype="bfloat16").start_grad()

    def execute(self, value):
        return (self.weight * value).sum()


def bfloat16_training_step(model, optimizer):
    loss = model(jt.array([0.5, -0.25], dtype="bfloat16").stop_grad())
    value = float(loss.item())
    optimizer.backward(loss)
    gradient = optimizer.param_groups[0]["grads"][0].numpy().copy()
    optimizer.step()
    jt.sync_all(True)
    assert str(model.weight.dtype) == "bfloat16"
    group = optimizer.param_groups[0]
    assert str(group["values"][0].dtype) == "bfloat16"
    return {"weight": jt.reinterpret_view(model.weight, (2,), "uint16").numpy().copy(),
            "momentum": group["values"][0].numpy().copy(),
            "gradient": gradient, "loss": value,
            "step": group["n_step"]}


def check_bfloat16_training_resume(directory):
    model = Bfloat16TrainingModel()
    optimizer = jt.optim.SGD(model.parameters(), lr=0.125, momentum=0.5)
    parameter = np.array([0.25, -0.5], dtype=np.float32)
    gradient = np.array([0.5, -0.25], dtype=np.float32)
    momentum = np.zeros(2, dtype=np.float32)

    def check_reference(trace, index):
        nonlocal parameter, momentum
        loss = float(np.sum(parameter * gradient))
        momentum = gradient.copy() if index == 0 else 0.5 * momentum + gradient
        parameter -= 0.125 * momentum
        # All values in this dyadic fixture are exactly representable in BF16;
        # compare raw bits against an independent NumPy FP32 SGD trajectory.
        assert np.all((parameter.view(np.uint32) & 0xffff) == 0)
        bits = (parameter.view(np.uint32) >> 16).astype(np.uint16)
        np.testing.assert_array_equal(trace["weight"], bits)
        np.testing.assert_array_equal(trace["momentum"], momentum)
        np.testing.assert_array_equal(trace["gradient"], gradient)
        assert trace["step"] == index + 1 and trace["loss"] == loss

    for index in range(3):
        check_reference(bfloat16_training_step(model, optimizer), index)
    checkpoint.save_training_state(model, optimizer, directory)
    expected = []
    for index in range(3, 5):
        trace = bfloat16_training_step(model, optimizer)
        check_reference(trace, index)
        expected.append(trace)
    fresh = Bfloat16TrainingModel()
    resumed = jt.optim.SGD(fresh.parameters(), lr=0.7, momentum=0.0)
    checkpoint.load_training_state(fresh, resumed, directory)
    for want in expected:
        actual = bfloat16_training_step(fresh, resumed)
        for field in want:
            np.testing.assert_array_equal(actual[field], want[field])
    return fresh, resumed
