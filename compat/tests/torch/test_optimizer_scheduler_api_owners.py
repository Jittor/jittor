"""Stable optimizer/scheduler owners retain update and state trajectories."""
import ast
from functools import partial
import importlib
import inspect
import os
import pickle
import textwrap

import numpy as np
import pytest
import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner


def _linear_resume_factor(step, *, total_steps):
    return max(0.0, float(total_steps - step) / float(total_steps))


def test_optimizer_scheduler_installers_only_bind():
    for module_name, name in (
        ("lr_scheduler", "_install_lr_scheduler"),
        ("optimizers", "_install_optimizers"),
        ("optimizers", "install_module_keys"),
        ("optim_frontend", "make_optimizer_frontend"),
    ):
        module = importlib.import_module("jittor.compat.torch." + module_name)
        node = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name)))).body[0]
        assert not [child for child in ast.walk(node) if child is not node and
                    isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda))]


def test_scheduler_and_optimizer_implementation_identity():
    torch = compatibility_owner(jt)
    schedulers = importlib.import_module("jittor.compat.torch.lr_scheduler")
    updates = importlib.import_module("jittor.compat.torch.optimizer_api")
    for name in ("LRScheduler", "LambdaLR", "MultiplicativeLR", "ConstantLR", "LinearLR",
                 "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
                 "PolynomialLR", "OneCycleLR", "SequentialLR", "ChainedScheduler",
                 "ReduceLROnPlateau"):
        actual = getattr(torch.optim.lr_scheduler, name)
        assert actual is getattr(schedulers, name)
        assert pickle.loads(pickle.dumps(actual)) is actual
        assert "<locals>" not in actual.__qualname__
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan"):
        actual = getattr(torch.optim, name).step
        assert actual is getattr(updates, name.lower() + "_step")
        assert pickle.loads(pickle.dumps(actual)) is actual
    assert torch.optim.Optimizer.state_dict is updates._state_dict_torch
    assert torch.optim.Optimizer.load_state_dict is updates._lsd


@pytest.mark.parametrize("name", ("SGD", "Adam", "AdamW"))
def test_update_scheduler_and_restore_trajectory(name):
    torch = compatibility_owner(jt)
    parameter = torch.tensor([1., -2.], requires_grad=True)
    options = {"lr": 0.1, "weight_decay": 0.03}
    if name == "SGD":
        options["momentum"] = 0.9
    else:
        options.update(betas=(0.8, 0.9), eps=1e-6)
    optimizer = getattr(torch.optim, name)([parameter], **options)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    expected = np.array([1., -2.], dtype=np.float64)
    momentum = np.zeros(2)
    variance = np.zeros(2)
    for step, values in enumerate(([0.2, -0.4], [0.1, 0.3]), 1):
        learning_rate = 0.1 * 0.5 ** (step - 1)
        gradient = np.array(values, dtype=np.float64)
        parameter.grad = torch.tensor(values)
        published = parameter.grad
        if name == "SGD":
            gradient = gradient + 0.03 * expected
            momentum = 0.9 * momentum + gradient
            expected = expected - learning_rate * momentum
        else:
            if name == "Adam":
                gradient = gradient + 0.03 * expected
            else:
                expected = expected * (1 - learning_rate * 0.03)
            momentum = 0.8 * momentum + 0.2 * gradient
            variance = 0.9 * variance + 0.1 * gradient ** 2
            expected -= learning_rate * (momentum / (1 - 0.8 ** step)) / (
                np.sqrt(variance / (1 - 0.9 ** step)) + 1e-6)
        optimizer.step()
        assert parameter.grad is published
        np.testing.assert_allclose(parameter.numpy(), expected, rtol=4e-6, atol=4e-6)
        schedule.step()
    assert schedule.get_last_lr() == [0.025]
    saved = optimizer.state_dict()
    restored_parameter = torch.tensor(parameter.numpy(), requires_grad=True)
    restored = getattr(torch.optim, name)([restored_parameter], **options)
    restored.load_state_dict(saved)
    assert restored_parameter.requires_grad
    assert restored.param_groups[0]["lr"] == 0.025
    for key, value in optimizer.state[parameter].items():
        other = restored.state[restored_parameter][key]
        if hasattr(value, "numpy"):
            np.testing.assert_allclose(other.numpy(), value.numpy())
        else:
            assert other == value
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None
    parameter.requires_grad_(False)
    restored_parameter.requires_grad_(False)
    from jittor.compat.torch.nested import _torch_prune_leaf_registry
    _torch_prune_leaf_registry()


def test_lambda_scheduler_state_round_trips_through_weights_only_checkpoint(tmp_path):
    torch = compatibility_owner(jt)
    factor = partial(_linear_resume_factor, total_steps=4)
    parameter = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    optimizer.step()
    scheduler.step()

    state = scheduler.state_dict()
    assert state["lr_lambdas"] == [{}]
    checkpoint = tmp_path / "scheduler.pt"
    torch.save(state, os.fspath(checkpoint))
    loaded = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
    assert loaded["lr_lambdas"] == [{}]

    resumed_parameter = torch.tensor([1.0], requires_grad=True)
    resumed_optimizer = torch.optim.SGD(
        [resumed_parameter], lr=optimizer.param_groups[0]["lr"]
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, factor)
    resumed_scheduler.load_state_dict(loaded)
    assert resumed_scheduler.lr_lambdas[0] is factor

    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert scheduler.state_dict() == resumed_scheduler.state_dict()
    assert optimizer.param_groups[0]["lr"] == resumed_optimizer.param_groups[0]["lr"]


def test_swa_ema_average_factories_keep_state_and_importable_math():
    schedulers = importlib.import_module("jittor.compat.torch.lr_scheduler")
    average = schedulers.get_swa_avg_fn()
    assert average is schedulers.get_swa_avg_fn()
    assert pickle.loads(pickle.dumps(average)) is average
    assert average(2., 8., 2) == 4.
    ema = schedulers.get_ema_avg_fn(0.75)
    assert ema(2., 8., 2) == 3.5
    assert pickle.loads(pickle.dumps(ema))(2., 8., 2) == 3.5
