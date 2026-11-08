"""Public FSDP backward and grad aliases do not depend on optimizer existence."""
import os

import numpy as np
import pytest
import torch
import jittor as jt

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from jittor._runtime.fallback import forbid_backend_fallbacks


def _full(value):
    method = getattr(value, "full_tensor", None)
    return method() if callable(method) else value


def _array(value):
    return _full(value).detach().float().cpu().numpy()


def _initialize(model, device):
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            data = np.arange(parameter.numel(), dtype=np.float32).reshape(tuple(parameter.shape))
            parameter.copy_(torch.tensor((data % 7 - 3) / 32 + (index + 1) / 64, device=device))


def _sequential_reference(parameters, data):
    first = data @ parameters["0.weight"].T + parameters["0.bias"]
    hidden = np.tanh(first)
    output = hidden @ parameters["2.weight"].T + parameters["2.bias"]
    dy = (2 * output) / output.size
    dh = dy @ parameters["2.weight"]
    da = dh * (1 - hidden * hidden)
    return output, {
        "0.weight": da.T @ data,
        "0.bias": da.sum(axis=0),
        "2.weight": dy.T @ hidden,
        "2.bias": dy.sum(axis=0),
    }, da @ parameters["0.weight"]


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("optimizer_before_forward", [False, True])
def test_nested_public_backward_preserves_snapshot_and_retired_alias_gradients(
        flat, optimizer_before_forward, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3),
    ).to(device)
    _initialize(model, device)
    retired = dict(model.named_parameters())
    initial = {name: _array(parameter).copy() for name, parameter in retired.items()}
    mesh = init_device_mesh(device, (1,))
    for child in (model[0], model[2]):
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    snapshot = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01) if optimizer_before_forward else None
    data = np.arange(32, dtype=np.float32).reshape(4, 8) / 100
    expected_output, expected_gradients, expected_input = _sequential_reference(initial, data[:, ::2])
    leaf = torch.tensor(data, dtype=torch.float32, device=device, requires_grad=True)
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        prediction = model(leaf[:, ::2])
        loss = prediction.square().mean()
        loss.backward()
        np.testing.assert_allclose(_array(prediction), expected_output, atol=2e-6, rtol=2e-5)
        expected_leaf = np.zeros_like(data)
        expected_leaf[:, ::2] = expected_input
        np.testing.assert_allclose(_array(leaf.grad), expected_leaf, atol=2e-6, rtol=2e-5)
        current = dict(model.named_parameters())
        for name, parameter in snapshot.items():
            gradient = parameter.grad
            assert gradient is not None and gradient.dtype == torch.float32
            assert gradient.device.type == device.split(":")[0]
            assert parameter._jittor_fsdp2_entry is current[name]._jittor_fsdp2_entry
            assert gradient is current[name].grad
            assert retired[name].grad is current[name].grad
            np.testing.assert_allclose(_array(gradient), expected_gradients[name], atol=2e-6, rtol=2e-5)
        # Preserve the late-construction path: a valid public backward must not
        # require an optimizer to have existed during that backward.
        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.step()
        for name, parameter in model.named_parameters():
            np.testing.assert_allclose(
                _array(parameter), initial[name] - .01 * expected_gradients[name],
                atol=2e-6, rtol=2e-5)
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("optimizer_before_forward", [False, True])
def test_public_grad_assignment_and_none_follow_current_entry(
        flat, optimizer_before_forward, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = torch.nn.Linear(2, 2).to(device)
    retired = dict(model.named_parameters())
    fully_shard(model, mesh=init_device_mesh(device, (1,)))
    snapshot = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01) if optimizer_before_forward else None
    model(torch.ones((2, 2), device=device)).square().mean().backward()
    for name, old in retired.items():
        current = dict(model.named_parameters())[name]
        assert old.grad is current.grad is snapshot[name].grad
        saved = current.grad.clone()
        old.grad = None
        assert old.grad is None and current.grad is None and snapshot[name].grad is None
        old.grad = saved
        assert old.grad is saved and current.grad is saved and snapshot[name].grad is saved
        current.grad = None
        assert old.grad is None and current.grad is None


class _Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(2, 2)
        self.unused = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Parameter(torch.tensor([.25, .5]), requires_grad=False)

    def forward(self, value, alternate=False):
        return (self.unused(value) if alternate else self.used(value)) + self.frozen


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("optimizer_before_forward", [False, True])
def test_unused_frozen_and_explicit_none_gradients_are_not_materialized(
        flat, optimizer_before_forward, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = _Branch().to(device)
    retired = dict(model.named_parameters())
    fully_shard(model, mesh=init_device_mesh(device, (1,)))
    snapshot = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01) if optimizer_before_forward else None
    for alternate in (False, True):
        for parameter in snapshot.values():
            parameter.grad = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        leaf = torch.ones((3, 2), device=device, requires_grad=True)
        model(leaf, alternate=alternate).square().mean().backward()
        used_prefix = "unused." if alternate else "used."
        for name, parameter in snapshot.items():
            if name.startswith(used_prefix):
                assert parameter.grad is not None
                assert parameter.grad.dtype == torch.float32
                assert parameter.grad.device.type == device.split(":")[0]
            else:
                assert parameter.grad is None
            assert retired[name].grad is parameter.grad
        assert leaf.grad is not None and np.isfinite(_array(leaf.grad)).all()


@pytest.mark.parametrize("flat", [False, True])
def test_no_optimizer_backward_accumulates_public_gradients_in_place(flat, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = torch.nn.Linear(2, 2).to(device)
    fully_shard(model, mesh=init_device_mesh(device, (1,)))
    snapshot = dict(model.named_parameters())
    data = torch.ones((3, 2), device=device)
    model(data).square().mean().backward()
    first = {name: parameter.grad for name, parameter in snapshot.items()}
    values = {name: _array(gradient).copy() for name, gradient in first.items()}
    model(data).square().mean().backward()
    for name, parameter in snapshot.items():
        assert parameter.grad is first[name]
        np.testing.assert_allclose(_array(parameter.grad), 2 * values[name], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("flat", [False, True])
def test_no_optimizer_mixed_policy_keeps_fp32_master_gradients(input_dtype, flat, monkeypatch):
    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)).to(device)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        output_dtype=torch.float32, cast_forward_inputs=True)
    mesh = init_device_mesh(device, (1,))
    for child in (model[0], model[2]):
        fully_shard(child, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    snapshot = dict(model.named_parameters())
    leaf = torch.ones((2, 4), dtype=input_dtype, device=device, requires_grad=True)
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        output = model(leaf)
        output.square().mean().backward()
        assert output.dtype == torch.float32 and leaf.grad.dtype == input_dtype
        for parameter in snapshot.values():
            assert parameter.grad is not None and parameter.grad.dtype == torch.float32
            assert parameter.grad.device.type == device.split(":")[0]
            assert np.isfinite(_array(parameter.grad)).all()
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def test_ordinary_backward_without_optimizer_keeps_its_public_contract():
    device = os.environ.get("FSDP_PUBLIC_GRAD_TEST_DEVICE", "cpu")
    model = _Branch().to(device)
    data = torch.ones((2, 2), device=device, requires_grad=True)
    model(data).square().mean().backward()
    assert model.used.weight.grad is not None and model.used.bias.grad is not None
    assert model.unused.weight.grad is None and model.unused.bias.grad is None
    assert model.frozen.grad is None and data.grad is not None
