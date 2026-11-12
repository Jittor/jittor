"""Ordinary linear training keeps live dispatch owners, operands and gradients."""
import cProfile
import importlib
import weakref
from unittest import mock

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not (
        torch.cuda.is_available() and getattr(jt.compiler, "is_cuda", False)
    ):
        pytest.skip("CUDA build and allocated CUDA device required")
    with jt.flag_scope(auto_graph_replay=0):
        yield request.param


def _linear_frames(module, x):
    owner = importlib.import_module("jittor.nn.functional.linear")
    profile = cProfile.Profile()
    profile.enable()
    try:
        result = module(x)
    finally:
        profile.disable()
    count = sum(entry.callcount for entry in profile.getstats()
                if entry.code is owner._NATIVE_LINEAR_FUNCTION.__code__)
    return result, count


def _prepared(device):
    layer = torch.nn.Linear(32, 32).to(device)
    outer = torch.nn.Sequential(layer)
    x = torch.ones((2, 3, 32), device=device, requires_grad=True)
    outer(x).sum().item()  # publish leaves before checking nested dispatch
    return layer, outer, x


def test_ordinary_linear_values_input_weight_and_bias_gradients(device):
    rng = np.random.default_rng(31)
    for shape in ((8, 32), (2, 8, 32)):
        for input_grad in (False, True):
            for bias_grad in (False, True):
                values = rng.integers(-8, 9, shape).astype("float32") / 8
                weights = rng.integers(-8, 9, (16, 32)).astype("float32") / 8
                bias = rng.integers(-8, 9, (16,)).astype("float32") / 8
                cotangent = rng.integers(-8, 9, shape[:-1] + (16,)).astype("float32") / 8
                layer = torch.nn.Linear(32, 16).to(device)
                with torch.no_grad():
                    layer.weight.copy_(torch.tensor(weights, device=device))
                    layer.bias.copy_(torch.tensor(bias, device=device))
                layer.bias.requires_grad_(bias_grad)
                x = torch.tensor(values, device=device, requires_grad=input_grad)
                outer = torch.nn.Sequential(layer)
                outer(x).sum().item()
                output, frames = _linear_frames(outer, x)
                assert frames == (0 if device == "cuda" else 1)
                assert output.device.type == device
                np.testing.assert_allclose(output.detach().cpu().numpy(),
                                           values @ weights.T + bias, rtol=1e-5, atol=1e-5)
                inputs = [x, layer.weight, layer.bias]
                targets = [p for p in inputs if p.requires_grad]
                gradients = torch.autograd.grad(
                    (output * torch.tensor(cotangent, device=device)).sum(), targets)
                expected = [cotangent @ weights,
                            cotangent.reshape(-1, 16).T @ values.reshape(-1, 32),
                            cotangent.reshape(-1, 16).sum(0)]
                for gradient, reference in zip(gradients,
                        [ref for p, ref in zip(inputs, expected) if p.requires_grad]):
                    assert gradient.device.type == device
                    np.testing.assert_allclose(gradient.detach().cpu().numpy(), reference,
                                               rtol=1e-5, atol=1e-5)


def test_new_parameters_are_read_live_and_old_holders_are_released(device):
    layer, outer, x = _prepared(device)
    old = weakref.ref(layer.weight)
    layer.weight = torch.nn.Parameter(torch.full((32, 32), 2., device=device))
    assert old() is None
    output, frames = _linear_frames(outer, x)
    assert frames == (0 if device == "cuda" else 1)
    np.testing.assert_allclose(output.detach().cpu().numpy(),
                               np.full((2, 3, 32), 64.) + layer.bias.detach().cpu().numpy())
    output.sum().backward()
    np.testing.assert_array_equal(layer.weight.grad.cpu().numpy(), np.full((32, 32), 6.))


def test_accumulator_override_is_selected_once(device):
    from jittor._runtime import dispatch
    _, outer, x = _prepared(device)
    calls = []
    def supports(*args):
        calls.append("supports")
        return True
    def provider(value, weight, bias):
        calls.append("provider")
        return value + 13
    with dispatch.override_kernel("linear_bias_accumulator", device, provider,
                                  supports=supports, priority=100):
        result = outer(x)
    assert calls == ["supports", "provider"]
    np.testing.assert_array_equal(result.detach().cpu().numpy(), np.full((2, 3, 32), 14.))


def test_declining_matmul_predicate_is_not_repeated(device):
    from jittor._runtime import dispatch
    from jittor.compat.torch import nn_frontend
    _, outer, x = _prepared(device)
    calls = []
    def supports(*args):
        calls.append(1)
        return False
    def unexpected(*args):
        raise AssertionError("declined provider ran")
    dispatch.register_kernel("matmul", device, unexpected,
                             supports=supports, priority=100)
    try:
        # A CPU build without a BLAS relay can retry after dtype resolution.
        # Native module dispatch must preserve that owner's call count.
        with mock.patch.object(nn_frontend, "_NATIVE_CALL", None):
            reference = outer(x)
        owner_count = len(calls)
        calls.clear()
        result = outer(x)
    finally:
        dispatch.unregister_kernel("matmul", device, unexpected)
    assert len(calls) == owner_count
    if device == "cuda":
        assert calls == [1]
    np.testing.assert_array_equal(result.detach().cpu().numpy(), reference.detach().cpu().numpy())
    _, frames = _linear_frames(outer, x)
    assert frames == (0 if device == "cuda" else 1)


def test_custom_provider_error_is_propagated_once(device):
    from jittor._runtime import dispatch
    _, outer, x = _prepared(device)
    class ProviderError(ValueError):
        pass
    calls = []
    def failure(*args):
        calls.append(1)
        raise ProviderError("linear provider failure")
    before = jt.core._current_tensor_placement()
    with dispatch.override_kernel("linear_bias_accumulator", device, failure, priority=100):
        with pytest.raises(ProviderError, match="linear provider failure"):
            outer(x)
    assert calls == [1]
    assert jt.core._current_tensor_placement() == before


def test_replaced_linear_and_tensor_methods_keep_their_owner(device):
    owner = importlib.import_module("jittor.nn.modules.linear")
    _, outer, x = _prepared(device)
    with mock.patch.object(owner, "linear", side_effect=lambda value, *args: value + 9) as call:
        result = outer(x)
    assert call.call_count == 1
    np.testing.assert_array_equal(result.detach().cpu().numpy(), np.full((2, 3, 32), 10.))
    original = torch.Tensor._storage_is_contiguous
    calls = []
    def storage(value):
        calls.append(1)
        return original(value)
    with mock.patch.object(torch.Tensor, "_storage_is_contiguous", storage):
        result, frames = _linear_frames(outer, x)
    assert frames == 1 and calls
    assert tuple(result.shape) == (2, 3, 32)


def test_tensor_subclasses_and_strided_inputs_keep_python_dispatch(device):
    _, outer, x = _prepared(device)
    class Wrapped(torch.Tensor):
        pass
    wrapped = torch.Tensor._make_subclass(Wrapped, x, True)
    result, frames = _linear_frames(outer, wrapped)
    assert frames == 1
    np.testing.assert_array_equal(result.detach().cpu().numpy(), outer(x).detach().cpu().numpy())
    sliced = torch.ones((2, 3, 64), device=device, requires_grad=True)[..., ::2]
    result, frames = _linear_frames(outer, sliced)
    assert frames == 1
    np.testing.assert_array_equal(result.detach().cpu().numpy(), outer(x).detach().cpu().numpy())
    offset = torch.ones((3, 3, 32), device=device, requires_grad=True)[1:]
    result, frames = _linear_frames(outer, offset)
    assert frames == 1
    np.testing.assert_array_equal(result.detach().cpu().numpy(), outer(x).detach().cpu().numpy())


@pytest.mark.parametrize("kind", ("reshape", "prefix"))
@pytest.mark.parametrize("train_input", (False, True))
def test_dense_input_views_keep_root_gradients_and_hooks(device, kind, train_input):
    rng = np.random.default_rng(317)
    values = rng.integers(-8, 9, (2, 3, 32)).astype("float32") / 8
    weights = rng.integers(-8, 9, (16, 32)).astype("float32") / 8
    bias = rng.integers(-8, 9, (16,)).astype("float32") / 8
    root = torch.tensor(values, device=device, requires_grad=train_input)
    x = root.reshape(6, 32) if kind == "reshape" else root[:1]
    inputs = values.reshape(6, 32) if kind == "reshape" else values[:1]
    layer = torch.nn.Linear(32, 16).to(device)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights, device=device))
        layer.bias.copy_(torch.tensor(bias, device=device))
    outer = torch.nn.Sequential(layer)
    outer(x).sum().item()
    events = []
    if train_input:
        def view_hook(gradient):
            events.append("view")
            return gradient * .5
        def root_hook(gradient):
            events.append("root")
            return gradient * .25
        x.register_hook(view_hook)
        root.register_hook(root_hook)
    result, frames = _linear_frames(outer, x)
    assert frames == (0 if device == "cuda" else 1)
    cotangent = rng.integers(-8, 9, inputs.shape[:-1] + (16,)).astype("float32") / 8
    targets = (root, layer.weight, layer.bias) if train_input else (layer.weight, layer.bias)
    gradients = torch.autograd.grad(
        (result * torch.tensor(cotangent, device=device)).sum(), targets)
    expected = [cotangent.reshape(-1, 16).T @ inputs.reshape(-1, 32),
                cotangent.reshape(-1, 16).sum(0)]
    if train_input:
        root_gradient = np.zeros_like(values)
        if kind == "reshape":
            root_gradient = (cotangent @ weights).reshape(values.shape) * .125
        else:
            root_gradient[:1] = (cotangent @ weights) * .125
        expected.insert(0, root_gradient)
        assert events == ["view", "root"]
    np.testing.assert_allclose(result.detach().cpu().numpy(), inputs @ weights.T + bias,
                               rtol=1e-5, atol=1e-5)
    for gradient, reference in zip(gradients, expected):
        assert gradient.device.type == device
        np.testing.assert_allclose(gradient.detach().cpu().numpy(), reference,
                                   rtol=1e-5, atol=1e-5)


def test_dense_view_reads_rebased_values_and_keeps_its_source_gradient(device):
    layer, outer, _ = _prepared(device)
    root = torch.ones((6, 32), device=device, requires_grad=True)
    base = root * 2
    x = base.reshape(2, 3, 32)
    with torch.no_grad():
        base.add_(1)
    output, frames = _linear_frames(outer, x)
    assert frames == (0 if device == "cuda" else 1)
    expected = np.full((2, 3, 32), 3.) @ layer.weight.detach().cpu().numpy().T
    expected += layer.bias.detach().cpu().numpy()
    np.testing.assert_allclose(output.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
    dx, dw, db = torch.autograd.grad(output.sum(), (root, layer.weight, layer.bias))
    assert dx.device.type == dw.device.type == db.device.type == device
    expected_dx = np.broadcast_to(2 * layer.weight.detach().cpu().numpy().sum(0), (6, 32))
    np.testing.assert_allclose(dx.detach().cpu().numpy(), expected_dx, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(dw.detach().cpu().numpy(), np.full((32, 32), 18.))
    np.testing.assert_array_equal(db.detach().cpu().numpy(), np.full((32,), 6.))
