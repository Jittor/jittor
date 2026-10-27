"""Direct contracts for the native rematerialization behind Torch checkpoint."""

import functools
import gc
import os
import weakref

import numpy as np
import pytest
import torch
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(autouse=True)
def native_device():
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA checkpoint acceptance requires a real CUDA device")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        yield device
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def checkpoint(fn, *args, **kwargs):
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True, **kwargs)


def snapshot(tensor):
    # Copy immediately: retained NumPy data views have a separate lifetime
    # regression and must not obscure the derivative under test here.
    return tensor.detach().cpu().numpy().copy()


@pytest.mark.parametrize("metadata", [[], (), {"tag": 1}])
def test_capture_releases_replay_tensors_without_cyclic_gc(native_device, metadata):
    from jittor.autograd._checkpoint_capture import Capture

    leaf = torch.ones(1, device=native_device)
    leaf.sync()
    watched = weakref.ref(leaf)
    replacements = {leaf.id: leaf}
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert Capture.__new__(Capture)._rebind(metadata, replacements, {}) is metadata
        del replacements, leaf
        assert watched() is None, "capture lookup retained a replay tensor in a reference cycle"
    finally:
        if enabled:
            gc.enable()


class Layer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.25, -0.5], [0.75, 0.125]], device=device))
        self.alias = self.weight
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2], device=device))
        self.calls = 0

    def forward(self, value, *, scale=1.0):
        self.calls += 1
        assert isinstance(self.weight, torch.nn.Parameter)
        result = (value @ self.weight + value @ self.alias + self.bias) * scale
        return {"values": (result.tanh(), [result.square(), None]), "tag": 7}


def objective(output):
    return output["values"][0].sum() + output["values"][1][0].sum() * 0.1


@pytest.mark.parametrize("input_grad", [True, False])
def test_captured_module_aliases_nested_outputs_kwargs_and_hooks(native_device, input_grad):
    plain, checked = Layer(native_device), Layer(native_device)
    x = torch.tensor([[0.2, 0.4], [-0.1, 0.8]], device=native_device, requires_grad=input_grad)
    y = x.detach().clone().requires_grad_(input_grad)
    expected = plain(x, scale=0.7)
    objective(expected).backward()
    expected_grad = [snapshot(p.grad) for p in plain.parameters()]
    if input_grad:
        expected_x = snapshot(x.grad)
    original = checked.weight
    hooks = []
    handle = original.register_hook(lambda grad: hooks.append(snapshot(grad)))
    actual = checkpoint(checked.__call__, y, scale=0.7)
    assert checked.calls == 1
    np.testing.assert_allclose(snapshot(actual["values"][0]), snapshot(expected["values"][0]), rtol=1e-6, atol=1e-7)
    assert actual["tag"] == 7 and actual["values"][1][1] is None
    objective(actual).backward()
    assert checked.calls == 2
    assert checked.weight is original and checked.alias is original
    assert len(hooks) == 1
    for parameter, wanted in zip(checked.parameters(), expected_grad):
        np.testing.assert_allclose(snapshot(parameter.grad), wanted, rtol=1e-5, atol=1e-6)
    if input_grad:
        np.testing.assert_allclose(snapshot(y.grad), expected_x, rtol=1e-5, atol=1e-6)
    handle.remove()


def test_closure_tensor_captures_and_nested_keyword_inputs(native_device):
    weight = torch.tensor([2., 3.], device=native_device, requires_grad=True)
    x = torch.tensor([4., 5.], device=native_device, requires_grad=True)
    def fn(*, inputs):
        return inputs["x"] * weight + inputs["again"] * weight
    output = checkpoint(functools.partial(fn), inputs={"x": x, "again": x})
    output.sum().backward()
    np.testing.assert_allclose(snapshot(x.grad), [4., 6.], rtol=0, atol=0)
    np.testing.assert_allclose(snapshot(weight.grad), [8., 10.], rtol=0, atol=0)


def test_dropout_replays_mask_and_restores_current_rng(native_device):
    x = torch.ones((8, 16), device=native_device, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    def fn(value):
        return torch.nn.functional.dropout(value, p=0.5, training=True).square()
    torch.manual_seed(913)
    expected = fn(x)
    expected.sum().backward()
    expected_values, expected_grad = snapshot(expected), snapshot(x.grad)
    torch.manual_seed(913)
    actual = checkpoint(fn, y)
    # Backward must preserve draws made AFTER the checkpoint forward as well.
    snapshot(torch.rand(23, device=native_device))
    before_cpu = snapshot(torch.get_rng_state())
    before_device = snapshot(torch.cuda.get_rng_state()) if native_device == "cuda" else None
    actual.sum().backward()
    np.testing.assert_array_equal(snapshot(actual), expected_values)
    np.testing.assert_array_equal(snapshot(y.grad), expected_grad)
    np.testing.assert_array_equal(snapshot(torch.get_rng_state()), before_cpu)
    if before_device is not None:
        np.testing.assert_array_equal(snapshot(torch.cuda.get_rng_state()), before_device)


def test_buffer_updates_once_and_outer_no_grad(native_device):
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.], device=native_device))
            self.register_buffer("running", torch.zeros(1, device=native_device))
        def forward(self, value):
            self.running.add_(1)
            return value * self.weight + self.running
    module = Stateful()
    value = torch.tensor([3.], device=native_device)
    result = checkpoint(module, value)
    result.sum().backward()
    np.testing.assert_array_equal(snapshot(module.running), [1.])
    np.testing.assert_array_equal(snapshot(module.weight.grad), [3.])
    with torch.no_grad():
        result = checkpoint(module, value)
    assert not result.requires_grad
    np.testing.assert_array_equal(snapshot(module.running), [2.])


def test_nested_checkpoint_and_repeated_backward(native_device):
    calls = []
    def inner(value):
        calls.append("inner")
        return value.square()
    def outer(value):
        calls.append("outer")
        return checkpoint(inner, value) * 3
    x = torch.tensor([2., 4.], device=native_device, requires_grad=True)
    loss = checkpoint(outer, x).sum()
    first, = torch.autograd.grad(loss, (x,), retain_graph=True)
    second, = torch.autograd.grad(loss, (x,), retain_graph=True)
    np.testing.assert_array_equal(snapshot(first), [12., 24.])
    np.testing.assert_array_equal(snapshot(second), [12., 24.])
    assert calls.count("outer") == 3 and calls.count("inner") == 5


def test_replay_exception_restores_parameter_identity_and_rng(native_device):
    module = Layer(native_device)
    original = module.weight
    calls = []
    def fn(value):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("intentional replay failure")
        return module(value)["values"][0]
    x = torch.ones((2, 2), device=native_device, requires_grad=True)
    output = checkpoint(fn, x)
    before = snapshot(torch.cuda.get_rng_state()) if native_device == "cuda" else snapshot(torch.get_rng_state())
    with pytest.raises(RuntimeError, match="intentional replay failure"):
        output.sum().backward()
    assert module.weight is original and module.alias is original
    after = snapshot(torch.cuda.get_rng_state()) if native_device == "cuda" else snapshot(torch.get_rng_state())
    np.testing.assert_array_equal(before, after)


def test_nonreentrant_is_explicitly_unsupported(native_device):
    x = torch.ones(2, device=native_device, requires_grad=True)
    with pytest.raises(NotImplementedError, match="non-reentrant"):
        torch.utils.checkpoint.checkpoint(lambda value: value.square(), x, use_reentrant=False)


def test_identity_output_does_not_freeze_captured_parameter(native_device):
    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.], device=native_device))
        def forward(self):
            return {"weight": (self.weight, self.weight)}
    module = Identity()
    original = module.weight
    result = checkpoint(module)
    assert original.requires_grad and module.weight is original
    (result["weight"][0] + result["weight"][1]).sum().backward()
    assert original.requires_grad
    np.testing.assert_array_equal(snapshot(original.grad), [2.])


@pytest.mark.parametrize("parameter", [False, True])
def test_tracked_alias_mutation_before_replay_is_rejected(native_device, parameter):
    layer = Layer(native_device)
    x = torch.ones((2, 2), device=native_device, requires_grad=True)
    output = checkpoint(layer, x)
    target = layer.weight if parameter else x
    with torch.no_grad():
        target.detach().add_(1)
    with pytest.raises(RuntimeError, match="modified before backward"):
        objective(output).backward()


def test_autocast_dtype_and_gradients_replay(native_device):
    if native_device != "cuda":
        pytest.skip("CUDA autocast contract")
    x = torch.ones((2, 4), device=native_device, requires_grad=True)
    weight = torch.ones((4, 3), device=native_device, requires_grad=True)
    calls = []
    def fn(value):
        out = value @ weight
        calls.append((torch.is_autocast_enabled("cuda"), str(out.dtype)))
        return out
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = checkpoint(fn, x)
    result.float().sum().backward()
    assert calls == [(True, "torch.bfloat16"), (True, "torch.bfloat16")]
    np.testing.assert_allclose(snapshot(x.grad), np.full((2, 4), 3), rtol=0, atol=0)
    np.testing.assert_allclose(snapshot(weight.grad), np.full((4, 3), 2), rtol=0, atol=0)


def test_autocast_preserves_registered_buffer_dtype(native_device):
    if native_device != "cuda":
        pytest.skip("CUDA autocast contract")
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((4, 3), device=native_device))
            self.register_buffer("offset", torch.ones(3, dtype=torch.float32, device=native_device))
            self.buffer_dtypes = []
        def forward(self, value):
            self.buffer_dtypes.append(str(self.offset.dtype))
            assert self.offset.dtype == torch.float32
            return value @ self.weight + self.offset
    module = Buffered()
    x = torch.ones((2, 4), device=native_device, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = checkpoint(module, x)
    result.float().sum().backward()
    assert module.buffer_dtypes == ["torch.float32", "torch.float32"]
    assert module.offset.dtype == torch.float32
