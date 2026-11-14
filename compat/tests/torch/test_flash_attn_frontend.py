"""The public FlashAttention facade preserves the caller's tensor frontend."""
import importlib.util
from pathlib import Path

import pytest
import torch
import jittor as jt


@pytest.fixture
def facade(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "shim/resources/stubs/flash_attn/__init__.py"
    spec = importlib.util.spec_from_file_location("flash_frontend_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Simulate an extension allocating its result through the native factory.
    # The actual fused CUDA forward/backward has separate device acceptance.
    monkeypatch.setattr(module, "_native_function", lambda name: (
        lambda q, *args, **kwargs: jt.zeros(q.shape, dtype=q.dtype)))
    return module


def test_native_result_preserves_torch_tensor_interface(facade):
    query = torch.ones((1, 2, 1, 4), device="cpu")
    result = facade.flash_attn_func(query, query, query)
    assert type(result) is torch.Tensor
    assert result.device == query.device
    assert callable(result.square)
    # The frontend scope must not change subsequent native construction.
    assert type(jt.zeros((1,))) is jt.Var


def test_native_callers_keep_native_results(facade):
    query = jt.ones((1, 2, 1, 4))
    result = facade.flash_attn_func(query, query, query)
    assert type(result) is jt.Var


def test_native_failure_restores_frontend_scope(facade, monkeypatch):
    def fail(query, *args, **kwargs):
        assert type(jt.zeros(query.shape)) is torch.Tensor
        raise RuntimeError("extension failed")

    monkeypatch.setattr(facade, "_native_function", lambda name: fail)
    query = torch.ones((1, 2, 1, 4), device="cpu")
    with pytest.raises(RuntimeError, match="extension failed"):
        facade.flash_attn_func(query, query, query)
    assert type(jt.zeros((1,))) is jt.Var


@pytest.mark.parametrize('varlen', [False, True])
def test_official_custom_backward_connects_frozen_extension_outputs(monkeypatch, tmp_path, varlen):
    from types import SimpleNamespace
    from jittor.compat.shim.backends import flash_attention

    def forward(q, k, v, *args):
        # Extension factories publish values without a native autograd graph.
        output = (q + 2*k + 3*v).detach()
        assert not output.requires_grad
        return output, None, None, None

    def backward(gradient, *args):
        return gradient, 2*gradient, 3*gradient

    low = SimpleNamespace(fwd=forward, varlen_fwd=forward,
                          bwd=backward, varlen_bwd=backward)
    monkeypatch.setattr(flash_attention, '_native_supported_dtype', lambda value: True)
    backend = flash_attention._make_official_backend(low, tmp_path)
    values = [torch.tensor([[[[1., 2.]]]], requires_grad=True) for _ in range(3)]

    def call():
        if varlen:
            lengths = torch.tensor([0, 1], dtype=torch.int32)
            return backend.flash_attn_varlen_func(
                *(value.reshape(1, 1, 2) for value in values), lengths, lengths, 1, 1)
        return backend.flash_attn_func(*values)

    output = call()
    assert output.requires_grad
    output.sum().backward()
    for factor, value in zip((1., 2., 3.), values):
        torch.testing.assert_close(value.grad, torch.full_like(value, factor))
    with torch.no_grad():
        assert not call().requires_grad


def test_opaque_extension_replay_is_refused(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from jittor.compat.shim.backends import flash_attention

    def forward(q, k, v, *args):
        return (q + k + v).detach(), None, None, None

    low = SimpleNamespace(fwd=forward, varlen_fwd=forward,
                          bwd=lambda *args: (), varlen_bwd=lambda *args: ())
    monkeypatch.setattr(flash_attention, '_native_supported_dtype', lambda value: True)
    backend = flash_attention._make_official_backend(low, tmp_path)
    runner = jt.capture_step(lambda value: backend.flash_attn_func(value, value, value))
    try:
        for index in range(4):
            value = torch.full((1, 2, 1, 4), float(index), device='cpu')
            torch.testing.assert_close(runner(value), 3*value)
        assert 'Python extension boundary' in runner.refused
    finally:
        runner.invalidate()
