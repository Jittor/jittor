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
