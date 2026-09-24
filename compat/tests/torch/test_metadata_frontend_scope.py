"""Metadata reads do not enter compute policy or retain stale holder metadata."""

import os

import numpy as np
import pytest


@pytest.fixture
def runtime():
    import jittor as jt
    import torch
    device = os.environ.get("JITTOR_METADATA_TEST_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    if device == "cuda":
        from _helpers import capability
        if not capability.check_accelerator("cuda", backend=jt).enabled:
            pytest.skip("accelerator prerequisite: real CUDA required")
    with jt.flag_scope(use_cuda=int(device == "cuda"), auto_flush_ops=0):
        yield jt, torch, device


@pytest.mark.parametrize("subtype", [False, True])
@pytest.mark.parametrize("name", [
    "dtype", "shape", "ndim", "dim", "device_id", "placement_backend",
    "numel", "nbytes",
])
def test_metadata_does_not_resolve_compute_policy(runtime, monkeypatch, subtype, name):
    jt, torch, device = runtime
    cls = type("MetadataTensor", (torch.Tensor,), {
        "_frontend_backend": jt,
    }) if subtype else torch.Tensor
    value = cls([[1., 2., 3.], [4., 5., 6.]], device=device)
    assert value.location() == "none"
    before = jt.introspection.counters.live_vars
    calls = []
    original = cls._frontend_precision_policy

    def policy(owner):
        calls.append(owner)
        return original()

    monkeypatch.setattr(cls, "_frontend_precision_policy", classmethod(policy))
    expected = {
        "dtype": torch.float32, "shape": (2, 3), "ndim": 2, "dim": 2,
        "device_id": 0 if device == "cuda" else -1,
        "placement_backend": 1 if device == "cuda" else 0,
        "numel": 6, "nbytes": 24,
    }[name]
    for _ in range(8):
        actual = getattr(value, name)
        if name in ("dim", "numel"):
            actual = actual()
        if name == "shape":
            actual = tuple(actual)
        assert actual == expected
    assert calls == [], "reading metadata must not re-enter compute policy"
    assert jt.introspection.counters.live_vars == before
    assert value.location() == "none"


def test_metadata_tracks_holder_rebinding(runtime):
    jt, torch, device = runtime
    value = torch.tensor([[1., 2., 3.]], device=device)
    assert value.dtype == torch.float32 and tuple(value.shape) == (1, 3)
    replacement = torch.tensor([7, 8, 9, 10], dtype=torch.int64, device=device)
    jt.Var.assign(value, replacement)
    assert value.dtype == torch.int64
    assert tuple(value.shape) == (4,) and value.ndim == value.dim() == 1
    assert value.numel() == 4 and value.nbytes == 32
    np.testing.assert_array_equal(value.numpy(), [7, 8, 9, 10])


def test_parameter_metadata_preserves_nested_scope_and_tensor_result(runtime, monkeypatch):
    jt, torch, device = runtime
    from jittor.compat.torch.frontend import tensor_frontend
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    native_policy = tuple(jt.core.float32_precision_state())
    calls = []

    def policy(owner):
        calls.append(owner)
        return (2, 0)

    monkeypatch.setattr(torch.Tensor, "_frontend_precision_policy", classmethod(policy))
    with pytest.raises(RuntimeError, match="nested metadata probe"):
        with tensor_frontend(torch.Tensor):
            assert tuple(jt.core.float32_precision_state()) == ("medium", "highest")
            calls.clear()
            assert parameter.dtype == torch.float32
            assert tuple(parameter.shape) == (2,)
            assert parameter.ndim == parameter.dim() == 1
            assert parameter.numel() == 2 and parameter.nbytes == 8
            assert parameter.placement_backend == (1 if device == "cuda" else 0)
            assert parameter.device_id == (0 if device == "cuda" else -1)
            assert calls == []
            assert tuple(jt.core.float32_precision_state()) == ("medium", "highest")
            result = jt.Var.multiply(parameter, 3)
            assert calls and type(result) is torch.Tensor and result.requires_grad
            raise RuntimeError("nested metadata probe")
    assert tuple(jt.core.float32_precision_state()) == native_policy
    np.testing.assert_array_equal(result.numpy(), [3., 6.])


@pytest.mark.parametrize("subtype", [False, True])
def test_compute_still_reads_policy_and_restores_on_failure(runtime, monkeypatch, subtype):
    jt, torch, device = runtime
    cls = type("PolicyTensor", (torch.Tensor,), {
        "_frontend_backend": jt,
    }) if subtype else torch.Tensor
    value = cls([1., 2.], device=device, requires_grad=True)
    native_policy = tuple(jt.core.float32_precision_state())
    seen = []
    tier = [0, 1]

    def policy(owner):
        seen.append(tuple(tier))
        return tuple(tier)

    monkeypatch.setattr(cls, "_frontend_precision_policy", classmethod(policy))
    first = jt.Var.add(value, 1)
    assert seen and seen[-1] == (0, 1)
    tier[:] = [2, 0]
    second = jt.Var.multiply(value, 2)
    assert seen[-1] == (2, 0)
    assert type(first) is cls and type(second) is cls
    assert first.requires_grad and second.requires_grad
    assert tuple(jt.core.float32_precision_state()) == native_policy

    def broken(owner):
        raise RuntimeError("metadata policy probe")

    monkeypatch.setattr(cls, "_frontend_precision_policy", classmethod(broken))
    with pytest.raises(RuntimeError, match="metadata policy probe"):
        jt.Var.add(value, 1)
    assert tuple(jt.core.float32_precision_state()) == native_policy
    monkeypatch.setattr(cls, "_frontend_precision_policy", classmethod(policy))
    np.testing.assert_array_equal(first.numpy(), [2., 3.])
    np.testing.assert_array_equal(second.numpy(), [2., 4.])
    assert first.device.type == second.device.type == device
