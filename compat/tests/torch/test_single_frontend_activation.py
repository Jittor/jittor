"""Torch activation has one target and cannot install onto the native backend."""

from _helpers import capability as _test_capability
import os
import types
from unittest import mock

import numpy as np
import pytest


def test_removed_mode_arguments_and_environment_are_rejected_before_activation():
    from jittor.compat.shim import runtime
    from jittor._runtime.state import RuntimeContext, RuntimeState
    native = types.ModuleType("single_frontend_native")
    native.runtime = RuntimeState(RuntimeContext(types.SimpleNamespace()))
    before = dict(vars(native))
    with mock.patch.object(runtime, "_activate_once") as activate_once:
        with pytest.raises(RuntimeError, match="legacy.*removed"):
            runtime.activate(_root_module=native, independent_namespace=False)
        with mock.patch.dict(os.environ, {"JITTOR_TORCH_INDEPENDENT": "0"}):
            with pytest.raises(RuntimeError, match="removed legacy"):
                runtime.activate(_root_module=native)
        activate_once.assert_not_called()
    assert dict(vars(native)) == before
    assert runtime.activation_status(native).phase == "inactive"


def test_direct_install_rejects_native_and_activation_target_is_always_independent():
    import jittor as jt
    import torch
    from jittor.compat import torch as compatibility
    from jittor.compat.shim import runtime
    before = {owner: dict(vars(owner)) for owner in (jt.Var, jt.Module, jt.Function)}
    policy = jt.autograd.get_policy()
    with pytest.raises(RuntimeError, match=r"install\(jittor\).*no longer supported"):
        compatibility.install(jt)
    assert runtime._installation_target(jt) is torch
    assert torch is not jt and torch.Tensor is not jt.Var
    assert jt.autograd.get_policy() == policy
    for owner, attributes in before.items():
        assert dict(vars(owner)) == attributes


def test_removed_import_mode_fails_before_preflight_writes():
    from jittor.compat.shim import preflight
    environment = {"JITTOR_TORCH_SHIM": "1", "JITTOR_TORCH_INDEPENDENT": "0"}
    with mock.patch.object(preflight, "_ensure_dir") as mkdir:
        with pytest.raises(RuntimeError, match="removed legacy"):
            preflight.prepare_import_environment(environ=environment)
        mkdir.assert_not_called()
    assert environment == {"JITTOR_TORCH_SHIM": "1", "JITTOR_TORCH_INDEPENDENT": "0"}
    assert not preflight.prepare_import_environment(
        environ={"JITTOR_TORCH_INDEPENDENT": "0"}).active


def test_native_dtype_casts_and_independent_cpu_cuda_storage_and_gradients():
    import jittor as jt
    import torch
    assert callable(jt.float32)
    assert not callable(torch.float32)
    with pytest.raises(TypeError):
        torch.float32([1., 2.])
    devices = ["cpu"] + (["cuda"] if _test_capability.check_accelerator('cuda', backend=jt).enabled else [])
    for device in devices:
        value = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64,
                             requires_grad=True, device=device)
        view = value.t()
        loss = (view * view).sum()
        grad, = torch.autograd.grad(loss, value)
        grad.sync()
        assert type(view) is type(grad) is torch.Tensor
        assert view.device.type == grad.device.type == device
        assert grad.location() == ("cpu" if device == "cpu" else "device")
        np.testing.assert_array_equal(grad.numpy(), [[2., 4.], [6., 8.]])
    native = jt.array([1., 2.])
    assert type(native) is jt.Var and native.placement_backend == -1
    assert isinstance(native.data, np.ndarray)
