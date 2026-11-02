"""Optimizer internal buffers follow explicit parameter placement."""
import types

import pytest
import jittor as jt
import torch

from jittor._core.dtypes import dtype_name
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.optim._state_placement import optimizer_state_scope


_FIELDS = {"SGD": ("values",), "Adam": ("m", "values"),
           "AdamW": ("m", "values"), "RMSprop": ("values",),
           "Adan": ("m", "v", "d", "pre_grad")}


def _parameter(frontend, device, dtype):
    if frontend == "torch":
        return torch.nn.Parameter(torch.tensor([1., -2.], dtype=dtype, device=device))
    index = int(device.split(":")[1]) if device.startswith("cuda:") else 0
    backend = 1 if device.startswith("cuda:") else 0
    token = jt.core._push_native_tensor_placement(backend, index)
    try:
        param = jt.array([1., -2.], dtype=dtype_name(dtype)).start_grad()
        assert type(param) is jt.Var
        assert int(param.placement_backend) == backend
        assert int(param.device_id) == (index if backend else -1)
        assert tuple(jt.core.dispatch_context([param])) == (
            ("cuda", index) if backend else ("cpu", -1))
        return param
    finally:
        jt.core._pop_native_tensor_placement(token)


def _optimizer(frontend, name, params):
    owner = torch.optim if frontend == "torch" else jt.optim
    kwargs = {"momentum": .9} if name == "SGD" else {}
    return getattr(owner, name)(params, lr=.01, **kwargs)


def _assert_group_state(group, name, expected_device, parameter_dtype):
    backend = 1 if expected_device.startswith("cuda:") else 0
    index = int(expected_device.split(":")[1]) if backend else -1
    param = group["params"][0]
    assert int(param.placement_backend) == backend
    assert tuple(jt.core.dispatch_context([param])) == (("cuda", index) if backend else ("cpu", -1))
    for field in _FIELDS[name]:
        state = group[field][0]
        assert type(state) is getattr(type(param), "_frontend_result_type", type(param))
        assert int(state.placement_backend) == backend
        if backend:
            assert int(state.device_id) == index
        assert tuple(jt.core.dispatch_context([state])) == (("cuda", index) if backend else ("cpu", -1))
        expected_dtype = dtype_name(parameter_dtype) if name == "SGD" else "float32"
        assert dtype_name(state.dtype) == expected_dtype
        if type(state) is torch.Tensor:
            assert str(state.device) == expected_device
        state.sync()
        assert state.location() == ("device" if backend else "cpu")
        if name == "SGD":
            assert state._storage_is_contiguous()


@pytest.mark.parametrize("frontend", ["torch", "native"])
@pytest.mark.parametrize("name", list(_FIELDS))
@pytest.mark.parametrize("ambient", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_parameter_state_ignores_conflicting_torch_device(frontend, name, ambient, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cpu = _parameter(frontend, "cpu", dtype)
        cuda = _parameter(frontend, "cuda:0", dtype)
        with torch.device(ambient):
            before = jt.core._current_tensor_placement()
            opt = _optimizer(frontend, name, [{"params": [cpu]}, {"params": [cuda]}])
            assert jt.core._current_tensor_placement() == before
        _assert_group_state(opt.param_groups[0], name, "cpu", dtype)
        _assert_group_state(opt.param_groups[1], name, "cuda:0", dtype)


@pytest.mark.parametrize("frontend", ["torch", "native"])
@pytest.mark.parametrize("name", list(_FIELDS))
def test_add_group_state_uses_each_explicit_parameter_device(frontend, name):
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cpu = _parameter(frontend, "cpu", torch.bfloat16)
        cuda = _parameter(frontend, "cuda:0", torch.bfloat16)
        with torch.device("cpu"):
            opt = _optimizer(frontend, name, [{"params": [cuda]}])
            opt.add_param_group({"params": [cpu]})
        _assert_group_state(opt.param_groups[0], name, "cuda:0", torch.bfloat16)
        _assert_group_state(opt.param_groups[1], name, "cpu", torch.bfloat16)


@pytest.mark.parametrize("frontend", ["torch", "native"])
def test_two_cuda_devices_remain_distinct_under_cpu_context(frontend):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two real visible CUDA devices")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        p0 = _parameter(frontend, "cuda:0", torch.bfloat16)
        p1 = _parameter(frontend, "cuda:1", torch.bfloat16)
        with torch.device("cpu"):
            opt = _optimizer(frontend, "AdamW", [{"params": [p1]}, {"params": [p0]}])
        _assert_group_state(opt.param_groups[0], "AdamW", "cuda:1", torch.bfloat16)
        _assert_group_state(opt.param_groups[1], "AdamW", "cuda:0", torch.bfloat16)


def test_scope_restores_nested_ambient_after_exception(monkeypatch):
    param = torch.nn.Parameter(torch.tensor([1., 2.], device="cpu"))
    token = jt.core._set_tensor_placement(0, 0)
    try:
        outer = jt.core._current_tensor_placement()
        before_device = int(jt.current_device())
        def fail(_param):
            raise RuntimeError("injected zero allocation failure")
        monkeypatch.setattr(jt, "zeros_like", fail)
        with torch.device("cpu"):
            with pytest.raises(RuntimeError, match="injected zero allocation failure"):
                torch.optim.AdamW([param], lr=.01)
            assert jt.core._current_tensor_placement() == outer
            assert int(jt.current_device()) == before_device
    finally:
        jt.core._reset_tensor_placement(token)


@pytest.mark.parametrize("frontend", ["torch", "native"])
def test_conflicting_ambient_restored_when_allocation_raises(monkeypatch, frontend):
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cpu = _parameter(frontend, "cpu", torch.bfloat16)
        token = jt.core._set_tensor_placement(1, 0)
        try:
            before = jt.core._current_tensor_placement()
            before_device = int(jt.current_device())
            def fail(_param):
                assert jt.core._current_tensor_placement() == (0, 0)
                raise RuntimeError("injected conflicting-scope failure")
            monkeypatch.setattr(jt, "zeros_like", fail)
            with torch.device("cuda:0"):
                with pytest.raises(RuntimeError, match="injected conflicting-scope failure"):
                    _optimizer(frontend, "AdamW", [{"params": [cpu]}])
                assert jt.core._current_tensor_placement() == before
                assert int(jt.current_device()) == before_device
        finally:
            jt.core._reset_tensor_placement(token)


def test_follow_runtime_scope_keeps_existing_ambient_unchanged():
    token = jt.core._set_tensor_placement(0, 0)
    try:
        before = jt.core._current_tensor_placement()
        with optimizer_state_scope(types.SimpleNamespace(placement_backend=-1, device_id=0)):
            assert jt.core._current_tensor_placement() == before
        assert jt.core._current_tensor_placement() == before
    finally:
        jt.core._reset_tensor_placement(token)


def test_public_zeros_like_explicit_device_still_overrides_reference():
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cuda = torch.tensor([1., 2.], device="cuda:0")
        cpu = torch.zeros_like(cuda, device="cpu")
        assert str(cpu.device) == "cpu"
        assert tuple(jt.core.dispatch_context([cpu])) == ("cpu", -1)



def test_native_factory_restores_nested_placement_after_exception(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        assert jt.core._current_tensor_placement() is None
        device_before = int(jt.current_device())
        outer = jt.core._push_native_tensor_placement(0, 0)
        try:
            before = jt.array([1.])
            assert type(before) is jt.Var and int(before.placement_backend) == 0
            gpu = _parameter("native", "cuda:0", torch.float16)
            assert int(gpu.placement_backend) == 1
            restored = jt.array([2.])
            assert type(restored) is jt.Var and int(restored.placement_backend) == 0
            with monkeypatch.context() as patch:
                def fail(*args, **kwargs):
                    raise RuntimeError("injected native factory failure")
                patch.setattr(jt, "array", fail)
                with pytest.raises(RuntimeError, match="injected native factory failure"):
                    _parameter("native", "cuda:0", torch.float16)
            after_failure = jt.array([3.])
            assert type(after_failure) is jt.Var and int(after_failure.placement_backend) == 0
            assert tuple(jt.core.dispatch_context([after_failure])) == ("cpu", -1)
            assert jt.core._current_tensor_placement() is None
            assert int(jt.current_device()) == device_before
        finally:
            jt.core._pop_native_tensor_placement(outer)



def test_optimizer_scope_restores_native_and_frontend_nested_on_exception():
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cpu = _parameter("native", "cpu", torch.float32)
        cuda = _parameter("torch", "cuda:0", torch.float32)
        previous = jt.core._current_tensor_placement()
        native_outer = jt.core._push_native_tensor_placement(1, 0)
        python_outer = jt.core._set_tensor_placement(1, 0)
        try:
            with pytest.raises(RuntimeError, match="outer injected failure"):
                with optimizer_state_scope(cpu):
                    assert jt.core._current_tensor_placement() == (0, 0)
                    native = jt.array([1.])
                    assert type(native) is jt.Var and int(native.placement_backend) == 0
                    with pytest.raises(ValueError, match="inner injected failure"):
                        with optimizer_state_scope(cuda):
                            assert jt.core._current_tensor_placement() == (1, 0)
                            frontend = jt.array([2.])
                            assert type(frontend) is torch.Tensor
                            assert int(frontend.placement_backend) == 1
                            raise ValueError("inner injected failure")
                    assert jt.core._current_tensor_placement() == (0, 0)
                    native_after = jt.array([3.])
                    assert type(native_after) is jt.Var and int(native_after.placement_backend) == 0
                    raise RuntimeError("outer injected failure")
            assert jt.core._current_tensor_placement() == (1, 0)
            restored = jt.array([4.])
            assert type(restored) is jt.Var and int(restored.placement_backend) == 1
        finally:
            jt.core._reset_tensor_placement(python_outer)
            jt.core._pop_native_tensor_placement(native_outer)
        assert jt.core._current_tensor_placement() == previous


@pytest.mark.parametrize("name", ["SGD", "AdamW"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_public_optimizer_state_retains_torch_type_and_parameter_device(name, device):
    if not torch.cuda.is_available():
        pytest.skip("requires actual CUDA")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        param = _parameter("torch", device, torch.float32)
        conflict = "cuda:0" if device == "cpu" else "cpu"
        with torch.device(conflict):
            opt = _optimizer("torch", name, [param])
        (param.float() * param.float()).sum().backward()
        opt.step()
        state = opt.state[param]
        fields = ("momentum_buffer",) if name == "SGD" else ("exp_avg", "exp_avg_sq")
        for field in fields:
            value = state[field]
            assert type(value) is torch.Tensor
            assert str(value.device) == device
            assert tuple(jt.core.dispatch_context([value])) == (
                ("cpu", -1) if device == "cpu" else ("cuda", 0))
            value.sync()
            assert value.location() == ("cpu" if device == "cpu" else "device")
