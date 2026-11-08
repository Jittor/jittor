"""NPU namespace contracts, with isolated backend routing and real ACL work."""

import importlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _api():
    return importlib.import_module("jittor.compat.torch.installers.cuda.api")


def _backend(count=2):
    return SimpleNamespace(
        compiler=SimpleNamespace(has_acl=True),
        get_device_count=mock.Mock(return_value=count),
        current_device=mock.Mock(return_value=1),
        set_device=mock.Mock(),
        sync_all=mock.Mock(),
        core=SimpleNamespace(
            device_memory_info=mock.Mock(return_value=(1234, 9876)),
            device_memory_used=mock.Mock(return_value=256),
            device_memory_reserved=mock.Mock(return_value=512),
            device_memory_peak=mock.Mock(return_value=384),
            device_memory_reserved_peak=mock.Mock(return_value=768),
            device_memory_allocated_total=mock.Mock(return_value=1024),
            reset_device_memory_peak=mock.Mock(),
            get_device_rng_state=mock.Mock(return_value="JITTOR_ACL_RNG_V1\n42 16\n"),
            validate_device_rng_state=mock.Mock(),
            set_device_rng_state=mock.Mock(),
            set_device_seed=mock.Mock(),
            get_device_initial_seed=mock.Mock(return_value=(1 << 63) + 7),
        ),
    )


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    return _backend()


def test_npu_discovery_uses_acl_without_cuda_visibility(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend), mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
        assert torch.npu.is_available()
        assert torch.npu.device_count() == 2
        assert torch.npu.current_device() == 1
        backend.get_device_count.assert_called()


def test_npu_missing_backend_is_unavailable_and_refuses_execution(fake_backend):
    backend = fake_backend
    backend.compiler.has_acl = False
    with mock.patch.object(_api(), "jt", backend):
        assert not torch.npu.is_available()
        assert torch.npu.device_count() == 0
        with pytest.raises(RuntimeError, match="NPU|ACL|npu"):
            torch.npu.synchronize()
        backend.sync_all.assert_not_called()


def test_npu_set_device_validates_and_reaches_backend(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend):
        torch.npu.set_device(device="npu:0")
        backend.set_device.assert_called_once_with(0)
        with pytest.raises((ValueError, RuntimeError), match="device|ordinal"):
            torch.npu.set_device("cpu")
        with pytest.raises((ValueError, RuntimeError), match="device|ordinal"):
            torch.npu.set_device("cuda:0")
        with pytest.raises(ValueError, match="ordinal|index"):
            torch.npu.set_device(5)
        assert backend.set_device.call_count == 1


def test_npu_synchronize_calls_backend_and_preserves_failure(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend):
        torch.npu.synchronize("npu:0")
        backend.sync_all.assert_called_once_with(True)
        error = RuntimeError("ACL stream failure")
        backend.sync_all.side_effect = error
        with pytest.raises(RuntimeError) as caught:
            torch.npu.synchronize()
        assert caught.value is error
        with pytest.raises((ValueError, RuntimeError), match="device|ordinal"):
            torch.npu.synchronize("cpu")


def test_npu_memory_is_per_device_and_driver_info_is_not_pool_estimate(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend):
        assert torch.npu.memory_allocated("npu:0") == 256
        assert torch.npu.memory_reserved(1) == 512
        assert torch.npu.max_memory_allocated(0) == 384
        assert torch.npu.max_memory_reserved(1) == 768
        assert torch.npu.mem_get_info(0) == (1234, 9876)
        backend.core.device_memory_used.assert_called_with(0)
        backend.core.device_memory_reserved.assert_called_with(1)
        backend.core.device_memory_info.assert_called_once_with(0)
        torch.npu.reset_peak_memory_stats(1)
        backend.core.reset_device_memory_peak.assert_called_once_with(1)


@pytest.mark.parametrize("keyword", ["device", "device_name"])
def test_npu_device_name_uses_provider_and_restores_current_device(fake_backend, keyword):
    backend = fake_backend
    dispatch = importlib.import_module("jittor._runtime.dispatch")
    query = mock.Mock(return_value="Ascend910B3")
    with mock.patch.object(_api(), "jt", backend), mock.patch.object(
            dispatch, "registered_kernel", return_value=query) as lookup:
        assert torch.npu.get_device_name(**{keyword: 0}) == "Ascend910B3"
        lookup.assert_called_once_with("runtime.device_name", "acl")
        query.assert_called_once_with()
        assert backend.set_device.call_args_list == [mock.call(0), mock.call(1)]


def test_npu_bfloat16_discovery_requires_provider_evidence(fake_backend):
    backend = fake_backend
    dispatch = importlib.import_module("jittor._runtime.dispatch")
    with mock.patch.object(_api(), "jt", backend), mock.patch.object(
            dispatch, "registered_kernel") as lookup:
        lookup.return_value = None
        with pytest.raises(RuntimeError, match="BF16 capability query"):
            torch.npu.is_bf16_supported()
        query = mock.Mock(return_value=False)
        lookup.return_value = query
        assert not torch.npu.is_bf16_supported()
        query.return_value = True
        assert torch.npu.is_bf16_supported()
        lookup.assert_called_with("runtime.supports_bfloat16", "acl")
        query.side_effect = RuntimeError("CANN capability failed")
        with pytest.raises(RuntimeError, match="CANN capability failed"):
            torch.npu.is_bf16_supported()


def test_native_driver_memory_rejects_host_and_unavailable_accelerator():
    with pytest.raises(RuntimeError, match="Invalid accelerator device ordinal"):
        jt.core.device_memory_info(-1)
    if not jt.get_device_count():
        with pytest.raises(RuntimeError, match="Invalid accelerator device ordinal"):
            jt.core.device_memory_info(0)


def test_npu_device_context_restores_previous_index_after_error(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend):
        with pytest.raises(RuntimeError, match="body failure"):
            with torch.npu.device("npu:0"):
                raise RuntimeError("body failure")
        assert backend.set_device.call_args_list == [mock.call(0), mock.call(1)]


def test_npu_rng_state_routes_to_native_acl_owner(fake_backend):
    backend = fake_backend
    owner = importlib.import_module("jittor.compat.torch.installers.core")
    with mock.patch.object(_api(), "jt", backend), mock.patch.object(
            owner, "_encode_rng_state", side_effect=lambda text: ("opaque", text)), mock.patch.object(
            owner, "_decode_rng_state", side_effect=lambda value: value[1]):
        state = torch.npu.get_rng_state("npu:0")
        assert state == ("opaque", "JITTOR_ACL_RNG_V1\n42 16\n")
        backend.core.get_device_rng_state.assert_called_once_with("acl", 0)
        torch.npu.set_rng_state(state, "npu:1")
        backend.core.set_device_rng_state.assert_called_once_with("acl", 1, state[1])
        assert torch.npu.initial_seed() == (1 << 63) + 7
        backend.core.get_device_initial_seed.assert_called_once_with("acl", 1)


def test_npu_rng_all_validates_every_state_before_mutation(fake_backend):
    backend = fake_backend
    owner = importlib.import_module("jittor.compat.torch.installers.core")
    with mock.patch.object(_api(), "jt", backend), mock.patch.object(
            owner, "_decode_rng_state", side_effect=lambda value: value):
        backend.core.validate_device_rng_state.side_effect = [None, RuntimeError("invalid ACL RNG state")]
        with pytest.raises(RuntimeError, match="invalid ACL RNG state"):
            torch.npu.set_rng_state_all(["good", "bad"])
        assert backend.core.validate_device_rng_state.call_args_list == [
            mock.call("acl", "good"), mock.call("acl", "bad")]
        backend.core.set_device_rng_state.assert_not_called()
        with pytest.raises(ValueError, match="visible device count"):
            torch.npu.set_rng_state_all(["only one"])


def test_npu_manual_seed_preserves_uint64_and_device_scope(fake_backend):
    backend = fake_backend
    with mock.patch.object(_api(), "jt", backend):
        torch.npu.manual_seed(-1)
        backend.core.set_device_seed.assert_called_once_with("acl", 1, (1 << 64) - 1)
        backend.core.set_device_seed.reset_mock()
        torch.npu.manual_seed_all((1 << 63) + 9)
        assert backend.core.set_device_seed.call_args_list == [
            mock.call("acl", 0, (1 << 63) + 9), mock.call("acl", 1, (1 << 63) + 9)]
        backend.core.set_device_seed.reset_mock()
        with pytest.raises(ValueError, match="Overflow"):
            torch.npu.manual_seed(1 << 64)
        backend.core.set_device_seed.assert_not_called()


def test_npu_random_namespace_shares_public_rng_owners():
    random = importlib.import_module("torch.npu.random")
    for name in ("get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all",
                 "manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed"):
        assert getattr(random, name) is getattr(torch.npu, name)


def test_npu_amp_namespace_uses_npu_device_and_shared_owner():
    amp = importlib.import_module("torch.npu.amp")
    mode = importlib.import_module("torch.npu.amp.autocast_mode")
    scaler_module = importlib.import_module("torch.npu.amp.grad_scaler")
    assert mode.autocast is amp.autocast
    assert scaler_module.GradScaler is amp.GradScaler
    assert issubclass(amp.autocast, torch.amp.autocast)
    scaler = amp.GradScaler(enabled=False)
    assert not scaler.is_enabled()
    if not jt.compiler.has_acl:
        with pytest.raises(RuntimeError, match="unsupported autocast device_type 'npu'"):
            amp.autocast(enabled=False, dtype=torch.bfloat16)
        return
    with amp.autocast(enabled=False, dtype=torch.bfloat16) as region:
        assert region.device == "npu"
        assert region.fast_dtype == "bfloat16"
        assert not torch.is_autocast_enabled("npu")


def test_npu_real_acl_bfloat16_autocast_forward_and_backward():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    assert torch.npu.is_available()
    assert torch.npu.device_count() == jt.get_device_count()
    assert "Ascend" in torch.npu.get_device_name()
    assert torch.npu.is_bf16_supported()
    assert torch.bfloat16 in torch.npu.get_amp_supported_dtype()
    source = np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)
    with jt.runtime.scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        before = torch.npu.memory_allocated()
        left = torch.tensor(source, device="npu", requires_grad=True)
        right = torch.tensor(source, device="npu", requires_grad=True)
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            output = left @ right
            loss = output.float().sum()
            assert loss.dtype == torch.float32
        grads = torch.autograd.grad(loss, [left, right])
        torch.npu.synchronize()
        assert str(output.dtype) == "torch.bfloat16"
        for value in [output] + list(grads):
            assert value.location() == "device"
            assert value.placement_backend == 2
            assert value.device.type == "npu"
        assert torch.npu.memory_allocated() > before
        assert torch.npu.max_memory_allocated() >= torch.npu.memory_allocated()
        free, total = torch.npu.mem_get_info()
        assert 0 < free <= total
        np.testing.assert_allclose(output.float().cpu().numpy(), source @ source)
        np.testing.assert_allclose(grads[0].cpu().numpy(), np.ones((2, 2)) @ source.T)
        np.testing.assert_allclose(grads[1].cpu().numpy(), source.T @ np.ones((2, 2)))


def test_npu_real_rng_restore_preserves_exact_next_draw_and_cpu_stream():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.runtime.scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        torch.npu.manual_seed_all((1 << 63) + 11)
        assert torch.npu.initial_seed() == (1 << 63) + 11
        state = torch.npu.get_rng_state()
        assert state.dtype == torch.uint8 and state.device.type == "cpu"
        cpu_before = jt.get_cpu_rng_state()
        first = torch.rand(64, device="npu")
        first.sync()
        assert first.location() == "device" and first.placement_backend == 2
        expected = first.cpu().numpy().copy()
        torch.npu.set_rng_state(state)
        assert jt.get_cpu_rng_state() == cpu_before
        repeated = torch.rand(64, device="npu")
        repeated.sync()
        assert repeated.location() == "device" and repeated.placement_backend == 2
        np.testing.assert_array_equal(repeated.cpu().numpy(), expected)
