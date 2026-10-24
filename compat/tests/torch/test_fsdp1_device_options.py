"""Real world-1 FSDP1 placement/regression tests, not a full FSDP1 claim.

CPU and NPU run independently. NumPy is only the written forward/gradient/SGD
oracle; all tested forward/backward/update work executes on the selected device.
"""
import numpy as np
import pytest
import jittor as jt
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], device="cpu"))
        self.bias = torch.nn.Parameter(torch.tensor([1., -1.], device="cpu"))
        self.register_buffer("offset", torch.tensor([.5, -.5], device="cpu"))
        self.register_buffer("counter", torch.tensor([7], device="cpu", dtype=torch.int64))
        self.register_buffer("scratch", torch.zeros(2, device="cpu"), persistent=False)
        self.config = {"fixture": "fsdp1-device-options"}

    def forward(self, value):
        return value @ self.weight + self.bias + self.offset + self.scratch


def _placement(value, device):
    value.sync()
    assert value.location() == ("cpu" if device == "cpu" else "device")
    if device != "cpu":
        assert value.get_device() == jt.core.current_device()


def _array(value):
    return np.array(value.detach().cpu().numpy(), copy=True)


def _exercise(device):
    if device == "npu" and not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("requires a real NPU")
    with forbid_backend_fallbacks():
        model = _Toy()
        for value in tuple(model.parameters()) + tuple(model.buffers()):
            _placement(value, "cpu")
        parameters = dict(model.named_parameters())
        buffer_dtypes = {name: value.dtype for name, value in model.named_buffers()}
        mesh = init_device_mesh(device, (1,), mesh_dim_names=("fsdp",))
        original_mode = jt.runtime.use_cuda
        target = "cpu" if device == "cpu" else "npu:%d" % jt.core.current_device()
        wrapped = FSDP(
            model, device_id=target, device_mesh=mesh, use_orig_params=True,
            mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                           buffer_dtype=torch.float32),
            sync_module_states=True, auto_wrap_policy=None,
        )
        assert jt.runtime.use_cuda == original_mode
        assert wrapped.config is model.config  # Retain wrapper attribute forwarding.
        assert wrapped._fsdp_state.mesh is mesh
        assert wrapped._fsdp_state.fsdp1_execution == "unsharded_world_size_1"
        assert wrapped._fsdp_state.true_fsdp_initialized is False
        assert set(dict(wrapped.named_parameters())) == set(parameters)
        for name, value in wrapped.named_parameters():
            assert value is parameters[name], "device migration replaced a Parameter"
            _placement(value, device)
        assert set(dict(wrapped.named_buffers())) == {"offset", "counter", "scratch"}
        for name, value in wrapped.named_buffers():
            _placement(value, device)
            assert value.dtype == buffer_dtypes[name]
        inputs = torch.tensor([[1., 2.], [3., 4.]], device=target, requires_grad=True)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.1)
        output = wrapped(inputs)
        _placement(output, device)
        output.sum().backward()
        for value in list(wrapped.parameters()) + [inputs]:
            assert value.grad is not None
            _placement(value.grad, device)
        np.testing.assert_allclose(_array(output), [[8.5, 8.5], [16.5, 20.5]], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_array(model.weight.grad), [[4., 4.], [6., 6.]], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_array(model.bias.grad), [2., 2.], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_array(inputs.grad), [[3., 7.], [3., 7.]], rtol=1e-6, atol=1e-6)
        optimizer.step()
        for value in wrapped.parameters():
            _placement(value, device)
        np.testing.assert_allclose(_array(model.weight), [[.6, 1.6], [2.4, 3.4]], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_array(model.bias), [.8, -1.2], rtol=1e-6, atol=1e-6)
        assert _array(model.counter).tolist() == [7]
        # Parameter migration must not turn ordinary factory calls into NPU calls.
        _placement(torch.empty(1), "cpu")


def test_cpu_original_parameters_buffers_and_update():
    with jt.runtime.scope(use_cuda=0):
        _exercise("cpu")


def test_npu_cpu_module_moves_all_parameters_and_buffers():
    _exercise("npu")


def test_unsupported_precision_rejected_before_fsdp_mutation():
    with jt.runtime.scope(use_cuda=0):
        model = _Toy()
        original = tuple(model.parameters())
        with pytest.raises(NotImplementedError, match="param_dtype"):
            FSDP(model, use_orig_params=True, device_id="cpu",
                 mixed_precision=MixedPrecision(param_dtype=torch.float16))
        assert "_fsdp_state" not in vars(model)
        assert all(a is b for a, b in zip(original, model.parameters()))


def test_no_options_wrapper_keeps_historical_cpu_forward():
    with jt.runtime.scope(use_cuda=0):
        model = _Toy()
        wrapped = FSDP(model)
        assert wrapped.config is model.config
        np.testing.assert_allclose(_array(wrapped(torch.tensor([[1., 2.]]))), [[8.5, 8.5]])


def _exercise_nested_precision_guard(device):
    if device == "npu" and not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("requires a real NPU")

    class NestedEcho(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.], device="cpu"))
            self.forward_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return value

    with forbid_backend_fallbacks():
        target = "cpu" if device == "cpu" else "npu:%d" % jt.core.current_device()
        model = NestedEcho()
        wrapped = FSDP(model, device_id=target, use_orig_params=True,
                       mixed_precision=MixedPrecision(param_dtype=torch.float32,
                                                      reduce_dtype=torch.float32,
                                                      buffer_dtype=torch.float32))
        _placement(model.weight, device)
        offsets = torch.tensor([0, 2, 3], device=target, dtype=torch.int64)
        for dtype in (torch.float16, torch.float32, torch.int64):
            requires_grad = dtype != torch.int64
            payload = torch.tensor([1, 2, 3], device=target, dtype=dtype,
                                   requires_grad=requires_grad)
            nested = torch.nested.nested_tensor_from_jagged(payload, offsets, jagged_dim=1)
            assert nested.is_nested and not isinstance(nested, jt.Var)
            _placement(nested.values(), device)
            calls_before = model.forward_calls
            if dtype == torch.float16:
                with pytest.raises(NotImplementedError, match="input casting"):
                    wrapped(value=nested)
                assert model.forward_calls == calls_before
            else:
                assert wrapped(value=nested) is nested
                assert model.forward_calls == calls_before + 1
            # The guard must neither convert the nested payload nor detach it.
            assert nested.values() is payload and payload.dtype == dtype
            assert payload.requires_grad == requires_grad
            _placement(payload, device)
        assert model.forward_calls == 2


def test_cpu_nested_precision_guard_rejects_cast_before_forward():
    with jt.runtime.scope(use_cuda=0):
        _exercise_nested_precision_guard("cpu")


def test_npu_nested_precision_guard_rejects_cast_before_forward():
    _exercise_nested_precision_guard("npu")
