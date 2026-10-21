"""Mixed-dtype standard RMSNorm must retain its public forward contract."""
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability
from jittor.compat.torch.installers.nn import module_methods


DEVICES = ["cpu"] + (["cuda"] if capability.check_accelerator("cuda", backend=jt).enabled else [])


class MixedRMSNorm(torch.nn.Module):
    def __init__(self, weight, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32, device=device))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        input_dtype = value.dtype
        value = value.to(torch.float32)
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * value.to(input_dtype)


def array(value):
    return value.detach().float().cpu().numpy().copy()


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_mixed_dtype_declines_all_standard_rmsnorm_fastpaths(input_dtype):
    module = MixedRMSNorm(np.ones(8, dtype=np.float32), "cpu")
    value = torch.ones((2, 8), dtype=input_dtype, device="cpu")
    def unexpected(*args, **kwargs):
        raise AssertionError("mixed dtype reached a native RMSNorm kernel")
    hooks = SimpleNamespace(rms_norm_training_cuda=unexpected, rms_norm_cuda=unexpected)
    with patch.object(module_methods, "_acl_bfloat16_rms_norm", side_effect=AssertionError("mixed dtype reached ACL")), \
         patch.object(module_methods, "_backend_hooks", hooks):
        assert module_methods._standard_rms_norm(module, (value,), {}) is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_mixed_dtype_eval_and_training_match_original_forward(device, input_dtype):
    source = np.arange(1, 17, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    weight = np.arange(4, 12, dtype=np.float32) * np.float32(.125)
    coefficient = np.arange(16, 0, -1, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    observed = {}
    with jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with forbid_backend_fallbacks():
            for route in ("module", "forward"):
                module = MixedRMSNorm(weight, device)
                value = torch.tensor(source, dtype=input_dtype, device=device, requires_grad=True)
                module.eval()
                with torch.no_grad():
                    output = module(value) if route == "module" else module.forward(value)
                assert output.dtype == torch.float32
                assert output.device.type == device
                assert jt.core.dispatch_context([output])[0] == device
                module.train()
                trained = module(value) if route == "module" else module.forward(value)
                loss = (trained * torch.tensor(coefficient, device=device)).sum()
                input_grad, weight_grad = torch.autograd.grad(loss, (value, module.weight))
                assert trained.dtype == torch.float32
                assert input_grad.dtype == input_dtype
                assert weight_grad.dtype == torch.float32
                observed[route] = [array(tensor) for tensor in (output, trained, input_grad, weight_grad)]
            for dispatched, original in zip(observed["module"], observed["forward"]):
                np.testing.assert_array_equal(dispatched, original)
                assert np.isfinite(dispatched).all()


class SourceOrderRMSNorm(MixedRMSNorm):
    def __init__(self, weight, dtype, device):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=dtype, device=device))
        self.variance_epsilon = 1e-6
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        return super().forward(value)


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_acl_standard_hf_norm_keeps_original_forward_and_full_gradients(device, dtype):
    if device == "npu":
        capability.require_accelerator("acl")
    source = np.arange(1, 17, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    weight = np.arange(4, 12, dtype=np.float32) * np.float32(.125)
    coefficient = np.arange(16, 0, -1, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    settings = {"use_cuda": int(device == "npu"), "backend_fallback": "error"}
    if getattr(jt.compiler, "has_acl", 0):
        settings["use_acl"] = int(device == "npu")
    observed = {}
    from jittor._runtime.fallback import forbid_backend_fallbacks
    with jt.runtime.scope(**settings), forbid_backend_fallbacks():
        for route in ("module", "forward"):
            module = SourceOrderRMSNorm(weight, dtype, device)
            value = torch.tensor(source, dtype=dtype, device=device, requires_grad=True)
            module.eval()
            with torch.no_grad():
                output = module(value) if route == "module" else module.forward(value)
            assert module.forward_calls == 1
            module.train()
            trained = module(value) if route == "module" else module.forward(value)
            assert module.forward_calls == 2
            loss = (trained * torch.tensor(coefficient, dtype=dtype, device=device)).float().sum()
            input_grad, weight_grad = torch.autograd.grad(loss, (value, module.weight))
            for tensor in (output, trained, input_grad, weight_grad):
                assert tensor.dtype == dtype
                assert tensor.device.type == device
                expected = ("acl", 0) if device == "npu" else ("cpu", -1)
                assert tuple(jt.core.dispatch_context([tensor])) == expected
            observed[route] = [array(tensor) for tensor in (output, trained, input_grad, weight_grad)]
        for dispatched, original in zip(observed["module"], observed["forward"]):
            np.testing.assert_array_equal(dispatched, original)
            assert np.isfinite(dispatched).all()


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_acl_standard_rmsnorm_bf16_autocast_preserves_hf_fp32_math_and_gradients(input_dtype):
    capability.require_accelerator("acl")
    source = np.arange(1, 17, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    weight = np.arange(4, 12, dtype=np.float32) * np.float32(.125)
    coefficient = np.arange(16, 0, -1, dtype=np.float32).reshape(2, 8) * np.float32(.125)
    from jittor._runtime.fallback import forbid_backend_fallbacks
    with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
        reference = SourceOrderRMSNorm(weight, torch.float32, "npu")
        reference_value = torch.tensor(source, dtype=input_dtype, device="npu", requires_grad=True)
        with jt.flag_scope(amp_reg=0):
            expected = reference.forward(reference_value)
            expected_loss = (expected * torch.tensor(coefficient, device="npu")).sum()
            expected_grads = torch.autograd.grad(expected_loss, (reference_value, reference.weight))

        observed_module = SourceOrderRMSNorm(weight, torch.float32, "npu")
        observed_value = torch.tensor(source, dtype=input_dtype, device="npu", requires_grad=True)
        with torch.autocast("npu", dtype=torch.bfloat16):
            observed = observed_module(observed_value)
        observed_loss = (observed * torch.tensor(coefficient, device="npu")).sum()
        observed_grads = torch.autograd.grad(observed_loss, (observed_value, observed_module.weight))

        assert observed_module.forward_calls == 1
        assert observed.dtype == torch.float32
        assert observed_value.dtype == expected_grads[0].dtype == input_dtype
        assert observed_module.weight.dtype == expected_grads[1].dtype == torch.float32
        np.testing.assert_array_equal(array(observed), array(expected))
        for actual, expected_grad in zip(observed_grads, expected_grads):
            np.testing.assert_array_equal(array(actual), array(expected_grad))
