"""The native Tensor subclass factory used by third-party Parameter types."""

import numpy as np
import pytest

import jittor as jt
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_make_subclass_shares_data_but_starts_a_new_cuda_gradient_leaf():
    calls = []

    class WrappedParameter(torch.nn.Parameter):
        def __new__(cls, data, requires_grad=True):
            calls.append("new")
            return torch.Tensor._make_subclass(cls, data, requires_grad)

    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        donor = torch.tensor([1., 2.], device="cuda", requires_grad=True)
        produced = donor * 2.
        with torch.no_grad():
            parameter = WrappedParameter(produced, requires_grad=True)
        assert calls == ["new"]
        assert type(parameter) is WrappedParameter
        assert parameter.device.type == "cuda"
        assert parameter.is_leaf and parameter.requires_grad
        assert parameter.data_ptr() == produced.data_ptr()
        (parameter * 3.).sum().backward()
        np.testing.assert_allclose(parameter.grad.numpy(), [3., 3.])
        assert donor.grad is None
        frozen = torch.Tensor._make_subclass(WrappedParameter, produced, False)
        assert type(frozen) is WrappedParameter
        assert frozen.is_leaf and not frozen.requires_grad
        assert frozen.data_ptr() == produced.data_ptr()
        assert calls == ["new"]
        assert jt.core.device_pool_stats(0)[2] > 0
    assert jt.core.backend_fallback_count() == before


def test_parameter_subclass_can_accept_constructor_metadata():
    class WithMetadata(torch.nn.Parameter):
        def __new__(cls, data, *, tag):
            value = torch.Tensor._make_subclass(cls, data, False)
            value.tag = tag
            return value

    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        value = WithMetadata(torch.tensor([2.], device="cuda"), tag="quantized")
        assert type(value) is WithMetadata
        assert value.tag == "quantized"
        assert value.is_leaf and not value.requires_grad
        assert value.item() == 2.
        with pytest.raises(TypeError, match="unexpected keyword"):
            torch.nn.Parameter(torch.tensor([1.], device="cuda"), tag="no")
    assert jt.core.backend_fallback_count() == before


def test_module_to_invokes_parameter_subclass_conversion():
    class Convertible(torch.nn.Parameter):
        def __new__(cls, data):
            value = torch.Tensor._make_subclass(cls, data, True)
            value.conversions = 0
            return value

        def to(self, *args, **kwargs):
            self.conversions += 1
            return self

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Convertible(torch.tensor([2.], device="cuda"))

    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        module = Holder()
        parameter = module.weight
        assert module.to("cuda") is module
        assert module.weight is parameter
        assert parameter.conversions == 1
        assert parameter.device.type == "cuda"

        plain = torch.nn.Module()
        plain.weight = torch.nn.Parameter(torch.ones(2, device="cuda"))
        plain.register_buffer("integer", torch.ones(2, dtype=torch.int64, device="cuda"))
        plain.to(dtype=torch.float64)
        assert plain.weight.dtype is torch.float64
        assert plain.integer.dtype is torch.int64
    assert jt.core.backend_fallback_count() == before


def test_make_subclass_rejects_non_tensor_inputs():
    with pytest.raises(TypeError, match="subclass"):
        torch.Tensor._make_subclass(str, torch.tensor([1.]))
    with pytest.raises(TypeError, match="Tensor data"):
        torch.Tensor._make_subclass(torch.nn.Parameter, [1.])
