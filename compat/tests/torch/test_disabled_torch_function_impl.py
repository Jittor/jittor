"""The disabled eager hook still invokes the supplied function."""

import torch


def test_disabled_torch_function_impl_calls_with_args_and_kwargs():
    calls = []

    def function(value, *, scale):
        calls.append((value, scale))
        return value * scale

    result = torch._C._disabled_torch_function_impl(
        function, (torch.Tensor,), (3,), {"scale": 4})

    assert result == 12
    assert calls == [(3, 4)]


def test_disabled_torch_function_impl_accepts_missing_or_none_kwargs():
    def function(value=5):
        return value + 1

    assert torch._C._disabled_torch_function_impl(function, ()) == 6
    assert torch._C._disabled_torch_function_impl(function, (), (2,), None) == 3
