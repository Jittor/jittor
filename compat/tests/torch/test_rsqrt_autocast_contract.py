"""Public rsqrt uses the CUDA autocast FP32 policy and preserves input gradients."""
import numpy as np
import pytest
import torch
from _release_test_helpers import public_device_scope, on_device


@pytest.mark.parametrize("name", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_rsqrt_dtype_gradient_and_inplace(name, autocast_enabled):
    with public_device_scope() as device:
        dtype = getattr(torch, name)
        x = torch.tensor([.25, 1., 4., 16.], dtype=dtype, device=device, requires_grad=True)
        expected_dtype = (torch.float32 if device == "cuda" and autocast_enabled
                          and name in ("float16", "bfloat16") else dtype)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast_enabled):
            public = torch.rsqrt(input=x)
            method = x.rsqrt()
            assert public.dtype == method.dtype == expected_dtype
            on_device(public, device)
            np.testing.assert_array_equal(public.detach().float().cpu().numpy(), [.5**-1, 1., .5, .25])
            np.testing.assert_array_equal(method.detach().float().cpu().numpy(), public.detach().float().cpu().numpy())
            with torch.no_grad():
                inplace = x.detach().clone()
                assert inplace.rsqrt_() is inplace
                assert inplace.dtype == dtype
                np.testing.assert_array_equal(inplace.float().cpu().numpy(), [2., 1., .5, .25])
        gradient, = torch.autograd.grad(public.sum(), (x,))
        assert gradient.dtype == dtype
        on_device(gradient, device)
        np.testing.assert_array_equal(gradient.float().cpu().numpy(), [-4., -.5, -.0625, -.0078125])
