"""Kaiming delegates its bounded draw to the explicit generator owner."""

import math

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_kaiming_explicit_generator_advances_only_its_stream(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device=device).manual_seed(71)
    before = generator.get_state().cpu().numpy().copy()
    global_before = torch.get_rng_state().cpu().numpy().copy()
    value = torch.empty((9, 7), device=device, dtype=dtype, requires_grad=True)
    alias = value.detach()
    pointer = value.data_ptr()
    result = torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
    assert result is value and value.data_ptr() == pointer and value.requires_grad
    actual = value.detach().float().cpu().numpy().copy()
    np.testing.assert_array_equal(alias.float().cpu().numpy(), actual)
    np.testing.assert_array_equal(torch.get_rng_state().cpu().numpy(), global_before)
    after = generator.get_state()
    assert not np.array_equal(after.cpu().numpy(), before)
    generator.set_state(torch.tensor(before, dtype=torch.uint8, device="cpu"))
    expected = torch.empty_like(value)
    bound = 1 / math.sqrt(7)
    torch.nn.init.uniform_(expected, -bound, bound, generator=generator)
    np.testing.assert_array_equal(actual, expected.float().cpu().numpy())
    np.testing.assert_array_equal(generator.get_state().cpu().numpy(), after.cpu().numpy())
    (value.float() * 2).sum().backward()
    assert value.grad is not None and value.grad.dtype == dtype
    np.testing.assert_array_equal(value.grad.float().cpu().numpy(), np.full((9, 7), 2.))
