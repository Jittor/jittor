"""Mean VJPs retain the requested scalar precision and placement."""

import numpy as np
import pytest

import jittor as jt
from _helpers.capability import require_accelerator
from _helpers.devices import explicit_tensor_placement


@pytest.mark.parametrize("device", ["cpu", "cuda", "cpu_ambient_cuda"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("count", [7, 71])
def test_mean_gradient_preserves_wide_reciprocal(device, dtype, count):
    if device != "cpu":
        require_accelerator("cuda")
    data = np.arange(3 * count * 5).reshape(3, count, 5).astype(dtype)
    if dtype == "complex128":
        data += 1j * np.flip(data.real, axis=1)
    backend = int(device == "cuda")
    with jt.flag_scope(use_cuda=int(device != "cpu"), auto_convert_64_to_32=0,
                       backend_fallback="error"):
        with explicit_tensor_placement(backend):
            x = jt.array(data, dtype=dtype).start_grad()
        output = x.mean([1])
        loss = output.real.sum() if dtype == "complex128" else output.sum()
        gradient = jt.grad(loss, x)
        gradient.sync()
        assert gradient.location() == ("device" if device == "cuda" else "cpu")
        assert str(gradient.dtype) == dtype
        np.testing.assert_allclose(gradient.numpy(), np.full(data.shape, 1. / count, dtype=dtype),
                                   atol=1e-15, rtol=1e-15)
