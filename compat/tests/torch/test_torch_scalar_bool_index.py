"""A scalar boolean index adds a leading selection dimension."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_scalar_boolean_index_values_shape_and_gradient(device, mask, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    data = np.arange(1, 1 + int(np.prod(shape)), dtype=np.float32).reshape(shape)
    x = torch.tensor(data, device=device, requires_grad=True)
    selected = x[torch.tensor(mask, device=device)]
    expected = np.expand_dims(data, 0) if mask else np.empty((0, *shape), dtype=np.float32)
    assert selected.device.type == device
    assert tuple(selected.shape) == expected.shape
    np.testing.assert_array_equal(selected.detach().cpu().numpy(), expected)
    selected.sum().backward()
    np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), np.ones_like(data) if mask else np.zeros_like(data))
