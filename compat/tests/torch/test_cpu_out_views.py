"""CPU out= views must keep their placement in an accelerator process."""
from contextlib import nullcontext

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("device_context", [False, True])
def test_add_cpu_out_view_under_cuda(count, device_context):
    # Keep a real CUDA allocation alive so this exercises a GPU process.
    gpu = torch.ones((4,), device="cuda")
    lhs = torch.tensor([2, 5, 9, 13], dtype=torch.int32, device="cpu")
    rhs = torch.from_numpy(np.arange(count, dtype=np.int32) + 7)
    base = torch.full((4,), -1, dtype=torch.int32, device="cpu")
    view = base[:count]
    with torch.device("cuda") if device_context else nullcontext():
        result = torch.add(lhs[:count], rhs, out=view)
    assert result is view
    assert result.device.type == "cpu" and base.device.type == "cpu"
    expected = np.array([-1] * 4, dtype=np.int32)
    expected[:count] = np.array([2, 5, 9, 13])[:count] + np.arange(count) + 7
    np.testing.assert_array_equal(base.cpu().numpy(), expected)
    np.testing.assert_array_equal(gpu.cpu().numpy(), 1)


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("retained_view", [False, True])
@pytest.mark.parametrize("method", ["fill_", "zero_"])
def test_inplace_fill_keeps_target_device_with_opposite_default(target, retained_view, method):
    base = torch.ones((4,), device=target)
    tensor = base[::2] if retained_view else base
    previous = str(torch.get_default_device())
    torch.set_default_device("cpu" if target == "cuda" else "cuda")
    try:
        result = tensor.fill_(7) if method == "fill_" else tensor.zero_()
        assert result is tensor
        assert result.device.type == target
        assert base.device.type == target
        expected = np.ones(4, dtype=np.float32)
        expected[::2] = 7 if method == "fill_" else 0
        if not retained_view:
            expected[:] = 7 if method == "fill_" else 0
        np.testing.assert_array_equal(base.cpu().numpy(), expected)
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("scalars", ["left", "right", "both"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("inside_module", [False, True])
def test_where_scalar_literals_follow_tensor_device_with_opposite_default(target, scalars, count, inside_module):
    class Selection(torch.nn.Module):
        def forward(self, condition, left, right):
            return torch.where(condition, left, right)
    condition = torch.tensor([True, False][:count], device=target)
    value = torch.tensor([2., 3.][:count], device=target)
    previous = str(torch.get_default_device())
    torch.set_default_device("cpu" if target == "cuda" else "cuda")
    try:
        left = 7. if scalars in ("left", "both") else value
        right = 9. if scalars == "both" else (7. if scalars == "right" else value)
        result = Selection()(condition, left, right) if inside_module else torch.where(condition, left, right)
        assert result.device.type == target
        expected = {"left": [7., 3.], "right": [2., 7.], "both": [7., 9.]}[scalars][:count]
        np.testing.assert_array_equal(result.cpu().numpy(), expected)
    finally:
        torch.set_default_device(previous)
