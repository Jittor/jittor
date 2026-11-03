"""An indexed write to a stopped tensor records the differentiable RHS."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("index_kind", ["slice", "mask", "paired"])
def test_indexed_write_promotes_stopped_target_and_keeps_rhs_gradient(device, index_kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device, requires_grad=True)
    if index_kind == "paired":
        target = torch.zeros((2, 3, 3), device=device)
        rows = torch.tensor([0, 0, 1], device=device)
        cols = torch.tensor([1, 2, 2], device=device)
        target[:, rows, cols] = source
    elif index_kind == "mask":
        target = torch.zeros((3, 3), device=device)
        mask = torch.tensor([True, False, True], device=device)
        target[mask] = source
    else:
        target = torch.zeros((3, 3), device=device)
        target[:2] = source
    assert target.requires_grad and not target.is_leaf
    target.sum().backward()
    assert source.grad is not None
    np.testing.assert_array_equal(source.grad.cpu().numpy(), np.ones((2, 3)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indexed_leaf_write_is_still_rejected(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    target = torch.ones(3, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf"):
        target[:] = torch.ones(3, device=device)
