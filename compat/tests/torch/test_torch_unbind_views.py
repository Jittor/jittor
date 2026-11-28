"""Torch unbind exposes storage views, including large interleaved slices."""

import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.cuda
@pytest.mark.parametrize("functional", [False, True])
def test_cuda_unbind_preserves_interleaved_views(functional):
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"), torch.no_grad():
        x = torch.zeros((1024, 1024, 2), device="cuda", dtype=torch.bfloat16)
        parts = torch.unbind(x, dim=-1) if functional else x.unbind(dim=-1)
        assert isinstance(parts, tuple)
        left, right = parts
        assert tuple(left.stride()) == (2048, 2)
        assert left.device.type == right.device.type == "cuda"
        left.fill_(3)
        x[..., 1] = 7
        np.testing.assert_array_equal(x[0, 0].float().cpu().numpy(), [3, 7])
        np.testing.assert_array_equal(right[0, :2].float().cpu().numpy(), [7, 7])


def test_unbind_axis_and_gradient():
    x = torch.ones(2, 3, device="cpu", requires_grad=True)
    first, second = torch.unbind(x, 0)
    gradient, = torch.autograd.grad(first.sum() + 2 * second.sum(), (x,))
    np.testing.assert_array_equal(gradient.numpy(), [[1, 1, 1], [2, 2, 2]])
    assert torch.unbind(torch.empty(0, 3, device="cpu")) == ()
    with pytest.raises(IndexError):
        x.unbind(2)
