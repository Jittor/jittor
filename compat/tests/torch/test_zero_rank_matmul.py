"""Batched multiplication through an empty contraction still has output rows."""

import torch

from _helpers.device_types import instantiate_device_type_tests


class TestZeroRankMatmul:
    def test_matmul_after_zero_width_projection(self, device):
        values = torch.ones((4, 1, 4), device=device, requires_grad=True)
        down = torch.empty((0, 4), device=device, requires_grad=True)
        up = torch.empty((4, 0), device=device, requires_grad=True)
        compact = values @ down.T
        assert compact.shape == (4, 1, 0)

        output = compact @ up.T
        assert output.device.type == device and output.shape == (4, 1, 4)
        assert torch.equal(output, torch.zeros_like(values))
        output.sum().backward()
        assert torch.equal(values.grad, torch.zeros_like(values))
        assert down.grad.shape == down.shape and up.grad.shape == up.shape
        assert down.grad.device.type == up.grad.device.type == device


instantiate_device_type_tests(TestZeroRankMatmul, globals(), only_for=("cpu", "cuda"))
