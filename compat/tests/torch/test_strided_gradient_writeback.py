"""Pending reshape/transpose writes preserve logical values and derivatives."""
import numpy as np
import jittor as jt
import torch
import pytest

from _helpers.device_types import instantiate_device_type_tests


class TestStridedGradientWriteback:
    def test_pending_reshape_transpose_copy(self, device):
        for broadcast in (False, True):
            with jt.flag_scope(transpose_storage_view=1):
                base = torch.empty((2, 2), device=device)
                target = base.reshape(-1).as_strided((2, 2), (1, 2))
                source = (torch.tensor(2., device=device).expand(2, 2) if broadcast else
                          torch.tensor([[1., 2.], [3., 4.]], device=device))
                # No pointer query or sync: the destination is still pending.
                target.copy_(source)
                expected = np.full((2, 2), 2.) if broadcast else np.array([[1., 2.], [3., 4.]])
                np.testing.assert_array_equal(target.cpu().numpy(), expected)
                np.testing.assert_array_equal(base.cpu().numpy(), expected.T)
                assert target.stride() == (1, 2)

    def test_constant_gradient_initialization_and_accumulation(self, device):
        parameter = torch.tensor([[1., 2.], [3., 4.]], device=device).t().detach().requires_grad_(True)
        (parameter * 2.).sum().backward()
        published = parameter.grad
        assert published.stride() == parameter.stride() == (1, 2)
        np.testing.assert_array_equal(published.cpu().numpy(), np.full((2, 2), 2.))
        (parameter * 3.).sum().backward()
        assert parameter.grad is published
        np.testing.assert_array_equal(published.cpu().numpy(), np.full((2, 2), 5.))
        assert not published.requires_grad

    def test_pending_copy_preserves_backed_base_and_rejects_overlap(self, device):
        root = torch.tensor(np.arange(16.).reshape(4, 4), device=device, requires_grad=True)
        pointer = root.data_ptr()
        destination, source = root[:, ::2], root[:, 1::2]
        with torch.no_grad():
            destination.copy_(source)
        assert root.data_ptr() == pointer
        expected = np.arange(16.).reshape(4, 4)
        expected[:, ::2] = expected[:, 1::2].copy()
        np.testing.assert_array_equal(root.detach().cpu().numpy(), expected)
        flat = root.view(-1)
        with torch.no_grad(), pytest.raises(RuntimeError, match="overlap|single memory"):
            flat[1:].copy_(flat[:-1])
        np.testing.assert_array_equal(root.detach().cpu().numpy(), expected)

    def test_writeback_copy_keeps_second_derivatives(self, device):
        source = torch.tensor([[1., 2.], [3., 4.]], device=device, requires_grad=True)
        with jt.flag_scope(transpose_storage_view=1):
            base = torch.empty((2, 2), device=device)
            target = base.reshape(-1).as_strided((2, 2), (1, 2))
            target.copy_(source)
            first = torch.autograd.grad(target.square().sum(), source, create_graph=True)[0]
            np.testing.assert_array_equal(first.detach().cpu().numpy(), [[2., 4.], [6., 8.]])
            second = torch.autograd.grad(first.sum(), source)[0]
            np.testing.assert_array_equal(second.cpu().numpy(), np.full((2, 2), 2.))


instantiate_device_type_tests(TestStridedGradientWriteback, globals(), only_for=("cpu", "cuda"))
