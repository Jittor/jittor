"""Dense gradient packing preserves the inputs' autograd contract."""
import numpy as np
import torch
from torch._utils import _flatten_dense_tensors

from _helpers.device_types import instantiate_device_type_tests


class TestFlattenDenseTensors:
    def test_gradient_bucket_is_not_a_trainable_leaf(self, device):
        for dtype in (torch.float32, torch.bfloat16):
            a = torch.tensor([1., 2.], dtype=dtype, device=device)
            b = torch.tensor([[3., 4.]], dtype=dtype, device=device)
            flat = _flatten_dense_tensors([a, b])
            assert not flat.requires_grad
            assert flat.dtype == dtype and flat.device.type == device
            flat.copy_(torch.tensor([5., 6., 7., 8.], dtype=dtype, device=device))
            np.testing.assert_array_equal(flat.float().cpu().numpy(), [5., 6., 7., 8.])
            np.testing.assert_array_equal(a.float().cpu().numpy(), [1., 2.])
            np.testing.assert_array_equal(b.float().cpu().numpy(), [[3., 4.]])

    def test_explicit_cuda_bucket_uses_native_concat(self, device):
        if device != "cuda":
            return
        from jittor._runtime import dispatch

        values = [
            torch.tensor([1., 2.], device=device),
            torch.tensor([3., 4.], device=device),
        ]
        assert all(value.placement_backend == 1 for value in values)
        original = dispatch.select_kernel._native_concat
        calls = []

        def recording(tensors, dim):
            calls.append((len(tensors), dim))
            return original(tensors, dim)

        dispatch.select_kernel._native_concat = recording
        try:
            flat = _flatten_dense_tensors(values)
        finally:
            dispatch.select_kernel._native_concat = original
        np.testing.assert_array_equal(
            flat.cpu().numpy(), [1., 2., 3., 4.])
        assert calls == [(2, 0)]

    def test_cat_requires_grad_comes_only_from_inputs(self, device):
        for count in (2, 65):
            values = [torch.tensor([float(i)], device=device) for i in range(count)]
            result = torch.cat(values)
            assert not result.requires_grad
            np.testing.assert_array_equal(result.cpu().numpy(), np.arange(count))
            for index in (0, count - 1):
                leaves = [value.detach().clone() for value in values]
                leaves[index].requires_grad_(True)
                result = torch.cat(leaves)
                assert result.requires_grad and not result.is_leaf
                result.sum().backward()
                np.testing.assert_array_equal(leaves[index].grad.cpu().numpy(), [1.])
                assert all(value.grad is None for i, value in enumerate(leaves) if i != index)
                with torch.no_grad():
                    assert not torch.cat(leaves).requires_grad

    def test_trainable_inputs_keep_their_gradient_edges(self, device):
        a = torch.tensor([1., 2.], device=device, requires_grad=True)
        b = torch.tensor([[3., 4.]], device=device, requires_grad=True)
        flat = _flatten_dense_tensors([a, b])
        assert flat.requires_grad and not flat.is_leaf
        flat.square().sum().backward()
        np.testing.assert_array_equal(a.grad.cpu().numpy(), [2., 4.])
        np.testing.assert_array_equal(b.grad.cpu().numpy(), [[6., 8.]])


instantiate_device_type_tests(TestFlattenDenseTensors, globals(), only_for=("cpu", "cuda"))
