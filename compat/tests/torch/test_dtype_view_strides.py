"""Dtype views change element units while preserving the original byte storage."""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestDtypeViewStrides:
    def test_widen_strided_bytes_with_offset_and_write_through(self, device):
        original = np.arange(104, dtype=np.uint8)
        base = torch.tensor(original, device=device)
        source = base[4:100].reshape(3, 4, 8)[:2, :3, :]
        viewed = source.view(torch.int32)
        expected = np.ndarray((2, 3, 2), np.int32, original, 4, (32, 8, 4))
        assert tuple(viewed.shape) == (2, 3, 2)
        assert viewed.stride() == (8, 2, 1)
        assert viewed.storage_offset() == 1
        assert viewed.data_ptr() == source.data_ptr()
        np.testing.assert_array_equal(viewed.cpu().numpy(), expected)
        viewed[1, 2, 0] = 0
        original[52:56] = 0
        np.testing.assert_array_equal(base.cpu().numpy(), original)
        # The original source sees the dtype view's write, too.
        np.testing.assert_array_equal(source[1, 2, :4].cpu().numpy(), np.zeros(4, np.uint8))

    def test_narrow_strided_float_keeps_byte_strides(self, device):
        original = np.arange(24, dtype=np.float32)
        base = torch.tensor(original, device=device)
        source = base[1:13].reshape(2, 6)[:, :2]
        viewed = source.view(torch.uint8)
        expected = np.ndarray((2, 8), np.uint8, original, 4, (24, 1))
        assert tuple(viewed.shape) == (2, 8)
        assert viewed.stride() == (24, 1)
        assert viewed.storage_offset() == 4
        assert viewed.data_ptr() == source.data_ptr()
        np.testing.assert_array_equal(viewed.cpu().numpy(), expected)
        viewed[1, :4] = 0
        original[7] = 0
        np.testing.assert_array_equal(base.cpu().numpy(), original)

    def test_equal_itemsize_preserves_transposed_and_same_dtype_views(self, device):
        original = np.arange(12, dtype=np.float32).reshape(3, 4)
        base = torch.tensor(original, device=device)
        source = base.transpose(0, 1)
        for dtype in (torch.float32, torch.int32):
            viewed = source.view(dtype)
            assert viewed.stride() == (1, 4)
            assert tuple(viewed.shape) == (4, 3)
            assert viewed.data_ptr() == source.data_ptr()
            expected = original.T if dtype == torch.float32 else original.T.view(np.int32)
            np.testing.assert_array_equal(viewed.cpu().numpy(), expected)
        source.view(torch.int32)[1, 2] = 0
        original[2, 1] = 0
        np.testing.assert_array_equal(base.cpu().numpy(), original)

    def test_byte_view_of_layer_head_token_cache(self, device):
        # vLLM's per-layer allocation preserves the shared raw byte storage.
        base = torch.arange(256, dtype=torch.int32, device=device).view(torch.uint8)
        layer = base[32:544].reshape(2, 4, 4, 16).permute(0, 2, 1, 3)[:, :2]
        cache = layer.view(torch.float32)
        assert tuple(cache.shape) == (2, 2, 4, 4)
        assert cache.stride() == (64, 4, 16, 1)
        assert cache.storage_offset() == 8
        key, value = cache.transpose(1, 2).split(2, dim=-1)
        assert key.stride() == value.stride() == (64, 16, 4, 1)
        assert key.data_ptr() == cache.data_ptr()
        assert value.data_ptr() - key.data_ptr() == 8
        # Exercise the dtype view's tracked write owner. The split K/V pointers
        # above are what the external attention kernel reads and writes.
        cache[1, 1, 2, 2] = 0
        # Byte offset: 32 + block*256 + head*16 + token*64 + value*8.
        expected = np.arange(256, dtype=np.int32).view(np.uint8)
        expected[440:444] = 0
        np.testing.assert_array_equal(base.cpu().numpy(), expected)

    def test_invalid_reinterpretations_fail_instead_of_copying(self, device):
        base = torch.arange(64, dtype=torch.uint8, device=device)
        for source, dtype in (
            (base[:18].reshape(2, 9)[:, :8], torch.int32),  # outer stride not divisible
            (base[1:33].reshape(2, 16)[:, :8], torch.int32),  # offset not divisible
            (base[:32].reshape(2, 16)[:, :8:2], torch.int16),  # last stride not one
            (base[:32].reshape(2, 16)[:, :7], torch.int32),  # last size not divisible
        ):
            with pytest.raises(RuntimeError):
                source.view(dtype)
        # Contiguity alone ignores singleton-axis strides, but dtype-view must
        # validate the actual last stride when changing the element size.
        singleton = torch.arange(8, dtype=torch.float32, device=device).reshape(1, 8).t()[:2]
        assert singleton.is_contiguous()
        with pytest.raises(RuntimeError):
            singleton.view(torch.uint8)

    def test_pending_dtype_view_write_reaches_original_storage(self, device):
        base = torch.empty(64, dtype=torch.uint8, device=device)
        source = base[4:36].reshape(2, 16)[:, :8]
        viewed = source.view(torch.int32)
        # Do not materialize or query data_ptr before the mutation.
        viewed.fill_(0)
        np.testing.assert_array_equal(source.cpu().numpy(), np.zeros((2, 8), np.uint8))
        np.testing.assert_array_equal(base[4:12].cpu().numpy(), np.zeros(8, np.uint8))
        np.testing.assert_array_equal(base[20:28].cpu().numpy(), np.zeros(8, np.uint8))

    def test_noncontiguous_arbitrary_shape_is_not_silently_reinterpreted(self, device):
        source = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).t()
        with pytest.raises(RuntimeError, match="non-contiguous|contiguous storage"):
            jt.reinterpret_view(source, (12,), "int32")

    def test_scalar_same_itemsize_and_zero_size_metadata(self, device):
        scalar = torch.tensor(1., dtype=torch.float32, device=device)
        assert scalar.view(torch.int32).item() == np.array(1., np.float32).view(np.int32).item()
        with pytest.raises(RuntimeError):
            scalar.view(torch.uint8)
        empty = torch.empty((0, 2, 8), dtype=torch.uint8, device=device)
        viewed = empty.view(torch.int32)
        assert tuple(viewed.shape) == (0, 2, 2)
        assert viewed.stride() == (4, 2, 1)

    def test_dtype_view_is_nondifferentiable_even_when_source_becomes_trainable(self, device):
        for dtype in (torch.float32, torch.int32, torch.uint8):
            base = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
            viewed = base.view(dtype)
            assert not viewed.requires_grad
            assert viewed.grad_fn is None
        base = torch.tensor([1., 2.], device=device)
        viewed = base.view(torch.float32)
        base.requires_grad_(True)
        assert not viewed.requires_grad
        assert viewed.grad_fn is None
        with torch.no_grad():
            viewed[0] = 4.
        np.testing.assert_array_equal(base.detach().cpu().numpy(), [4., 2.])

    def test_dtype_view_retains_storage_after_original_name_dies(self, device):
        base = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
        source = base[1:3, :2]
        viewed = source.view(torch.uint8)
        sibling = base[1:3, :2]
        del source, base
        viewed[1, :4] = 0
        np.testing.assert_array_equal(sibling.cpu().numpy(), [[6., 7.], [0., 13.]])

    def test_dtype_view_shares_native_mutation_version(self, device):
        base = torch.tensor([1., 2.], device=device)
        viewed = base.view(torch.int32)
        before = base._autograd_version()
        assert viewed._autograd_version() == before
        viewed[0] = 0
        assert base._autograd_version() > before
        assert viewed._autograd_version() == base._autograd_version()


instantiate_device_type_tests(TestDtypeViewStrides, globals())
