"""Five-dimensional channels-last copies use real storage layout."""
import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def test_contiguous_and_clone_keep_logical_values_and_gradient(device):
    data = np.arange(240, dtype=np.float32).reshape(2, 3, 2, 4, 5)
    x = torch.tensor(data, device=device, requires_grad=True)
    fmt = torch.channels_last_3d
    result = x.contiguous(memory_format=fmt)
    assert result.device.type == device
    assert result.stride() == (120, 1, 60, 15, 3)
    assert result.is_contiguous(memory_format=fmt) and not result.is_contiguous()
    assert result.contiguous(memory_format=fmt) is result
    assert result._base is None and result.data_ptr() != x.data_ptr()
    np.testing.assert_array_equal(result.detach().cpu().numpy(), data)
    result.sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), np.ones_like(data))

    cloned = x.detach().clone(memory_format=fmt)
    assert cloned._base is None and cloned.data_ptr() != x.data_ptr()
    assert cloned.stride() == result.stride()
    cloned.fill_(-1)
    np.testing.assert_array_equal(x.detach().cpu().numpy(), data)


@pytest.mark.parametrize("shape, stride", [
    ((2, 3, 2, 4, 5), (120, 1, 60, 15, 3)),
    ((2, 1, 2, 4, 5), (40, 1, 20, 5, 1)),
    ((2, 3, 0, 4, 5), (0, 1, 60, 15, 3)),
])
def test_empty_factories_with_explicit_layout(device, shape, stride):
    fmt = torch.channels_last_3d
    x = torch.empty(size=shape, memory_format=fmt, device=device, requires_grad=True)
    y = torch.empty_like(x, memory_format=fmt, dtype=torch.float64)
    assert x.stride() == y.stride() == stride
    assert x.device.type == y.device.type == device
    assert x.is_leaf and x.requires_grad and y.is_leaf and not y.requires_grad
    assert x._base is None and y._base is None
    assert x.is_contiguous(memory_format=fmt) and y.is_contiguous(memory_format=fmt)
    assert y.dtype == torch.float64


def test_to_layout_reuses_views_and_reorders_new_allocations(device):
    fmt = torch.channels_last_3d
    x = torch.ones((2, 3, 2, 4, 5), device=device)
    formatted = x.to(memory_format=fmt)
    assert formatted.stride() == (120, 1, 60, 15, 3)
    assert formatted.to(memory_format=fmt) is formatted
    cropped = formatted[:, :, :, 1:3, :]
    assert not cropped.is_contiguous(memory_format=fmt)
    assert cropped.to(memory_format=fmt) is cropped

    converted = cropped.to(dtype=torch.float64, memory_format=fmt, copy=True)
    assert converted.device.type == device and converted.dtype == torch.float64
    assert converted.stride() == (60, 1, 30, 15, 3)
    np.testing.assert_array_equal(converted.cpu().numpy(), np.ones((2, 3, 2, 2, 5)))

    single_channel = x[:, :1].contiguous()
    assert single_channel.is_contiguous(memory_format=fmt)
    assert single_channel.to(memory_format=fmt).stride() == (40, 1, 20, 5, 1)


def test_invalid_rank_has_no_channels_last_3d_layout(device):
    fmt = torch.channels_last_3d
    x = torch.empty((2, 3, 4), device=device)
    assert not x.is_contiguous(memory_format=fmt)
    with pytest.raises(RuntimeError, match="required rank 5"):
        x.contiguous(memory_format=fmt)
    with pytest.raises(RuntimeError, match="required rank 5"):
        torch.empty_like(x, memory_format=fmt)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nonoverlapping_conv3d_packs_channels_and_spatial_axes(device, dtype):
    rng = np.random.default_rng(17)
    x = rng.integers(-2, 3, size=(2, 8, 4, 9, 4)).astype(np.float32)
    weight = rng.integers(-2, 3, size=(16, 8, 2, 3, 2)).astype(np.float32)
    bias = np.arange(16, dtype=np.float32) / 4
    from jittor._runtime.fallback import forbid_backend_fallbacks

    with torch.no_grad(), forbid_backend_fallbacks():
        out = torch.nn.functional.conv3d(
            torch.tensor(x, device=device, dtype=dtype),
            torch.tensor(weight, device=device, dtype=dtype),
            torch.tensor(bias, device=device, dtype=dtype),
            stride=(2, 3, 2),
        )
        assert out.dtype == dtype and out.device.type == device
        if device == "cuda":
            assert out.is_contiguous(memory_format=torch.channels_last_3d)
        tiles = x.reshape(2, 8, 2, 2, 3, 3, 2, 2).transpose(0, 2, 4, 6, 1, 3, 5, 7)
        expected = tiles.reshape(-1, 96) @ weight.reshape(16, 96).T + bias
        expected = torch.tensor(expected, dtype=dtype, device="cpu").float().numpy()
        actual = out.permute(0, 2, 3, 4, 1).reshape(-1, 16).float().cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
