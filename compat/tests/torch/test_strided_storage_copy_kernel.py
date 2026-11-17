"""Storage copies preserve bytes and untouched regions for arbitrary layouts."""
import numpy as np
import pytest
import jittor as jt
import torch

from _rng_init_support import require_rng_device


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return require_rng_device(request.param)


def _numpy(value):
    value = value.detach().clone()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.cpu().numpy().copy()


def _select(value, layout):
    if layout == "inner_gap":
        return value[..., ::2]
    if layout == "layer_gap":
        return value[:, :, 1, :]
    if layout == "transpose":
        return value.transpose(0, 3)
    return value[:0]


@pytest.mark.parametrize("dtype", ["bool", "uint8", "int16", "int32", "int64",
                                  "float16", "bfloat16", "float32", "float64",
                                  "complex64", "complex128"])
@pytest.mark.parametrize("layout", ["inner_gap", "layer_gap", "transpose", "empty"])
def test_copy_retains_storage_and_untouched_bytes(device, dtype, layout):
    np_dtype = np.dtype("float32" if dtype == "bfloat16" else dtype)
    expected = np.full((3, 4, 5, 6), 7, dtype=np_dtype)
    root = torch.tensor(expected, dtype=getattr(torch, dtype), device=device)
    destination = _select(root, layout)
    # numpy.transpose takes an axes tuple, while Tensor.transpose swaps axes.
    selected = (expected.swapaxes(0, 3) if layout == "transpose"
                else _select(expected, layout))
    data = (np.arange(selected.size).reshape(selected.shape) % 13).astype(np_dtype)
    if np_dtype.kind == "c":
        data = data + 1j * (data + 1)
    source = torch.tensor(data, dtype=getattr(torch, dtype), device=device)
    root.sync(); source.sync(); destination.sync()
    pointer = root.data_ptr()
    with torch.no_grad():
        destination._copy_into(source)
    selected[...] = data
    np.testing.assert_array_equal(_numpy(root), expected)
    assert root.data_ptr() == pointer
    assert root.device.type == device
    assert root.location() == ("device" if device == "cuda" else "cpu")


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
def test_copy_is_bit_preserving(device, dtype):
    # Large integers, signed zero and noncanonical NaN payloads cannot survive
    # an implementation that converts through floating-point arithmetic.
    bits = np.array([0x8000000000000000, 0x7ff8000000000123,
                     0x002000000000000b, 0xffffffffffffffff], dtype=np.uint64)
    data = np.resize(bits, 12).view(dtype).reshape(3, -1)
    tensor_dtype = getattr(torch, np.dtype(dtype).name)
    source = torch.tensor(data.copy(), dtype=tensor_dtype, device=device)
    root = torch.zeros((3, data.shape[1] * 2), dtype=tensor_dtype, device=device)
    destination = root[:, ::2]
    with torch.no_grad():
        destination._copy_into(source)
    actual = _numpy(destination)
    np.testing.assert_array_equal(actual.view(np.uint64), data.view(np.uint64))
    np.testing.assert_array_equal(_numpy(root[:, 1::2]), np.zeros_like(data))


def test_high_rank_copy_coalesces_singleton_axes(device):
    shape = (2, 3) + (1,) * 10
    root = torch.zeros(shape, dtype=torch.int64, device=device)
    destination = root.transpose(0, 1)
    data = np.arange(6, dtype=np.int64).reshape(tuple(destination.shape))
    source = torch.tensor(data, device=device)
    with torch.no_grad():
        destination._copy_into(source)
    np.testing.assert_array_equal(_numpy(root), data.swapaxes(0, 1))


def test_recorded_copy_reads_fresh_source_values(device):
    if device != "cuda":
        pytest.skip("accelerator prerequisite: CUDA graph capture")
    root = torch.zeros((3, 8), device=device)
    destination = root[:, ::2]
    source = torch.ones((3, 4), device=device)
    root.sync(); source.sync(); destination.sync()
    with torch.no_grad():
        destination._copy_into(source, False)
        jt.sync_all(True)
        assert jt.core.graph_capture_begin()
        try:
            destination._copy_into(source, False)
        finally:
            handle = jt.core.graph_capture_end()
        assert handle
        try:
            for value in (2., -3., 9.):
                source._copy_into(torch.full((3, 4), value, device=device))
                jt.core.graph_launch(handle)
                expected = np.zeros((3, 8), dtype=np.float32)
                expected[:, ::2] = value
                np.testing.assert_array_equal(_numpy(root), expected)
        finally:
            jt.core.graph_release(handle)
