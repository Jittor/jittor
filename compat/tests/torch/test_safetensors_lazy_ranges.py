"""SafeTensor slices read only their byte range and BF16 stays 16-bit."""

import json
import struct

import numpy as np
import pytest
import torch

pytest.importorskip("safetensors")

from jittor.compat.torch.serialization import safetensors as codec  # noqa: E402


def _write(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path))
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size)), 8 + size


def _bf16_bits(tensor):
    return tensor.view(torch.int16).numpy().astype(np.uint16)


def test_metadata_and_row_slices_do_not_read_the_whole_tensor(tmp_path, monkeypatch):
    from safetensors import safe_open

    values = np.arange(4096, dtype=np.float32)
    matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    path = tmp_path / "lazy.safetensors"
    _write(path, {"weight": torch.tensor(values), "matrix": torch.tensor(matrix)})
    reads = []
    original = codec._read_range

    def tracking(filename, offset, length):
        reads.append((offset, length))
        return original(filename, offset, length)

    monkeypatch.setattr(codec, "_read_range", tracking)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        handle = reader.get_slice("weight")
        assert handle.get_shape() == [4096]
        assert handle.get_dtype() == "F32"
        assert reads == []  # metadata is header-only
        part = handle[16:48]
        assert reads[-1][1] == 32 * 4  # only the requested rows
        np.testing.assert_array_equal(part.numpy(), values[16:48])
        scalar = handle[7]
        assert reads[-1][1] == 4
        assert scalar.item() == values[7]

        rows = reader.get_slice("matrix")[1:3]
        assert reads[-1][1] == 2 * 6 * 4
        np.testing.assert_array_equal(rows.numpy(), matrix[1:3])

        # A selection that is not one contiguous range still decodes the
        # whole tensor, exactly like the previous reader.
        columns = reader.get_slice("matrix")[:, 1:3]
        assert reads[-1][1] == matrix.nbytes
        np.testing.assert_array_equal(columns.numpy(), matrix[:, 1:3])


@pytest.mark.parametrize(
    "index",
    [Ellipsis, slice(None), slice(1, 3), 2, -1, slice(None, None, -1),
     (slice(None), slice(1, 3)), (Ellipsis, slice(1, 3))],
)
def test_slice_values_match_numpy(tmp_path, index):
    from safetensors import safe_open

    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    path = tmp_path / "slices.safetensors"
    _write(path, {"value": torch.tensor(values)})
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        got = reader.get_slice("value")[index]
    expected = values if index is Ellipsis else values[index]
    np.testing.assert_array_equal(got.numpy(), expected)


def test_bf16_payload_is_bit_exact_and_not_float32(tmp_path):
    from safetensors.torch import load, load_file

    bits = np.array([0x3F80, 0x0000, 0x8000, 0x7FC0, 0x0080], dtype=np.uint16)
    tensor = torch.tensor(bits, dtype="uint16").reinterpret_view((5,), "bfloat16")
    path = tmp_path / "bf16.safetensors"
    header, base = _write(path, {"w": tensor})
    raw = path.read_bytes()[base:base + bits.nbytes]
    assert raw == bits.tobytes()

    loaded = load_file(str(path))["w"]
    assert loaded.dtype is torch.bfloat16
    np.testing.assert_array_equal(_bf16_bits(loaded), bits)

    decoded = load(path.read_bytes())["w"]
    assert decoded.dtype is torch.bfloat16
    np.testing.assert_array_equal(_bf16_bits(decoded), bits)


def test_non_contiguous_bf16_still_round_trips(tmp_path):
    from safetensors.torch import load_file, save_file

    bits = np.arange(30, dtype=np.uint16).reshape(5, 6)
    base = torch.tensor(bits, dtype="uint16").reinterpret_view((5, 6), "bfloat16")
    view = base[:, ::2]
    assert not view.is_contiguous()
    path = tmp_path / "noncontiguous.safetensors"
    save_file({"w": view}, str(path))
    loaded = load_file(str(path))["w"]
    np.testing.assert_array_equal(_bf16_bits(loaded), bits[:, ::2])
