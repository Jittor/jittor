import math

import numpy as np
import pytest

from _helpers.capability import require_accelerator
import jittor as jt


@pytest.mark.cuda
@pytest.mark.parametrize(("shape", "axes", "dtype"), [
    ((2, 19, 5, 64), (0, 2, 1, 3), "float32"),
    ((1, 31, 7, 72), (0, 2, 1, 3), "float16"),
    ((1, 13, 3, 9, 5, 20), (0, 1, 3, 2, 4, 5), "bfloat16"),
    ((2, 3, 5, 2048), (0, 2, 1, 3), "float32"),
])
def test_packed_contiguous_copy(shape, axes, dtype):
    require_accelerator("cuda")
    data = (np.arange(math.prod(shape), dtype=np.float32) % 127).reshape(shape)
    with jt.flag_scope(use_cuda=1, transpose_storage_view=1, backend_fallback="error"), jt.no_grad():
        source = jt.array(data).cast(dtype).transpose(axes)
        assert not source._storage_is_contiguous()
        result = source.contiguous()
        result.sync()
        assert result.location() == "device" and result._storage_is_contiguous()
        assert result.dtype == source.dtype
        np.testing.assert_array_equal(result.float32().numpy(), data.transpose(axes))


@pytest.mark.cuda
def test_packed_contiguous_with_offset_and_broadcast():
    require_accelerator("cuda")
    data = np.arange(1 * 15 * 3 * 68, dtype=np.float32).reshape(1, 15, 3, 68)
    with jt.flag_scope(use_cuda=1, transpose_storage_view=1, backend_fallback="error"), jt.no_grad():
        base = jt.array(data).transpose(0, 2, 1, 3)
        for start in (0, 2, 3, 4):
            source = base[..., start:start + 64].expand(2, 3, 15, 64)
            result = source.contiguous()
            result.sync()
            assert result.location() == "device"
            expected = data.transpose(0, 2, 1, 3)[..., start:start + 64]
            np.testing.assert_array_equal(result.numpy(), np.broadcast_to(expected, result.shape))


@pytest.mark.cuda
def test_batched_short_transpose_copy():
    require_accelerator("cuda")
    shape = (2, 17, 23, 2, 2, 24)
    axes = (0, 5, 1, 3, 2, 4)
    data = (np.arange(math.prod(shape), dtype=np.float32) % 127).reshape(shape)
    with jt.flag_scope(use_cuda=1, transpose_storage_view=1, backend_fallback="error"), jt.no_grad():
        source = jt.array(data).bfloat16().transpose(axes)
        result = source.contiguous()
        result.sync()
        assert result.location() == "device" and result._storage_is_contiguous()
        np.testing.assert_array_equal(result.float32().numpy(), data.transpose(axes))
