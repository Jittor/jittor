"""Real CUDA coverage of the dense native DDP bucket storage boundary."""

import numpy as np
import pytest

import jittor as jt
from jittor.backends.cuda.kernels.misc.ddp_bucket import (
    pack_gradients, unpack_gradients,
)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_bucket_copies_preserve_dtype_shape_and_detached_storage(dtype):
    with jt.flag_scope(use_cuda=1):
        arrays = [np.arange(24).reshape(4, 6), np.arange(9).reshape(3, 3),
                  np.empty((0, 2))]
        members = [jt.array(a, dtype="float32").cast(dtype).cuda(0).stop_grad()
                   for a in arrays]
        members[0] = members[0].transpose(1, 0)
        expected = [arrays[0].T, arrays[1], arrays[2]]
        flat = pack_gradients(members)
        restored = unpack_gradients(flat, members, 2)
        for result, original, wanted in zip(restored, members, expected):
            assert result.dtype == original.dtype
            assert tuple(result.shape) == tuple(original.shape)
            assert not result.requires_grad
            assert result.device_id == 0
            np.testing.assert_array_equal(result.float64().numpy(), wanted / 2)
        # Rebinding one gradient must leave the other outputs and packed data
        # intact, including when the optimizer clears holders after a step.
        restored[0].update(jt.zeros_like(restored[0]))
        np.testing.assert_array_equal(restored[1].float64().numpy(), expected[1] / 2)
        np.testing.assert_array_equal(flat.float64().numpy(),
                                      np.concatenate([a.reshape(-1) for a in expected]))


def test_all_empty_bucket_executes_on_cuda():
    with jt.flag_scope(use_cuda=1):
        members = [jt.empty((0,), dtype="float32").cuda(0).stop_grad(),
                   jt.empty((2, 0), dtype="float32").cuda(0).stop_grad()]
        flat = pack_gradients(members)
        restored = unpack_gradients(flat, members, 2)
        jt.sync([flat] + restored)
        assert flat.numel() == 0
        assert [tuple(g.shape) for g in restored] == [(0,), (2, 0)]
        assert all(g.device_id == 0 for g in [flat] + restored)
