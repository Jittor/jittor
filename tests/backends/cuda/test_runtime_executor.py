"""Runtime-owned executor access from Python-generated CUDA code."""

import numpy as np
import pytest


def test_unique_generated_cuda_uses_runtime_executor_allocator():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")
    data = np.array([4, 1, 4, 2, 1, 3], dtype=np.int32)
    reference, inverse, counts = np.unique(data, return_inverse=True, return_counts=True)
    with jt.flag_scope(use_cuda=1):
        values, actual_inverse, actual_counts = jt.unique(
            jt.array(data), return_inverse=True, return_counts=True
        )
        values.sync()
        assert values.device_id == 0
        assert values.location() == "device"
        np.testing.assert_array_equal(values.numpy(), reference)
        np.testing.assert_array_equal(actual_inverse.numpy(), inverse)
        np.testing.assert_array_equal(actual_counts.numpy(), counts)
