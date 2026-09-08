"""Native tensor spellings retain their overloads after backend routing."""

import jittor as jt
import numpy as np
import pytest


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_registered_tensor_spellings_match_native_values(use_cuda):
    if use_cuda and not jt.has_cuda:
        pytest.skip("CUDA is unavailable")
    values = np.arange(6, dtype="float32").reshape(2, 3)
    with jt.flag_scope(use_cuda=use_cuda, backend_fallback="error"):
        x = jt.array(values)
        for output in (jt.transpose(x), x.transpose(), x.transpose((1, 0)), x.transpose(1, 0)):
            np.testing.assert_array_equal(output.numpy(), values.T)
        np.testing.assert_array_equal(x.transpose((0, 1)).numpy(), values)
        np.testing.assert_array_equal(x.transpose(-1, -2).numpy(), values.T)
        for actual, expected in zip(jt.index(x, dtype="int64"), np.indices(values.shape)):
            assert str(actual.dtype) == "int64"
            np.testing.assert_array_equal(actual.numpy(), expected)
        condition = x > 2
        for actual, expected in zip(jt.where(condition), np.where(values > 2)):
            np.testing.assert_array_equal(actual.numpy(), expected)
        np.testing.assert_array_equal(jt.where(condition, x, -x).numpy(),
                                      np.where(values > 2, values, -values))
        np.testing.assert_array_equal(jt.floor_int(x + 0.75).numpy(), values.astype("int32"))
        np.testing.assert_allclose(x.sigmoid().numpy(), 1 / (1 + np.exp(-values)), atol=1e-6)
