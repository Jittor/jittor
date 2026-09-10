"""Adaptive pooling retains native shape objects and integer-like sizes."""
import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_native_shape_and_numpy_integer_output_sizes(device):
    if device == "npu":
        require_accelerator("acl")
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        source = np.arange(20, dtype=np.float32).reshape(1, 1, 4, 5)
        x = jt.array(source)
        for size in (x.shape[-2:], [4, 5], (np.int64(4), np.int32(5))):
            output = jt.nn.adaptive_avg_pool2d(x, size)
            gradient = jt.grad(output.sum(), x)
            output.sync()
            assert output.location() == ("device" if device == "npu" else "cpu")
            np.testing.assert_array_equal(output.numpy(), source)
            np.testing.assert_array_equal(gradient.numpy(), np.ones_like(source))
        # A NumPy scalar is also a valid integer output-size argument.
        np.testing.assert_allclose(
            jt.nn.adaptive_avg_pool2d(x, np.int64(1)).numpy(),
            source.mean(axis=(-2, -1), keepdims=True))
        for invalid in (True, (False, 2), (np.float64(2), 3)):
            with pytest.raises(ValueError, match="positive integers"):
                jt.nn.adaptive_avg_pool2d(x, invalid)
