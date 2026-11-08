"""Native remainder gradients must multiply both branches by the cotangent."""

import jittor as jt
import numpy as np
import pytest

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["scalar", "broadcast"])
def test_mod_weighted_two_input_gradients(device, dtype, layout):
    if layout == "broadcast":
        left = np.asarray([[-7.5], [8.5]], dtype=dtype)
        right = np.asarray([[2, -3, 4, -5]], dtype=dtype)
        weights = np.asarray(
            [[0.25, -1.5, 2, 0.75], [-0.5, 1.25, -2.5, 3]], dtype=dtype
        )
        expected_x = weights.sum(axis=1, keepdims=True)
        expected_y = (-weights * np.floor(left / right)).sum(axis=0, keepdims=True)
    else:
        left, right, weights = (
            np.asarray(value, dtype=dtype) for value in (-7.5, 2, -1.75)
        )
        expected_x = weights
        expected_y = np.asarray(-weights * np.floor(left / right))
    # Stay away from remainder's discontinuities. Native mod uses floor,
    # unlike fmod: dx=w and dy=-w*floor(x/y), with broadcast axes reduced.
    assert np.all(left / right != np.floor(left / right))
    x, y, w = (jt.array(value, dtype=dtype) for value in (left, right, weights))
    value = jt.mod(x, y)
    gradient_x, gradient_y = jt.grad((value * w).sum(), [x, y])
    pairs = (
        (value, np.asarray(np.remainder(left, right))),
        (gradient_x, expected_x),
        (gradient_y, expected_y),
    )
    for actual, expected in pairs:
        assert tuple(actual.shape) == expected.shape
        assert str(actual.dtype) == dtype
        actual.sync()
        assert actual.location() == ("device" if device == "cuda" else "cpu")
    # All device assertions precede the first host fetch from this graph.
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    for actual, expected in pairs:
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=tolerance, atol=tolerance
        )
