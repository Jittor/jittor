"""Pooling's relocated geometry helper must reach real ACL forward/backward."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.skipif(
    not capability.check_accelerator("acl", backend=jt).enabled, reason="No ACL found")


@pytest.mark.parametrize("operation", ["maximum", "mean"])
@pytest.mark.parametrize("ceil_mode,include_pad", [(False, True), (False, False), (True, True), (True, False)])
def test_pooling_forward_backward_with_padding(operation, ceil_mode, include_pad):
    # Distinct signed values avoid max-pool ties; asymmetric weights catch
    # backward window placement as well as forward geometry.
    rng = np.random.RandomState(20260910)
    source = rng.permutation(16).astype("float32").reshape(1, 1, 4, 4) - 8
    extent = 3 if ceil_mode else 2
    weights = np.arange(1, extent * extent + 1, dtype="float32").reshape(1, 1, extent, extent)
    expected = np.empty_like(weights)
    gradient = np.zeros_like(source)
    for row in range(extent):
        for col in range(extent):
            top, left = row * 2 - 1, col * 2 - 1
            points = [(i, j) for i in range(top, top + 3)
                      for j in range(left, left + 3) if 0 <= i < 4 and 0 <= j < 4]
            if operation == "maximum":
                i, j = max(points, key=lambda point: source[0, 0, point[0], point[1]])
                expected[0, 0, row, col] = source[0, 0, i, j]
                gradient[0, 0, i, j] += weights[0, 0, row, col]
            else:
                divisor = (min(top + 3, 5) - max(top, -1)) * (min(left + 3, 5) - max(left, -1)) if include_pad else len(points)
                expected[0, 0, row, col] = sum(source[0, 0, i, j] for i, j in points) / divisor
                for i, j in points:
                    gradient[0, 0, i, j] += weights[0, 0, row, col] / divisor
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        value = jt.array(source)
        output = jt.nn.Pool(3, stride=2, padding=1, ceil_mode=ceil_mode,
                            count_include_pad=include_pad, op=operation)(value)
        backward = jt.grad((output * jt.array(weights)).sum(), value)
        for result in (output, backward):
            result.sync()
            assert result.location() == "device"
            assert result.device_id >= 0
            assert result.placement_backend in (-1, 2)
        actual, actual_gradient = jt.fetch_sync([output, backward])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-6, atol=1e-6)
