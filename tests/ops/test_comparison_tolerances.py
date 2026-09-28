"""Tensor precision survives NumPy staging without weakening other comparisons."""

import numpy as np
import pytest

from _helpers.common import JittorTestCase


class _StagedTensor:
    """A dtype-preserving tensor facade whose host transport uses float32."""

    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def numpy(self):
        return np.array([self.value], dtype=np.float32)


def test_bfloat16_uses_tensor_precision_before_host_conversion():
    case = JittorTestCase()
    case.assertEqual(_StagedTensor(0.388672, "bfloat16"), np.array([0.389076]))
    # A material error must still fail under the pre-existing BF16 policy.
    with pytest.raises(AssertionError):
        case.assertEqual(_StagedTensor(0.5, "bfloat16"), np.array([0.389076]))
    # Explicit caller tolerances continue to override the dtype defaults.
    with pytest.raises(AssertionError):
        case.assertEqual(_StagedTensor(0.388672, "bfloat16"), np.array([0.389076]),
                         atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float_precision_is_not_relaxed_by_bfloat16_support(dtype):
    with pytest.raises(AssertionError):
        JittorTestCase().assertEqual(_StagedTensor(0.388672, dtype), np.array([0.389076]))
