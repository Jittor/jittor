"""Python integer operands and storage writes preserve signed 64-bit values."""
import numpy as np
import pytest

import jittor as jt
from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests


class TestScalarIntegerValues(JittorTestCase):
    def test_large_operands_are_not_truncated_before_promotion(self, device):
        values = np.array([-(2**63), -(2**31)-1, 0, 2**31, 2**40+3, 2**63-1], dtype=np.int64)
        value = jt.array(values, dtype="int64")
        zero = jt.zeros((1,), dtype="int64")
        for scalar in values.tolist():
            np.testing.assert_array_equal(jt.ops.not_equal(value, scalar).numpy(), values != scalar)
            np.testing.assert_array_equal(jt.ops.equal(value, scalar).numpy(), values == scalar)
            assert (zero + scalar).item() == scalar

    def test_scalar_data_assignment_preserves_large_integers(self, device):
        value = jt.zeros((1,), dtype="int64")
        for scalar in (2**31, -(2**31)-1, 2**40+3, -(2**63), 2**63-1):
            value.data = scalar
            assert value.item() == scalar
        small = jt.zeros((1,), dtype="int32")
        for scalar in (0, -1, 2**31-1, -(2**31)):
            small.data = scalar
            assert small.item() == scalar

    def test_unrepresentable_python_integer_reports_overflow(self, device):
        value = jt.zeros((1,), dtype="int64")
        for scalar in (-(2**63)-1, 2**63, 2**100):
            with pytest.raises(OverflowError):
                jt.ops.not_equal(value, scalar)
            with pytest.raises(OverflowError):
                value.data = scalar
        assert value.item() == 0


instantiate_device_type_tests(TestScalarIntegerValues, globals())
