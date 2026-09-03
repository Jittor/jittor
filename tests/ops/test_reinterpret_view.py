import unittest

import numpy as np

import jittor as jt
from _helpers.assertions import expect_error


class TestReinterpretViewErrors(unittest.TestCase):
    def test_invalid_dtype_is_a_catchable_error(self):
        expect_error(
            lambda: jt.reinterpret_view(jt.array([1, 2]), [2], "not_a_dtype"),
            exc_type=RuntimeError,
            # pyjt rejects an unknown NanoString before the op constructor;
            # the public call remains a normal, catchable RuntimeError.
            match="Not a valid call",
        )

    def test_multiple_inferred_dimensions_is_a_catchable_user_error(self):
        expect_error(
            lambda: jt.reinterpret_view(jt.array([1, 2, 3, 4]), [-1, -1], "float32"),
            exc_type=RuntimeError,
            match="at most one -1 dimension",
        )

    def test_byte_size_mismatch_is_a_catchable_user_error(self):
        expect_error(
            lambda: jt.reinterpret_view(jt.array([1, 2]), [3], "float32"),
            exc_type=RuntimeError,
            match="byte size mismatch",
        )

    def test_complex_views_require_a_two_element_last_dimension(self):
        complex_value = jt.array(
            np.array([1 + 2j, 3 + 4j], dtype="complex64"))
        expect_error(
            lambda: jt.reinterpret_view(complex_value, [4], "float32"),
            exc_type=RuntimeError,
            match="complex64 -> float32.*shape",
        )

        real_value = jt.array(np.arange(4, dtype="float32"))
        expect_error(
            lambda: jt.reinterpret_view(real_value, [2], "complex64"),
            exc_type=RuntimeError,
            match="float32 -> complex64.*shape",
        )


if __name__ == "__main__":
    unittest.main()
