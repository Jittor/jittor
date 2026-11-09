"""Native dtype queries are available without installing a Torch frontend."""

import pytest
import jittor as jt


@pytest.mark.parametrize("dtype,floating,complex_,signed", [
    ("float16", True, False, True),
    ("bfloat16", True, False, True),
    ("float32", True, False, True),
    ("float64", True, False, True),
    ("int32", False, False, True),
    ("uint8", False, False, False),
    ("bool", False, False, False),
    ("complex64", False, True, True),
    ("complex128", False, True, True),
])
def test_native_dtype_queries(dtype, floating, complex_, signed):
    value = jt.array([0], dtype=dtype)
    assert value.is_floating_point() is floating
    assert value.is_complex() is complex_
    assert value.is_signed() is signed
