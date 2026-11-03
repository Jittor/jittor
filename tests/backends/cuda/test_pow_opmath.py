"""Weak Python bases must select the result tensor's FP32 CUDA arithmetic."""
import os

import jittor as jt
import numpy as np
import pytest

from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.parametrize("base", [1000000, 1000000.])
def test_weak_base_and_fp32_exponent_use_fp32_cuda_pow(base):
    if os.environ.get("JITTOR_TEST_DEVICE", "cpu") != "cuda":
        pytest.skip("CUDA pow overload contract")
    # Independent PyTorch 2.8 CUDA scalar-base pow outputs, including both
    # rounding directions where double pow followed by a cast disagrees.
    exponents = np.array([.28125, .34375, .578125, .78125], dtype="float32")
    expected = np.array([48.696754455566406, 115.47820281982422,
                         2942.72705078125, 48696.75], dtype="float32")
    with jt.flag_scope(use_cuda=1, amp_reg=0), forbid_backend_fallbacks():
        x = jt.array(exponents)
        y = base ** x
        y.sync()
        assert str(y.dtype) == "float32" and y.location() == "device"
        np.testing.assert_array_equal(y.numpy(), expected)


def test_mixed_and_float64_pow_preserve_wide_and_signed_values():
    with jt.flag_scope(use_cuda=int(os.environ.get("JITTOR_TEST_DEVICE", "cpu") == "cuda"),
                       amp_reg=0, auto_convert_64_to_32=0), forbid_backend_fallbacks():
        x = jt.array(np.array([-3., -2., .5, 2.], dtype="float64"))
        for exponent in (3, 3., jt.array(np.array([3., 3., 3., 3.], dtype="float64"))):
            result = x ** exponent
            assert str(result.dtype) == "float64"
            np.testing.assert_allclose(result.numpy(), np.array([-27., -8., .125, 8.]), rtol=2e-15, atol=0.)
