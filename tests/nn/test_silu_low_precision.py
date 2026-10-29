"""SiLU rounds reduced precision outputs and gradients after fp32 computation."""

import unittest

import numpy as np

import jittor as jt
from jittor._core.dtypes import var_dtype_name
from jittor._runtime.fallback import forbid_backend_fallbacks


def _quantize(value, dtype):
    value = np.asarray(value, dtype="float32")
    if dtype == "float16":
        return value.astype("float16").astype("float32")
    bits = value.view("uint32")
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return rounded.view("float32")


class _SiLULowPrecision:
    use_cuda = 0

    def setUp(self):
        self.scope = jt.runtime.scope(use_cuda=self.use_cuda, amp_reg=0,
                                      backend_fallback="error")
        self.scope.__enter__()
        self.forbid_fallback = forbid_backend_fallbacks()
        self.forbid_fallback.__enter__()
        self.fallback_before = jt.core.backend_fallback_count()

    def tearDown(self):
        jt.sync_all(True)
        self.assertEqual(jt.core.backend_fallback_count(), self.fallback_before)
        self.forbid_fallback.__exit__(None, None, None)
        self.scope.__exit__(None, None, None)

    def _check(self, dtype, amp_reg=0):
        # These exactly representable inputs also distinguish premature rounding
        # of sigmoid(x) from one output rounding: SiLU(bf16(4.59375)) = 4.5625.
        raw = np.array([-11., -8., -6., -4., -3., -2., -1.5, -1., -.5,
                        -.125, 0., .125, .5, 1., 1.5, 2., 3., 4., 4.59375,
                        6., 8., 10., 12.], dtype="float32")
        x = jt.array(raw).cast(dtype)
        with jt.flag_scope(amp_reg=amp_reg):
            output = jt.nn.silu(x)
            self.assertEqual(jt.flags.amp_reg, amp_reg)
        gradient = jt.grad(output.float32().sum(), x)
        for value in (x, output, gradient):
            self.assertEqual(var_dtype_name(value), dtype)
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value])[0],
                             "cuda" if self.use_cuda else "cpu")
        wide = raw.astype("float64")
        sigmoid = 1. / (1. + np.exp(-wide))
        expected = _quantize(wide * sigmoid, dtype)
        expected_gradient = _quantize(sigmoid * (1. + wide * (1. - sigmoid)), dtype)
        np.testing.assert_array_equal(output.float32().numpy(), expected)
        np.testing.assert_array_equal(gradient.float32().numpy(), expected_gradient)

    def test_forward_and_input_gradient_against_math(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                self._check(dtype)

    def test_compute_precision_under_amp(self):
        for dtype, register in (("float16", jt.amp_flags.prefer16),
                                ("bfloat16", jt.amp_flags.prefer_bfloat16)):
            with self.subTest(dtype=dtype):
                self._check(dtype, register)


class TestSiLULowPrecisionCPU(_SiLULowPrecision, unittest.TestCase):
    pass


@unittest.skipUnless(jt.has_cuda, "requires real CUDA")
class TestSiLULowPrecisionCUDA(_SiLULowPrecision, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
