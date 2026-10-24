"""SOURCE ONLY: finite CUDA softplus/scalar-division contracts, not PPO parity.

Independent expectations use Python math and explicit IEEE round-to-nearest,
not a second call to the same backend's softplus/sigmoid/division formula.
Python scalar metadata/reference evaluation is allowed; no CPU Tensor runs.
Unexecuted source is not CUDA evidence. No production function is patched.
"""
import ctypes
import math
import struct
import unittest

import pytest

pytestmark = pytest.mark.cuda

SOFTPLUS_X = [-16., -2., -.5, 0., .5, 2., 8., 24.]
SOFTPLUS_DY = [.125, -.25, .5, -1., 2., -.5, .25, .125]
DIVISION_X = [-3.25, -.5, .25, 1.125, 2., 7.75]
DIVISION_DY = [.125, -.5, 2., .25, -1., .5]


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _round(value, dtype_name):
    value = _f32(value)  # FP32 opmath before the final low-precision store.
    if dtype_name == "float32":
        return value
    if dtype_name == "float16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    if dtype_name != "bfloat16":
        raise ValueError("unsupported fixed-fixture reference dtype")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # Retain high 16 bits, ties to even. These fixtures are finite, no NaNs.
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _softplus_reference(values, weights, output_dtype, input_dtype):
    output, gradient = [], []
    for value, weight in zip(values, weights):
        if value > 20.:  # default beta=1, threshold=20: linear branch.
            result, derivative = value, 1.
        elif value >= 0.:
            z = math.exp(-value)
            result, derivative = value + math.log1p(z), 1. / (1. + z)
        else:
            z = math.exp(value)
            result, derivative = math.log1p(z), z / (1. + z)
        output.append(_round(result, output_dtype))
        gradient.append(_round(weight * derivative, input_dtype))
    return output, gradient


def run(torch, jt, pointer_probe):
    """Caller supplies real, already admitted CUDA and Driver pointer checks."""
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks
    check = unittest.TestCase()
    check.assertTrue(torch.cuda.is_available(), "real CUDA is required")
    check.assertEqual(torch.cuda.current_device(), 0)
    diagnostics_before = counts()
    rows = []

    def flags():
        return (torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
                torch.is_autocast_cache_enabled(), torch.is_grad_enabled(),
                int(jt.flags.amp_reg), int(jt.flags.use_cuda))

    def prove(value):
        value.sync()
        pointer_probe(jt, value)
        check.assertEqual(str(value.device), "cuda:0")
        check.assertTrue(value.is_cuda)
        check.assertEqual((int(value.placement_backend), int(value.device_id)), (1, 0))
        check.assertEqual(int(jt.core.backend_fallback_count()), 0)

    def tensor(data, dtype, grad=False):
        with torch.autocast("cuda", enabled=False):
            value = torch.tensor(data, device="cuda:0", dtype=dtype, requires_grad=grad)
        prove(value)
        check.assertIs(value.dtype, dtype)
        return value

    def compare(value, expected, dtype):
        check.assertIsNotNone(value)
        check.assertIs(value.dtype, dtype)
        prove(value)
        with torch.autocast("cuda", enabled=False):
            target = tensor(expected, dtype)
            # Binary RNE reference is exact for low-precision fixture stores;
            # FP32 permits only small transcendental/division kernel roundoff.
            tolerance = 3e-7 if dtype is torch.float32 else 0.
            torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)

    def softplus_case(dtype_name, enabled, label):
        before = flags()
        dtype = getattr(torch, dtype_name)
        source = tensor(SOFTPLUS_X, dtype)
        value = source.detach().clone().requires_grad_(True)
        prove(value)
        output_dtype_name = "float32" if enabled else dtype_name
        output_dtype = getattr(torch, output_dtype_name)
        expected, gradient = _softplus_reference(
            SOFTPLUS_X, SOFTPLUS_DY, output_dtype_name, dtype_name)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            inside = flags()
            actual = torch.nn.functional.softplus(value)
            check.assertEqual(flags(), inside)
            compare(actual, expected, output_dtype)
            check.assertEqual(flags(), inside)
        check.assertEqual(flags(), before)
        # Supply a complete nonuniform cotangent directly. No sum-only VJP.
        with torch.autocast("cuda", enabled=False):
            actual.backward(tensor(SOFTPLUS_DY, output_dtype))
        compare(value.grad, gradient, dtype)
        check.assertIs(value.dtype, dtype)
        prove(value)
        check.assertTrue(bool(torch.equal(value.detach(), source)))
        check.assertEqual(flags(), before)
        rows.append(dict(operation="softplus", case=label, input_dtype=dtype_name,
                         output_dtype=output_dtype_name, all_input_gradients_checked=True))

    def division_case(denominator, enabled, label):
        before = flags()
        source = tensor(DIVISION_X, torch.float32)
        value = source.detach().clone().requires_grad_(True)
        prove(value)
        expected = [_f32(x / denominator) for x in DIVISION_X]
        gradient = [_f32(w / denominator) for w in DIVISION_DY]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            inside = flags()
            actual = value / denominator
            check.assertEqual(flags(), inside)
            compare(actual, expected, torch.float32)
            check.assertEqual(flags(), inside)
        check.assertEqual(flags(), before)
        with torch.autocast("cuda", enabled=False):
            actual.backward(tensor(DIVISION_DY, torch.float32))
        compare(value.grad, gradient, torch.float32)
        check.assertIs(value.dtype, torch.float32)
        prove(value)
        check.assertTrue(bool(torch.equal(value.detach(), source)))
        check.assertEqual(flags(), before)
        rows.append(dict(operation="scalar_division", case=label, denominator=denominator,
                         input_dtype="float32", output_dtype="float32",
                         all_input_gradients_checked=True))

    with forbid_backend_fallbacks(), jt.runtime.scope(
            use_cuda=1, device_id=0, backend_fallback="error"), torch.enable_grad():
        entry = flags()
        try:
            for enabled in (False, True):
                for dtype_name in ("float32", "float16", "bfloat16"):
                    softplus_case(dtype_name, enabled, "amp_on" if enabled else "amp_off")
                for denominator in (1.0, 1.3):
                    division_case(denominator, enabled, "amp_on" if enabled else "amp_off")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outer = flags()
                with torch.autocast("cuda", enabled=False):
                    inner = flags()
                    softplus_case("bfloat16", False, "nested_disabled")
                    division_case(1.3, False, "nested_disabled")
                    check.assertEqual(flags(), inner)
                check.assertEqual(flags(), outer)
                # Invalid beta reaches the original native softplus while its
                # operation-local disabled autocast scope is active.
                source = tensor(SOFTPLUS_X, torch.float32)
                with check.assertRaises(TypeError):
                    torch.nn.functional.softplus(source, beta="invalid")
                check.assertEqual(flags(), outer)
                with check.assertRaises(TypeError):
                    torch.nn.functional.softplus(source, threshold="invalid")
                check.assertEqual(flags(), outer)
                with check.assertRaises(RuntimeError):
                    torch.div(source, 1.3, rounding_mode="invalid")
                check.assertEqual(flags(), outer)
        finally:
            check.assertEqual(flags(), entry)
    check.assertEqual(counts(), diagnostics_before)
    actual_fallback_count = int(jt.core.backend_fallback_count())
    check.assertEqual(actual_fallback_count, 0)
    check.assertEqual(len(rows), 12)
    return dict(scope="finite_CUDA_softplus_scalar_division_NOT_PPO",
                actual_cases=len(rows), expected_cases=12, cases=rows,
                core_matrix_cases=10, nested_disabled_cases=2,
                mathematical_reference="Python_math_IEEE_RNE_not_same_backend_formula",
                fallback_count=actual_fallback_count, no_new_diagnostics=True,
                payload_passed=True, full_PPO_or_L2_PASS_claimed=False)


class TestCudaAutocastSoftplusScalarDivision(unittest.TestCase):
    def test_cuda_softplus_scalar_division_dtype_and_weighted_gradients(self):
        import torch
        import jittor as jt

        def pointer_probe(jt, value):
            pointer = int(value.device_raw_ptr)
            self.assertGreater(pointer, 0)
            ordinal, memory = ctypes.c_int(-1), ctypes.c_int(-1)
            driver = ctypes.CDLL("libcuda.so.1")
            self.assertEqual(driver.cuPointerGetAttribute(
                ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
            self.assertEqual(driver.cuPointerGetAttribute(
                ctypes.byref(memory), 2, ctypes.c_void_p(pointer)), 0)
            self.assertEqual((ordinal.value, memory.value), (0, 2))

        report = run(torch, jt, pointer_probe)
        self.assertTrue(report["payload_passed"])
        self.assertEqual(report["actual_cases"], 12)
