"""CUDA scalar multiplication uses FP32 opmath with one low-precision cast.

Expected values use IEEE rounding in NumPy; the same non-dyadic inputs and
nonuniform cotangents were checked in a separate genuine PyTorch process on
CUDA. CPU checks protect FP32 arithmetic; low-precision CPU scalar opmath is
a separate native/Torch boundary and is unchanged by the CUDA policy.
"""
import ctypes
import unittest

import numpy as np
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _round(values, precision):
    values = np.asarray(values, dtype=np.float32)
    if precision == torch.float32:
        return values
    if precision == torch.float16:
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def _spellings():
    return (
        lambda x, scalar: x * scalar,
        lambda x, scalar: scalar * x,
        lambda x, scalar: x.mul(scalar),
        lambda x, scalar: x.multiply(other=scalar),
        lambda x, scalar: torch.mul(x, scalar),
        lambda x, scalar: torch.multiply(scalar, x),
        lambda x, scalar: torch.mul(input=x, other=scalar),
        lambda x, scalar: torch.multiply(input=scalar, other=x),
    )


class TestScalarMultiply(cu.JittorTestCase):
    def test_scalar_multiply_opmath_and_gradient(self, device):
        source = np.linspace(-7, 7, 136, dtype=np.float32).reshape(2, 4, 17)
        cotangent = np.random.RandomState(17831).normal(size=source.shape).astype(np.float32)
        with forbid_backend_fallbacks():
            precisions = (torch.float32,) if device == "cpu" else (torch.float16, torch.bfloat16, torch.float32)
            for precision in precisions:
                rounded = _round(source, precision)
                for scalar in (128 ** -.5, -0.143579, 0.):
                    expected = _round(rounded * np.float32(scalar), precision)
                    expected_grad = _round(_round(cotangent, precision) * np.float32(scalar), precision)
                    for enabled in (False, True):
                        for index, multiply in enumerate(_spellings()):
                            with self.subTest(precision=str(precision), scalar=scalar,
                                              enabled=enabled, spelling=index):
                                x = torch.nn.Parameter(torch.tensor(source, device=device).to(precision))
                                amp_dtype = torch.bfloat16 if precision == torch.float32 else precision
                                with torch.autocast(device, dtype=amp_dtype, enabled=enabled):
                                    y = multiply(x, scalar)
                                (y.float() * torch.tensor(cotangent, device=device)).sum().backward()
                                self.assertEqual(str(y.dtype), str(precision))
                                self.assertEqual(str(x.grad.dtype), str(precision))
                                for value in (y, x.grad):
                                    value.sync()
                                    self.assertEqual(value.location(), 'cpu' if device == 'cpu' else 'device')
                                    self.assertEqual(value.device.type, device)
                                np.testing.assert_array_equal(y.float().cpu().numpy(), expected)
                                np.testing.assert_array_equal(x.grad.float().cpu().numpy(), expected_grad)
                                np.testing.assert_array_equal(x.detach().float().cpu().numpy(), rounded)

    def test_scalar_multiply_keeps_nonambient_cuda_device(self, device):
        if device != 'cuda' or torch.cuda.device_count() < 2:
            self.skipTest('requires two visible CUDA devices')
        driver = ctypes.CDLL('libcuda.so.1')
        source = np.array([-.771, .413, 2.17], dtype=np.float32)
        with forbid_backend_fallbacks(), torch.cuda.device(0):
            for precision in (torch.float16, torch.bfloat16):
                for enabled in (False, True):
                    for multiply in _spellings():
                        x = torch.nn.Parameter(torch.tensor(source, device='cuda:1').to(precision))
                        with torch.autocast("cuda", dtype=precision, enabled=enabled):
                            y = multiply(x, 128 ** -.5)
                        y.float().sum().backward()
                        self.assertEqual(torch.cuda.current_device(), 0)
                        for value in (y, x.grad):
                            value.sync()
                            ordinal = ctypes.c_int(-1)
                            status = driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                                                 ctypes.c_void_p(value.device_raw_ptr))
                            self.assertEqual(status, 0)
                            self.assertEqual(ordinal.value, 1)
                            self.assertEqual(value.device.index, 1)
                        expected = _round(_round(source, precision) * np.float32(128 ** -.5), precision)
                        np.testing.assert_array_equal(y.float().cpu().numpy(), expected)


instantiate_device_type_tests(TestScalarMultiply, globals())

if __name__ == '__main__':
    unittest.main()
