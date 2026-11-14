"""Empty matrix contractions preserve shape, values and operand gradients."""
import itertools

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests


def _values(value):
    return value.float32().numpy() if str(value.dtype) == "bfloat16" else value.numpy()


class TestMatmulEmptyContractions(JittorTestCase):
    def test_zero_inner_dimension_fills_the_whole_result(self, device):
        for batch, dtype in itertools.product(
                ((), (2,)), ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")):
            with self.subTest(batch=batch, dtype=dtype):
                a = jt.zeros(batch + (2, 0), dtype=dtype)
                b = jt.zeros(batch + (0, 3), dtype=dtype)
                result = jt.matmul(a, b)
                # Reading a dependent op also checks that the zero fill was
                # ordered before its consumer on the accelerator stream.
                np.testing.assert_array_equal(_values(result + 1), np.ones(batch + (2, 3)))
                self.assertEqual(tuple(result.shape), batch + (2, 3))
                self.assertEqual(str(result.dtype), dtype)
                result.sync()
                self.assertEqual(result.location(), "device" if device == "cuda" else "cpu")
                np.testing.assert_array_equal(_values(result), np.zeros(batch + (2, 3)))

    def test_empty_outputs_and_contractions_have_zero_gradients(self, device):
        for batch, shape, dtype in itertools.product(
                ((), (2,), (0,)), ((2, 0, 3), (0, 4, 3), (2, 4, 0)), ("float32", "complex64")):
            with self.subTest(batch=batch, shape=shape, dtype=dtype):
                m, k, n = shape
                a = jt.ones(batch + (m, k), dtype=dtype)
                b = jt.ones(batch + (k, n), dtype=dtype)
                result = jt.matmul(a, b)
                loss = result.real.sum() if dtype.startswith("complex") else result.sum()
                with jt.flag_scope(missing_grad_error=1):
                    da, db = jt.grad(loss, [a, b])
                self.assertEqual(tuple(result.shape), batch + (m, n))
                np.testing.assert_array_equal(_values(result), np.zeros(batch + (m, n)))
                for gradient, operand in ((da, a), (db, b)):
                    self.assertEqual(tuple(gradient.shape), tuple(operand.shape))
                    self.assertEqual(str(gradient.dtype), dtype)
                    np.testing.assert_array_equal(_values(gradient), np.zeros(tuple(operand.shape)))
                    if gradient.numel():
                        self.assertEqual(gradient.location(), "device" if device == "cuda" else "cpu")

    def test_explicit_cuda_transposes_handle_empty_contractions(self, device):
        if device != "cuda":
            self.skipTest("explicit cuBLAS transpose flags require CUDA")
        from jittor._runtime.backend_libraries import get_library_ops
        ops = get_library_ops("cublas")
        for batch, trans_a, trans_b in itertools.product(((), (2,)), (False, True), (False, True)):
            with self.subTest(batch=batch, trans_a=trans_a, trans_b=trans_b):
                a = jt.zeros(batch + ((0, 2) if trans_a else (2, 0)))
                b = jt.zeros(batch + ((3, 0) if trans_b else (0, 3)))
                operation = ops.cublas_batched_matmul if batch else ops.cublas_matmul
                result = operation(a, b, trans_a, trans_b)
                np.testing.assert_array_equal(result.numpy(), np.zeros(batch + (2, 3)))
                self.assertEqual(result.location(), "device")

    def test_empty_bfloat16_contraction_fills_float32_accumulator_bytes(self, device):
        if device != "cuda":
            self.skipTest("explicit cuBLAS accumulator output requires CUDA")
        from jittor._runtime.backend_libraries import get_library_ops
        ops = get_library_ops("cublas")
        a = jt.zeros((2, 0), dtype="bfloat16")
        b = jt.zeros((0, 3), dtype="bfloat16")
        result = ops.cublas_matmul(a, b, False, False, fp32_output=True)
        self.assertEqual(str(result.dtype), "float32")
        np.testing.assert_array_equal(result.numpy(), np.zeros((2, 3)))
        self.assertEqual(result.location(), "device")


instantiate_device_type_tests(TestMatmulEmptyContractions, globals())
