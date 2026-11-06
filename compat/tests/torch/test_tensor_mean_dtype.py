"""Regression tests for Tensor.mean(dtype=...)."""

import unittest

import numpy as np
import torch


class TestTensorMeanDtype(unittest.TestCase):
    def test_dtype_casts_before_reduction(self):
        values = np.array([[1.25, 2.5], [4.0, 8.0]], dtype=np.float32)
        tensor = torch.tensor(values, requires_grad=True)

        result = tensor.mean(dim=1, dtype=torch.float64)

        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_allclose(result.detach().numpy(), values.astype(np.float64).mean(axis=1))

    def test_dtype_cast_preserves_gradient(self):
        values = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float32)
        tensor = torch.tensor(values, requires_grad=True)

        tensor.mean(dtype=torch.float64).backward()

        self.assertEqual(tensor.grad.dtype, torch.float32)
        np.testing.assert_allclose(tensor.grad.numpy(), np.full_like(values, 0.25))

    def test_product_dtype_casts_before_reduction(self):
        values = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
        result = torch.tensor(values).prod(dim=1, dtype=torch.float64)

        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_array_equal(result.numpy(), values.astype(np.float64).prod(axis=1))

    def test_mean_casts_before_accumulating(self):
        tensor = torch.tensor([[16777216.0, 1.0, -16777216.0]])

        result = tensor.mean(dim=1, dtype=torch.float64)

        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_allclose(result.numpy(), [1.0 / 3.0], rtol=1e-8)

    def test_mean_accepts_integral_inputs_with_floating_dtype(self):
        for input_dtype in (torch.int32, torch.bool):
            with self.subTest(input_dtype=input_dtype):
                tensor = torch.tensor([[0, 1], [1, 1]], dtype=input_dtype)
                result = tensor.mean(dim=(0, 1), keepdim=True, dtype=torch.float64)
                self.assertEqual(result.dtype, torch.float64)
                self.assertEqual(tuple(result.shape), (1, 1))
                np.testing.assert_array_equal(result.numpy(), [[0.75]])

    def test_mean_rejects_integral_output_dtype(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        for output_dtype in (torch.int32, torch.int64, torch.bool, int, bool):
            with self.subTest(output_dtype=output_dtype):
                with self.assertRaisesRegex(RuntimeError, "floating point or complex"):
                    tensor.mean(dim=-1, keepdim=True, dtype=output_dtype)

    def test_reductions_reject_invalid_dtype_arguments(self):
        tensor = torch.tensor([1.0, 3.0])
        for operation in (tensor.mean, tensor.prod):
            for output_dtype in ("float64", np.dtype("float64"), np.float64, 123, object()):
                with self.subTest(operation=operation.__name__, output_dtype=output_dtype):
                    with self.assertRaises(TypeError):
                        operation(dtype=output_dtype)

    def test_python_float_dtype_alias(self):
        tensor = torch.tensor([1.0, 3.0])
        for operation, expected in ((tensor.mean, 2.0), (tensor.prod, 3.0)):
            with self.subTest(operation=operation.__name__):
                result = operation(dtype=float)
                self.assertEqual(result.dtype, torch.float64)
                self.assertEqual(result.item(), expected)

    def test_complex_reduction_dtype(self):
        tensor = torch.tensor([1.0, 3.0])
        for output_dtype in (torch.complex64, torch.complex128, complex):
            expected_dtype = torch.complex128 if output_dtype is complex else output_dtype
            for operation, expected in ((tensor.mean, 2.0), (tensor.prod, 3.0)):
                with self.subTest(operation=operation.__name__, output_dtype=output_dtype):
                    result = operation(dtype=output_dtype)
                    self.assertEqual(result.dtype, expected_dtype)
                    np.testing.assert_array_equal(result.numpy(), complex(expected))

    def test_product_preserves_explicit_boolean_dtype(self):
        tensor = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
        for output_dtype in (torch.bool, bool):
            with self.subTest(output_dtype=output_dtype):
                result = tensor.prod(dim=-1, keepdim=True, dtype=output_dtype)
                self.assertEqual(result.dtype, torch.bool)
                self.assertEqual(tuple(result.shape), (2, 1))
                np.testing.assert_array_equal(result.numpy(), [[False], [True]])

    def test_product_cast_prevents_integer_overflow(self):
        tensor = torch.tensor([50000, 50000], dtype=torch.int32)

        result = tensor.prod(dtype=torch.int64)

        self.assertEqual(result.dtype, torch.int64)
        self.assertEqual(result.item(), 2500000000)

    def test_product_dtype_cast_preserves_gradient(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

        tensor.prod(dim=-1, keepdim=True, dtype=torch.float64).sum().backward()

        self.assertEqual(tensor.grad.dtype, torch.float32)
        np.testing.assert_allclose(tensor.grad.numpy(), [[6.0, 3.0, 2.0], [30.0, 24.0, 20.0]])


if __name__ == "__main__":
    unittest.main()
