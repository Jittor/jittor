"""Regression tests for SELU and CELU numerical behavior."""

import unittest

import numpy as np
import torch


class TestSeluCeluNumerics(unittest.TestCase):
    def test_functional_values_near_zero(self):
        values = np.array([-1e-7, 1e-7, -np.inf, np.inf], dtype=np.float32)
        celu = torch.nn.functional.celu(torch.tensor(values), alpha=1.0).numpy()
        selu = torch.nn.functional.selu(torch.tensor(values)).numpy()

        celu_expected = np.array([-1e-7, 1e-7, -1.0, np.inf], dtype=np.float32)
        scale, alpha = 1.0507009873554805, 1.6732632423543772
        selu_expected = np.array([-scale * alpha * 1e-7, scale * 1e-7,
                                  -scale * alpha, np.inf], dtype=np.float32)
        np.testing.assert_allclose(celu, celu_expected, rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(selu, selu_expected, rtol=2e-6, atol=1e-12)

        elu = torch.nn.functional.elu(torch.tensor(values)).numpy()
        elu_expected = np.array([-1e-7, 1e-7, -1.0, np.inf], dtype=np.float32)
        np.testing.assert_allclose(elu, elu_expected, rtol=2e-6, atol=1e-12)

    def test_gradients_on_each_branch(self):
        values = np.array([-1e-6, 0.0, 1e-6, 1.0], dtype=np.float32)
        celu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.celu(celu_input, alpha=1.0).sum().backward()
        celu_expected = np.where(values > 0, 1.0, np.exp(values))
        np.testing.assert_allclose(celu_input.grad.numpy(), celu_expected,
                                   rtol=2e-6, atol=2e-7)

        selu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.selu(selu_input).sum().backward()
        scale, alpha = 1.0507009873554805, 1.6732632423543772
        expected = scale * np.where(values > 0, 1.0, alpha * np.exp(values))
        np.testing.assert_allclose(selu_input.grad.numpy(), expected,
                                   rtol=2e-6, atol=2e-7)

        elu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.elu(elu_input).sum().backward()
        elu_expected = np.where(values > 0, 1.0, np.exp(values))
        np.testing.assert_allclose(elu_input.grad.numpy(), elu_expected,
                                   rtol=2e-6, atol=2e-7)

    def test_nonfinite_inputs_have_finite_branch_gradients(self):
        values = np.array([-np.inf, np.inf], dtype=np.float32)
        celu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.celu(celu_input).sum().backward()
        np.testing.assert_array_equal(celu_input.grad.numpy(), [0.0, 1.0])

        selu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.selu(selu_input).sum().backward()
        scale = 1.0507009873554805
        np.testing.assert_allclose(selu_input.grad.numpy(), [0.0, scale],
                                   rtol=1e-7, atol=1e-8)

        elu_input = torch.tensor(values, requires_grad=True)
        torch.nn.functional.elu(elu_input).sum().backward()
        np.testing.assert_array_equal(elu_input.grad.numpy(), [0.0, 1.0])

    def test_smallest_float32_subnormals_are_preserved(self):
        tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
        values = np.array([-tiny, tiny], dtype=np.float32)
        for activation in (torch.nn.functional.celu, torch.nn.functional.elu):
            np.testing.assert_array_equal(activation(torch.tensor(values)).numpy(), values)
        np.testing.assert_array_equal(
            torch.nn.functional.selu(torch.tensor(values)).numpy(),
            np.array([-2.0 * tiny, tiny], dtype=np.float32))

    def test_tail_boundary_and_gradients_in_float32_and_float64(self):
        scale, alpha = 1.0507009873554805, 1.6732632423543772
        for dtype, np_dtype, rtol in ((torch.float32, np.float32, 3e-6),
                                     (torch.float64, np.float64, 3e-13)):
            values = np.array([-2.0, -1.00001, -1.0, -0.99999, -1e-8,
                               0.0, 1e-8, 2.0], dtype=np_dtype)
            for name in ("celu", "selu", "elu"):
                value = torch.tensor(values, dtype=dtype, requires_grad=True)
                result = getattr(torch.nn.functional, name)(value)
                expected = np.where(values > 0, values, np.expm1(values))
                derivative = np.where(values > 0, 1.0, np.exp(values))
                if name == "selu":
                    expected = scale * np.where(values > 0, values, alpha * np.expm1(values))
                    derivative = scale * np.where(values > 0, 1.0, alpha * np.exp(values))
                np.testing.assert_allclose(result.detach().numpy(), expected, rtol=rtol, atol=0.0)
                result.sum().backward()
                np.testing.assert_allclose(value.grad.numpy(), derivative, rtol=rtol, atol=0.0)

    def test_modules_expose_inplace_option(self):
        self.assertTrue(torch.nn.CELU(inplace=True).inplace)
        self.assertTrue(torch.nn.SELU(inplace=True).inplace)

    def test_module_values_cover_near_zero_and_nonfinite_inputs(self):
        values = np.array([-1e-8, 1e-8, -np.inf, np.inf], dtype=np.float32)
        celu = torch.nn.CELU()(torch.tensor(values)).numpy()
        selu = torch.nn.SELU()(torch.tensor(values)).numpy()

        scale, alpha = 1.0507009873554805, 1.6732632423543772
        np.testing.assert_allclose(
            celu, [-1e-8, 1e-8, -1.0, np.inf], rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(
            selu, [-scale * alpha * 1e-8, scale * 1e-8,
                   -scale * alpha, np.inf], rtol=2e-6, atol=1e-12)

    def test_celu_rejects_zero_alpha(self):
        for activation in (torch.nn.functional.celu, torch.nn.CELU(alpha=0.0)):
            with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError: alpha cannot be 0"):
                if activation is torch.nn.functional.celu:
                    activation(torch.tensor([1.0]), alpha=0.0)
                else:
                    activation(torch.tensor([1.0]))

    def test_non_tensor_inputs_are_rejected_before_alpha_validation(self):
        for activation in (torch.nn.functional.celu, torch.nn.functional.selu,
                           torch.nn.CELU(), torch.nn.SELU(), torch.nn.CELU(alpha=0.0)):
            for value in ([1.0], np.array([1.0], dtype=np.float32), 1.0):
                with self.assertRaisesRegex(TypeError, "input.*Tensor"):
                    activation(value)
        with self.assertRaisesRegex(TypeError, "input.*Tensor"):
            torch.nn.functional.celu([1.0], alpha=0.0)

    def test_inplace_returns_the_input_and_changes_its_values(self):
        for name in ("celu", "selu"):
            functional = getattr(torch.nn.functional, name)
            module = getattr(torch.nn, name.upper())(inplace=True)
            for activation in (lambda value: functional(value, inplace=True), module):
                value = torch.tensor([-1.0, 1.0])
                alias = value[:]
                pointer = value.data_ptr()
                expected = functional(value.clone()).numpy()
                self.assertIs(activation(value), value)
                self.assertEqual(value.data_ptr(), pointer)
                np.testing.assert_allclose(value.numpy(), expected, rtol=2e-6, atol=1e-7)
                np.testing.assert_allclose(alias.numpy(), expected, rtol=2e-6, atol=1e-7)

    def test_inplace_preserves_nonleaf_gradients_and_rejects_trainable_leaf(self):
        values = np.array([-1.0, 1.0], dtype=np.float32)
        for name in ("celu", "selu"):
            functional = getattr(torch.nn.functional, name)
            module = getattr(torch.nn, name.upper())(inplace=True)
            for activation in (lambda value: functional(value, inplace=True), module):
                leaf = torch.tensor(values, requires_grad=True)
                value = leaf * 1.0
                self.assertIs(activation(value), value)
                value.sum().backward()
                expected = np.where(values > 0, 1.0, np.exp(values))
                if name == "selu":
                    expected = 1.0507009873554805 * np.where(
                        values > 0, 1.0, 1.6732632423543772 * np.exp(values))
                np.testing.assert_allclose(leaf.grad.numpy(), expected, rtol=2e-6, atol=1e-7)
                with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                    activation(torch.tensor(values, requires_grad=True))

    def test_celu_preserves_negative_alpha_behavior(self):
        values = np.array([-1.0, 1.0], dtype=np.float32)
        result = torch.nn.functional.celu(torch.tensor(values), alpha=-1.0).numpy()
        expected = np.maximum(0.0, values) + np.minimum(
            0.0, -1.0 * (np.exp(values / -1.0) - 1.0))
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
