"""CPU contracts for the canonical linear module implementations."""

import unittest

import numpy as np

import jittor as jt
from jittor import nn


class TestConv1dSpecialized(unittest.TestCase):
    def test_forward_and_backward_match_linear_reference(self):
        weight = np.array(
            [[0.25, -0.5, 0.75], [-1.0, 0.125, 0.5]], dtype=np.float32
        )
        bias = np.array([0.1, -0.2], dtype=np.float32)
        x_array = (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 9.0
        ) / 5.0
        mask = np.linspace(-0.4, 0.8, 16, dtype=np.float32).reshape(2, 2, 4)

        with jt.flag_scope(use_cuda=0):
            layer = nn.Conv1d_sp(3, 2, kernel_size=1, bias=True)
            layer.weight.assign(jt.array(weight))
            layer.bias.assign(jt.array(bias))
            x = jt.array(x_array)
            x.start_grad()
            output = layer(x)
            input_grad, weight_grad, bias_grad = jt.grad(
                (output * jt.array(mask)).sum(),
                [x, layer.weight, layer.bias],
            )
            actual = output.numpy()

        expected = np.einsum("ncl,oc->nol", x_array, weight) + bias.reshape(1, 2, 1)
        expected_input_grad = np.einsum("nol,oc->ncl", mask, weight)
        expected_weight_grad = np.einsum("nol,ncl->oc", mask, x_array)
        expected_bias_grad = mask.sum(axis=(0, 2))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            input_grad.numpy(), expected_input_grad, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            weight_grad.numpy(), expected_weight_grad, atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            bias_grad.numpy(), expected_bias_grad, atol=1e-6, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
