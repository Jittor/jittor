"""Torch-compatible scalar power backward order on Jittor Vars."""
import numpy as np
import torch


def test_pow_two_backward_method_function_and_operator():
    values = np.array([0.0, -0.125, 0.3, 1.0, 17.25], dtype=np.float32)
    upstream = np.array([0.25, -0.75, 1.5, -2.0, 0.125], dtype=np.float32)
    operations = (
        lambda value: value.pow(2),
        lambda value: torch.pow(value, 2),
        lambda value: value ** 2,
    )
    for operation in operations:
        value = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        grad_output = torch.tensor(upstream, dtype=torch.float32)
        result = operation(value)
        (actual,) = torch.autograd.grad(
            result, (value,), grad_outputs=grad_output)
        expected = (grad_output * 2) * value.pow(1)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(
            result.numpy(), (value.detach() * value.detach()).numpy())
