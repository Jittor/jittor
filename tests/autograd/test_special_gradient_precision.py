"""Special-function VJPs preserve double constants and their input device."""

import math

import numpy as np
import pytest

import jittor as jt
from _helpers.capability import require_accelerator
from _helpers.devices import explicit_tensor_placement


@pytest.mark.parametrize("device", ["cpu", "cuda", "cpu_ambient_cuda"])
@pytest.mark.parametrize("operation", ["erf", "erfinv"])
def test_special_function_double_first_and_second_gradients(device, operation):
    if device != "cpu":
        require_accelerator("cuda")
    data = np.array([-.5, -.25, 0., .25, .5], dtype="float64")
    with jt.flag_scope(use_cuda=int(device != "cpu"), auto_convert_64_to_32=0,
                       backend_fallback="error"):
        with explicit_tensor_placement(int(device == "cuda")):
            x = jt.array(data).start_grad()
        output = getattr(jt, operation)(x)
        first = jt.grad(output.sum(), x)
        second = jt.grad(first.sum(), x)
        first.sync()
        second.sync()
        if operation == "erf":
            expected = 2 / math.sqrt(math.pi) * np.exp(-data * data)
            expected_second = -2 * data * expected
        else:
            # Isolate VJP accuracy from the inverse function's approximation.
            y = output.numpy().copy()
            expected = math.sqrt(math.pi) / 2 * np.exp(y * y)
            expected_second = 2 * y * expected * expected
        for value, reference in ((first, expected), (second, expected_second)):
            assert str(value.dtype) == "float64"
            assert value.location() == ("device" if device == "cuda" else "cpu")
            np.testing.assert_allclose(value.numpy(), reference, atol=1e-14, rtol=1e-14)
