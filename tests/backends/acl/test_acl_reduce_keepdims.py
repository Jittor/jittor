"""Full reductions retain singleton dimensions in CANN descriptors and gradients."""

import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


def _reference(values, operation, keepdims):
    output = getattr(np, operation)(values, keepdims=keepdims)
    if operation == "sum":
        gradient = np.ones_like(values)
    elif operation == "mean":
        gradient = np.ones_like(values) / values.size
    elif operation in ("max", "min"):
        selected = values == output
        gradient = selected / selected.sum()
    else:
        # Products of every other factor handle one or several zeros directly;
        # output / input is not a valid oracle at zero.
        flat = values.reshape(-1)
        gradient = np.array([np.prod(np.delete(flat, i)) for i in range(flat.size)]).reshape(values.shape)
    return output, gradient * 1.5


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("operation", ["sum", "mean", "max", "min", "prod"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_full_reduce_shape_value_and_gradient(device, operation, keepdims):
    if device == "npu":
        require_accelerator("acl")
    with jt.flag_scope(use_acl=int(device == "npu"), use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        for shape in ((4,), (2, 2)):
            values = np.array([2., 2., 1., 1.], dtype=np.float32).reshape(shape)
            expected, expected_gradient = _reference(values, operation, keepdims)
            source = jt.array(values).start_grad()
            output = getattr(source, operation)(keepdims=keepdims)
            gradient = jt.grad((output * 1.5).sum(), source)
            assert tuple(output.shape) == ((1,) * len(shape) if keepdims else ())
            output.sync()
            gradient.sync()
            if device == "npu":
                assert output.location() == gradient.location() == "device"
            np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=1e-6, atol=1e-6)
