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


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("operation", ["add", "maximum", "minimum", "multiply"])
def test_mixed_keepdims_reduce_value_and_gradient(device, operation):
    """Broadcast gradients can retain one reduced axis while deleting another."""
    if device == "npu":
        require_accelerator("acl")
    values = np.linspace(0.8, 1.2, 48, dtype=np.float32).reshape(2, 2, 3, 4)
    reference_name = dict(add="sum", maximum="max",
                          minimum="min", multiply="prod")[operation]
    expected = getattr(np, reference_name)(values, axis=(1, 2)).reshape(2, 1, 4)
    if operation == "add":
        expected_grad = np.ones_like(values)
    elif operation == "multiply":
        expected_grad = expected.reshape(2, 1, 1, 4) / values
    else:
        expected_grad = (values == expected.reshape(2, 1, 1, 4)).astype(np.float32)
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        source = jt.array(values).start_grad()
        output = getattr(jt, "reduce_" + operation)(
            source, dims_mask=0b0110, keepdims_mask=0b0010)
        gradient = jt.grad(output.sum(), source)
        assert tuple(output.shape) == (2, 1, 4)
        for tensor in (output, gradient):
            tensor.sync()
            assert tensor.location() == ("device" if device == "npu" else "cpu")
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), expected_grad, rtol=1e-6, atol=1e-6)
