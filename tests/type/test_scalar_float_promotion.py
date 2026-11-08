"""Native rank-0 promotion must not depend on operand order.

This deliberately avoids the Torch compatibility layer. Both ArrayOp rank-0
values and full reductions carry the scalar flag used by the core dtype rules.
"""

import numpy as np
import pytest

import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def scalar_promotion_device(request):
    device = request.param
    if (
        device == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(device == "cuda")):
        yield device


def _scalar(dtype, value, source):
    data = np.asarray(value if source == "array" else [value], dtype=dtype)
    result = jt.array(data, dtype=dtype)
    if source == "reduction":
        result = result.sum()
    assert tuple(result.shape) == ()
    return result


def _check_outputs(outputs, expected, device):
    # Check every device before any numpy() call can fetch a shared allocation.
    for output in outputs:
        output.sync()
    for output in outputs:
        assert output.location() == ("device" if device == "cuda" else "cpu")
    for output, reference in zip(outputs, expected):
        np.testing.assert_allclose(output.numpy(), reference, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("source", ["array", "reduction"])
def test_two_scalar_float_widths_are_symmetric(scalar_promotion_device, source):
    # Preserve Jittor's existing width-based native float promotion. In
    # particular, float32/int32 -> float32 is not NumPy's promotion table.
    pairs = [
        ("float64", "float32", "float64"),
        ("float64", "int32", "float64"),
        ("float64", "int64", "float64"),
        ("float64", "bool", "float64"),
        ("float32", "int32", "float32"),
        ("float32", "int64", "float64"),
        ("float32", "bool", "float32"),
    ]
    outputs, expected = [], []
    for left_dtype, right_dtype, dtype in pairs:
        left = _scalar(left_dtype, 1.25, source)
        right_value = 1 if right_dtype == "bool" else 2
        right = _scalar(right_dtype, right_value, source)
        for x, y in ((left, right), (right, left)):
            for output, value in (
                (x + y, 1.25 + right_value),
                (x * y, 1.25 * right_value),
            ):
                assert str(output.dtype) == dtype, (left_dtype, right_dtype, source)
                assert tuple(output.shape) == ()
                outputs.append(output)
                expected.append(value)
    _check_outputs(outputs, expected, scalar_promotion_device)


@pytest.mark.parametrize("source", ["array", "reduction"])
def test_scalar_left_preserves_float64_mantissa(scalar_promotion_device, source):
    value = 1.0 + 2.0**-35
    x = _scalar("float64", value, source)
    single = _scalar("float32", 3.0, source)
    outputs = [3 * x, x * 3, 3 + x, x + 3, single * x, x * single]
    expected = [3 * value, value * 3, 3 + value, value + 3, 3 * value, value * 3]
    for output in outputs:
        assert str(output.dtype) == "float64"
    _check_outputs(outputs, expected, scalar_promotion_device)


@pytest.mark.parametrize("source", ["array", "reduction"])
def test_scalar_ternary_preserves_wider_branch(scalar_promotion_device, source):
    # TernaryOp uses dtype_infer rather than binary_dtype_infer.
    value = 1.0 + 2.0**-35
    wider = _scalar("float64", value, source)
    narrower = _scalar("float32", 2.0, source)
    yes = _scalar("bool", True, "array")
    no = _scalar("bool", False, "array")
    outputs = [
        jt.ternary(yes, wider, narrower),
        jt.ternary(no, narrower, wider),
        jt.ternary(no, wider, narrower),
        jt.ternary(yes, narrower, wider),
    ]
    for output in outputs:
        assert str(output.dtype) == "float64"
        assert tuple(output.shape) == ()
    _check_outputs(outputs, [value, value, 2.0, 2.0], scalar_promotion_device)


def test_single_scalar_keeps_existing_tensor_rules(scalar_promotion_device):
    outputs, expected = [], []
    for dtype in ("float16", "float32", "float64"):
        tensor = jt.array(np.asarray([1.5, 2.0], dtype=dtype), dtype=dtype)
        for output in (2 * tensor, tensor * 2, 2.0 * tensor, tensor * 2.0):
            assert str(output.dtype) == dtype
            outputs.append(output)
            expected.append([3.0, 4.0])
    # A Python floating scalar lifts an integer tensor to the default float;
    # the new two-scalar rule must not widen this existing float32 result.
    tensor = jt.array(np.asarray([3, 4], dtype=np.int64), dtype="int64")
    for output in (2.0 * tensor, tensor * 2.0):
        assert str(output.dtype) == "float32"
        outputs.append(output)
        expected.append([6.0, 8.0])
    _check_outputs(outputs, expected, scalar_promotion_device)


@pytest.mark.parametrize("scalar_input", [False, True])
def test_scalar_weighted_loss_and_double_backward(
    scalar_promotion_device, scalar_input
):
    data = np.asarray(
        1.0 + 2.0**-30 if scalar_input else [1.0 + 2.0**-30, -0.25],
        dtype=np.float64,
    )
    base = jt.array(data, dtype="float64")
    left, right = base.clone(), base.clone()
    left_norm = (left * left).sum()
    right_norm = (right * right).sum()
    loss = left_norm + 3 * right_norm
    first_left, first_right = jt.grad(loss, [left, right])
    second_left, second_right = jt.grad(
        first_left.sum() + first_right.sum(), [left, right]
    )
    outputs = [loss, first_left, first_right, second_left, second_right]
    for output in outputs:
        assert str(output.dtype) == "float64"
    expected = [
        4 * np.sum(data * data),
        2 * data,
        6 * data,
        np.full_like(data, 2.0),
        np.full_like(data, 6.0),
    ]
    _check_outputs(outputs, expected, scalar_promotion_device)
