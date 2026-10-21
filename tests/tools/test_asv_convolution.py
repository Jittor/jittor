"""Independent finite differences and failure contracts for the ASV oracle."""

import numpy as np
import pytest

from benchmarks.asv import convolution as suite


def test_reference_keeps_depthwise_channels_and_bias_separate():
    x = np.arange(8, dtype="float64").reshape(1, 2, 2, 2)
    weight = np.array([2.0, -3.0]).reshape(2, 1, 1, 1)
    bias = np.array([1.0, -1.0])
    cotangent = np.array([1.0, 2.0]).reshape(1, 2, 1, 1) * np.ones_like(x)
    output, (dx, dw, db) = suite._reference(x, weight, bias, cotangent, 1, 0, 2)
    np.testing.assert_array_equal(
        output, np.array([1, 3, 5, 7, -13, -16, -19, -22]).reshape(x.shape)
    )
    np.testing.assert_array_equal(dx, np.array([2, 2, 2, 2, -6, -6, -6, -6]).reshape(x.shape))
    np.testing.assert_array_equal(dw, np.array([6, 44]).reshape(weight.shape))
    np.testing.assert_array_equal(db, [4, 8])


@pytest.mark.parametrize("groups,stride,padding,kernel", [(1, 1, 1, 3), (1, 2, 0, 1), (2, 1, 1, 3)])
def test_reference_gradients_match_finite_differences(groups, stride, padding, kernel):
    rng = np.random.default_rng(17)
    x = rng.standard_normal((1, 2, 4, 4))
    weight = rng.standard_normal((2, 2 // groups, kernel, kernel))
    bias = rng.standard_normal(2)
    size = (4 + 2 * padding - kernel) // stride + 1
    cotangent = rng.standard_normal((1, 2, size, size))
    args = (stride, padding, groups)
    output, gradients = suite._reference(x, weight, bias, cotangent, *args)
    # A perturbation of any requested leaf must match its returned derivative.
    for value, gradient in zip((x, weight, bias), gradients):
        direction = rng.standard_normal(value.shape)
        original = value.copy()
        value[:] = original + 1e-5 * direction
        plus = suite._reference(x, weight, bias, cotangent, *args)[0]
        value[:] = original - 1e-5 * direction
        minus = suite._reference(x, weight, bias, cotangent, *args)[0]
        value[:] = original
        derivative = ((plus - minus) * cotangent).sum() / 2e-5
        np.testing.assert_allclose(derivative, (gradient * direction).sum(), rtol=1e-8, atol=1e-8)
    assert np.isfinite(output).all()


def test_validation_rejects_missing_wrong_and_nonfinite_results():
    expected = np.ones((2, 2))
    gradients = (np.ones(2), np.ones((2, 2)), np.ones(2))
    suite._assert_result(expected, gradients, expected, gradients)
    with pytest.raises(RuntimeError, match="every requested gradient"):
        suite._assert_result(expected, gradients[:2], expected, gradients)
    with pytest.raises(RuntimeError, match="weight gradient differs"):
        suite._assert_result(
            expected, (gradients[0], np.zeros((2, 2)), gradients[2]), expected, gradients
        )
    with pytest.raises(RuntimeError, match="wrong shape"):
        suite._assert_result(np.ones(2), (), expected, ())
    with pytest.raises(RuntimeError, match="non-finite"):
        suite._assert_result(np.full((2, 2), np.nan), (), expected, ())


def test_setup_rejects_invalid_device_and_mode_before_backend_import():
    bench = suite.Conv2dBenchmarks()
    with pytest.raises(ValueError, match="unsupported convolution device"):
        bench.setup("jittor", "npu", "standard_3x3", "forward")
    with pytest.raises(ValueError, match="unsupported convolution mode"):
        bench.setup("jittor", "cpu", "standard_3x3", "unknown")
