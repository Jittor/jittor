"""310P half arg reductions keep values and indices on the same input row."""
import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("shape,dim", [((7,), 0), ((7,), -1), ((3, 7), 0), ((2, 3, 7), 1),
    ((1, 8400, 80), -1), ((1, 8400, 96), -1)])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_half_arg_reduce_values_indices_and_gradient(device, shape, dim, op, keepdims):
    if device == "npu":
        require_accelerator("acl")
    rng = np.random.RandomState(20261030)
    source = rng.uniform(-5, 5, shape).astype("float16")
    axis = dim % len(shape)
    ids = getattr(np, "arg" + op)(source, axis=axis)
    selected = np.take_along_axis(source, np.expand_dims(ids, axis), axis=axis)
    expected_values = selected if keepdims else np.squeeze(selected, axis=axis)
    expected_ids = np.expand_dims(ids, axis) if keepdims else ids
    if len(shape) == 1 and not keepdims:
        expected_ids = np.asarray(expected_ids).reshape(1)
        expected_values = np.asarray(expected_values).reshape(1)
    weights = (np.arange(ids.size).reshape(ids.shape) % 7 + 1).astype("float16")
    expected_gradient = np.zeros_like(source)
    np.put_along_axis(expected_gradient, np.expand_dims(ids, axis),
        np.expand_dims(weights, axis), axis=axis)
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(device == "npu"), use_acl=int(device == "npu")), forbid_backend_fallbacks():
        value = jt.array(source, dtype="float16").start_grad()
        indices, values = jt.arg_reduce(value, op, dim, keepdims)
        scaled = weights if not keepdims else np.expand_dims(weights, axis)
        gradient = jt.grad((values * jt.array(scaled, dtype="float16")).sum(), value)
        for output in (indices, values, gradient):
            output.sync()
            assert output.location() == ("device" if device == "npu" else "cpu")
        assert str(values.dtype) == str(gradient.dtype) == "float16"
        np.testing.assert_array_equal(indices.numpy(), expected_ids)
        np.testing.assert_array_equal(values.numpy(), expected_values)
        np.testing.assert_array_equal(gradient.numpy(), expected_gradient)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.npu
@pytest.mark.parametrize("op", ["max", "min"])
def test_half_arg_reduce_gathers_nonfinite_values_without_saturation(op):
    require_accelerator("acl")
    # One NaN per row avoids the separate existing SDK multiple-NaN tie policy.
    source = np.array([[np.inf, 1, -np.inf], [1, np.nan, 2]], dtype="float16")
    expected_ids = [0, 1] if op == "max" else [2, 1]
    expected = [np.inf, np.nan] if op == "max" else [-np.inf, np.nan]
    with jt.runtime.scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        indices, values = jt.arg_reduce(jt.array(source, dtype="float16"), op, -1)
        jt.sync([indices, values], True)
        assert indices.location() == values.location() == "device"
        np.testing.assert_array_equal(indices.numpy(), expected_ids)
        np.testing.assert_array_equal(values.numpy(), np.array(expected, dtype="float16"))


@pytest.mark.npu
@pytest.mark.parametrize("soc", ["Ascend310P3", "Ascend910B3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_half_arg_reduce_widening_is_scoped_to_310p_half(soc, dtype, monkeypatch):
    require_accelerator("acl")
    from jittor.backends.acl.kernels import tensor
    seen = []
    original = tensor.ArgReduceACL.__call__
    def observe(self, source, *args):
        seen.append(str(source.dtype))
        return original(self, source, *args)
    monkeypatch.setattr(tensor, "soc_name", lambda: soc)
    monkeypatch.setattr(tensor.ArgReduceACL, "__call__", observe)
    with jt.runtime.scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        indices, values = jt.arg_reduce(jt.array([[1, 2, 2]], dtype=dtype), "max", -1)
        jt.sync([indices, values], True)
        np.testing.assert_array_equal(indices.numpy(), [1])
        np.testing.assert_array_equal(values.numpy(), [2])
    expected = "float32" if soc.startswith("Ascend310P") else dtype
    assert seen == [expected]


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("dim", [0, -1])
def test_native_scalar_arg_reduce_keeps_existing_dimension_error(device, dim):
    if device == "npu":
        require_accelerator("acl")
    with jt.runtime.scope(use_cuda=int(device == "npu"), use_acl=int(device == "npu")), forbid_backend_fallbacks():
        scalar = jt.array(np.array(3, dtype="float16"), dtype="float16")
        with pytest.raises(RuntimeError, match="Invalid dim for arg_reduce"):
            jt.arg_reduce(scalar, "max", dim)
