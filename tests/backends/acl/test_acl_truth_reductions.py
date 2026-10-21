"""Shared any/all dispatch keeps ACL boolean values and public reduction shapes."""
import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("operation", ["any", "all"])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dim", [None, 0, -1, (0, 1)])
@pytest.mark.parametrize("dtype", ["bool", "float32"])
def test_truth_reduce_full_values_shape_and_residency(device, operation, keepdims, dim, dtype):
    if device == "npu":
        require_accelerator("acl")
    values = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=dtype)
    if dtype == "float32":
        values[1, 1] = np.nan
    expected = getattr(np, operation)(values, axis=dim, keepdims=keepdims)
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        source = jt.array(values)
        # None labels the documented default full reduction: native dim=().
        options = {"keepdims": keepdims}
        if dim is not None:
            options["dim"] = dim
        output = getattr(jt, operation)(source, **options)
        output.sync()
        assert tuple(output.shape) == expected.shape
        assert str(output.dtype) == "bool"
        assert output.location() == ("device" if device == "npu" else "cpu")
        if device == "npu":
            assert tuple(jt.core.dispatch_context([output])) == ("acl", 0)
        np.testing.assert_array_equal(output.numpy(), expected)
