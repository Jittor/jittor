"""Audited OpInfo NPU dtypes must execute; excluded float64 must reject clearly."""

import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from _helpers.child_process import run_child_script
from jittor._runtime.fallback import forbid_backend_fallbacks
from opinfo.database import op_db


def _abs_info():
    return next(info for info in op_db if info.full_name == "abs")


@pytest.mark.npu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_abs_declared_npu_dtype_executes(dtype):
    require_accelerator("acl")
    info = _abs_info()
    assert info.npu_dtypes_are_explicit
    assert dtype in info.supported_dtypes("npu")
    with jt.flag_scope(use_acl=1, use_cuda=1), forbid_backend_fallbacks():
        for sample in info.sample_inputs("npu", dtype):
            expected = np.abs(sample.input.numpy())
            output = info.op(sample.input)
            assert str(output.dtype) == dtype
            output.sync()
            assert output.location() == "device"
            assert output.device_id >= 0
            assert output.placement_backend in (-1, 2)
            np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.npu
def test_abs_float64_rejects_npu_without_silent_conversion():
    require_accelerator("acl")
    info = _abs_info()
    assert "float64" not in info.supported_dtypes("npu")
    assert "float64" in info.supported_dtypes("cpu")
    result = run_child_script(
        r'''
import gc
import jittor as jt
assert jt.compiler.has_acl
with jt.runtime.scope(backend_fallback="error"):
    with jt.flag_scope(use_acl=1, use_cuda=1):
        before = jt.core.backend_fallback_count()
        source = jt.array([-2.0, 0.0, 3.0], dtype="float64")
        output = jt.abs(source)
        assert str(source.dtype) == str(output.dtype) == "float64"
        try:
            output.sync()
        except RuntimeError as error:
            message = str(error).lower()
            assert "fallback" in message and "abs" in message, message
            assert "does not support input dtype" in message, message
        else:
            raise AssertionError("ACL Abs unexpectedly accepted float64")
        assert jt.core.backend_fallback_count() - before == 1
        del output, source
        gc.collect()
print("ACL-ABS-FLOAT64-REJECTED", flush=True)
''',
        text=True, crash_isolated=True, without_torch_mode=True,
        name="acl_abs_float64_rejection",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACL-ABS-FLOAT64-REJECTED" in result.stdout
