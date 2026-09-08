"""Real CUDA execution through registered kernel and codegen callbacks."""

from _helpers import capability as _test_capability

import numpy as np
import pytest


def test_fused_cuda_execution_uses_registered_fragments_and_jit_kernel():
    import jittor as jt
    from _helpers.native_op_dispatch import _check_fused_callbacks

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    _check_fused_callbacks(jt, True)


def test_code_op_cuda_selects_cuda_source_and_keeps_device_residency():
    import jittor as jt
    from _helpers.native_op_dispatch import _CPU_CODE, _CUDA_CODE

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    data = np.arange(17, dtype=np.float32)
    with jt.flag_scope(use_cuda=1):
        source = jt.array(data)
        result = jt.code(source.shape, source.dtype, [source],
                         cpu_src=_CPU_CODE, cuda_src=_CUDA_CODE)
        result.sync()
        assert result.location() == "device"
        np.testing.assert_array_equal(result.numpy(), data + 23)
        supported = set(jt.core.backend_supported_ops("cuda"))
        assert {"binary", "fused", "cublas_matmul"} <= supported
