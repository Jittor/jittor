"""Explicit CUDA source provenance is enforced before executing foreign code."""

from _helpers import capability as _test_capability

import gc

import numpy as np
import pytest

import jittor as jt


def _require_cuda():
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled or _test_capability.device_count('cuda', backend=jt) < 1:
        pytest.skip("CUDA source provenance requires a CUDA device")


def test_cuda_source_provenance_forward_and_backward():
    _require_cuda()
    with jt.flag_scope(use_cuda=1, lazy_execution=1, auto_flush_ops=0):
        x = jt.array(np.array([1, 2, 3], dtype=np.float32))
        y = jt.code(x.shape, x.dtype, [x], backend="cuda", cuda_src="""
            __global__ void scale_forward(const float* input, float* output, int n) {
                int i = threadIdx.x;
                if (i < n) output[i] = input[i] * 2;
            }
            CHECK(backend == "cuda");
            scale_forward<<<1, 32>>>(in0_p, out0_p, in0_shape0);
        """, cuda_grad_src=["""
            __global__ void scale_backward(const float* input, float* output, int n) {
                int i = threadIdx.x;
                if (i < n) output[i] = input[i] * 2;
            }
            CHECK(backend == "cuda");
            scale_backward<<<1, 32>>>(dout_p, out0_p, out0_shape0);
        """])
        gradient = jt.grad(y.sum(), x)
        jt.sync_all(True)
        assert y.location() == "device"
        assert gradient.location() == "device"
        np.testing.assert_array_equal(y.numpy(), [2, 4, 6])
        np.testing.assert_array_equal(gradient.numpy(), [2, 2, 2])


def test_foreign_backend_source_is_rejected_and_runtime_remains_usable():
    _require_cuda()
    with jt.flag_scope(use_cuda=1, lazy_execution=1, auto_flush_ops=0):
        x = jt.array(np.array([1, 2, 3], dtype=np.float32))
        bad = None
        try:
            with pytest.raises(RuntimeError, match="code backend source is for.*acl"):
                bad = jt.code(x.shape, x.dtype, [x], backend="acl",
                              cuda_src="this_is_not_valid_cuda_source();")
                bad.sync()
        finally:
            del bad
            gc.collect()
        result = x + 4
        jt.sync_all(True)
        assert result.location() == "device"
        np.testing.assert_array_equal(result.numpy(), [5, 6, 7])
