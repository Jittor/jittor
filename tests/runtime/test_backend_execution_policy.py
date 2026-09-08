"""Default CPU/CUDA math keeps user controls after backend policy extraction."""

import jittor as jt
import numpy as np


def test_native_reduction_policy_keeps_user_amp_and_parallel_controls():
    with jt.flag_scope(use_cuda=0, amp_reg=0, use_parallel_op_compiler=2,
                       use_cuda_host_allocator=0):
        values = jt.array([1, 2, 3, 4]).float16()
        total = values.sum()
        mean = values.mean()
        product = values.prod()
        np.testing.assert_allclose(total.numpy(), 10)
        np.testing.assert_allclose(mean.numpy(), 2.5)
        np.testing.assert_allclose(product.numpy(), 24)
        assert str(total.dtype) == "float16" and str(mean.dtype) == "float16"
        assert str(product.dtype) == "float32"
        assert jt.flags.amp_reg == 0
        assert jt.flags.use_parallel_op_compiler == 2
        assert jt.flags.use_cuda_host_allocator == 0


def test_explicit_keep_reduce_policy_remains_a_user_control_on_cpu():
    with jt.flag_scope(use_cuda=0, amp_reg=jt.amp_flags.keep_reduce):
        values = jt.array([1, 2, 3, 4]).float16()
        result = values.prod()
        assert str(result.dtype) == "float16"
        np.testing.assert_allclose(result.numpy(), 24)
        assert jt.flags.amp_reg == jt.amp_flags.keep_reduce
