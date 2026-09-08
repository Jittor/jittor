"""Moved native helpers still link into CPU and CUDA operator execution."""

from _helpers import capability as _test_capability

import jittor as jt
import numpy as np
import pytest


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_relocated_nan_checker_links_and_checks_device_storage(use_cuda):
    if use_cuda and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    with jt.runtime.scope(use_cuda=use_cuda):
        value = jt.array(np.arange(16, dtype=np.float32)) + 1
        value.sync()
        result = jt.code(
            [1], "int32", [value],
            cpu_header='#include "debug/nan_checker.h"',
            cuda_header='#include "debug/nan_checker.h"',
            cpu_src='@out(0) = jittor::check_nan(*this->inputs().begin(), this);',
            cuda_src='''
                int ok = jittor::check_nan(*this->inputs().begin(), this);
                ASSERT(cudaMemcpy(out0_p, &ok, sizeof(ok), cudaMemcpyHostToDevice) == cudaSuccess);
            ''',
        )
        result.sync()
        if use_cuda:
            assert result.location() == "device"
        assert result.item() == 1


def test_relocated_cpu_math_and_runtime_ring_buffer():
    with jt.runtime.scope(use_cuda=0):
        actual = jt.erfinv(jt.array([0.0, 0.5])).numpy()
        np.testing.assert_allclose(actual, [0.0, 0.4769362762], rtol=1e-6, atol=1e-7)
        queue = jt.core.RingBuffer(1024)
        queue.push(42)
        assert queue.pop() == 42
