"""Math-policy switches distinguish both standalone and fused CUDA kernels."""

from pathlib import Path

import jittor as jt
import numpy as np
import pytest


@pytest.mark.skipif(not jt.has_cuda, reason="CUDA is required")
@pytest.mark.parametrize("fused", [False, True])
def test_cuda_math_policy_has_distinct_compiled_keys(fused):
    filenames = {}
    with jt.flag_scope(use_cuda=1, enable_tuner=0):
        x = jt.array(np.arange(16, dtype=np.float32))
        x.sync()
        for policy in ("default", "strict", "backend", "default"):
            with jt.flag_scope(cuda_kernel_math=policy):
                with jt.profile_scope() as report:
                    if fused:
                        y = x * 1.25 + 2
                    else:
                        y = jt.code(x.shape, x.dtype, [x], cuda_src=r'''
                            __global__ static void kernel(@ARGS_DEF) {
                                @PRECALC
                                int i = threadIdx.x;
                                if (i < in0_shape0) @out(i) = @in0(i) * 1.25f + 2;
                            }
                            kernel<<<1, 32>>>(@ARGS);
                        ''')
                    np.testing.assert_allclose(y.numpy(), np.arange(16) * 1.25 + 2)
                    del y
                source_files = {
                    cell for row in report[1:] for cell in row
                    if isinstance(cell, str) and cell.endswith(".cc")
                    and Path(cell).is_file()
                }
                matching = {
                    path for path in source_files
                    if f"#define JIT_cuda_math {policy}" in Path(path).read_text()
                }
                assert matching, (policy, report)
                if policy in filenames:
                    assert matching == filenames[policy]
                filenames[policy] = matching
    assert filenames["default"].isdisjoint(filenames["strict"])
    assert filenames["default"].isdisjoint(filenames["backend"])
    assert filenames["strict"].isdisjoint(filenames["backend"])


@pytest.mark.skipif(not jt.has_cuda, reason="CUDA is required")
@pytest.mark.parametrize("fused", [False, True])
def test_strict_math_changes_actual_cuda_rounding(fused):
    with jt.flag_scope(use_cuda=1, enable_tuner=0):
        a = jt.array(np.full(16, 1 + 2 ** -13, dtype=np.float32))
        b = jt.array(np.full(16, 1 - 2 ** -13, dtype=np.float32))
        c = jt.array(np.full(16, -1, dtype=np.float32))
        jt.sync_all(True)
        results = {}
        for policy in ("default", "strict"):
            with jt.runtime.scope(cuda_kernel_math=policy):
                if fused:
                    y = a * b + c
                else:
                    y = jt.code(a.shape, a.dtype, [a, b, c], cuda_src=r'''
                        __global__ static void kernel(@ARGS_DEF) {
                            @PRECALC
                            int i = threadIdx.x;
                            if (i < in0_shape0) @out(i) = @in0(i) * @in1(i) + @in2(i);
                        }
                        kernel<<<1, 32>>>(@ARGS);
                    ''')
                y.sync()
                assert y.location() == "device"
                results[policy] = y.numpy().copy()
                del y
        np.testing.assert_array_equal(results["strict"], np.zeros(16, dtype=np.float32))
        # The default startup flags allow NVCC's multiply-add contraction.
        if "--fmad=false" not in jt.config.nvcc_flags:
            np.testing.assert_array_equal(results["default"],
                                          np.full(16, -(2 ** -26), dtype=np.float32))
