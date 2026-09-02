# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``use_tensorcore`` must pick the tensor-op algorithm, not turn it off.

``cublasGemmEx`` takes the compute type and the algorithm as two separate
arguments, and both ops used to select them with opposite senses: the compute
type followed ``use_tensorcore`` while the algorithm was inverted, so enabling
tensor cores asked cuBLAS for ``CUBLAS_GEMM_DEFAULT`` and disabling them asked
for ``CUBLAS_GEMM_DEFAULT_TENSOR_OP``.

Nothing about that is visible in the output values -- on CUDA >= 11 the
tensor-op algorithm hint is advisory -- so the choice is read back from the
op's own log line rather than inferred from numerics.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers.logs import find_log_with_re


#: matches the line each cuBLAS gemm op emits just before it calls cuBLAS.
_SELECT_RE = r"algo select: use_tensorcore=(\S+) computeType=(\S+) algo=(\S+)"

DEFAULT = "CUBLAS_GEMM_DEFAULT"
TENSOR_OP = "CUBLAS_GEMM_DEFAULT_TENSOR_OP"


def _capture(prefix, build):
    """Run ``build`` and return the (tensorcore, compute, algo) it selected."""
    with jt.log_capture_scope(log_silent=1, log_v=0,
                              log_vprefix="%s=100" % prefix) as raw_log:
        build().sync()
    found = find_log_with_re(raw_log, _SELECT_RE)
    assert found, "no %s selection log captured" % prefix
    return found[-1]


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCublasTensorcoreAlgo(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.use_tensorcore,
                       jt.flags.cuda_allow_tf32)
        jt.flags.use_cuda = 1
        jt.flags.cuda_allow_tf32 = 0

    def tearDown(self):
        jt.sync_all()
        (jt.flags.use_cuda, jt.flags.use_tensorcore,
         jt.flags.cuda_allow_tf32) = self._saved

    def _matmul(self, dtype):
        a = jt.random((32, 48)).cast(dtype)
        b = jt.random((48, 64)).cast(dtype)
        return lambda: jt.matmul(a, b)

    def _batched(self, dtype):
        a = jt.random((3, 32, 48)).cast(dtype)
        b = jt.random((3, 48, 64)).cast(dtype)
        return lambda: jt.matmul(a, b)

    def _check(self, prefix, make, dtype, expected):
        for tensorcore, compute, algo in expected:
            jt.flags.use_tensorcore = tensorcore
            got_tc, got_compute, got_algo = _capture(prefix, make(dtype))
            self.assertEqual(int(got_tc), tensorcore)
            self.assertEqual(got_algo, algo,
                             "%s %s use_tensorcore=%d picked %s" %
                             (prefix, dtype, tensorcore, got_algo))
            self.assertEqual(got_compute, compute,
                             "%s %s use_tensorcore=%d picked %s" %
                             (prefix, dtype, tensorcore, got_compute))

    # ---- float16: the algorithm follows use_tensorcore ------------------
    FP16 = [
        (0, "CUBLAS_COMPUTE_32F", DEFAULT),
        (1, "CUBLAS_COMPUTE_16F", TENSOR_OP),
        (2, "CUBLAS_COMPUTE_16F", TENSOR_OP),
        (3, "CUBLAS_COMPUTE_16F", TENSOR_OP),
    ]

    # ---- bfloat16 -------------------------------------------------------
    BF16 = [
        (0, "CUBLAS_COMPUTE_32F", DEFAULT),
        (1, "CUBLAS_COMPUTE_32F_FAST_16BF", TENSOR_OP),
        (2, "CUBLAS_COMPUTE_32F_FAST_16BF", TENSOR_OP),
        (3, "CUBLAS_COMPUTE_32F_FAST_16BF", TENSOR_OP),
    ]

    # ---- float32: only the compute type moves, the algorithm stays ------
    FP32 = [
        (0, "CUBLAS_COMPUTE_32F", DEFAULT),
        (1, "CUBLAS_COMPUTE_32F_FAST_TF32", DEFAULT),
        (2, "CUBLAS_COMPUTE_32F_FAST_16BF", DEFAULT),
        (3, "CUBLAS_COMPUTE_32F_FAST_16F", DEFAULT),
    ]

    def test_matmul_float16(self):
        self._check("cublas_matmul", self._matmul, "float16", self.FP16)

    def test_matmul_bfloat16(self):
        self._check("cublas_matmul", self._matmul, "bfloat16", self.BF16)

    def test_matmul_float32(self):
        self._check("cublas_matmul", self._matmul, "float32", self.FP32)

    def test_batched_matmul_float16(self):
        self._check("cublas_batched_matmul", self._batched, "float16",
                    self.FP16)

    def test_batched_matmul_bfloat16(self):
        self._check("cublas_batched_matmul", self._batched, "bfloat16",
                    self.BF16)

    def test_batched_matmul_float32(self):
        self._check("cublas_batched_matmul", self._batched, "float32",
                    self.FP32)

    def test_values_stay_correct_under_every_setting(self):
        """The algorithm hint must not change what the gemm computes."""
        a = np.random.RandomState(0).randn(32, 48).astype("float32")
        b = np.random.RandomState(1).randn(48, 64).astype("float32")
        want = a @ b
        for tensorcore in (0, 1, 2, 3):
            jt.flags.use_tensorcore = tensorcore
            got = jt.matmul(jt.array(a), jt.array(b)).numpy()
            # use_tensorcore >= 1 deliberately lowers the fp32 accumulate, so
            # the tolerance has to admit tf32/bf16/fp16 compute here.
            tol = 1e-4 if tensorcore == 0 else 3e-1
            np.testing.assert_allclose(got, want, atol=tol, rtol=tol)


if __name__ == "__main__":
    unittest.main()
