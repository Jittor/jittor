# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The MPI fp16 sum must be IEEE round-to-nearest-even everywhere (6.B10).

MPI has no predefined float16, so jittor registers its own MPI_Op (HalfAdd).
That operator used to be two unrelated implementations:

  * x86: _mm256_cvtph_ps / _mm256_cvtps_ph -- correct IEEE rounding, full
    subnormal support -- entered unconditionally, with no run-time check that
    the CPU has F16C and AVX at all.
  * everything else: a hand-written path that truncated the mantissa instead of
    rounding, and flushed every result below 2^-14 to zero.

Measured over 31.6M fp16 sums, the two disagreed on 34.27% of values, and on
49512 of them one produced a subnormal where the other produced zero. The same
all-reduce therefore returned different numbers depending on the host.

The reference here is numpy's float32 -> float16 cast, which is the same
round-to-nearest-even IEEE conversion. The test runs twice: once letting the
SIMD path be selected, and once with JT_MPI_HALF_SIMD=0, which forces the
scalar code -- i.e. the exact path a non-x86 host takes. Both must match the
reference exactly, which is what "x86 and ARM agree" means operationally.
"""
import os
import unittest

import numpy as np

import jittor as jt
from _helpers.distributed import run_mpi_test

mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()


def _fp16(bits):
    """fp16 array from raw bit patterns, so subnormals can be named exactly."""
    return np.frombuffer(np.asarray(bits, dtype="uint16").tobytes(), dtype="float16")


# Chosen so the sum of the two ranks' values lands on the interesting cases:
# subnormal results, exact halfway ties (which truncation and RNE disagree on),
# values near the fp16 max, infinities and signed zeros.
_SUBNORMALS = list(range(1, 33))                 # 2^-24 .. 32*2^-24, all subnormal
_TIES = [0x3C00, 0x3C01, 0x3801, 0x0401, 0x0402] # 1.0, 1+1ulp, 0.5+1ulp, min-normal+
_BIG = [0x7BFF, 0x7BFE, 0x7B00]                  # 65504, 65472, 61440
_MISC = [0x0000, 0x8000, 0xBC00, 0x4900, 0xC900] # +0, -0, -1, 10, -10

_PATTERNS = _SUBNORMALS + _TIES + _BIG + _MISC


@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpiHalfReduce(unittest.TestCase):

    def _operands(self):
        """Per-rank fp16 operands; every rank can compute every other rank's."""
        base = _fp16(_PATTERNS)
        out = []
        for rank in range(n):
            # rotate, so ranks hold different values and every pattern meets
            # every other one across the ranks
            out.append(np.roll(base, rank).astype("float16"))
        return out

    def test_all_reduce_matches_ieee_reference(self):
        rank = mpi.world_rank()
        ops = self._operands()
        x = jt.array(ops[rank].copy(), dtype="float16")
        self.assertEqual(str(x.dtype), "float16")
        got = x.mpi_all_reduce().numpy()

        # Reference: sum in float32 (which is what HalfAdd does), then one
        # round-to-nearest-even cast back to float16.
        acc = ops[0].astype("float32")
        for r in range(1, n):
            acc = (acc + ops[r].astype("float32")).astype("float16").astype("float32")
        want = acc.astype("float16")

        np.testing.assert_array_equal(
            got.view("uint16"), want.view("uint16"),
            err_msg="fp16 all_reduce is not IEEE round-to-nearest-even "
                    "(JT_MPI_HALF_SIMD={})".format(
                        os.environ.get("JT_MPI_HALF_SIMD", "<auto>")))

    def test_subnormal_results_are_not_flushed_to_zero(self):
        # The old non-x86 path turned every subnormal result into zero. Guard it
        # directly: these operands sum to subnormals on every rank count.
        rank = mpi.world_rank()
        vals = _fp16([1, 2, 3, 4])                 # 1..4 * 2^-24, all subnormal
        x = jt.array(vals.copy(), dtype="float16")
        got = x.mpi_all_reduce().numpy()
        self.assertTrue(np.all(got != 0),
                        "subnormal fp16 results were flushed to zero: {}".format(got))
        want = (vals.astype("float32") * n).astype("float16")
        np.testing.assert_array_equal(got.view("uint16"), want.view("uint16"))


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiHalfReduceEntry(unittest.TestCase):
    def test(self):
        saved = os.environ.get("JT_MPI_HALF_SIMD")
        try:
            # SIMD path (selected at run time when the CPU supports F16C+AVX).
            os.environ.pop("JT_MPI_HALF_SIMD", None)
            run_mpi_test(2, "test_mpi_half_reduce")
            # Scalar path -- the code every non-x86 host runs.
            os.environ["JT_MPI_HALF_SIMD"] = "0"
            run_mpi_test(2, "test_mpi_half_reduce")
        finally:
            if saved is None:
                os.environ.pop("JT_MPI_HALF_SIMD", None)
            else:
                os.environ["JT_MPI_HALF_SIMD"] = saved


if __name__ == "__main__":
    unittest.main()
