# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Dtype coverage for the MPI collectives.

Regression test for 6.B01: the per-operator dtype tables in
mpi_all_reduce_op.cc / mpi_reduce_op.cc / mpi_broadcast_op.cc mapped int64 to
``MPI_DOUBLE_INT``. That is the MAXLOC (double, int) pair -- 16 bytes wide with
padding, not an integer type -- so passing ``count = num`` made MPI read 2x past
the end of the buffer and the reduction produced garbage (in practice: all
zeros, on every rank).

These tests compare against numpy expectations computed from the rank index, so
they fail loudly on a wrong datatype rather than on a wrong-looking number.
"""
import unittest

import numpy as np

import jittor as jt
from _helpers.distributed import run_mpi_test

mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()


@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpiCollectiveDtypes(unittest.TestCase):
    """Every dtype the MPI collectives claim to support, checked numerically."""

    # (jittor dtype, numpy dtype). uint8 is kept small so the sum cannot wrap.
    CASES = [
        ("int32", "int32"),
        ("int64", "int64"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("uint8", "uint8"),
    ]

    def _payload(self, np_dtype, rank):
        # Distinct per rank, and large enough that a datatype whose element is
        # wider than the real one walks off the end of the buffer.
        return (np.arange(1, 33) * (rank + 1)).astype(np_dtype)

    def test_all_reduce(self):
        rank = mpi.world_rank()
        for jt_dtype, np_dtype in self.CASES:
            with self.subTest(dtype=jt_dtype):
                base = self._payload(np_dtype, rank)
                x = jt.array(base.copy(), dtype=jt_dtype)
                self.assertEqual(str(x.dtype), jt_dtype)
                got = x.mpi_all_reduce().data
                # sum over ranks of arange * (r+1) == arange * n(n+1)/2
                want = (np.arange(1, 33) * (n * (n + 1) // 2)).astype(np_dtype)
                np.testing.assert_array_equal(got, want)

    def test_broadcast(self):
        rank = mpi.world_rank()
        for jt_dtype, np_dtype in self.CASES:
            with self.subTest(dtype=jt_dtype):
                want = self._payload(np_dtype, 0)
                if rank == 0:
                    x = jt.array(want.copy(), dtype=jt_dtype)
                else:
                    x = jt.zeros([32], dtype=jt_dtype)
                got = x.mpi_broadcast(0).data
                np.testing.assert_array_equal(got, want)

    def test_broadcast_from_nonzero_root(self):
        # 6.B06 lives in var_broadcast, but the operator path takes root too and
        # must honour it for every dtype.
        if n < 2:
            self.skipTest("needs at least 2 ranks")
        root = n - 1
        rank = mpi.world_rank()
        for jt_dtype, np_dtype in self.CASES:
            with self.subTest(dtype=jt_dtype):
                want = self._payload(np_dtype, root)
                if rank == root:
                    x = jt.array(want.copy(), dtype=jt_dtype)
                else:
                    x = jt.zeros([32], dtype=jt_dtype)
                got = x.mpi_broadcast(root).data
                np.testing.assert_array_equal(got, want)

    def test_reduce(self):
        rank = mpi.world_rank()
        for jt_dtype, np_dtype in self.CASES:
            with self.subTest(dtype=jt_dtype):
                base = self._payload(np_dtype, rank)
                x = jt.array(base.copy(), dtype=jt_dtype)
                got = x.mpi_reduce(root=0).data
                if rank == 0:
                    want = (np.arange(1, 33) * (n * (n + 1) // 2)).astype(np_dtype)
                else:
                    want = np.zeros(32, dtype=np_dtype)
                np.testing.assert_array_equal(got, want)

    def test_var_all_reduce_matches_op(self):
        # jt.mpi.var_all_reduce keeps its own call into the same table; it must
        # agree with the operator for the same input.
        rank = mpi.world_rank()
        for jt_dtype, np_dtype in self.CASES:
            with self.subTest(dtype=jt_dtype):
                base = self._payload(np_dtype, rank)
                v = jt.array(base.copy(), dtype=jt_dtype)
                v.sync()
                mpi.var_all_reduce(v)
                want = (np.arange(1, 33) * (n * (n + 1) // 2)).astype(np_dtype)
                np.testing.assert_array_equal(v.data, want)

    def test_unsupported_dtype_raises(self):
        # A dtype with no MPI datatype must report it, not expand to nothing or
        # silently send the wrong width. Symmetric across ranks, so no deadlock.
        x = jt.array(np.ones(8), dtype="bool")
        with self.assertRaises(Exception):
            x.mpi_all_reduce().sync()


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiCollectiveDtypesEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(2, "test_mpi_dtypes")


if __name__ == "__main__":
    unittest.main()
