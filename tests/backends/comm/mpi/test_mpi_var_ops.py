# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""In-place MPI helpers on a VarHolder: var_broadcast / var_reduce / var_all_reduce.

Regression test for 6.B06: var_broadcast took a ``root`` argument and then
passed a hardcoded 0 to MPI_Bcast. Any broadcast from a non-zero root silently
did the wrong thing -- every rank got rank 0's buffer, and the real root's data
was overwritten -- with no error anywhere.
"""
import unittest

import numpy as np

import jittor as jt
from _helpers.distributed import run_mpi_test

mpi = jt.compile_extern.mpi
if mpi:
    n = mpi.world_size()


@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpiVarOps(unittest.TestCase):

    def _payload(self, rank):
        return (np.arange(1, 33) * (rank + 1)).astype("float32")

    def test_var_broadcast_from_each_root(self):
        rank = mpi.world_rank()
        for root in range(n):
            with self.subTest(root=root):
                want = self._payload(root)
                # Only the root holds the payload; everyone else holds a value
                # that is distinguishable from both the payload and from zero,
                # so a broadcast that quietly used the wrong root is visible.
                start = want.copy() if rank == root else np.full(32, -1.0, "float32")
                v = jt.array(start, dtype="float32")
                v.sync()
                mpi.var_broadcast(v, root)
                np.testing.assert_array_equal(v.data, want)

    def test_var_broadcast_rejects_out_of_range_root(self):
        v = jt.array(np.zeros(4, "float32"), dtype="float32")
        v.sync()
        with self.assertRaises(Exception):
            mpi.var_broadcast(v, n + 5)

    def test_var_reduce_to_each_root(self):
        rank = mpi.world_rank()
        total = (np.arange(1, 33) * (n * (n + 1) // 2)).astype("float32")
        for root in range(n):
            with self.subTest(root=root):
                v = jt.array(self._payload(rank), dtype="float32")
                v.sync()
                mpi.var_reduce(v, root)
                if rank == root:
                    np.testing.assert_allclose(v.data, total, rtol=1e-5)
                else:
                    # Non-root buffers are not defined by MPI_Reduce; only the
                    # root's result is asserted.
                    pass

    def test_var_all_reduce(self):
        rank = mpi.world_rank()
        v = jt.array(self._payload(rank), dtype="float32")
        v.sync()
        mpi.var_all_reduce(v)
        total = (np.arange(1, 33) * (n * (n + 1) // 2)).astype("float32")
        np.testing.assert_allclose(v.data, total, rtol=1e-5)


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiVarOpsEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(3, "test_mpi_var_ops")


if __name__ == "__main__":
    unittest.main()
