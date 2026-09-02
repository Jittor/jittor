# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuSPARSE SpMM must compute in the dtype it was handed.

Both spmm ops worked out the cuSPARSE data type of every operand and then
passed ``CUDA_R_32F`` as the compute type with ``float`` alpha/beta anyway.
``cusparseSpMM`` reads the scalars as raw memory of the compute type, so an
fp64 product was computed in single precision *and* scaled by whatever eight
bytes began at ``&alpha``.  The signature was already in the tree: the fp64 CSR
case in ``test_cusparse_op.py`` had been commented out rather than fixed.

The COO variant additionally called ``cusparseSpMM`` with a NULL external
buffer, its buffer-size query commented out, while the CSR variant next door
did ask -- undefined behaviour on any algorithm that wants a buffer.  Before
the fix the fp64 COO case below did not fail, it *aborted the interpreter*,
which is why running it took the whole pytest session with it.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers.logs import find_log_with_re


cusparse_ops = None


def setUpModule():
    global cusparse_ops
    if not jt.has_cuda:
        raise unittest.SkipTest("No CUDA found")
    cusparse_ops = getattr(jt.compile_extern, "cusparse_ops", None)
    if cusparse_ops is None:
        raise unittest.SkipTest("cuSPARSE support is unavailable")


def _as(value, dtype):
    """A jittor array that really holds ``dtype``.

    ``jt.array(np.ones(4, "float64"))`` silently produces float32, so an fp64
    test written the obvious way compares an fp64 reference against an fp32
    computation and reports a difference the backend is not responsible for.
    """
    var = jt.array(value.astype(dtype)).cast(dtype)
    assert str(var.dtype) == dtype, (str(var.dtype), dtype)
    return var


class TestCusparseDtype(unittest.TestCase):
    """Sparse 6x6 times dense 6x4, against a dense numpy product.

    The reference is built from the values *jittor actually holds*, read back
    after the cast, so a dtype the array layer narrows on the way in cannot be
    mistaken for a backend error.
    """

    def _csr_case(self, dtype, tol):
        rows = cols = 6
        rng = np.random.RandomState(0)
        row_offset = np.array([0, 2, 3, 5, 7, 8, 10], dtype="int32")
        col_indices = np.array([0, 3, 1, 2, 4, 0, 5, 2, 1, 4], dtype="int32")
        values = rng.rand(10) + 0.5
        x = rng.rand(cols, 4) + 0.5

        with jt.flag_scope(use_cuda=1, lazy_execution=0):
            jx = _as(x, dtype)
            jv = _as(values, dtype)
            dense = np.zeros((rows, cols), dtype="float64")
            held = jv.cast("float64").numpy()
            for row in range(rows):
                for k in range(row_offset[row], row_offset[row + 1]):
                    dense[row, col_indices[k]] += held[k]
            want = dense @ jx.cast("float64").numpy()

            output = jt.zeros((rows, 4), dtype=dtype)
            with jt.log_capture_scope(log_silent=1, log_v=0,
                                      log_vprefix="cusparse_spmmcsr=100") as log:
                cusparse_ops.cusparse_spmmcsr(
                    output, jx, jt.array(col_indices), jv, jt.array(row_offset),
                    rows, cols, False, False).fetch_sync()
            got = output.data.astype("float64")
        np.testing.assert_allclose(got, want, atol=tol, rtol=tol)
        return log

    def _coo_case(self, dtype, tol):
        rows = cols = 6
        rng = np.random.RandomState(1)
        row_indices = np.array([0, 0, 1, 2, 2, 3, 4, 5, 5, 5], dtype="int32")
        col_indices = np.array([1, 4, 2, 0, 5, 3, 1, 0, 2, 4], dtype="int32")
        values = rng.rand(10) + 0.5
        x = rng.rand(cols, 4) + 0.5

        with jt.flag_scope(use_cuda=1, lazy_execution=0):
            jx = _as(x, dtype)
            jv = _as(values, dtype)
            dense = np.zeros((rows, cols), dtype="float64")
            for value, r, c in zip(jv.cast("float64").numpy(), row_indices, col_indices):
                dense[r, c] += value
            want = dense @ jx.cast("float64").numpy()

            output = jt.zeros((rows, 4), dtype=dtype)
            with jt.log_capture_scope(log_silent=1, log_v=0,
                                      log_vprefix="cusparse_spmmcoo=100") as log:
                cusparse_ops.cusparse_spmmcoo(
                    output, jx, jt.array(row_indices), jt.array(col_indices), jv,
                    rows, cols, False, False).fetch_sync()
            got = output.data.astype("float64")
        np.testing.assert_allclose(got, want, atol=tol, rtol=tol)
        return log

    def test_csr_float32(self):
        self._csr_case("float32", 1e-5)

    def test_csr_float64(self):
        # 1e-10 is the point: it is far below anything a single-precision
        # accumulation can reach, so this tolerance is what distinguishes
        # "computed in fp64" from "computed in fp32 and rounded back".
        self._csr_case("float64", 1e-10)

    def test_coo_float32(self):
        self._coo_case("float32", 1e-5)

    def test_coo_float64(self):
        self._coo_case("float64", 1e-10)


    # ---- The numbers agreeing is not proof the right path was taken ----
    #
    # fp64 with an fp32 compute type is wrong by a lot, so the parity cases
    # above do catch it. The buffer is a different matter: the COO variant
    # passed NULL and produced correct results on every shape anyone had
    # tried, because the algorithm cuSPARSE picked for them happened not to
    # need one. So read the choice out of the op rather than inferring it.

    def _selection(self, dtype, coo):
        tol = 1e-5 if dtype == "float32" else 1e-10
        log = self._coo_case(dtype, tol) if coo else self._csr_case(dtype, tol)
        found = find_log_with_re(
            log,
            r"select: compute=(\S+) buffer_bytes=(\d+) buffer_from=(\S+)")
        self.assertTrue(found, "no selection log captured")
        return found[-1]

    def test_compute_type_follows_dtype(self):
        for coo in (False, True):
            self.assertEqual(self._selection("float32", coo)[0], "CUDA_R_32F")
            self.assertEqual(self._selection("float64", coo)[0], "CUDA_R_64F")

    def test_external_buffer_comes_from_the_temp_allocator(self):
        """Not cudaMalloc/cudaFree per call.

        cudaFree synchronizes the whole device, so the old pair cost a
        full-device sync on every SpMM. Whether a buffer is needed at all is
        cuSPARSE's decision and shape-dependent, so the assertion is
        conditional on the size it asked for -- but "asked for none" is itself
        only observable because the query now runs at all.
        """
        seen_any = False
        for coo in (False, True):
            for dtype in ("float32", "float64"):
                _, size, origin = self._selection(dtype, coo)
                if int(size) > 0:
                    seen_any = True
                    self.assertNotEqual(origin, "none")
                    self.assertIn("temp", origin.lower())
                else:
                    self.assertEqual(origin, "none")
        self.assertTrue(seen_any,
            "no case needed an external buffer, so nothing was proved about "
            "where it comes from; pick shapes that do")


if __name__ == "__main__":
    unittest.main()
