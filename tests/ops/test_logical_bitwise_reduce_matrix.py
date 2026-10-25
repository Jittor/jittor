# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

"""The published logical and bitwise reductions, over every dtype, on both devices.

KI-BACKEND-009 was a family of reductions that the CPU answered and CUDA could
not compile. Six operations are published -- ``logical_and`` (``all_``),
``logical_or`` (``any_``), ``logical_xor`` and the three bitwise folds -- and
they are legal for eleven dtypes, so there are 66 cells that have to agree
between the devices.

What did not compile, and why:

* A reduction's per-thread accumulator is rewritten by the parallel and
  atomic-tuner passes into ``a = a & bool(b)`` / ``a = a | bool(b)`` /
  ``a = a ^ bool(b)`` and fenced with a CUDA ``atomicAnd``/``atomicOr``/
  ``atomicXor``. CUDA only has those for 32- and 64-bit words. The 8-bit
  ``uint8``/``int8`` words and bool were already given packed-word CAS forms
  in ``cuda_atomic.h``; ``short``, ``unsigned short`` and ``unsigned long``
  (``uint64`` on LP64, which is not the ``unsigned long long`` CUDA takes)
  were not, so five dtypes' cells failed with "no instance of overloaded
  function ``atomicAnd``".
* A floating-point logical reduction ran in its input dtype. Its accumulator
  then reached ``float & bool`` -- "expression must have integral or unscoped
  enum type" -- and its flush would have needed a floating-point atomic AND.
  The answer to a logical reduction is a truth value: NumPy's ``logical_*.
  reduce`` returns bool for every input dtype and torch's ``any``/``all`` do
  too. ``reduce_dtype_infer`` now says bool for a floating-point input, so the
  accumulator is boolean and the existing bool atomics apply. Integer inputs
  keep their dtype, because the published ``any_``/``all_`` document an int32
  result.
* The bitwise folds over floating-point input have no meaning. They are
  refused at graph construction on *both* devices with a sentence naming the
  operation and the dtype (see ``test_bitwise_dtype_guard.py``), which is the
  resolution the ledger allows. This file checks the refusal in the same
  matrix so a future change cannot silently make one device accept it.

The tests execute the operation on the device they name and assert
``location()`` before reading a value -- ``numpy()`` migrates the result to the
host, so it must not be consulted first. ``forbid_backend_fallbacks`` then
holds the whole matrix to "no host execution".
"""

import unittest

import numpy as np

import jittor as jt

from _helpers import capability as _test_capability


LOGICAL = ("logical_and", "logical_or", "logical_xor")
BITWISE = ("bitwise_and", "bitwise_or", "bitwise_xor")
OPERATIONS = LOGICAL + BITWISE

NUMPY = {
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
}

FLOAT_DTYPES = ("float32", "float64")
DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
          "uint32", "uint64") + FLOAT_DTYPES

#: The values make the two families disagree: ``bitwise_or`` leaves 1|2 == 3,
#: but ``logical_or`` is True. A table that only used 0/1 would not tell them
#: apart, and a kernel that accidentally ran the other fold would pass.
VALUES = {
    "bool": [True, False, True, True],
    "int8": [1, 0, 2, -1],
    "int16": [1, 0, 2, -1],
    "int32": [1, 0, 2, -1],
    "int64": [1, 0, 2, -1],
    "uint8": [1, 0, 2, 1],
    "uint16": [1, 0, 2, 1],
    "uint32": [1, 0, 2, 1],
    "uint64": [1, 0, 2, 1],
    "float32": [1., 0., 2., -1.],
    "float64": [1., 0., 2., -1.],
}


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class _ReductionMatrix:

    device_flag = 0
    device_label = "cpu"

    def _dev(self):
        return jt.flag_scope(use_cuda=self.device_flag)

    def _var(self, values, dtype):
        # jt.array narrows int64/float64 unless asked not to; the 64-bit cells
        # of this table are about the 64-bit reduction, not about the default.
        with jt.flag_scope(auto_convert_64_to_32=0):
            var = jt.array(np.asarray(values, dtype=dtype))
        self.assertEqual(str(var.dtype), dtype)
        return var

    def _expected_dtype(self, dtype, operation):
        # ReduceOp maps a bool input to int32 before the dtype is inferred.
        if dtype == "bool":
            return "int32"
        # A logical fold over a float answers a truth value on both devices.
        if operation in LOGICAL and dtype in FLOAT_DTYPES:
            return "bool"
        return dtype

    def _check_cell(self, dtype, operation):
        values = np.asarray(VALUES[dtype], dtype=dtype)
        with self._dev():
            var = self._var(values, dtype)
            result = var.reduce(operation, 0)
            result.sync()
            # Read the location *before* numpy(): the copy to the host moves
            # the result, so a CPU fallback would otherwise look like a pass.
            self.assertEqual(result.location(), self.device_label)
            self.assertEqual(str(result.dtype),
                             self._expected_dtype(dtype, operation))
            got = np.asarray(result.numpy())
        want = NUMPY[operation].reduce(values)
        np.testing.assert_array_equal(
            got.astype(bool), np.asarray(want).astype(bool),
            err_msg="%s over %s" % (operation, dtype))

    def _check_refusal(self, dtype, operation):
        """A bitwise fold over a float is refused, in words, on both devices."""
        with self._dev():
            with self.assertRaises(RuntimeError) as caught:
                self._var(VALUES[dtype], dtype).reduce(operation, 0).sync()
        message = str(caught.exception)
        self.assertIn(operation, message)
        self.assertIn(dtype, message)
        # The compiler wall is also a RuntimeError; this one has no transcript.
        self.assertNotIn("error:", message)
        self.assertNotIn("Compile fused operator", message)

    def test_the_whole_operation_dtype_matrix(self):
        for dtype in DTYPES:
            for operation in OPERATIONS:
                with self.subTest(dtype=dtype, operation=operation):
                    if dtype in FLOAT_DTYPES and operation in BITWISE:
                        self._check_refusal(dtype, operation)
                    else:
                        self._check_cell(dtype, operation)

    def test_no_cell_falls_back_to_the_host(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with self._dev(), forbid_backend_fallbacks():
            for dtype in DTYPES:
                for operation in OPERATIONS:
                    if dtype in FLOAT_DTYPES and operation in BITWISE:
                        continue
                    values = np.asarray(VALUES[dtype], dtype=dtype)
                    self._var(values, dtype).reduce(operation, 0).sync()

    def test_the_same_matrix_reduces_rows(self):
        """``dim=1`` reaches the same accumulators with several outputs."""
        rng = np.random.RandomState(0)
        for dtype in DTYPES:
            if dtype == "bool":
                raw = rng.rand(17, 513) < 0.5
            elif dtype in FLOAT_DTYPES:
                raw = (rng.rand(17, 513) * 6 - 3).astype(dtype)
            elif dtype.startswith("uint"):
                raw = rng.randint(0, 5, size=(17, 513)).astype(dtype)
            else:
                raw = rng.randint(-3, 3, size=(17, 513)).astype(dtype)
            with jt.flag_scope(auto_convert_64_to_32=0):
                var = jt.array(raw)
            for operation in OPERATIONS:
                with self.subTest(dtype=dtype, operation=operation):
                    if dtype in FLOAT_DTYPES and operation in BITWISE:
                        continue
                    with self._dev():
                        result = var.reduce(operation, 1)
                        result.sync()
                        self.assertEqual(result.location(), self.device_label)
                        got = np.asarray(result.numpy())
                    want = NUMPY[operation].reduce(raw, axis=1)
                    np.testing.assert_array_equal(got.astype(bool),
                                                  np.asarray(want).astype(bool))

    def test_a_wide_reduction_keeps_narrow_neighbours(self):
        """8/16-bit outputs share an aligned 32-bit word.

        The narrow atomics update one byte or halfword of that word with a CAS
        loop, so a wrong mask would corrupt a neighbouring output. Many rows
        and a dense value set make that neighbour occupied and contended.
        """
        rng = np.random.RandomState(1)
        for dtype in ("uint8", "int8", "uint16", "int16", "uint32", "uint64"):
            raw = np.tile(np.array(VALUES[dtype], dtype=dtype), 2048)
            rng.shuffle(raw)
            with jt.flag_scope(auto_convert_64_to_32=0):
                var = jt.array(raw)
            for operation in OPERATIONS:
                with self.subTest(dtype=dtype, operation=operation):
                    with self._dev():
                        result = var.reduce(operation, 0)
                        result.sync()
                        self.assertEqual(result.location(), self.device_label)
                        got = np.asarray(result.numpy())
                    want = NUMPY[operation].reduce(raw)
                    np.testing.assert_array_equal(got.astype(bool),
                                                  np.asarray(want).astype(bool))

    def test_half_logical_folds_answer_a_truth_value(self):
        # Not part of the 66 published cells, but the same bool-result rule:
        # a half logical accumulator is boolean, so the bool atomics apply.
        values = np.array([[0, 0, 0], [0, 2, 0], [1, 1, 1]], "float32")
        with self._dev():
            for dtype in ("float16", "bfloat16"):
                var = jt.array(values).cast(dtype)
                reference = var.float32().numpy()
                for operation in LOGICAL:
                    with self.subTest(dtype=dtype, operation=operation):
                        result = var.reduce(operation, 1)
                        result.sync()
                        self.assertEqual(result.location(), self.device_label)
                        self.assertEqual(str(result.dtype), "bool")
                        np.testing.assert_array_equal(
                            np.asarray(result.numpy()).astype(bool),
                            NUMPY[operation].reduce(reference, axis=1).astype(bool))

    def test_public_any_and_all_are_unchanged(self):
        # The raw folds above now answer bool for a float; the public helpers
        # already cast, so their contract (bool, NaN true) must not move.
        with self._dev():
            for dtype in ("float16", "bfloat16", "float32", "float64"):
                var = jt.array(np.array([0., 2., 0.], "float32")).cast(dtype)
                self.assertEqual(str(var.any().dtype), "bool")
                self.assertEqual(str(var.all().dtype), "bool")
                self.assertTrue(bool(var.any().numpy()))
                self.assertFalse(bool(var.all().numpy()))
                nan = jt.array(np.array([np.nan, 0.], "float32")).cast(dtype)
                self.assertTrue(bool(nan.any().numpy()))


class TestLogicalBitwiseReduceCpu(_ReductionMatrix, unittest.TestCase):
    device_flag = 0
    device_label = "cpu"


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestLogicalBitwiseReduceCuda(_ReductionMatrix, unittest.TestCase):
    device_flag = 1
    device_label = "device"


if __name__ == "__main__":
    unittest.main()
