# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Var.item()`` must return the stored value for every dtype.

``item()`` copies ``dtype.dsize()`` bytes into an ``ItemData`` and hands the
8-byte payload to the Python converter.  Two independent defects made the
unsigned dtypes return garbage:

  1. ``ItemData`` is a POD and was default-initialized, so the bytes above
     ``dsize`` kept whatever was on the stack;
  2. the converter listed only the signed and floating dtypes and fell through
     to ``PyLong_FromLongLong``, which reads all 8 bytes.

Together they made ``jt.array(np.uint8([200])).item()`` return a random large
integer -- a silent wrong value with no exception.  Both are asserted here.

Values are pinned with ``jt.array(x, dtype=...)`` because a bare ``jt.array``
silently narrows int64 to int32 and float64 to float32.

Run::  python -m pytest tests/core/test_item.py
"""

import unittest

import numpy as np

import jittor as jt


#: dtype -> values that must survive a round trip through ``item()``.
#: The unsigned entries deliberately include values whose top bit is set, which
#: is where a signed reinterpretation would show up as a negative number.
INTEGER_CASES = {
    "uint8": [0, 1, 127, 200, 255],
    "uint16": [0, 200, 32768, 65535],
    "uint32": [0, 200, 2147483648, 4294967295],
    "uint64": [0, 200, 2147483648, 4294967295, 9223372036854775808],
    "int8": [-128, -1, 0, 127],
    "int16": [-32768, -1, 0, 200, 32767],
    "int32": [-2147483648, -1, 0, 200, 2147483647],
    "int64": [-9223372036854775808, -1, 0, 200, 9223372036854775807],
}

FLOAT_CASES = {
    "float16": [0.0, -1.5, 200.0],
    "float32": [0.0, -1.5, 200.0, 3.5e30],
    "float64": [0.0, -1.5, 200.0, 3.5e30],
}


class TestItem(unittest.TestCase):
    def test_integer_dtypes_round_trip(self):
        for dtype, values in INTEGER_CASES.items():
            for value in values:
                array = np.array([value], dtype=dtype)
                got = jt.array(array, dtype=dtype).item()
                self.assertIsInstance(got, int, (dtype, value))
                self.assertEqual(got, int(array[0]), (dtype, value))

    def test_float_dtypes_round_trip(self):
        for dtype, values in FLOAT_CASES.items():
            for value in values:
                array = np.array([value], dtype=dtype)
                got = jt.array(array, dtype=dtype).item()
                self.assertIsInstance(got, float, (dtype, value))
                self.assertEqual(got, float(array[0]), (dtype, value))

    def test_bool_dtype_round_trip(self):
        for value in (False, True):
            got = jt.array(np.array([value], dtype="bool"), dtype="bool").item()
            self.assertIsInstance(got, bool, value)
            self.assertEqual(got, value)

    def test_item_is_deterministic_across_calls(self):
        # The uninitialized high bytes made repeated reads of the *same* value
        # differ from each other whenever the stack underneath changed.
        for dtype in ("uint8", "uint16", "uint32"):
            var = jt.array(np.array([200], dtype=dtype), dtype=dtype)
            first = var.item()
            self.assertEqual([first] * 8, [var.item() for _ in range(8)], dtype)

    def test_item_matches_numpy_for_unsigned_reductions(self):
        # A reduced value takes the same path but is produced by a kernel
        # rather than by the host copy, so the high bytes come from a
        # different allocation.
        data = np.array([200, 55], dtype="uint8")
        var = jt.array(data, dtype="uint8")
        self.assertEqual(var.max().item(), 200)
        self.assertEqual(var.min().item(), 55)


@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestItemCuda(TestItem):
    def setUp(self):
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()
