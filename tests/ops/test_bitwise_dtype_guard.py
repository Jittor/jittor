# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Bitwise operations on a float say so, instead of failing to compile.

A bitwise operation has no meaning for a float, and the answer to asking for
one should be a sentence. What used to happen is that the request was accepted,
a kernel was generated, and the mistake was noticed by g++ several layers down:

    parallel_compiler.cc:339: Error happened during compilation:
     Compile fused operator(0/1)failed:[Op(3:...,unary.bitwise_not->4),]
    Reason: log.cc:1011: compilation failed with exit code 256.
      src/ops/unary_op.cc:1091:49: error: wrong type argument to bit-complement
       1091 |         index_t xi = i;

994 characters that never say "dtype", and the quoted line is not the line: the
generated kernel's numbering does not map back to the template, so `index_t xi
= i;` is simply the text that happens to live at that offset of the source
file. The reduction path gave the same thing pointed at `Ty rcount = y->num*1.0
/ x->num;`.

`BinaryOp` already rejected float operands for the bitwise binaries with a
sentence. The unary (`bitwise_not`) and the reductions did not, so which
message a user got depended on which of three spellings they reached for.

Asserting only that an exception is raised would not hold anything down -- the
compiler wall is an exception too. These cases assert the shape of the message:
one line, naming the operation and the dtype, with no compiler output in it.
"""

import unittest

import numpy as np

import jittor as jt

from _helpers import capability as _test_capability


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


FLOAT = np.array([1., 2., 3., 4., 5., 6., 7., 8.], dtype="float32")
INT = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int32")
BOOL = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype="bool")

#: The reductions that are integral-only, and what NumPy answers for ``INT``.
INTEGRAL_REDUCTIONS = (
    ("bitwise_and", np.bitwise_and.reduce(INT)),
    ("bitwise_or", np.bitwise_or.reduce(INT)),
    ("bitwise_xor", np.bitwise_xor.reduce(INT)),
)


class _BitwiseDtypeGuard:

    device_flag = 0

    def _expect_clean_rejection(self, call, operation):
        """Run ``call``, require a readable refusal, and return the message."""
        with jt.flag_scope(use_cuda=self.device_flag):
            with self.assertRaises(RuntimeError) as caught:
                call().sync()
        message = str(caught.exception)
        self.assertNotIn("error:", message, "compiler output reached the user")
        self.assertNotIn("Compile fused operator", message,
                         "compiler output reached the user")
        self.assertIn(operation, message, "message does not name the operation")
        self.assertIn("float32", message, "message does not name the dtype")
        # One sentence plus the macro's own suffix. The wall was 14-16 lines.
        self.assertLessEqual(
            message.count("\n"), 1, "message is a report, not a sentence: %r" % message)
        return message

    def test_bitwise_not_on_a_float_is_refused_in_words(self):
        self._expect_clean_rejection(
            lambda: jt.bitwise_not(jt.array(FLOAT)), "bitwise_not")

    def test_bitwise_reductions_on_a_float_are_refused_in_words(self):
        for operation, _ in INTEGRAL_REDUCTIONS:
            with self.subTest(operation=operation):
                self._expect_clean_rejection(
                    lambda op=operation: jt.array(FLOAT).reduce(op, 0), operation)

    def test_the_integer_and_boolean_cases_still_work(self):
        """The guard must refuse floats without refusing what is well defined.

        A check written one predicate too wide would pass every case above and
        break every real use of these operations, so the working cases are part
        of the same file rather than assumed elsewhere.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            np.testing.assert_array_equal(
                jt.bitwise_not(jt.array(INT)).numpy(), np.bitwise_not(INT))
            np.testing.assert_array_equal(
                jt.bitwise_not(jt.array(BOOL)).numpy(), np.bitwise_not(BOOL))
            for operation, expected in INTEGRAL_REDUCTIONS:
                with self.subTest(operation=operation):
                    got = jt.array(INT).reduce(operation, 0)
                    got.sync()
                    self.assertEqual(
                        got.location(),
                        "device" if self.device_flag else "cpu")
                    np.testing.assert_array_equal(got.numpy(), expected)


class TestBitwiseDtypeGuardCpu(_BitwiseDtypeGuard, unittest.TestCase):
    device_flag = 0


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestBitwiseDtypeGuardCuda(_BitwiseDtypeGuard, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
