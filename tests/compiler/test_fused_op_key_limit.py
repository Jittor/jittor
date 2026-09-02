# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A fusion whose edge ids overflow the jit key must fail loudly.

``FusedOp::do_jit_prepare`` encodes every fusion edge as
``hex2(producer) hex1(out_slot) hex2(consumer) hex1(in_slot)`` -- 8 bits for the
op ids, 4 for the slots.  Producer ids run over ``ops.size() + var_index``
(``executor.cc``), so a fusion holding more than 255 ops plus external input
vars wraps.  Two structurally different fusions then hash to the same key, the
kernel cache lookup hits, and an unrelated compiled kernel runs: a silently
wrong result with no diagnostic.

This is a temporary guard.  Task 3.02 replaces the fixed-width edge encoding
with a variable-length one and this file should then assert that the large
fusion *computes correctly* rather than that it is refused.

Run::  python -m pytest tests/compiler/test_fused_op_key_limit.py
"""

import unittest

import numpy as np

import jittor as jt


def chain_sum(n):
    """One fusion candidate: n external input vars and n-1 element-wise adds."""
    xs = [jt.array(np.full((4,), i + 1, dtype="float32"), dtype="float32")
          for i in range(n)]
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total


class TestFusedOpKeyLimit(unittest.TestCase):
    def test_fusion_within_the_limit_is_exact(self):
        for n in (2, 8, 64):
            got = chain_sum(n).numpy()
            expected = n * (n + 1) // 2
            np.testing.assert_allclose(got, np.full((4,), expected, "float32"))

    def test_fusion_past_the_limit_raises(self):
        # 300 inputs + 299 adds: the producer id of the last external input is
        # far past 255. Before the guard this silently built an aliasing key.
        with self.assertRaises(Exception) as caught:
            chain_sum(300).numpy()
        message = str(caught.exception)
        self.assertIn("jit key encoding", message, message[:2000])

    def test_the_limit_is_reported_not_silently_wrapped(self):
        # Whatever the exact threshold, crossing it must never produce a value.
        # Walk up until it trips and check every value below it is exact.
        n = 64
        while n <= 512:
            try:
                got = chain_sum(n).numpy()
            except Exception as error:
                self.assertIn("jit key encoding", str(error), str(error)[:2000])
                return
            np.testing.assert_allclose(
                got, np.full((4,), n * (n + 1) // 2, "float32"), err_msg=str(n))
            n *= 2
        self.fail("the fused-op edge id guard never tripped")


if __name__ == "__main__":
    unittest.main()
