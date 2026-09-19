# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np


class TestFuser(unittest.TestCase):
    def test_wrong_fuse(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = (a * b).sum() + (a + 1)
        print(c)

    def test_wrong_fuse2(self):
        a = jt.array([1])
        b = jt.random([10,])
        c = jt.random([100,])
        bb = a*b
        cc = a*c
        jt.sync([bb,cc])
        np.testing.assert_allclose(b.data, bb.data)
        np.testing.assert_allclose(c.data, cc.data)

    def test_for_fuse(self):
        arr = []
        x = 0
        for i in range(100):
            arr.append(jt.array(1))
            x += arr[-1]
        x.sync()
        for i in range(100):
            # print(arr[i].debug_msg())
            assert ",0)" not in arr[i].debug_msg()

    def test_array_bc(self):
        # a = jt.array(1)
        with jt.profile_scope() as rep:
            b = jt.array(1).broadcast([10])
            b.sync()
        assert len(rep) == 2

    # `fuse_op_limit` bounds how many operators one fused kernel may hold, and
    # it ships on because the bound is free: it fires only on half-precision
    # chains, since a float32 chain stops itself on its own dtype casts and
    # bounding it just buys a write-out (8.1% on a 40-operator float32 chain).
    #
    # The mechanism is pinned in tests/codegen/test_fuse_op_limit.py: that limit
    # 16 splits a float16 chain, that a float32 chain survives even limit 1, and
    # that the result is bit-identical either way. What that file does not pin
    # is the *default* -- every one of its cases sets the flag by hand, so
    # flipping the default back to 0 would break the H3 decode again and leave
    # the suite green. Hence one assertion here.

    def test_fuse_op_limit_ships_enabled(self):
        # A bound that defaults to 0 removes nothing, so assert "on", not a
        # particular value: the sweep in section 47 of
        # docs/results/2026-09-14-vllm-omni-h3-enablement.md shows 8 is a little
        # faster than 16 and is not free, so the number is measured, not fixed.
        self.assertGreater(jt.flags.fuse_op_limit, 0)


if __name__ == "__main__":
    unittest.main()