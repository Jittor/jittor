# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests


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



def _plans_and_agrees(build, limit):
    """``build`` under ``fuse_op_limit=limit`` plans, and agrees with unbounded."""
    with jt.flag_scope(fuse_op_limit=0):
        unbounded = [v.float32().numpy() for v in build()]
    with jt.flag_scope(fuse_op_limit=limit):
        bounded = [v.float32().numpy() for v in build()]
    for a, b in zip(unbounded, bounded):
        # Not bit-identical on every backend: a wider kernel can keep an
        # intermediate in float32 that a narrower one writes out as half.
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-3)


def _finished_input(dtype):
    x = jt.array(np.linspace(0.1, 1, 32, dtype="float32")).cast(dtype)
    x.sync()        # finished: a boundary of the batch below, as a weight is
    return x


class TestFuseOpLimitKeepsThePlanAcyclic(JittorTestCase):
    """A bounded fusion partition must still order: its groups form a DAG.

    ``count_fuse`` unions same-level neighbours into fused groups, and
    ``build_exec_plan`` phase 4 topologically sorts those groups. Unbounded,
    a whole level component becomes one group, so no path can leave a group and
    come back. ``fuse_op_limit`` keeps only part of a half-precision component
    together, and a greedy per-edge bound used to union A with C while refusing
    the B on A -> B -> C -- or, one level up, union two ops that only a third
    group connects. Phase 4 then dequeued nothing of the cycle and died on
    ``exec_plan.cc: [check failed: queue.size() == roots.size()]``. Under
    torch-shim ``autocast`` that is what reading the gradients of
    ``nn.Sequential(Linear, GELU, Linear)`` reached at the default limit
    (compat/tests/torch/test_torch_amp_training_loop.py); these graphs reach
    it natively with a small limit. The bug is in the planner, so it is
    device-independent and runs on every device.
    """

    def test_an_op_on_the_path_between_two_merged_ops(self, device):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                def build():
                    x = _finished_input(dtype)
                    a = x * x
                    b = a * a
                    c = b + a
                    d = x + c
                    return [a + d]
                _plans_and_agrees(build, 2)

    def test_two_groups_joined_only_through_a_third(self, device):
        """Group {cast, x*n5, x+x} is entered at one op and left from another.

        No op-level path runs through it, so a check that walks ops finds
        nothing; the group-level quotient still has the cycle.
        """
        def build():
            x = _finished_input("float16")
            n1 = -x
            n2 = x + x
            n3 = n2 * n1
            n4 = n2 * n3            # a sink of the same batch
            n5 = n1 + n1
            n6 = x * n5
            return [n6, n4]
        _plans_and_agrees(build, 3)


instantiate_device_type_tests(TestFuseOpLimitKeepsThePlanAcyclic, globals())


if __name__ == "__main__":
    unittest.main()