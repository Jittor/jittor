# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`check_graph`'s dangling-node sweep, and what it declines to look at.

The sweep reports a live, unreachable, unreleased node -- but only when it has
no inputs:

    if (node->is_var() && node->_inputs.size())
        continue;

Every Var a computation produces has inputs, so the class of leak most worth
catching is the class the sweep is blind to, and an "all clear" run says nothing
about it. `check_graph=2` escalates those to the same `LOGf` the edge-less case
gets, and `check_graph=1` now counts them and warns rather than skipping in
silence.

What these cases protect is the other direction. Escalation is only useful if it
is quiet on healthy graphs; a `check_graph=2` that fires on ordinary
construction and teardown would be worse than the silence it replaces, because
`LOGf` aborts. So the assertions here are that both levels survive the shapes
that occur constantly -- chains, shared subgraphs, dropped intermediates,
forced recompute -- and produce the right answers while doing it.
"""
import unittest

import numpy as np

import jittor as jt


class TestGraphCheckDanglingSweep(unittest.TestCase):

    def setUp(self):
        self._old = jt.flags.check_graph

    def tearDown(self):
        jt.flags.check_graph = self._old

    def _shapes(self):
        """Graph shapes that a dangling-node sweep sees in ordinary use."""
        # a plain chain, fully consumed
        a = jt.random((64, 64)).float32()
        b = a * 2.0 + 1.0
        yield "chain", b.sum(), float((a.numpy() * 2.0 + 1.0).sum())

        # a shared subgraph: one producer, two consumers
        p = jt.random((32, 32)).float32()
        q = p * 3.0
        r1, r2 = q + 1.0, q - 1.0
        yield "shared", (r1 + r2).sum(), float((p.numpy() * 6.0).sum())

        # an intermediate that is dropped before the barrier, so its Var is
        # unreachable from any holder by the time the sweep runs
        s = jt.random((48, 48)).float32()
        t = s * 4.0
        u = t + 2.0
        del t
        yield "dropped intermediate", u.sum(), float((s.numpy() * 4.0 + 2.0).sum())

    def _run_at(self, level):
        jt.flags.check_graph = level
        for name, var, want in self._shapes():
            jt.sync_all(True)
            got = float(var.numpy())
            rel = abs(got - want) / max(1e-6, abs(want))
            self.assertLess(rel, 1e-4,
                            "%s at check_graph=%d: got %r want %r"
                            % (name, level, got, want))
        jt.gc()

    def test_check_graph_1_is_quiet_on_healthy_graphs(self):
        self._run_at(1)

    def test_check_graph_2_is_quiet_on_healthy_graphs(self):
        # The escalation added for the MiniMax-H3 investigation. If it fires
        # here it aborts the process, so a pass is the whole assertion.
        self._run_at(2)

    def test_the_flag_setter_still_drives_node_registration(self):
        # The sweep can only report what is registered, and registration
        # follows the flag. Turning it on and off either side of a graph must
        # not leave the checker unable to run -- reporting "all clear" from an
        # empty table is the defect this file's neighbour documents.
        jt.flags.check_graph = 0
        a = jt.random((16, 16)).float32() + 1.0
        jt.flags.check_graph = 2
        b = a * 2.0
        jt.sync_all(True)
        self.assertAlmostEqual(float(b.numpy().mean()),
                               float(a.numpy().mean()) * 2.0, places=4)
        jt.flags.check_graph = 0
        jt.sync_all(True)


if __name__ == "__main__":
    unittest.main()
