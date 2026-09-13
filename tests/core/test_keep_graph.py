# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``keep_graph`` leaves an executed batch's nodes unfinished.

Normally the executor finishes a batch's nodes once it has run them, which
releases their pending liveness and lets the graph be collected. With the
flag set the nodes stay, so the same graph can be handed back to the
executor and run again -- the point being that the second run skips graph
construction entirely, which is the dominant cost of a small step.

Re-running is only meaningful if it actually recomputes, so the tests feed
a new input between runs and require the answer to follow it.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


@_test_preserve_policy(jt, 'keep_graph')
class TestKeepGraph(unittest.TestCase):

    def setUp(self):
        jt.flags.keep_graph = 0

    def _leaf(self, value):
        """A materialized leaf.

        The graph is only re-runnable down to nodes that are already finished:
        a re-run re-executes everything still pending, and a leaf's own
        producer -- here an array op -- no longer has its host staging by
        then. Every caller of ``keep_graph`` owes the graph this.
        """
        x = jt.array(value)
        # This var only, and no weak sync: a process-wide sync_all would also
        # try to run whatever another test left pending, and a graph that
        # cannot run (a numpy_code op with no cupy, say) would fail here for
        # reasons that have nothing to do with keep_graph.
        x.sync(True, False)
        return x

    def _chain(self, x):
        return ((x * 2 + 1) * 3).sum()

    def test_a_kept_graph_recomputes_from_the_current_input(self):
        x = self._leaf(np.zeros((8,), dtype="float32"))
        jt.flags.keep_graph = 1
        out = self._chain(x)
        out.sync()
        seen = []
        for step in range(4):
            x._write_inplace(np.full((8,), step, dtype="float32"))
            out.sync()
            seen.append(float(out.numpy()))
        jt.flags.keep_graph = 0
        # sum((v*2+1)*3) over 8 elements = 8 * (6v + 3)
        self.assertEqual(seen, [8 * (6 * v + 3) for v in range(4)])

    def test_the_kept_answer_matches_a_freshly_built_graph(self):
        rs = np.random.RandomState(0)
        feed = [rs.randn(16).astype("float32") for _ in range(3)]
        x = self._leaf(feed[0])
        jt.flags.keep_graph = 1
        held = self._chain(x)
        held.sync()
        replayed = []
        for v in feed:
            x._write_inplace(v)
            held.sync()
            replayed.append(float(held.numpy()))
        jt.flags.keep_graph = 0
        for v, got in zip(feed, replayed):
            xx = jt.array(v)
            self.assertAlmostEqual(got, float(self._chain(xx).numpy()), places=3)

    def test_the_flag_is_off_by_default(self):
        self.assertEqual(jt.flags.keep_graph, 0)

    def test_a_graph_that_tracks_gradients_replays_too(self):
        # keep_graph is about node lifetime, not about autograd: a graph built
        # with gradient tracking on replays the same way.
        x = self._leaf(np.zeros((8,), dtype="float32"))
        x.requires_grad = True
        jt.flags.keep_graph = 1
        try:
            out = ((x * 2 + 1) * 3).sum()
            out.sync()
            seen = []
            for step in range(3):
                x._write_inplace(np.full((8,), step, dtype="float32"))
                out.sync()
                seen.append(float(out.numpy()))
        finally:
            jt.flags.keep_graph = 0
        self.assertEqual(seen, [8 * (6 * v + 3) for v in range(3)])

    def test_a_graph_built_while_off_is_still_finished(self):
        # The flag must not leak into batches built without it: an ordinary
        # graph still has to be collectable once it has run.
        x = self._leaf(np.ones((8,), dtype="float32"))
        out = self._chain(x)
        out.sync()
        self.assertEqual(float(out.numpy()), 8 * 9)
        del out
        jt.gc()


@unittest.skipIf(not _test_capability.machine_has_accelerator("cuda"),
                 "no CUDA device")
@_test_preserve_policy(jt, 'keep_graph')
class TestKeepGraphCuda(TestKeepGraph):

    def setUp(self):
        jt.flags.keep_graph = 0
        jt.flags.use_cuda = 1


if __name__ == "__main__":
    unittest.main()
