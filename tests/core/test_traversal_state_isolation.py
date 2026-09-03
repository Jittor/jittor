# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Two graph traversals must not be able to destroy each other's bookkeeping.

``Node`` carries one general-purpose ``int`` (``custom_data``) that several
unrelated algorithms use as scratch: ``Executor::run_sync`` keeps each op's and
var's index in it, ``FusedOp::update_ops`` bit-packs "cannot fuse", "visited"
and a var index into it, ``grad()`` keeps gradient-var indices in it, and the
topological sorts in ``graph.h`` used it for in-degrees.  Nothing marks whose
turn it is.  Correctness rested on each caller *remembering* not to run while
another was mid-flight -- and where that could not be arranged, on copying the
field out and putting it back by hand.

That is not hypothetical.  ``MemoryProfiler::check()`` is called from inside
``run_sync``'s op loop (the ``profile_memory_enable`` branch), i.e. while the
executor's indices are live, and it ran a topological sort.  It got away with it
only because it saved the whole field first::

    vector<int> backup_custom_data;                      # memory_profiler.cc
    ...
    toplogical_sort_forward(queue, queue2, [](Node*){});
    for (int i=0; i<queue.size(); i++)
        queue[i]->custom_data = backup_custom_data[i];

Delete those lines and a fused graph dies with ``Check failed: outputs().size()``
in ``FusedOp::update_ops``: the "cannot fuse" bit the executor left in the field
comes back as an in-degree, so the fused op decides it has no outputs.  The sort
keeps its own in-degrees now, so there is nothing left to save.

Five of the six users keep their own storage now.  The sixth -- the executor's
own batch numbering -- stayed on the node, because moving it costs about
0.075 us per op of planning time (see the ``[2.02 下半]`` commit message).  What
changed there is the part that was broken: the field has one owner, it is
written with the stamp of the batch that wrote it, and every read names the
batch it believes it is in.

So there are two things to hold, and they are different:

* **the behaviour** -- a full traversal running inside ``run_sync``'s op loop
  does not change the answer.  That is the first case below.
* **the mechanism** -- a synchronous inner traversal restores every outer mark
  before returning.  That is the second case, and it is the one that keeps
  holding when somebody writes a new traversal next month.

Run::  python -m pytest tests/core/test_traversal_state_isolation.py
"""

import unittest

import numpy as np
import jittor as jt


def build(x):
    """A graph with enough element-wise ops to be fused into one kernel."""
    a = jt.array(x)
    b = a * 2 + 1
    c = b.sqr() - a
    return (c * b).sum() + (c + b).mean()


def expected(x):
    b = x * 2 + 1
    c = b ** 2 - x
    return float((c * b).sum() + (c + b).mean())


class TestBatchIndexIsChecked(unittest.TestCase):
    """The mechanism: a stale reader is caught, not served.

    ``jt.tests`` exposes the C++ unit tests; this one stamps a node as one
    batch, restamps it as another, and asserts that reading with the first
    stamp raises. It is the general form of the failure that made the field
    dangerous -- the reader who was there first keeps reading and gets somebody
    else's index -- and unlike an interleaving scenario it does not depend on
    which traversals happen to call each other today.
    """

    def test_reading_a_batch_index_with_a_stale_stamp_is_an_error(self):
        jt.tests.node_batch_index_is_checked()

    def test_a_nested_traversal_restores_the_outer_one_s_marks(self):
        # The other shared slot: Node::tflag. A traversal that starts while
        # another is walking temporarily borrows its marks. TraversalEpoch logs
        # only those nested writes and restores them before the outer resumes.
        jt.tests.traversal_epoch_restores_outer_marks()

    def test_the_table_the_other_traversals_use_behaves(self):
        # The same question for NodeIndex: a reference stays valid across
        # further inserts (the topological sorts do `--index[node]`), and a
        # node never indexed is distinguishable from one indexed as 0.
        jt.tests.node_index_table()


class TestTraversalStateIsolation(unittest.TestCase):
    def setUp(self):
        self.previous = jt.flags.profile_memory_enable

    def tearDown(self):
        jt.flags.profile_memory_enable = self.previous

    def test_a_traversal_inside_run_sync_does_not_break_fusion(self):
        x = np.random.RandomState(0).rand(96, 96).astype("float32")
        reference = build(x).item()
        np.testing.assert_allclose(reference, expected(x), rtol=1e-5)

        jt.flags.profile_memory_enable = 1
        try:
            # MemoryProfiler::check() walks the whole graph once per op that
            # run_sync executes, from inside run_sync.
            with_profiler = build(x).item()
        finally:
            jt.flags.profile_memory_enable = 0
        np.testing.assert_allclose(with_profiler, reference, rtol=1e-5)

    def test_backward_survives_a_traversal_inside_its_own_forward(self):
        """``grad()`` sorts the graph, then keeps indices across op building.

        Building the backward ops can re-enter ``run_sync`` (``Op::init`` does,
        for a vary-shape op), and the profiler's traversal rides along with it.
        """
        x = np.random.RandomState(1).rand(32, 32).astype("float32")
        weight = jt.array(x)
        loss = (weight * weight + weight).sum()
        reference = jt.grad(loss, weight).numpy()

        jt.flags.profile_memory_enable = 1
        try:
            weight2 = jt.array(x)
            loss2 = (weight2 * weight2 + weight2).sum()
            with_profiler = jt.grad(loss2, weight2).numpy()
        finally:
            jt.flags.profile_memory_enable = 0
        np.testing.assert_allclose(with_profiler, reference, rtol=1e-5)
        np.testing.assert_allclose(reference, 2 * x + 1, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
