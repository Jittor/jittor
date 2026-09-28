# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A kept graph is run by its owner, and by nobody else.

`keep_graph` leaves a batch's nodes unfinished so the same graph can be run
again. Unfinished also means *pending*, and three different places collect
pending work as a matter of course:

  * `top_weak_sync` widens a weak sync with older pending holder Vars,
  * `sync_all` collects every pending holder Var,
  * the auto-flush in `Executor::submit_pending` collects them mid-construction.

Each of those would re-execute a kept graph behind its owner's back. That is
wasted work in every case, and worse than wasted when the graph writes into a
buffer it also reads -- `share_with` makes exactly that shape, which is how a
replay advances state in place -- because then an unrequested run is a silent
state change.

These tests use that shape deliberately: `y = leaf * 0.5` with `y` sharing
`leaf`'s storage, so one execution halves the leaf. Counting is then just
reading the value, and an extra run is impossible to miss.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
import unittest

import numpy as np

import jittor as jt


@_test_preserve_policy(jt, 'keep_graph')
class TestKeptGraphBystander(unittest.TestCase):

    def setUp(self):
        jt.flags.keep_graph = 0
        self.leaf = jt.empty([8], "float32")
        self.leaf.sync(False, False)
        self.leaf._write_inplace(np.ones(8, "float32"))
        jt.sync_all(True)
        jt.flags.keep_graph = 1
        try:
            with jt.flag_scope(auto_flush_ops=1 << 30, auto_flush_bytes=1 << 40):
                self.y = self.leaf * 0.5
                self.y.share_with(self.leaf)
            # The first run is what marks the nodes kept; before it there is
            # nothing for any of these paths to skip.
            jt.sync([self.y], False, False)
        finally:
            jt.flags.keep_graph = 0

    def tearDown(self):
        try:
            self.y._release_kept()
            jt.sync_all(True)
        except Exception:
            pass

    def _value(self):
        # Reading the leaf, never the kept output: the leaf is a finished Var,
        # so this cannot itself run anything.
        return float(self.leaf.numpy()[0])

    def _run_once(self):
        jt.flags.keep_graph = 1
        try:
            jt.sync([self.y], False, False)
        finally:
            jt.flags.keep_graph = 0

    def test_the_owner_can_still_run_it(self):
        before = self._value()
        self._run_once()
        self.assertAlmostEqual(self._value(), before * 0.5, places=6)

    def test_sync_all_does_not_run_it(self):
        before = self._value()
        jt.sync_all(True)
        self.assertEqual(self._value(), before)

    def test_an_unrelated_weak_sync_does_not_run_it(self):
        before = self._value()
        other = jt.ones([4]) + 1          # pending work of somebody else's
        other.sync()                      # a weak sync, the default
        self.assertEqual(self._value(), before)

    def test_building_unrelated_work_does_not_run_it(self):
        # The auto-flush path: enough new operators to trip it, which is what
        # happens in the middle of the next call's construction.
        before = self._value()
        acc = jt.ones([64])
        for _ in range(400):
            acc = acc * 1.000001
        acc.sync()
        self.assertEqual(self._value(), before)

    def test_a_consumer_still_pulls_it_in(self):
        # The skip must not break the graph: asking for something downstream
        # of the kept output has to run it, or the answer would be stale.
        before = self._value()
        jt.flags.keep_graph = 1
        try:
            consumer = self.y + 0.0
            consumer.sync(False, False)
            got = float(consumer.numpy()[0])
        finally:
            jt.flags.keep_graph = 0
        self.assertAlmostEqual(got, before * 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
