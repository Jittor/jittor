# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Recording a replay as a device graph instead of re-issuing its launches.

A replay still costs about 4 us of host time per operator -- the plan walk,
the per-operator scopes, the allocation check, the launch -- and a decode step
has of the order of a hundred operators. A recorded graph pays that once and
every later call is a single launch.

What is pinned down here is the ways it can be silently wrong, because they
all look like success: a recording that contains nothing answers with whatever
the buffer last held, and a recording whose input copy is left outside answers
the first input's result for every later one.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor._runtime.graph_replay import graph_replay


class _Net(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def execute(self, x):
        return self.l2(nn.relu(self.l1(x)))


@unittest.skipIf(not _test_capability.machine_has_accelerator("cuda"),
                 "no CUDA device")
@_test_preserve_policy(jt, 'keep_graph', 'auto_graph_replay')
class TestGraphCapture(unittest.TestCase):

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        jt.flags.keep_graph = 0
        self.model = _Net()
        rs = np.random.RandomState(0)
        self.feed = [jt.array(rs.randn(2, 16).astype("float32")) for _ in range(8)]
        for f in self.feed:
            f.sync(True, False)

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    def _eager(self, x):
        before = jt.flags.auto_graph_replay
        jt.flags.auto_graph_replay = 0
        try:
            with jt.no_grad():
                return self.model(x).numpy().copy()
        finally:
            jt.flags.auto_graph_replay = before

    def test_supported_on_cuda(self):
        self.assertTrue(jt.graph_capture_supported())

    def test_a_recorded_replay_answers_exactly_like_eager(self):
        replay = graph_replay(self.model)
        for x in self.feed:
            want = self._eager(x)
            with jt.no_grad():
                got = replay(x).numpy()
            np.testing.assert_array_equal(got, want)
        # The recording is taken on a later call, so by the end of eight calls
        # it has to have been used -- otherwise this test proves nothing about
        # graphs at all and would keep passing if recording never happened.
        self.assertGreater(replay.stats["graph"], 0)

    def test_each_input_gets_its_own_answer(self):
        # The failure this rules out is the loudest-looking and the most
        # silent: a recording re-issues fixed pointers, so if the input copy
        # were left out of the call path every launch would recompute the
        # capture-time input and hand back the same answer for ever.
        replay = graph_replay(self.model)
        for x in self.feed:
            with jt.no_grad():
                replay(x)
        answers = []
        for x in self.feed:
            with jt.no_grad():
                answers.append(replay(x).numpy().copy())
        for x, got in zip(self.feed, answers):
            np.testing.assert_array_equal(got, self._eager(x))
        self.assertGreater(len({a.tobytes() for a in answers}), 1)

    def test_the_returned_var_survives_the_next_call(self):
        replay = graph_replay(self.model)
        with jt.no_grad():
            for _ in range(5):
                replay(self.feed[0])
            held = replay(self.feed[0])
            kept = held.numpy().copy()
            replay(self.feed[1])
        np.testing.assert_array_equal(held.numpy(), kept)

    def test_invalidate_releases_the_recording(self):
        replay = graph_replay(self.model)
        with jt.no_grad():
            for _ in range(5):
                replay(self.feed[0])
        self.assertNotEqual(replay._cuda_graph, 0)
        replay.invalidate()
        self.assertEqual(replay._cuda_graph, 0)
        # And it still answers, by recapturing.
        with jt.no_grad():
            got = replay(self.feed[2]).numpy()
        np.testing.assert_array_equal(got, self._eager(self.feed[2]))

    def test_a_refused_recording_still_answers(self):
        replay = graph_replay(self.model)
        replay._graph_refused = "pretend the device said no"
        with jt.no_grad():
            for x in self.feed:
                np.testing.assert_array_equal(replay(x).numpy(), self._eager(x))
        self.assertEqual(replay.stats["graph"], 0)

    def test_copy_into_without_syncing_the_source_reads_what_is_there(self):
        a = jt.array(np.ones(8, "float32")); a.sync(True, False)
        b = jt.empty(a.shape, a.dtype); b.sync(False, False)
        b._copy_into(a, False)
        np.testing.assert_array_equal(b.numpy(), np.ones(8, "float32"))

    def test_copy_into_without_syncing_refuses_an_unexecuted_source(self):
        pending = jt.empty([8], "float32")
        dst = jt.empty([8], "float32"); dst.sync(False, False)
        with self.assertRaises(Exception):
            dst._copy_into(pending, False)


if __name__ == "__main__":
    unittest.main()
