# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Replaying a captured inference graph instead of rebuilding it.

The timing check is off by default, so the guards below are exercised
whatever this machine would have decided about this small model. The opt-in
timing path has its own test.

The speedup is only worth anything if the answer is still right, and the
ways a captured graph can stop being right are all silent: it keeps
answering with whatever it last computed. So most of what is pinned down
here is the refusals -- every guard, and what happens when it fires.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import dataclasses
import unittest

import numpy as np

import jittor as jt
from jittor import nn
from jittor._runtime.graph_replay import graph_replay


class _Net(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def execute(self, x):
        return self.l2(nn.relu(self.l1(x)))


class _Random(nn.Module):
    def execute(self, x):
        return x + jt.rand(x.shape)


class _Stack(nn.Module):
    def execute(self, x):
        h = (x * 2 + 1).tanh()
        return jt.stack([h, h * 3], 0)


@dataclasses.dataclass
class _Result:
    sample: object = None
    scale: float = 1.0


class _Structured(nn.Module):
    """Keyword input, and every result shape replay has to rebuild."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 8)

    def execute(self, x, *, bias=None, extra=None):
        h = self.l1(x)
        y = h + bias
        z = jt.concat([h, y], 1)
        return {"y": y, "pair": (h, z), "same": [y, y],
                "result": _Result(sample=y * extra["k"], scale=2.0), "none": None}


@_test_preserve_policy(jt, 'keep_graph', 'auto_graph_replay')
class TestGraphReplay(unittest.TestCase):

    def setUp(self):
        jt.flags.keep_graph = 0
        self.model = _Net()
        rs = np.random.RandomState(0)
        self.feed = [jt.array(rs.randn(2, 8).astype("float32")) for _ in range(4)]
        for f in self.feed:
            f.sync(True, False)

    def _eager(self, x):
        # Genuinely eager: the automatic policy would otherwise replay the
        # reference too, and then this compares a replay against a replay.
        before = jt.flags.auto_graph_replay
        jt.flags.auto_graph_replay = 0
        try:
            with jt.no_grad():
                return self.model(x).numpy().copy()
        finally:
            jt.flags.auto_graph_replay = before

    def test_it_answers_for_each_input_not_just_the_captured_one(self):
        want = [self._eager(f) for f in self.feed]
        replay = graph_replay(self.model, self.feed[0])
        got = [replay(f).numpy().copy() for f in self.feed]
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        # The real failure mode is one answer repeated, so check that too.
        self.assertEqual(len({g.tobytes() for g in got}), 4)

    def test_the_returned_var_survives_the_next_call(self):
        replay = graph_replay(self.model, self.feed[0])
        held = replay(self.feed[0])
        snapshot = held.numpy().copy()
        replay(self.feed[1])
        np.testing.assert_array_equal(held.numpy(), snapshot)

    def test_a_shape_change_is_answered_correctly(self):
        replay = graph_replay(self.model, self.feed[0])
        replay(self.feed[0])
        wide = jt.array(np.random.RandomState(1).randn(5, 8).astype("float32"))
        wide.sync(True, False)
        np.testing.assert_allclose(replay(wide).numpy(), self._eager(wide),
                                   rtol=1e-5, atol=1e-5)

    def test_a_finished_capture_is_noticed_and_retaken(self):
        # Reading the captured output can finish the graph -- whether it does
        # depends on what else the batch collected, so this asserts the
        # contract rather than the mechanism: the answer stays right, and if
        # the graph did get finished, the capture was retaken rather than
        # replayed. A finished graph still answers, with the value it last
        # computed, which is exactly the silent failure the guard exists for.
        replay = graph_replay(self.model, self.feed[0])
        replay(self.feed[0])
        capture = replay._capture
        float(capture.outputs[0].numpy().sum())
        # Read the state *before* the call: the call itself finishes the graph
        # on its way to retaking it, so asking afterwards always says "finished"
        # and would demand a retake that was never needed.
        was_finished = capture.outputs[0].is_finished
        before = replay.stats["captured"]
        np.testing.assert_allclose(replay(self.feed[2]).numpy(),
                                   self._eager(self.feed[2]), rtol=1e-5, atol=1e-5)
        if was_finished:
            self.assertGreater(replay.stats["captured"], before)

    def test_a_replaced_parameter_is_noticed(self):
        replay = graph_replay(self.model, self.feed[0])
        replay(self.feed[0])
        before = replay.stats["captured"]
        # An optimizer step rebinds the holder; the captured graph still reads
        # the Var it captured, so this must not answer with the old weights.
        self.model.l1.weight.update(self.model.l1.weight * 2)
        # This Var only: a process-wide sync_all would also try to run
        # whatever another test left pending.
        self.model.l1.weight.sync(True, False)
        np.testing.assert_allclose(replay(self.feed[0]).numpy(),
                                   self._eager(self.feed[0]), rtol=1e-5, atol=1e-5)
        self.assertGreater(replay.stats["captured"], before)

    def test_a_stacked_result_follows_each_input(self):
        # `setitem_gopt` computes each stacked operand straight into its slice
        # of the result and makes the setitem a no-op. A replay that freed an
        # operand between runs recomputed it into a fresh buffer nothing
        # copied, and on CUDA every call answered with the first input's
        # result -- the automatic policy, which is on by default, included.
        model = _Stack()
        feeds = [jt.array(np.full((4,), float(i), np.float32)) for i in range(5)]
        expected = [np.stack([np.tanh(2.0 * i + 1), 3 * np.tanh(2.0 * i + 1)])
                    for i in range(5)]
        replay = graph_replay(model)
        for x, want in zip(feeds, expected):
            np.testing.assert_allclose(replay(x).numpy()[:, 0], want, rtol=1e-5)
        self.assertIsNone(replay.refused)
        self.assertGreaterEqual(replay.stats["replayed"], 4)
        jt.flags.auto_graph_replay = 1
        with jt.no_grad():
            for x, want in zip(feeds, expected):
                np.testing.assert_allclose(model(x).numpy()[:, 0], want, rtol=1e-5)

    def test_keyword_arguments_and_structured_results(self):
        model = _Structured()
        rs = np.random.RandomState(3)
        calls = [(jt.array(rs.randn(2, 8).astype("float32")),
                  jt.array(rs.randn(8).astype("float32")),
                  jt.array(rs.randn(1).astype("float32"))) for _ in range(4)]

        def eager(x, b, k):
            jt.flags.auto_graph_replay = 0
            try:
                with jt.no_grad():
                    out = model(x, bias=b, extra={"k": k})
                    return (out["y"].numpy().copy(), out["pair"][1].numpy().copy(),
                            out["result"].sample.numpy().copy())
            finally:
                jt.flags.auto_graph_replay = 1

        replay = graph_replay(model)
        for x, b, k in calls + calls:
            want = eager(x, b, k)
            out = replay(x, bias=b, extra={"k": k})
            self.assertEqual(set(out), {"y", "pair", "same", "result", "none"})
            self.assertIsInstance(out["pair"], tuple)
            self.assertIsInstance(out["result"], _Result)
            self.assertEqual(out["result"].scale, 2.0)
            self.assertIsNone(out["none"])
            # One Var returned twice comes back as one Var twice.
            self.assertIs(out["same"][0], out["same"][1])
            self.assertIs(out["same"][0], out["y"])
            np.testing.assert_allclose(out["y"].numpy(), want[0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(out["pair"][1].numpy(), want[1], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(out["result"].sample.numpy(), want[2],
                                       rtol=1e-5, atol=1e-6)
        self.assertIsNone(replay.refused)
        self.assertEqual(replay.stats["captured"], 1)

    def test_a_result_that_cannot_be_rebuilt_is_refused(self):
        class _Opaque(nn.Module):
            def execute(self, x):
                return object(), x * 2
        replay = graph_replay(_Opaque())
        _, doubled = replay(self.feed[0])
        np.testing.assert_allclose(doubled.numpy(), self.feed[0].numpy() * 2)
        self.assertIn("cannot rebuild", replay.refused)

    def test_a_random_graph_is_refused_rather_than_repeated(self):
        replay = graph_replay(_Random(), self.feed[0])
        self.assertIsNotNone(replay.refused)
        self.assertIn("random", replay.refused)
        # And it still works, eagerly: two calls must not agree.
        a = replay(self.feed[0]).numpy().copy()
        b = replay(self.feed[0]).numpy().copy()
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(replay.stats["replayed"], 0)

    def test_a_graph_that_replay_does_not_help_falls_back(self):
        # With the timing on, the verdict is the machine's to make -- what has
        # to hold either way is that the answer is right and that a refusal is
        # stated rather than silently costing the speedup.
        replay = graph_replay(self.model, self.feed[0], measure=True)
        np.testing.assert_allclose(replay(self.feed[1]).numpy(),
                                   self._eager(self.feed[1]), rtol=1e-5, atol=1e-5)
        if replay.refused is not None:
            self.assertIn("slower", replay.refused)
            self.assertEqual(replay.stats["replayed"], 0)

    def test_the_flag_is_left_as_it_was_found(self):
        self.assertEqual(jt.flags.keep_graph, 0)
        replay = graph_replay(self.model, self.feed[0])
        replay(self.feed[1])
        self.assertEqual(jt.flags.keep_graph, 0)


# `check_accelerator(...).enabled`, not `machine_has_accelerator`: the
# question a gate asks is whether CUDA is usable in *this build*, not
# whether the machine has a card. The helper's own docstring says so --
# it is "the question to ask before reporting that something is
# unverifiable here". A CPU-only build on a GPU box answered yes, the
# class ran, and `jt.flags.use_cuda = 1` raised `No CUDA found`. It also
# keeps the loud path: a CUDA build that FAILED still asserts rather
# than skipping (see tests/_helpers/capability.py).
@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
@_test_preserve_policy(jt, 'keep_graph', 'auto_graph_replay')
class TestGraphReplayCuda(TestGraphReplay):

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        super().setUp()

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda


if __name__ == "__main__":
    unittest.main()
