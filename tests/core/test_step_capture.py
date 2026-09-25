# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Replaying a whole training step: forward, backward and the state update.

What a step capture has to get right is that a replay *is* a step: the state
it updates moves exactly as it would have, every call, and nothing -- a
refusal, a re-capture, reading the state in between -- takes a step twice or
not at all. Each test runs the same step eagerly on a twin and compares.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor._runtime import step_capture


def _state(seed=0):
    rs = np.random.RandomState(seed)
    w = jt.array(rs.randn(8, 4).astype("float32"))
    b = jt.array(np.zeros(4, "float32"))
    jt.sync([w, b])
    return w, b


def _feeds(n=8):
    rs = np.random.RandomState(1)
    return [(jt.array(rs.randn(16, 8).astype("float32")),
             jt.array(rs.randn(16, 4).astype("float32"))) for _ in range(n)]


def _make_step(w, b, lr, readback=False):
    def step(x, y):
        loss = ((jt.matmul(x, w) + b - y) ** 2).mean()
        gw, gb = jt.grad(loss, [w, b])
        if readback:
            float(loss.numpy())
        # Not in place: each update is a new Var the holder is rebound to,
        # which a replay has to write back into the state the graph reads.
        w.update(w - lr() * gw)
        b.update(b - lr() * gb)
        return loss
    return step


@_test_preserve_policy(jt, "keep_graph", "auto_graph_replay")
class TestStepCapture(unittest.TestCase):

    def setUp(self):
        jt.flags.keep_graph = 0
        jt.flags.auto_graph_replay = 0

    def _twins(self, readback=False, lr=lambda: 0.1):
        ew, eb = _state()
        cw, cb = _state()
        eager = _make_step(ew, eb, lr, readback)
        captured = jt.capture_step(_make_step(cw, cb, lr, readback))
        return (eager, ew, eb), (captured, cw, cb)

    def _check(self, eager_state, captured_state, feeds, between=None):
        eager, ew, eb = eager_state
        captured, cw, cb = captured_state
        for i, (x, y) in enumerate(feeds):
            want = float(eager(x, y).numpy())
            got = float(captured(x, y).numpy())
            self.assertAlmostEqual(got, want, places=5, msg=f"step {i}")
            if between is not None:
                between(i, cw, ew)
        np.testing.assert_allclose(cw.numpy(), ew.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cb.numpy(), eb.numpy(), rtol=1e-5, atol=1e-6)
        return captured

    def test_replays_update_state_like_the_step_does(self):
        eager, captured = self._twins()
        cap = self._check(eager, captured, _feeds())
        self.assertIsNone(cap.refused)
        self.assertEqual(cap.stats["captured"], 1)
        self.assertEqual(cap.stats["replayed"], len(_feeds()) - 2)

    def test_reading_the_state_between_steps_changes_nothing(self):
        # Reading a device Var back migrates it; a device recording that kept
        # the address it saw would update memory that is no longer the state.
        def between(i, cw, ew):
            np.testing.assert_allclose(cw.numpy(), ew.numpy(), rtol=1e-5, atol=1e-6)
        eager, captured = self._twins()
        self._check(eager, captured, _feeds(), between)

    def test_a_refused_step_still_steps_exactly_once(self):
        # A step that reads a value back is refused -- after it ran once
        # under capture. That call must not take a second step.
        eager, captured = self._twins(readback=True)
        cap = self._check(eager, captured, _feeds(5))
        self.assertIsNotNone(cap.refused)
        self.assertIn("read a value back", cap.refused)

    def test_a_guarded_scalar_that_changes_is_recaptured(self):
        rate = {"lr": 0.1}

        def lr():
            step_capture.guard(lambda: rate["lr"])
            return rate["lr"]
        eager, captured = self._twins(lr=lr)

        def between(i, cw, ew):
            if i == 4:
                rate["lr"] = 0.05
        cap = self._check(eager, captured, _feeds(), between)
        self.assertEqual(cap.stats["captured"], 2)

    def test_invalidating_takes_no_step(self):
        eager, captured = self._twins()
        feeds = _feeds(6)

        def between(i, cw, ew):
            if i == 3:
                captured[0].invalidate()
        self._check(eager, captured, feeds, between)


    def _replay_matches(self, step, feeds):
        captured = jt.capture_step(step)
        for x in feeds:
            want = [v.numpy() for v in step(x)]
            got = [v.numpy() for v in captured(x)]
            for w, g in zip(want, got):
                np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)
        self.assertGreater(captured.stats["replayed"], 0)
        return captured

    def test_a_constant_built_inside_the_step_survives_a_replay(self):
        # `jt.array` hands its data to its output on the first run; a re-run
        # found nothing left and crashed, or read an uninitialized buffer --
        # the causal-LM loss of a captured Qwen3 step went to ln(vocab).
        rs = np.random.RandomState(2)

        def step(x):
            positions = jt.array(np.arange(8, dtype=np.float32))
            return [(x * positions).sum(1)]
        self._replay_matches(step, [jt.array(rs.randn(4, 8).astype("float32"))
                                    for _ in range(5)])

    def test_results_of_custom_functions_survive_a_replay(self):
        # A `jt.Function`'s result is a tape of its output, scheduled late by
        # the `Tapes` node it also depends on; a re-run could allocate it after
        # the output had given its buffer back, and it then read memory
        # nothing wrote: two such results added together replayed as 0.
        class Double(jt.Function):
            def execute(self, x):
                return x * 2

            def grad(self, g):
                return g * 2
        rs = np.random.RandomState(3)
        w = jt.array(rs.randn(64, 32).astype("float32"))
        w.sync()

        def step(x):
            return [(Double.apply(x * w) + Double.apply(x + w)).sum()]
        self._replay_matches(step, [jt.array(rs.randn(64, 32).astype("float32"))
                                    for _ in range(5)])

    def test_random_draws_differ_on_every_replay(self):
        # A step that draws random numbers draws new ones every call, the
        # way it does when it runs as written.
        def step(x):
            return [x + jt.random(x.shape)]
        captured = jt.capture_step(step)
        x = jt.zeros((4, 4))
        draws = [captured(x)[0].numpy() for _ in range(6)]
        self.assertIsNone(captured.refused)
        self.assertGreater(captured.stats["replayed"], 0)
        for a, b in zip(draws, draws[1:]):
            self.assertFalse(np.allclose(a, b))
        for d in draws:
            self.assertTrue(((d >= 0) & (d < 1)).all())


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
@_test_preserve_policy(jt, "keep_graph", "auto_graph_replay")
class TestStepCaptureCuda(TestStepCapture):

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        super().setUp()

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    def test_two_fast_path_sums_survive_a_replay(self):
        rs = np.random.RandomState(5)
        w = jt.array(rs.randn(256, 128).astype("float32"))
        w.sync()

        def step(x):
            return [((x * w) ** 2).sum() + ((x + w) ** 2).sum()]
        self._replay_matches(step, [jt.array(rs.randn(256, 128).astype("float32"))
                                    for _ in range(5)])

    def test_random_draws_differ_on_every_launch(self):
        # The device generator's offset is baked into a recorded launch; a
        # captured step draws through Philox with a position the replay
        # advances, so a device graph draws anew as well.
        def step(x):
            return [x + jt.random(x.shape, "float32", "normal")]
        captured = jt.capture_step(step)
        x = jt.zeros((256,))
        draws = [captured(x)[0].numpy() for _ in range(8)]
        self.assertIsNone(captured._graph_refused)
        self.assertGreater(captured.stats["graph"], 0)
        for a, b in zip(draws, draws[1:]):
            self.assertFalse(np.allclose(a, b))
        allofit = np.concatenate(draws)
        self.assertLess(abs(allofit.mean()), 0.15)
        self.assertLess(abs(allofit.std() - 1), 0.15)

    def test_the_step_is_recorded(self):
        eager, captured = self._twins()
        cap = self._check(eager, captured, _feeds(10))
        self.assertIsNone(cap._graph_refused)
        self.assertGreater(cap.stats["graph"], 0)


if __name__ == "__main__":
    unittest.main()
