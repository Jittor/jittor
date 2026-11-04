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

    # A captured or replayed result is a copy out of the recorded step, with
    # no autograd edge to the call's input or the step's parameters. A grad
    # through it was 4, 0, 0 for x = 2, 3, 4 with only a missing-gradient
    # warning; it raises now, while the eager first call stays differentiable.
    def test_external_grad_through_a_replayed_result_raises(self):
        step = jt.capture_step(lambda x: (x * x).sum(), record=False)
        grads = []
        for value in (2.0, 3.0, 4.0, 5.0):
            x = jt.array([value])
            y = step(x)
            np.testing.assert_allclose(y.numpy(), value * value)
            if not grads:
                grads.append(jt.grad(y, x).numpy())
                continue
            with self.assertRaisesRegex(RuntimeError, "capture_step call, which has no autograd"):
                jt.grad(y, x)
        np.testing.assert_allclose(grads[0], [4.0])
        self.assertEqual(step.stats["replayed"], 2)
        self.assertIsNone(step.refused)

    def test_external_grad_to_a_closure_parameter_raises(self):
        w = jt.array([3.0])
        step = jt.capture_step(lambda x: (w * x).sum(), record=False)
        for value in (1.0, 2.0, 3.0):
            out = step(jt.array([value]).stop_grad())
        with self.assertRaisesRegex(RuntimeError, "no autograd history"):
            jt.grad(out, w)

    # The step's part of a grad that also has an ordinary path used to be
    # dropped without a word: step(x) + x.sum() gave 1 where 2x + 1 is right.
    def test_a_grad_with_another_path_into_the_target_raises(self):
        w = jt.array([3.0])
        step = jt.capture_step(lambda x: ((x * x).sum(), {"scaled": w * x}), record=False)
        for value in (2.0, 3.0, 4.0):
            x = jt.array([value])
            total, parts = step(x)
            loss = total + x.sum()
            if step.stats["captured"] == 0:
                np.testing.assert_allclose(jt.grad(loss, x).numpy(), [2 * value + 1])
                continue
            np.testing.assert_allclose(loss.numpy(), value * value + value)
            for target, grad_of in ((x, loss), (w, parts["scaled"].sum() + w.sum()),
                                    (x, parts["scaled"].sum() + total)):
                with self.assertRaisesRegex(RuntimeError, "no autograd history"):
                    jt.grad(grad_of, target)
        self.assertEqual(step.stats["replayed"], 1)

    def test_a_grad_that_does_not_reach_the_step_is_untouched(self):
        # A target the step did not read, the result itself as the target,
        # and a stop_grad result: none of them loses a gradient to the replay.
        step = jt.capture_step(lambda x: (x * x).sum(), record=False)
        y = jt.array([5.0])
        for value in (2.0, 3.0, 4.0):
            x = jt.array([value])
            out = step(x)
            np.testing.assert_allclose(jt.grad(out * y.sum() + y.sum(), y).numpy(),
                                       [value * value + 1])
            np.testing.assert_allclose(jt.grad(out * 3 + y.sum(), out).numpy(), 3.0)
            frozen = out.stop_grad()
            np.testing.assert_allclose(jt.grad(frozen * x.sum(), x).numpy(), [value * value])
        self.assertEqual(step.stats["replayed"], 1)

    def test_constant_results_stay_usable(self):
        # Nothing the step reads asks for a gradient, or the call is under
        # no_grad: the result is a constant, and a grad that also reaches a
        # trainable Var downstream of it is the ordinary one.
        feature = jt.capture_step(lambda x: x * 2, record=False)
        head = jt.array([3.0])
        for value in (1.0, 2.0, 3.0):
            out = feature(jt.array([value]).stop_grad())
            np.testing.assert_allclose(jt.grad((out * head).sum(), head).numpy(), [2 * value])
        w = jt.array([3.0])
        step = jt.capture_step(lambda x: (w * x).sum(), record=False)
        with jt.no_grad():
            for value in (1.0, 2.0, 3.0):
                out = step(jt.array([value]))
        np.testing.assert_allclose(out.numpy(), 9.0)
        self.assertEqual(step.stats["replayed"], 1)

    def test_a_step_with_its_own_backward_still_replays(self):
        p = jt.array([1.0])

        def train(x):
            loss = ((p * x) ** 2).sum()
            p.update(p - 0.1 * jt.grad(loss, p))
            return loss
        step = jt.capture_step(train, record=False)
        losses = [float(step(jt.array([1.0]).stop_grad()).numpy()) for _ in range(4)]
        np.testing.assert_allclose(losses, [1.0, 0.64, 0.4096, 0.262144], rtol=1e-5)
        self.assertEqual(step.stats["replayed"], 2)

    def test_foreign_operation_refuses_step_replay(self):
        def step(value):
            jt.graph_replay_barrier("external step callback")
            return value * 2
        captured = jt.capture_step(step, record=False)
        for start in (1., 2., 3., 4.):
            np.testing.assert_array_equal(captured(jt.array([start])).numpy(), [start * 2])
        self.assertIn("external step callback", captured.refused)
        self.assertEqual(captured.stats["replayed"], 0)

    def test_nested_device_capture_does_not_advance_step_policy(self):
        from unittest.mock import patch
        captured = jt.capture_step(lambda value: value * 2, record=False)
        with patch.object(jt.core, "graph_capture_active", return_value=True):
            np.testing.assert_array_equal(captured(jt.array([3.])).numpy(), [6.])
        self.assertEqual(captured.stats["captured"], 0)
        self.assertEqual(captured.stats["replayed"], 0)
        self.assertIsNone(captured._seen)

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

    def test_state_replaced_from_outside_is_adopted(self):
        # A cache reset between two `generate()` calls, a `load_state_dict`:
        # the caller gives the state a new value of the same shape. The
        # capture copies it in rather than capturing again.
        def between(i, cw, ew):
            if i == 3:
                value = np.full((8, 4), 0.25, np.float32)
                ew.update(jt.array(value))
                cw.update(jt.array(value))
        eager, captured = self._twins()
        cap = self._check(eager, captured, _feeds(), between)
        self.assertEqual(cap.stats["captured"], 1)

    def test_a_pending_outside_forward_reads_the_forward_time_state(self):
        # A differentiable graph built between two replays -- an unrelated
        # forward from a parameter, left pending -- saves activations whose
        # inputs are the storages the next replay writes. Its backward must
        # see the values those storages held when it was built: directly,
        # through an unsaved intermediate, and through a view.
        cases = (
            ("direct", lambda w: (w + 1).exp().sum(),
             lambda w: np.exp(w + 1)),
            ("transitive", lambda w: (w * 2 + 1).exp().sum(),
             lambda w: 2 * np.exp(2 * w + 1)),
            ("alias", lambda w: (w.view(-1) + 1).exp().sum(),
             lambda w: np.exp(w + 1)),
        )
        for name, outside_of, expected in cases:
            with self.subTest(case=name):
                w, b = _state()
                captured = jt.capture_step(_make_step(w, b, lambda: 0.1))
                feeds = _feeds(4)
                for x, y in feeds[:3]:
                    float(captured(x, y).numpy())
                self.assertIsNone(captured.refused)
                self.assertEqual(captured.stats["replayed"], 1)
                before = w.numpy().copy()
                outside = outside_of(w)
                float(captured(*feeds[3]).numpy())
                gradient, = jt.grad(outside, [w])
                np.testing.assert_allclose(gradient.numpy(), expected(before),
                                           rtol=1e-5, atol=1e-6)

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

    def test_a_cast_then_scalar_multiply_reads_the_new_input(self):
        # The copies into the captured leaves used to run outside the kept-
        # graph window. The executor then treated them as an ordinary batch
        # and released the captured intermediate behind them, so the replay
        # answered with the value that intermediate held when it was captured:
        # a half input cast to float32 and multiplied by a Python float kept
        # returning the first feed's result. SDPA's scale fold is this shape.
        rs = np.random.RandomState(4)

        def step(x):
            return [(x.float32() * 2.0).sum(1)]
        self._replay_matches(
            step,
            [jt.array(rs.randn(64, 32).astype("float32")).float16()
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
@_test_preserve_policy(jt, "keep_graph", "auto_graph_replay", "auto_flush_ops",
                       "auto_flush_bytes")
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

    def test_captured_uniform_draws_are_uniform(self):
        def step(x):
            return [x + jt.random(x.shape)]
        captured = jt.capture_step(step)
        x = jt.zeros((100003,))       # not a multiple of the four a thread fills
        draws = [captured(x)[0].numpy() for _ in range(4)]
        self.assertGreater(captured.stats["replayed"], 0)
        d = draws[-1]
        self.assertTrue(((d >= 0) & (d < 1)).all())
        self.assertLess(abs(d.mean() - 0.5), 0.01)
        self.assertLess(abs(d.var() - 1 / 12), 0.005)
        self.assertFalse(np.allclose(draws[-1], draws[-2]))

    def test_a_dropout_keep_mask_is_the_draw_compared(self):
        # Drawn and compared in one kernel, and the same bits as the uniform
        # draw of the same slot compared afterwards.
        def step(x):
            keep = step_capture.random_keep(tuple(x.shape), 0.3)
            if keep is None:        # the first call runs as written
                keep = jt.random(x.shape) > 0.3
                return [keep, keep]
            step_capture._ACTIVE.random.draws -= 1      # the same slot again
            return [keep, jt.random(x.shape) > 0.3]
        captured = jt.capture_step(step)
        x = jt.zeros((1000,))
        masks = []
        for _ in range(5):
            keep, draw = (t.numpy() for t in captured(x))
            np.testing.assert_array_equal(keep, draw)
            masks.append(keep)
        self.assertGreater(captured.stats["replayed"], 0)
        self.assertFalse((masks[-1] == masks[-2]).all())
        self.assertTrue(0.6 < masks[-1].mean() < 0.8)

    def test_dropout_differentiates_through_its_captured_mask(self):
        w = jt.ones((4096,))
        w.sync()

        def step(x):
            y = jt.nn.dropout(x * w, 0.25, is_train=True)
            return [y, jt.grad(y.sum(), w)]
        captured = jt.capture_step(step)
        x = jt.ones((4096,))
        outs = [[t.numpy() for t in captured(x)] for _ in range(5)]
        self.assertGreater(captured.stats["replayed"], 0)
        for y, g in outs:
            np.testing.assert_allclose(y, g, rtol=1e-6)
            self.assertTrue(set(np.unique(y)) <= {0.0, np.float32(1 / 0.75)})
        self.assertFalse((outs[-1][0] == outs[-2][0]).all())

    def test_the_step_is_recorded(self):
        eager, captured = self._twins()
        cap = self._check(eager, captured, _feeds(10))
        self.assertIsNone(cap._graph_refused)
        self.assertGreater(cap.stats["graph"], 0)

    def test_capturing_runs_the_step_once(self):
        # CUDA launches what is pending every `auto_flush_ops` operators. In a
        # capture, each piece launched early was freed after its last use and
        # computed again when the whole step ran: a Qwen3 training step ran
        # its forward two and a half times to be captured.
        jt.flags.auto_flush_ops = 1
        jt.flags.auto_flush_bytes = 0
        rs = np.random.RandomState(6)
        w = jt.array((rs.randn(256, 256) / 16).astype("float32"))
        w.sync()

        def step(x):
            h = x
            for _ in range(12):
                h = jt.matmul(h, w)
            return [h.sum()]
        feeds = [jt.array(rs.randn(256, 256).astype("float32")) for _ in range(4)]
        jt.sync(feeds)
        allocated = step_capture._core.device_memory_allocated_total
        captured = jt.capture_step(step)
        captured(feeds[0])
        before = allocated(0)
        captured(feeds[1])[0].sync()
        capturing = allocated(0) - before
        before = allocated(0)
        captured(feeds[2])[0].sync()
        replaying = allocated(0) - before
        self.assertEqual(captured.stats["captured"], 1)
        self.assertLess(capturing, 1.5 * replaying)
        self._replay_matches(step, feeds)

    def test_a_shared_buffer_goes_after_its_last_use(self):
        # A `jt.Function` input and the tape that passes it on share one
        # buffer, neither a view of the other; a kept graph freed such a pair
        # never, and a captured training step held every Function input and
        # its embedding gradient -- 1.2 GB on a four-layer Qwen3.
        class Scale(jt.Function):
            def execute(self, x):
                return x * 3

            def grad(self, g):
                return g * 3
        rs = np.random.RandomState(7)
        w = jt.array(rs.randn(512, 512).astype("float32"))
        w.sync()

        def step(x):
            return [Scale.apply(x * w).sum()]
        feeds = [jt.array(rs.randn(512, 512).astype("float32")) for _ in range(5)]
        jt.sync(feeds)
        used = step_capture._core.device_memory_used
        jt.gc()
        before = used(0)
        # A recording holds its working set, as PyTorch's graph pools do; the
        # executor's replays are what free as they go.
        captured = jt.capture_step(step, record=False)
        for x in feeds:
            np.testing.assert_allclose(captured(x)[0].numpy(), step(x)[0].numpy(),
                                       rtol=1e-5)
        self.assertEqual(captured.stats["captured"], 1)
        self.assertGreater(captured.stats["replayed"], 0)
        jt.gc()
        # The capture keeps its private copy of the input, 1 MB, and a
        # scalar; the product, another 1 MB, goes after its last use.
        self.assertLess(used(0) - before, 1.5 * 512 * 512 * 4)

    def test_memory_a_recording_uses_is_not_handed_out(self):
        # A recording keeps its buffers' addresses, so what it freed and
        # reused must stay out of the pool while it lives -- without holding
        # every intermediate apart, which made it cost the sum of them.
        rs = np.random.RandomState(8)
        w = jt.array((rs.randn(256, 256) / 16).astype("float32"))
        w.sync()

        def step(x):
            h = x
            for _ in range(6):
                h = jt.matmul(h, w).tanh()
            return [h.sum(1)]
        feeds = [jt.array(rs.randn(256, 256).astype("float32")) for _ in range(10)]
        jt.sync(feeds)
        captured = jt.capture_step(step)
        outside = []
        for i, x in enumerate(feeds):
            want = step(x)[0].numpy()
            got = captured(x)[0].numpy()
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
            if captured.stats["graph"]:
                # Buffers of the intermediates' size, made between launches.
                made = [jt.full((256, 256), float(i)) for _ in range(6)]
                jt.sync(made)
                outside += [(float(i), v) for v in made]
        self.assertGreater(captured.stats["graph"], 1)
        for value, v in outside:
            np.testing.assert_array_equal(v.numpy(), np.full((256, 256), value, "float32"))


if __name__ == "__main__":
    unittest.main()
