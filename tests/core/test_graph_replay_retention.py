# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A replayed graph costs what a normal call peaks at, not the sum of it.

A capture keeps its graph re-runnable, and it used to keep every intermediate
allocated as well. The policy only looked at how large a call's *inputs* are:
an SD1.5 VAE decode takes a 32 KB latent -- under the 64 KB input bound -- and
its capture held 6.2 GB after the second call, where the decode itself peaks
at 0.5 GB, against 0.77 GB for PyTorch. A capture now replays with
``keep_graph=2``, which returns each intermediate's memory after its last use
in the run. Only recording it as a device graph keeps every buffer, and the
automatic policy does that only up to ``auto_graph_replay_retain_bytes``.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
from _helpers import capability as _test_capability
import gc
import unittest
import weakref

import numpy as np

import jittor as jt
from jittor import nn


def _held_bytes():
    """Bytes the pools hold for Vars, host and every device together."""
    total = jt.core.device_memory_used(-1)
    for device in range(jt.core.get_device_count()):
        total += jt.core.device_memory_used(device)
    return total


_MIB = 1 << 20


class _Wide(nn.Module):
    """A 256-byte input whose graph allocates four 32 MiB intermediates.

    Run normally, about two of them are alive at once; a graph that keeps its
    buffers holds all four. The first one also goes through two storage views,
    which do not run and must alias their input again on every replay.
    """

    def __init__(self):
        super().__init__()
        rs = np.random.RandomState(0)
        self.up = jt.array(rs.randn(1 << 20, 8).astype("float32") * 1e-3)
        self.down = jt.array(rs.randn(8, 1 << 20).astype("float32") * 1e-3)

    def execute(self, x):
        h = jt.nn.matmul(self.up, x)
        h = h.reshape((1 << 19, 16)).reshape((1 << 20, 8))
        for _ in range(3):
            h = jt.nn.matmul(h, x)
        return jt.nn.matmul(self.down, h)

    def reference(self, x):
        h = self.up.numpy() @ x
        for _ in range(3):
            h = h @ x
        return self.down.numpy() @ h


class _Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 8)

    def execute(self, x):
        return self.l2(nn.relu(self.l1(x)))


@_test_preserve_policy(jt, "auto_graph_replay", "auto_graph_replay_retain_bytes")
class TestReplayRetention(unittest.TestCase):
    def setUp(self):
        jt.flags.auto_graph_replay = 1
        jt.flags.auto_graph_replay_retain_bytes = 16 << 20
        self.x = jt.array(np.random.RandomState(1).randn(8, 8).astype("float32"))

    def _calls(self, model, n=6):
        """Call `model` n times on a different input each time.

        Returns every answer with its input, and what the pools still hold
        afterwards beyond the parameters.
        """
        rs = np.random.RandomState(1)
        feeds = [jt.array(rs.randn(8, 8).astype("float32")) for _ in range(n)]
        for leaf in feeds + model.parameters():
            leaf.sync(True, False)
        jt.sync_all(True)
        base = _held_bytes()
        answers = []
        with jt.no_grad():
            for x in feeds:
                answers.append((x.numpy(), model(x).numpy()))
        jt.sync_all(True)
        jt.gc()
        return answers, _held_bytes() - base

    def test_changing_the_budget_releases_and_recaptures_an_existing_graph(self):
        model = _Small()
        feeds = [self.x + step / 100 for step in range(16)]
        for feed in feeds:
            feed.sync()
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0):
            expected = [model(feed).numpy().copy() for feed in feeds]
        with jt.no_grad():
            for index, feed in enumerate(feeds[:8]):
                np.testing.assert_allclose(model(feed).numpy(), expected[index],
                                           atol=1e-6, rtol=1e-5)
            old = model.__dict__["_auto_graph_replay"].replay
            if jt.flags.use_cuda:
                self.assertTrue(old._cuda_graph, old._graph_refused)
            with jt.flag_scope(auto_graph_replay_retain_bytes=1):
                for index in range(8, 12):
                    np.testing.assert_allclose(model(feeds[index]).numpy(), expected[index],
                                               atol=1e-6, rtol=1e-5)
                lowered = model.__dict__["_auto_graph_replay"].replay
                self.assertIsNot(lowered, old)
                self.assertIsNone(old._capture)
                self.assertEqual(old._cuda_graph, 0)
                self.assertEqual(lowered._max_retained_bytes, 1)
                self.assertEqual(lowered._cuda_graph, 0)
                self.assertIn("recording would keep", lowered._graph_refused)
            for index in range(12, 16):
                np.testing.assert_allclose(model(feeds[index]).numpy(), expected[index],
                                           atol=1e-6, rtol=1e-5)
            restored = model.__dict__["_auto_graph_replay"].replay
            self.assertIsNot(restored, lowered)
            self.assertEqual(restored._max_retained_bytes, 16 << 20)
            self.assertIsNone(lowered._capture)

    def test_changing_the_budget_recaptures_state_without_repeating_updates(self):
        class Cache:
            def __init__(self):
                self.value = jt.zeros((8, 8))

        class Stateful(nn.Module):
            def execute(self, x, cache=None):
                cache.value.assign((cache.value + x).tanh())
                return cache.value + 1

        model, cache = Stateful(), Cache()
        feeds = [jt.array(np.full((8, 8), step / 100, dtype=np.float32))
                 for step in range(16)]
        for feed in feeds + [cache.value]:
            feed.sync()
        expected = np.zeros((8, 8), dtype=np.float32)

        def call(index):
            nonlocal expected
            expected = np.tanh(expected + index / 100)
            np.testing.assert_allclose(model(feeds[index], cache=cache).numpy(),
                                       expected + 1, atol=1e-6, rtol=1e-5)

        def step():
            entries = model.__dict__["_auto_graph_replay"].entries.values()
            return next(entry.step for entry in entries if entry.step is not None)

        with jt.no_grad():
            for index in range(8):
                call(index)
            old = step()
            with jt.flag_scope(auto_graph_replay_retain_bytes=1):
                for index in range(8, 12):
                    call(index)
                lowered = step()
                self.assertIsNot(lowered, old)
                self.assertIsNone(old._capture)
                self.assertEqual(lowered._max_retained_bytes, 1)
                self.assertEqual(lowered._cuda_graph, 0)
            for index in range(12, 16):
                call(index)
            self.assertEqual(step()._max_retained_bytes, 16 << 20)
            self.assertIsNone(lowered._capture)

    def test_no_grad_capture_does_not_replace_a_differentiable_call(self):
        model = nn.Linear(8, 8)
        with jt.no_grad():
            for _ in range(8):
                model(self.x).sync()
        replay = model.__dict__["_auto_graph_replay"].replay
        before = dict(replay.stats)
        output = model(self.x)
        self.assertTrue(output.requires_grad)
        weight_grad, bias_grad = jt.grad(output.sum(), [model.weight, model.bias])
        np.testing.assert_allclose(weight_grad.numpy(),
                                   np.ones((8, 8), dtype=np.float32) @ self.x.numpy(),
                                   atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(bias_grad.numpy(), np.full(8, 8, dtype=np.float32))
        self.assertEqual(replay.stats, before)

    def test_captured_graph_dies_with_its_module(self):
        model = _Small()
        with jt.no_grad():
            for _ in range(8):
                model(self.x).sync()
        model_ref = weakref.ref(model)
        replay_ref = weakref.ref(model.__dict__["_auto_graph_replay"].replay)
        del model
        gc.collect()
        jt.gc()
        self.assertIsNone(model_ref())
        self.assertIsNone(replay_ref())

    def test_a_large_graph_replays_without_keeping_its_intermediates(self):
        model = _Wide()
        answers, held = self._calls(model)
        # Each intermediate is 32 MiB; the inputs and the answer are bytes.
        self.assertLess(held, _MIB, "the capture kept %.1f MiB alive" % (held / _MIB))
        state = model.__dict__["_auto_graph_replay"]
        self.assertFalse(state.give_up)
        self.assertGreaterEqual(state.replay.stats["replayed"], 3)
        # Too large to record: that would keep all of it.
        self.assertEqual(state.replay._cuda_graph, 0)
        self.assertIn("recording would keep", state.replay._graph_refused)
        # Every replay recomputed its intermediates for its own input.
        for x, got in answers:
            np.testing.assert_allclose(got, model.reference(x),
                                       rtol=1e-2, atol=1e-4)  # TF32 on CUDA

    def test_a_small_graph_is_still_replayed(self):
        # What the policy is for: a graph that keeps little and is rebuilt on
        # every call.
        model = _Small()
        answers, _ = self._calls(model)
        state = model.__dict__["_auto_graph_replay"]
        self.assertFalse(state.give_up)
        self.assertGreater(state.replay.stats["replayed"], 0)
        with jt.no_grad():
            before = jt.flags.auto_graph_replay
            jt.flags.auto_graph_replay = 0
            try:
                for x, got in answers:
                    np.testing.assert_allclose(got, model(jt.array(x)).numpy(),
                                               rtol=1e-5, atol=1e-6)
            finally:
                jt.flags.auto_graph_replay = before


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
@_test_preserve_policy(jt, "auto_graph_replay", "auto_graph_replay_retain_bytes", "use_cuda")
class TestReplayRetentionCuda(TestReplayRetention):
    def setUp(self):
        jt.flags.use_cuda = 1
        super().setUp()

    def test_a_replay_peaks_like_a_normal_call(self):
        # Kept buffers would be all four intermediates at once, 128 MiB.
        model = _Wide()
        for leaf in model.parameters():
            leaf.sync(True, False)
        jt.sync_all(True)
        base = jt.core.device_memory_used(0)
        jt.core.reset_device_memory_peak(0)
        self._calls(model)
        peak = jt.core.device_memory_peak(0) - base
        self.assertLess(peak, 100 * _MIB, "peaked %.0f MiB above the weights" % (peak / _MIB))

    def test_a_recording_is_bounded_by_what_the_call_holds_at_once(self):
        # A recording hands a block the call frees to a later allocation of the
        # same recording, so it keeps what the call holds at its most -- here
        # about two of the four 32 MiB intermediates -- not everything the call
        # was handed. Counting the latter refused it under this bound.
        jt.flags.auto_graph_replay_retain_bytes = 100 * _MIB
        model = _Wide()
        answers, _ = self._calls(model)
        replay = model.__dict__["_auto_graph_replay"].replay
        self.assertLess(replay._graph_bytes, 100 * _MIB)
        self.assertGreater(replay.stats["graph"], 0, replay._graph_refused)
        for x, got in answers:
            np.testing.assert_allclose(got, model.reference(x),
                                       rtol=1e-2, atol=1e-4)  # TF32 on CUDA


if __name__ == "__main__":
    unittest.main()
