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
from jittor._runtime.graph_replay import GraphReplay, graph_replay


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


class _GroupNormTransposed(nn.Module):
    """GroupNorm over channels-last input, the way an attention block applies it."""

    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(2, 8)

    def execute(self, x):
        return self.norm(x.transpose(0, 2, 1)).transpose(0, 2, 1)


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


class _Keywords(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 8)

    def execute(self, x=None, bias=None, scale=1.0):
        return self.l1(x) * scale + bias


class _KVCache:
    """What Transformers passes a decode step: tensors the step writes in place."""

    def __init__(self, length=6, d=8):
        self.keys = jt.zeros((length, d))
        self.scale = jt.ones((d,))
        self.length = length


class _Decoder(nn.Module):
    """One decode step: writes its row of the cache, reads all of it."""

    def __init__(self, d=8, draw=False, count=False):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.draw = draw
        self.count = count

    def execute(self, x, position, cache=None):
        h = self.l1(x) * getattr(cache, "temperature", 1.0)
        if self.draw:
            h = h + jt.rand(h.shape)
        if self.count:
            cache.calls += 1
            h = h * cache.calls
        cache.keys[position] = h[0]
        return (cache.keys.sum(0, keepdims=True) * 0.5 + h * cache.scale).tanh()


class _FlushObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def execute(self, x):
        self.seen.append(jt.flags.auto_flush_ops)
        return (x * 2 + 1).tanh()


class _Strided(nn.Module):
    """A result that is a strided view of storage the call computed."""

    def __init__(self, strided):
        super().__init__()
        self.strided = strided

    def execute(self, x):
        h = (x * 2 + 1).tanh() * 3
        if not self.strided:
            return h
        with jt.flag_scope(transpose_storage_view=1):
            return h.transpose(1, 0)


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

    def test_a_pass_through_op_follows_each_input(self):
        # GroupNorm runs through a `tape`, which shares its input's storage and
        # launches nothing. Freed between runs like any intermediate, it came
        # back in a fresh buffer nobody wrote, and every replay answered with
        # uninitialized memory.
        model = _GroupNormTransposed()
        rs = np.random.RandomState(4)
        feeds = [jt.array(rs.randn(2, 5, 8).astype("float32")) for _ in range(4)]
        jt.flags.auto_graph_replay = 0
        with jt.no_grad():
            expected = [model(x).numpy() for x in feeds]
        replay = graph_replay(model)
        for x, want in zip(feeds, expected):
            np.testing.assert_allclose(replay(x).numpy(), want, rtol=1e-5, atol=1e-5)
        self.assertIsNone(replay.refused)

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

    def test_the_automatic_policy_replays_a_keyword_call(self):
        # Transformers calls every model by keyword. The policy used to give up
        # on any keyword argument, so no Hugging Face model was ever replayed.
        model = _Keywords()
        rs = np.random.RandomState(5)
        calls = [(jt.array(rs.randn(2, 8).astype("float32")),
                  jt.array(rs.randn(8).astype("float32"))) for _ in range(3)]
        jt.flags.auto_graph_replay = 0
        with jt.no_grad():
            want = [model(x=x, bias=b, scale=2.0).numpy().copy() for x, b in calls]
        jt.flags.auto_graph_replay = 1
        with jt.no_grad():
            for _ in range(2):
                for (x, b), expected in zip(calls, want):
                    np.testing.assert_allclose(model(x=x, bias=b, scale=2.0).numpy(),
                                               expected, rtol=1e-5, atol=1e-5)
        replay = model.__dict__["_auto_graph_replay"].replay
        self.assertIsNotNone(replay)
        self.assertGreaterEqual(replay.stats["replayed"], 3)

    def test_the_automatic_policy_leaves_an_object_argument_alone(self):
        # A KV cache is the same object on every decode step while what it
        # holds changes; matched by identity, a capture would keep answering
        # for the first step. One that is handed a new tensor every step --
        # a dynamic cache -- is never captured at all.
        class _Cache:
            def __init__(self):
                self.value = jt.zeros(8)

        class _Cached(nn.Module):
            def execute(self, x, cache=None):
                return x + cache.value

        model, cache = _Cached(), _Cache()
        x = self.feed[0]
        jt.flags.auto_graph_replay = 1
        with jt.no_grad():
            for step in range(4):
                cache.value = jt.full((8,), float(step))
                np.testing.assert_allclose(model(x, cache=cache).numpy(),
                                           x.numpy() + step, rtol=1e-6)
        state = model.__dict__["_auto_graph_replay"]
        self.assertTrue(all(e.step is None for e in state.entries.values()))

    def _decode(self, model, cache, flag, steps=6, reset=False, between=None):
        """Greedy-decode-shaped calls: each output is the next input."""
        before = jt.flags.auto_graph_replay
        jt.flags.auto_graph_replay = flag
        x = self.feed[0][:1]
        outs = []
        try:
            with jt.no_grad():
                for t in range(steps):
                    if reset and t == 0:
                        cache.keys.assign(jt.zeros_like(cache.keys))
                    if between is not None:
                        between(t, cache)
                    x = model(x, jt.array([t % cache.length]), cache=cache)
                    outs.append(x.numpy().copy())
        finally:
            jt.flags.auto_graph_replay = before
        return np.concatenate(outs), cache.keys.numpy().copy()

    def _stateful_steps(self, model):
        state = model.__dict__.get("_auto_graph_replay")
        if state is None:
            return []
        return [e.step for e in state.entries.values() if e.step is not None]

    def test_the_automatic_policy_replays_a_call_that_updates_its_cache_in_place(self):
        # Transformers' static-cache decode: the same cache object every step,
        # whose tensors the step writes in place. Captured as a step, the
        # write is replayed as a state update.
        jt.set_global_seed(3)
        model = _Decoder()
        want, want_keys = self._decode(model, _KVCache(), 0, steps=12)
        got, got_keys = self._decode(model, _KVCache(), 1, steps=12)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_keys, want_keys, rtol=1e-5, atol=1e-6)
        steps = self._stateful_steps(model)
        self.assertEqual(len(steps), 1)
        self.assertIsNone(steps[0].refused)
        self.assertGreaterEqual(steps[0].stats["replayed"], 6)

    def test_a_cache_reset_between_runs_is_taken_over(self):
        # `generate` resets its static cache in place before every run; the
        # capture takes the reset tensors over instead of capturing again.
        jt.set_global_seed(3)
        model = _Decoder()
        ref_cache, cache = _KVCache(), _KVCache()
        for run in range(3):
            want, want_keys = self._decode(model, ref_cache, 0, reset=True)
            got, got_keys = self._decode(model, cache, 1, reset=True)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(got_keys, want_keys, rtol=1e-5, atol=1e-6)
        steps = self._stateful_steps(model)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].stats["captured"], 1)

    def test_a_tensor_the_cache_holds_rebound_from_outside_is_noticed(self):
        # The step reads `cache.scale` without writing it. Rebinding it from
        # outside leaves the captured graph reading the old one, unless the
        # replay notices.
        def rescale(t, cache):
            if t == 7:
                cache.scale.assign(jt.full((8,), 3.0))
        jt.set_global_seed(3)
        model = _Decoder()
        want, _ = self._decode(model, _KVCache(), 0, steps=10, between=rescale)
        got, _ = self._decode(model, _KVCache(), 1, steps=10, between=rescale)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_a_scalar_the_cache_holds_changed_from_outside_is_noticed(self):
        # The step bakes `cache.temperature` in as a constant. Set again to
        # the same value it must still replay; to another, re-capture.
        def adjust(t, cache):
            if t == 6:
                cache.temperature = float("1.0")
            if t == 8:
                cache.temperature = 0.25 * 2
        jt.set_global_seed(3)
        model = _Decoder()
        ref_cache, cache = _KVCache(), _KVCache()
        ref_cache.temperature = cache.temperature = 1.0
        want, _ = self._decode(model, ref_cache, 0, steps=11, between=adjust)
        got, _ = self._decode(model, cache, 1, steps=11, between=adjust)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        steps = self._stateful_steps(model)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].stats["captured"], 2)

    def test_a_call_that_changes_its_object_on_the_host_is_not_captured(self):
        jt.set_global_seed(3)
        model = _Decoder(count=True)
        ref_cache, cache = _KVCache(), _KVCache()
        ref_cache.calls = cache.calls = 0
        want, _ = self._decode(model, ref_cache, 0, steps=8)
        got, _ = self._decode(model, cache, 1, steps=8)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        self.assertEqual(self._stateful_steps(model), [])

    def test_a_call_with_state_that_draws_random_numbers_is_refused(self):
        # A capture would draw from its own stream, where running the call as
        # written draws from the caller's generator.
        model = _Decoder(draw=True)
        self._decode(model, _KVCache(), 1, steps=8)
        steps = self._stateful_steps(model)
        self.assertEqual(len(steps), 1)
        self.assertIn("random", steps[0].refused)
        self.assertEqual(steps[0].stats["replayed"], 0)

    def test_the_token_step_survives_the_prompt_between_runs(self):
        # `generate` calls the model on the prompt, then token by token, and
        # again on every run. One slot would throw the token step's capture
        # away at every prompt; the prompt, once a run, is not captured.
        jt.set_global_seed(3)
        model = _Decoder()
        prompt = self.feed[1]

        def run(flag, cache):
            before = jt.flags.auto_graph_replay
            jt.flags.auto_graph_replay = flag
            outs = []
            try:
                with jt.no_grad():
                    for _ in range(4):
                        cache.keys.assign(jt.zeros_like(cache.keys))
                        x = model(prompt, jt.array([0, 1]), cache=cache)[1:]
                        for t in range(2, 6):
                            x = model(x, jt.array([t]), cache=cache)
                            outs.append(x.numpy().copy())
            finally:
                jt.flags.auto_graph_replay = before
            return np.concatenate(outs)

        want = run(0, _KVCache())
        got = run(1, _KVCache())
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        steps = self._stateful_steps(model)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].stats["captured"], 1)
        self.assertGreaterEqual(steps[0].stats["replayed"], 10)

    def test_a_cache_that_is_dropped_is_not_kept_alive(self):
        # Not captured -- it changes every call -- so nothing may hold it.
        import gc
        import weakref
        model = _Decoder(count=True)
        cache = _KVCache()
        cache.calls = 0
        self._decode(model, cache, 1, steps=4)
        ref = weakref.ref(cache)
        del cache
        gc.collect()
        self.assertIsNone(ref())

    def test_the_call_is_captured_as_one_graph(self):
        # Auto-flush would launch the traced call in pieces, and each piece's
        # results would stay held for as long as the capture lives.
        model = _FlushObserver()
        before = jt.flags.auto_flush_ops
        replay = graph_replay(model, self.feed[0])
        replay(self.feed[1])
        self.assertEqual(model.seen[-1], 0)
        self.assertEqual(jt.flags.auto_flush_ops, before)

    def test_a_strided_result_is_replayed_once(self):
        x = self.feed[1]
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0):
            want = (x * 2 + 1).tanh().numpy().T * 3
        counts = []
        for strided in (False, True):
            # Through the executor, never recorded: that is the path where a
            # copy that densified built on the kept graph and re-ran it.
            replay = GraphReplay(_Strided(strided), max_retained_bytes=1)
            for f in self.feed[2:]:
                replay(f).sync()
            jt.sync_all(True)
            with jt.profile() as p:
                got = replay(x)
                got.sync()
                jt.sync_all(True)
            np.testing.assert_allclose(got.numpy(), want.T if not strided else want,
                                       rtol=1e-5, atol=1e-5)
            counts.append(len(p.result.kernel_records))
        if jt.flags.use_cuda:
            # Densified inside the graph: one copy more, not the graph again.
            self.assertEqual(counts[1], counts[0] + 1, counts)

    def test_a_private_input_copy_keeps_requires_grad(self):
        # The capture computes on copies of the inputs; a copy that asked for a
        # gradient its input did not made everything built from it ask too.
        from jittor._runtime.graph_replay import _empty_like
        frozen = jt.array(np.ones((2, 3), "float32"))
        frozen.requires_grad = False
        self.assertFalse(_empty_like(frozen).requires_grad)
        live = jt.array(np.ones((2, 3), "float32"))
        self.assertTrue(_empty_like(live).requires_grad)

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
